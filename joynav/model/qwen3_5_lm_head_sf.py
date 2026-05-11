from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Optional

import torch

from .base_argument import BaseArguments
from .geometry_encoder import DepthAnythingEncoder, GeometryEncoderConfig
from .qwen3_5_lm_head import JoyNav_Qwen3_5ForCausalLM, QWEN3_5_AVAILABLE
from .spatial_forcing import (
    SpatialForcingProjector,
    add_spatial_positional_embedding,
    cosine_alignment_loss,
    parse_spatial_forcing_layers,
    resize_spatial_features_to_grid,
)


DEFAULT_DA2_CHECKPOINT = "joynav/model/geometry_encoder/depth_anything_v2_vitl.pth"


@dataclass
class JoyNav_Qwen3_5SpatialForcingArguments(BaseArguments):
    sf_enabled: bool = field(default=True, metadata={"help": "Enable Spatial Forcing alignment loss."})
    sf_alpha: float = field(default=0.1, metadata={"help": "Weight for the Spatial Forcing alignment loss."})
    sf_align_layers: str = field(default="18", metadata={"help": "Comma-separated hidden-state indices for SF, negative allowed."})
    sf_geometry_encoder_path: str = field(
        default=DEFAULT_DA2_CHECKPOINT,
        metadata={"help": "Depth Anything V2 checkpoint used as frozen spatial target encoder."},
    )
    sf_target_dim: int = field(default=1024, metadata={"help": "Depth Anything V2 patch feature dimension."})
    sf_projector_hidden_dim: Optional[int] = field(default=None)
    sf_add_pos_embed: bool = field(default=True)


if not QWEN3_5_AVAILABLE:

    class JoyNav_Qwen3_5SpatialForcingForCausalLM(JoyNav_Qwen3_5ForCausalLM):
        ARGUMENT_CLASS = JoyNav_Qwen3_5SpatialForcingArguments

else:

    class JoyNav_Qwen3_5SpatialForcingForCausalLM(JoyNav_Qwen3_5ForCausalLM):
        ARGUMENT_CLASS = JoyNav_Qwen3_5SpatialForcingArguments
        _keys_to_ignore_on_save = [r"spatial_forcing_geometry_encoder\..*"]
        _keys_to_ignore_on_load_missing = [r"spatial_forcing_geometry_encoder\..*"]

        def __init__(self, config):
            super().__init__(config)
            self.sf_enabled = bool(getattr(config, "sf_enabled", True))
            self.sf_alpha = float(getattr(config, "sf_alpha", 0.1))
            self.sf_target_dim = int(getattr(config, "sf_target_dim", 1024))
            self.sf_add_pos_embed = bool(getattr(config, "sf_add_pos_embed", True))
            self.sf_projector_hidden_dim = getattr(config, "sf_projector_hidden_dim", None)
            self.sf_align_layer_spec = getattr(config, "sf_align_layers", "18")
            self.spatial_forcing_projector = SpatialForcingProjector(
                input_dim=config.text_config.hidden_size,
                target_dim=self.sf_target_dim,
                hidden_dim=self.sf_projector_hidden_dim,
            ) if self.sf_enabled else None
            self.__dict__["spatial_forcing_geometry_encoder"] = None
            self.__dict__["_sf_debug_forward_count"] = 0

        def post_update_model(self):
            loading_info = getattr(self, "_hf_loading_info", {})
            missing_keys = loading_info.get("missing_keys", [])
            missing_projector = any(key.startswith("spatial_forcing_projector.") for key in missing_keys)
            if self.spatial_forcing_projector is not None and missing_projector:
                self.spatial_forcing_projector.initialize_weights()

        def state_dict(self, *args, **kwargs):
            state_dict = super().state_dict(*args, **kwargs)
            if bool(getattr(self.config, "sf_save_geometry_encoder", False)):
                return state_dict
            return {
                key: value
                for key, value in state_dict.items()
                if not key.startswith("spatial_forcing_geometry_encoder.")
            }

        def _resolve_sf_hidden_state_indices(self, num_hidden_states: int) -> list[int]:
            return parse_spatial_forcing_layers(self.sf_align_layer_spec, num_hidden_layers=num_hidden_states)

        def _get_geometry_encoder(self, device: torch.device) -> DepthAnythingEncoder:
            if self.spatial_forcing_geometry_encoder is not None:
                return self.spatial_forcing_geometry_encoder

            ckpt_path = Path(getattr(self.config, "sf_geometry_encoder_path", DEFAULT_DA2_CHECKPOINT))
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Depth Anything V2 checkpoint not found at {ckpt_path}.")

            encoder_config = GeometryEncoderConfig(
                encoder_type="da2",
                model_path=str(ckpt_path),
                freeze_encoder=True,
                out_hidden_size=self.sf_target_dim,
                spatial_merge_size=getattr(self.config.vision_config, "spatial_merge_size", 2),
            )
            encoder = DepthAnythingEncoder(encoder_config).to(device)
            encoder.eval()
            for param in encoder.parameters():
                param.requires_grad = False
            self.__dict__["spatial_forcing_geometry_encoder"] = encoder
            return encoder

        def _build_spatial_targets(
            self,
            sf_image_tensors,
            image_grid_thw: torch.LongTensor,
            device: torch.device,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            encoder = self._get_geometry_encoder(device)
            spatial_merge_size = getattr(self.config.vision_config, "spatial_merge_size", 2)

            if isinstance(sf_image_tensors, (list, tuple)):
                targets = []
                for image_tensor, grid_thw in zip(sf_image_tensors, image_grid_thw):
                    image_tensor = image_tensor.to(device=device)
                    if image_tensor.dim() == 3:
                        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
                    elif image_tensor.dim() == 4:
                        image_tensor = image_tensor.unsqueeze(0)
                    source_hw = (image_tensor.shape[-2] // 14, image_tensor.shape[-1] // 14)
                    with torch.no_grad():
                        da2_features = encoder.encode(image_tensor).reshape(1, source_hw[0] * source_hw[1], self.sf_target_dim)
                    targets.append(
                        resize_spatial_features_to_grid(
                            da2_features.to(dtype=dtype),
                            source_hw=source_hw,
                            image_grid_thw=grid_thw.to(device).unsqueeze(0),
                            spatial_merge_size=spatial_merge_size,
                        )
                    )
                target_features = torch.cat(targets, dim=0)
            else:
                sf_image_tensors = sf_image_tensors.to(device=device)
                if sf_image_tensors.dim() == 4:
                    sf_image_tensors = sf_image_tensors.unsqueeze(0)

                with torch.no_grad():
                    source_hw = (sf_image_tensors.shape[-2] // 14, sf_image_tensors.shape[-1] // 14)
                    da2_features = encoder.encode(sf_image_tensors).reshape(-1, source_hw[0] * source_hw[1], self.sf_target_dim)

                target_features = resize_spatial_features_to_grid(
                    da2_features.to(dtype=dtype),
                    source_hw=source_hw,
                    image_grid_thw=image_grid_thw.to(device),
                    spatial_merge_size=spatial_merge_size,
                )
            if self.sf_add_pos_embed:
                target_features = add_spatial_positional_embedding(
                    target_features,
                    image_grid_thw=image_grid_thw.to(device),
                    spatial_merge_size=spatial_merge_size,
                )
            return target_features

        def _compute_spatial_forcing_loss(
            self,
            hidden_states: tuple[torch.Tensor, ...],
            input_ids: torch.LongTensor,
            image_grid_thw: torch.LongTensor,
            sf_image_tensors,
        ) -> torch.Tensor:
            image_mask = input_ids == self.config.image_token_id
            visual_token_count = int(image_mask.sum().item())
            if visual_token_count == 0:
                return next(self.spatial_forcing_projector.parameters()).sum() * 0.0

            first_hidden = hidden_states[-1]
            target_features = self._build_spatial_targets(
                sf_image_tensors=sf_image_tensors,
                image_grid_thw=image_grid_thw,
                device=first_hidden.device,
                dtype=first_hidden.dtype,
            )
            if target_features.shape[0] != visual_token_count:
                raise ValueError(
                    "Spatial Forcing target/image-token mismatch: "
                    f"{target_features.shape[0]} DA2 target tokens vs {visual_token_count} Qwen image tokens."
                )

            losses = []
            for layer_idx in self._resolve_sf_hidden_state_indices(len(hidden_states)):
                visual_tokens = hidden_states[layer_idx][image_mask]
                projected_tokens = self.spatial_forcing_projector(visual_tokens)
                losses.append(cosine_alignment_loss(projected_tokens, target_features))
            return torch.stack(losses).mean()

        def forward(self, *args, sf_image_tensors=None, **kwargs):
            input_ids = kwargs.get("input_ids")
            labels = kwargs.get("labels")
            image_grid_thw = kwargs.get("image_grid_thw")
            use_spatial_forcing = (
                self.training
                and self.sf_enabled
                and labels is not None
                and sf_image_tensors is not None
                and image_grid_thw is not None
                and input_ids is not None
            )

            if use_spatial_forcing:
                kwargs["output_hidden_states"] = True

            outputs = super().forward(*args, **kwargs)

            spatial_forcing_loss = None
            if use_spatial_forcing:
                spatial_forcing_loss = self._compute_spatial_forcing_loss(
                    hidden_states=outputs.hidden_states,
                    input_ids=input_ids,
                    image_grid_thw=image_grid_thw,
                    sf_image_tensors=sf_image_tensors,
                )
                outputs.loss = spatial_forcing_loss * self.sf_alpha if outputs.loss is None else outputs.loss + spatial_forcing_loss * self.sf_alpha
                if os.environ.get("JOYNAV_SF_DEBUG", "0") == "1" and int(os.environ.get("LOCAL_RANK", "0")) == 0:
                    if self.__dict__["_sf_debug_forward_count"] < 10:
                        print(
                            "[SpatialForcing] "
                            f"layer={self.sf_align_layer_spec} "
                            f"align_loss={float(spatial_forcing_loss.detach()):.6f} "
                            f"alpha={self.sf_alpha:.6f} "
                            f"total_loss={float(outputs.loss.detach()):.6f}",
                            flush=True,
                        )
                    self.__dict__["_sf_debug_forward_count"] += 1

            outputs.spatial_forcing_loss = spatial_forcing_loss.detach() if spatial_forcing_loss is not None else None
            return outputs
