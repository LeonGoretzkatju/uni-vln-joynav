from dataclasses import dataclass, field

import torch

from .geometry_encoder import GeometryEncoderConfig
from .geometry_encoder.vggt_omega_encoder import VGGTOmegaEncoder, resolve_vggt_omega_mode
from .qwen3_5_lm_head_sf import (
    JoyNav_Qwen3_5SpatialForcingArguments,
    JoyNav_Qwen3_5SpatialForcingForCausalLM,
)
from .spatial_forcing import add_spatial_positional_embedding, resize_spatial_features_to_grid


@dataclass
class JoyNav_Qwen3_5OmegaSpatialForcingArguments(JoyNav_Qwen3_5SpatialForcingArguments):
    sf_geometry_encoder_path: str = field(
        default="",
        metadata={"help": "Optional VGGT-Omega checkpoint override. Defaults depend on omega_mode."},
    )
    sf_target_dim: int = field(default=2048, metadata={"help": "VGGT-Omega cached patch token dimension."})
    sf_teacher_layers: str = field(default="23", metadata={"help": "Cached VGGT-Omega aggregator layers to align."})
    sf_add_pos_embed: bool = field(default=False)
    omega_mode: str = field(default="512_1b", metadata={"help": "VGGT-Omega mode: 512_1b or text_align."})


class JoyNav_Qwen3_5OmegaSpatialForcingForCausalLM(JoyNav_Qwen3_5SpatialForcingForCausalLM):
    ARGUMENT_CLASS = JoyNav_Qwen3_5OmegaSpatialForcingArguments

    def __init__(self, config):
        super().__init__(config)
        self.sf_teacher_layers = str(getattr(config, "sf_teacher_layers", "23"))
        self.omega_mode = str(getattr(config, "omega_mode", "512_1b"))

    def _get_geometry_encoder(self, device: torch.device) -> VGGTOmegaEncoder:
        if self.spatial_forcing_geometry_encoder is not None:
            return self.spatial_forcing_geometry_encoder

        encoder_config = GeometryEncoderConfig(
            encoder_type="vggt_omega",
            model_path=resolve_vggt_omega_mode(
                self.omega_mode,
                getattr(self.config, "sf_geometry_encoder_path", ""),
            ).checkpoint_path,
            freeze_encoder=True,
            out_hidden_size=self.sf_target_dim,
            encoder_kwargs={
                "patch_size": 16,
                "embed_dim": self.sf_target_dim // 2,
                "teacher_layers": self.sf_teacher_layers,
                "omega_mode": self.omega_mode,
            },
        )
        encoder = VGGTOmegaEncoder(encoder_config).to(device)
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False
        self.__dict__["spatial_forcing_geometry_encoder"] = encoder
        return encoder

    def _encode_omega_sequence(
        self,
        encoder: VGGTOmegaEncoder,
        image_sequence: torch.Tensor,
        image_grid_thw: torch.LongTensor,
        dtype: torch.dtype,
        spatial_merge_size: int,
    ) -> torch.Tensor:
        if image_sequence.dim() == 4:
            image_sequence = image_sequence.unsqueeze(0)

        source_hw = (
            image_sequence.shape[-2] // encoder.patch_size,
            image_sequence.shape[-1] // encoder.patch_size,
        )
        with torch.no_grad():
            omega_features = encoder.encode(image_sequence, teacher_layer_spec=self.sf_teacher_layers)

        if self.omega_mode == "text_align_force_qwen":
            return omega_features.to(dtype=dtype).reshape(-1, omega_features.shape[-1])

        return resize_spatial_features_to_grid(
            omega_features.to(dtype=dtype),
            source_hw=source_hw,
            image_grid_thw=image_grid_thw,
            spatial_merge_size=spatial_merge_size,
        )

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
            grid_offset = 0
            for image_sequence in sf_image_tensors:
                image_sequence = image_sequence.to(device=device)
                num_frames = image_sequence.shape[0] if image_sequence.dim() == 4 else image_sequence.shape[1]
                grids = image_grid_thw[grid_offset : grid_offset + num_frames].to(device)
                targets.append(
                    self._encode_omega_sequence(
                        encoder=encoder,
                        image_sequence=image_sequence,
                        image_grid_thw=grids,
                        dtype=dtype,
                        spatial_merge_size=spatial_merge_size,
                    )
                )
                grid_offset += num_frames
            target_features = torch.cat(targets, dim=0)
        else:
            sf_image_tensors = sf_image_tensors.to(device=device)
            target_features = self._encode_omega_sequence(
                encoder=encoder,
                image_sequence=sf_image_tensors,
                image_grid_thw=image_grid_thw.to(device),
                dtype=dtype,
                spatial_merge_size=spatial_merge_size,
            )

        if self.sf_add_pos_embed:
            target_features = add_spatial_positional_embedding(
                target_features,
                image_grid_thw=image_grid_thw.to(device),
                spatial_merge_size=spatial_merge_size,
            )
        return target_features
