"""VGGT-Omega aggregator wrapper for Spatial Forcing targets."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .base import BaseGeometryEncoder, GeometryEncoderConfig
from ..spatial_forcing import parse_spatial_forcing_layers


DEFAULT_VGGT_OMEGA_CHECKPOINT = (
    "/mnt/nas5/xiangchen/vlacode/vggt-omega/facebook/VGGT-Omega/vggt_omega_1b_512.pt"
)
DEFAULT_VGGT_OMEGA_TEXT_ALIGN_CHECKPOINT = (
    "/mnt/nas5/xiangchen/vlacode/vggt-omega/facebook/VGGT-Omega/vggt_omega_1b_256_text.pt"
)


@dataclass(frozen=True)
class VGGTOmegaModeConfig:
    checkpoint_path: str
    image_resolution: int
    enable_alignment: bool


def resolve_vggt_omega_mode(
    omega_mode: str,
    checkpoint_path: str = "",
    image_resolution: Optional[int] = None,
) -> VGGTOmegaModeConfig:
    if omega_mode == "512_1b":
        return VGGTOmegaModeConfig(
            checkpoint_path=checkpoint_path or DEFAULT_VGGT_OMEGA_CHECKPOINT,
            image_resolution=image_resolution or 512,
            enable_alignment=False,
        )
    if omega_mode == "text_align":
        return VGGTOmegaModeConfig(
            checkpoint_path=checkpoint_path or DEFAULT_VGGT_OMEGA_TEXT_ALIGN_CHECKPOINT,
            image_resolution=image_resolution or 256,
            enable_alignment=True,
        )
    raise ValueError(f"Unsupported omega_mode: {omega_mode}")


def select_vggt_omega_patch_tokens(
    aggregated_tokens_list: list[Optional[torch.Tensor]],
    patch_token_start: int,
    teacher_layer_spec: str,
    source_hw: Tuple[int, int],
) -> torch.Tensor:
    """Select cached Omega patch tokens and flatten them as per-image features."""
    source_h, source_w = source_hw
    expected_patches = source_h * source_w
    layer_indices = parse_spatial_forcing_layers(teacher_layer_spec, len(aggregated_tokens_list))

    layer_features = []
    for layer_idx in layer_indices:
        tokens = aggregated_tokens_list[layer_idx]
        if tokens is None:
            raise ValueError(f"VGGT-Omega layer {layer_idx} is not cached by the aggregator.")
        patch_tokens = tokens[:, :, patch_token_start:, :]
        if patch_tokens.shape[2] != expected_patches:
            raise ValueError(
                f"VGGT-Omega produced {patch_tokens.shape[2]} patch tokens, "
                f"but source_hw={source_hw} implies {expected_patches}."
            )
        layer_features.append(patch_tokens)

    features = torch.stack(layer_features, dim=0).mean(dim=0)
    batch_size, num_frames, num_patches, feature_dim = features.shape
    return features.reshape(batch_size * num_frames, num_patches, feature_dim)


class VGGTOmegaEncoder(BaseGeometryEncoder):
    """Frozen VGGT-Omega aggregator that exposes cached 3D patch tokens."""

    def __init__(self, config: GeometryEncoderConfig):
        super().__init__(config)

        from vggt_omega.models.aggregator import Aggregator

        self.patch_size = int(config.encoder_kwargs.get("patch_size", 16))
        self.embed_dim = int(config.encoder_kwargs.get("embed_dim", 1024))
        self.default_teacher_layers = str(config.encoder_kwargs.get("teacher_layers", "23"))
        self.omega_mode = str(config.encoder_kwargs.get("omega_mode", "512_1b"))
        self.mode_config = resolve_vggt_omega_mode(self.omega_mode, config.model_path or "")
        self.aggregator = Aggregator(patch_size=self.patch_size, embed_dim=self.embed_dim)

        self.load_model(self.mode_config.checkpoint_path)
        self.aggregator.eval()
        if self.freeze_encoder:
            for param in self.aggregator.parameters():
                param.requires_grad = False

    def load_model(self, model_path: str) -> None:
        state_dict = torch.load(model_path, map_location="cpu")
        if self.mode_config.enable_alignment:
            from vggt_omega.models import VGGTOmega

            model = VGGTOmega(enable_alignment=True)
            model.load_state_dict(state_dict)
            self.aggregator = model.aggregator
            del model, state_dict
            return

        aggregator_state = {
            key.removeprefix("aggregator."): value
            for key, value in state_dict.items()
            if key.startswith("aggregator.")
        }
        self.aggregator.load_state_dict(aggregator_state)
        del state_dict

    def encode(self, images: torch.Tensor, teacher_layer_spec: Optional[str] = None) -> torch.Tensor:
        if images.dim() == 4:
            images = images.unsqueeze(0)

        _, _, _, height, width = images.shape
        source_hw = (height // self.patch_size, width // self.patch_size)
        teacher_layer_spec = teacher_layer_spec or self.default_teacher_layers

        use_cuda_amp = images.is_cuda and torch.cuda.is_available()
        amp_dtype = torch.bfloat16 if use_cuda_amp and torch.cuda.is_bf16_supported() else torch.float16
        autocast_context = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_cuda_amp else nullcontext()

        with torch.no_grad():
            with autocast_context:
                aggregated_tokens_list, patch_token_start = self.aggregator(images)

        return select_vggt_omega_patch_tokens(
            aggregated_tokens_list=aggregated_tokens_list,
            patch_token_start=patch_token_start,
            teacher_layer_spec=teacher_layer_spec,
            source_hw=source_hw,
        )

    def get_feature_dim(self) -> int:
        return 2 * self.embed_dim
