"""Train-time Spatial Forcing alignment utilities."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_spatial_forcing_layers(layer_spec: str, num_hidden_layers: int) -> list[int]:
    """Parse comma-separated decoder layer indices, allowing negative indexing."""
    layers = []
    for raw in layer_spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        layer_idx = int(raw)
        if layer_idx < 0:
            layer_idx = num_hidden_layers + layer_idx
        if layer_idx < 0 or layer_idx >= num_hidden_layers:
            raise ValueError(
                f"Spatial forcing layer {raw} resolves to {layer_idx}, "
                f"outside [0, {num_hidden_layers - 1}]"
            )
        layers.append(layer_idx)
    if not layers:
        raise ValueError("At least one spatial forcing layer must be provided.")
    return layers


def cosine_alignment_loss(
    projected_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return mean `1 - cosine_similarity` over valid token pairs."""
    if projected_tokens.shape != target_tokens.shape:
        raise ValueError(
            "projected_tokens and target_tokens must have the same shape, "
            f"got {tuple(projected_tokens.shape)} and {tuple(target_tokens.shape)}"
        )

    projected_tokens = projected_tokens.float()
    target_tokens = target_tokens.float()
    projected_tokens = F.normalize(projected_tokens, dim=-1, eps=eps)
    target_tokens = F.normalize(target_tokens, dim=-1, eps=eps)
    loss = 1.0 - (projected_tokens * target_tokens).sum(dim=-1)

    if mask is not None:
        if mask.shape != loss.shape:
            raise ValueError(f"mask shape {tuple(mask.shape)} does not match loss shape {tuple(loss.shape)}")
        loss = loss[mask]
        if loss.numel() == 0:
            return projected_tokens.sum() * 0.0

    return loss.mean()


def _get_2d_sincos_pos_embed(
    height: int,
    width: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError(f"Sin-cos position embedding dimension must be divisible by 4, got {dim}")

    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    omega = torch.arange(dim // 4, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(dim // 4, 1)))

    out_y = y.reshape(-1, 1) * omega.reshape(1, -1)
    out_x = x.reshape(-1, 1) * omega.reshape(1, -1)
    pos = torch.cat([out_y.sin(), out_y.cos(), out_x.sin(), out_x.cos()], dim=-1)
    return pos.to(dtype=dtype)


def resize_spatial_features_to_grid(
    features: torch.Tensor,
    source_hw: Tuple[int, int],
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> torch.Tensor:
    """Resize per-image spatial features to Qwen's merged visual-token grid."""
    if features.ndim != 3:
        raise ValueError(f"Expected features with shape [num_images, tokens, dim], got {tuple(features.shape)}")

    source_h, source_w = source_hw
    if source_h * source_w != features.shape[1]:
        raise ValueError(
            f"source_hw={source_hw} implies {source_h * source_w} tokens, "
            f"but features have {features.shape[1]}"
        )
    if image_grid_thw.shape[0] != features.shape[0]:
        raise ValueError(
            f"image_grid_thw has {image_grid_thw.shape[0]} rows but features have {features.shape[0]} images"
        )

    resized_features = []
    for feature, grid in zip(features, image_grid_thw.to(features.device)):
        temporal, grid_h, grid_w = [int(v.item()) for v in grid]
        target_h = grid_h // spatial_merge_size
        target_w = grid_w // spatial_merge_size
        if target_h <= 0 or target_w <= 0:
            raise ValueError(f"Invalid target grid ({target_h}, {target_w}) from image_grid_thw row {grid.tolist()}")

        feature = feature.reshape(source_h, source_w, features.shape[-1]).permute(2, 0, 1).unsqueeze(0)
        feature = F.interpolate(feature.float(), size=(target_h, target_w), mode="bilinear", align_corners=False)
        feature = feature.squeeze(0).permute(1, 2, 0).to(dtype=features.dtype)
        feature = feature.reshape(target_h * target_w, features.shape[-1])
        if temporal > 1:
            feature = feature.repeat(temporal, 1)
        resized_features.append(feature)

    return torch.cat(resized_features, dim=0)


def add_spatial_positional_embedding(
    features: torch.Tensor,
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> torch.Tensor:
    """Add deterministic 2D sin-cos position embedding per image grid."""
    chunks = []
    offset = 0
    for grid in image_grid_thw.to(features.device):
        temporal, grid_h, grid_w = [int(v.item()) for v in grid]
        target_h = grid_h // spatial_merge_size
        target_w = grid_w // spatial_merge_size
        token_count = temporal * target_h * target_w
        chunk = features[offset : offset + token_count]
        pos = _get_2d_sincos_pos_embed(target_h, target_w, features.shape[-1], features.device, features.dtype)
        if temporal > 1:
            pos = pos.repeat(temporal, 1)
        chunks.append(chunk + pos)
        offset += token_count

    if offset != features.shape[0]:
        raise ValueError(f"Position embedding consumed {offset} tokens, but features contain {features.shape[0]}")
    return torch.cat(chunks, dim=0)


class SpatialForcingProjector(nn.Module):
    """Two-layer MLP used to project VLA visual tokens to geometry-feature space."""

    def __init__(self, input_dim: int, target_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or target_dim * 2
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, target_dim)
        self.initialize_weights()

    def initialize_weights(self):
        for module in (self.fc1, self.fc2):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        original_shape = visual_tokens.shape
        visual_tokens = visual_tokens.reshape(-1, original_shape[-1]).to(self.fc1.weight.dtype)
        visual_tokens = self.fc1(visual_tokens)
        visual_tokens = self.act(visual_tokens)
        visual_tokens = self.fc2(visual_tokens)
        return visual_tokens.reshape(*original_shape[:-1], -1).float()
