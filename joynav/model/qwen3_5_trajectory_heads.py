import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from transformers.feature_extraction_utils import BatchFeature
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from .action_latent.modeling_action_latent import ActionLatent, ActionLatent_Config
from .nextdit.nextdit_crossattn_traj import NextDiTCrossAttn, NextDiTCrossAttnConfig
from .qwen3_5_lm_head_sf_omega import (
    JoyNav_Qwen3_5OmegaSpatialForcingArguments,
    JoyNav_Qwen3_5OmegaSpatialForcingForCausalLM,
)


def normalize_angle_tensor(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def trajectory_mse_loss(pred_actions: torch.Tensor, target_actions: torch.Tensor) -> torch.Tensor:
    target_actions = target_actions.to(device=pred_actions.device, dtype=pred_actions.dtype)
    if pred_actions.shape != target_actions.shape:
        raise ValueError(f"trajectory shape mismatch: pred={tuple(pred_actions.shape)} target={tuple(target_actions.shape)}")
    if pred_actions.shape[-1] != 3:
        raise ValueError(f"expected x,y,yaw trajectory dim 3, got {pred_actions.shape[-1]}")

    xy_error = pred_actions[..., :2] - target_actions[..., :2]
    yaw_error = normalize_angle_tensor(pred_actions[..., 2] - target_actions[..., 2]).unsqueeze(-1)
    return torch.cat([xy_error, yaw_error], dim=-1).pow(2).mean()


def select_action_token_features(hidden_states: torch.Tensor, select_mask: torch.Tensor) -> torch.Tensor:
    if select_mask is None:
        raise ValueError("select_mask is required for trajectory supervision")
    if hidden_states.dim() != 3:
        raise ValueError(f"hidden_states must be [B,S,H], got {tuple(hidden_states.shape)}")
    if select_mask.dim() == 3 and select_mask.shape[-1] == 1:
        select_mask = select_mask.squeeze(-1)
    if select_mask.dim() != 2:
        raise ValueError(f"select_mask must be [B,S], got {tuple(select_mask.shape)}")
    if tuple(select_mask.shape) != tuple(hidden_states.shape[:2]):
        raise ValueError(
            f"select_mask shape {tuple(select_mask.shape)} does not match hidden states {tuple(hidden_states.shape[:2])}"
        )

    select_mask = select_mask.to(device=hidden_states.device, dtype=torch.bool)
    selected_per_sample = select_mask.sum(dim=1)
    if torch.any(selected_per_sample == 0):
        raise ValueError(f"expected at least one action token per sample, got {selected_per_sample.tolist()}")
    return hidden_states[select_mask]


class TrajectoryMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_points: int, point_dim: int = 3):
        super().__init__()
        self.num_points = int(num_points)
        self.point_dim = int(point_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.num_points * self.point_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.view(*x.shape[:-1], self.num_points, self.point_dim)


class TrajectoryDiTHead(nn.Module):
    def __init__(self, latent_config: ActionLatent_Config):
        super().__init__()
        self.action_latent = ActionLatent(latent_config)

    @property
    def action_horizon(self) -> int:
        return self.action_latent.action_horizon

    @property
    def action_dim(self) -> int:
        return self.action_latent.action_dim

    def _normalize_vl_features(self, vl_features: torch.Tensor) -> torch.Tensor:
        if vl_features.dim() == 2:
            vl_features = vl_features.unsqueeze(1)
        if vl_features.dim() != 3:
            raise ValueError(f"vl_features must be [B,H] or [B,S,H], got {tuple(vl_features.shape)}")
        return vl_features

    def _embodiment_id(self, batch_size: int, device, embodiment_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        if embodiment_id is None:
            return torch.zeros(batch_size, device=device, dtype=torch.long)
        return embodiment_id.to(device=device, dtype=torch.long)

    def _backbone_output(self, vl_features: torch.Tensor) -> BatchFeature:
        vl_features = self._normalize_vl_features(vl_features)
        return BatchFeature(
            data={
                "backbone_features": vl_features,
                "backbone_attention_mask": torch.ones(
                    vl_features.shape[:2],
                    device=vl_features.device,
                    dtype=torch.bool,
                ),
            }
        )

    def _action_input(
        self,
        target_actions: torch.Tensor,
        embodiment_id: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
    ) -> BatchFeature:
        if (
            target_actions.dim() != 3
            or target_actions.shape[1] != self.action_horizon
            or target_actions.shape[-1] != self.action_dim
        ):
            raise ValueError(
                f"target_actions must be [B,{self.action_horizon},{self.action_dim}], got {tuple(target_actions.shape)}"
            )
        if action_mask is None:
            action_mask = torch.ones(
                target_actions.shape[:2],
                device=target_actions.device,
                dtype=target_actions.dtype,
            )
        else:
            action_mask = action_mask.to(device=target_actions.device, dtype=target_actions.dtype)
        return BatchFeature(
            data={
                "action": target_actions,
                "action_mask": action_mask,
                "embodiment_id": self._embodiment_id(target_actions.shape[0], target_actions.device, embodiment_id),
            }
        )

    def flow_matching_loss(
        self,
        vl_features: torch.Tensor,
        target_actions: torch.Tensor,
        embodiment_id: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        vl_features = self._normalize_vl_features(vl_features)
        target_actions = target_actions.to(device=vl_features.device, dtype=vl_features.dtype)
        output = self.action_latent(
            backbone_output=self._backbone_output(vl_features),
            action_input=self._action_input(target_actions, embodiment_id=embodiment_id, action_mask=action_mask),
        )
        return output["loss"]

    @torch.no_grad()
    def forward(self, vl_features: torch.Tensor, embodiment_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        vl_features = self._normalize_vl_features(vl_features)
        action_latent = self.action_latent
        batch_size = vl_features.shape[0]
        device = vl_features.device
        output = action_latent.get_action(
            backbone_output=self._backbone_output(vl_features),
            action_input=BatchFeature(
                data={
                    "embodiment_id": self._embodiment_id(batch_size, device, embodiment_id),
                }
            ),
        )
        return output["action_pred"]


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = int(embedding_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float()
        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent * (torch.log(torch.tensor(10000.0, device=timesteps.device)) / max(half_dim, 1))
        freqs = timesteps.unsqueeze(-1) * exponent.exp()
        enc = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)
        if enc.shape[-1] < self.embedding_dim:
            enc = F.pad(enc, (0, self.embedding_dim - enc.shape[-1]))
        return enc


class TrajectoryNextDiTHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_horizon: int,
        action_dim: int = 3,
        dim: int = 384,
        layers: int = 12,
        heads: int = 6,
        kv_heads: int = 6,
        num_inference_steps: int = 10,
        num_sample_trajs: int = 1,
        guidance_scale: float = 1.0,
    ):
        super().__init__()
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.dim = int(dim)
        self.num_inference_steps = int(num_inference_steps)
        self.num_sample_trajs = int(num_sample_trajs)
        self.guidance_scale = float(guidance_scale)

        self.cond_projector = nn.Sequential(
            nn.Linear(input_dim, self.dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.dim, self.dim),
        )
        self.action_encoder = nn.Linear(self.action_dim, self.dim, bias=True)
        self.pos_encoding = SinusoidalPositionalEncoding(self.dim)
        self.action_decoder = nn.Linear(self.dim, self.action_dim, bias=True)
        self.traj_dit = NextDiTCrossAttn(
            NextDiTCrossAttnConfig(
                input_size=self.action_horizon,
                patch_size=1,
                in_channels=self.dim,
                dim=self.dim,
                n_layers=int(layers),
                n_heads=int(heads),
                n_kv_heads=int(kv_heads),
                latent_embedding_size=self.dim,
                learn_sigma=False,
            )
        )
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler()

    def _encode_condition(self, vl_features: torch.Tensor) -> torch.Tensor:
        if vl_features.dim() == 2:
            vl_features = vl_features.unsqueeze(1)
        if vl_features.dim() != 3:
            raise ValueError(f"vl_features must be [B,H] or [B,S,H], got {tuple(vl_features.shape)}")
        return self.cond_projector(vl_features)

    def _encode_actions(self, actions: torch.Tensor) -> torch.Tensor:
        action_features = self.action_encoder(actions)
        pos_ids = torch.arange(actions.shape[1], device=actions.device).reshape(1, -1).repeat(actions.shape[0], 1)
        return action_features + self.pos_encoding(pos_ids).to(device=actions.device, dtype=action_features.dtype)

    def _predict_velocity(
        self,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        action_features = self._encode_actions(actions)
        model_output = self.traj_dit(x=action_features, timestep=timesteps, z_latents=condition)
        return self.action_decoder(model_output)

    def _get_sigmas(self, timesteps: torch.Tensor, device, n_dim: int, dtype: torch.dtype) -> torch.Tensor:
        sigmas = self.noise_scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler.timesteps.to(device=device)
        timesteps = timesteps.to(device=device)
        step_indices = []
        for timestep in timesteps:
            matches = (schedule_timesteps == timestep).nonzero()
            if matches.numel() == 0:
                step_indices.append(int((schedule_timesteps - timestep).abs().argmin().item()))
            else:
                step_indices.append(int(matches.flatten()[0].item()))
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def flow_matching_loss(self, vl_features: torch.Tensor, target_actions: torch.Tensor) -> torch.Tensor:
        if (
            target_actions.dim() != 3
            or target_actions.shape[1] != self.action_horizon
            or target_actions.shape[-1] != self.action_dim
        ):
            raise ValueError(
                f"target_actions must be [B,{self.action_horizon},{self.action_dim}], got {tuple(target_actions.shape)}"
            )
        condition = self._encode_condition(vl_features)
        target_actions = target_actions.to(device=condition.device, dtype=condition.dtype)
        noise = torch.randn_like(target_actions)
        batch_size = target_actions.shape[0]
        indices = (
            torch.rand(size=(batch_size,), device=condition.device)
            * self.noise_scheduler.config.num_train_timesteps
        ).long()
        indices = indices.clamp(max=self.noise_scheduler.config.num_train_timesteps - 1)
        timesteps = self.noise_scheduler.timesteps.to(device=condition.device)[indices]
        sigmas = self._get_sigmas(timesteps, condition.device, n_dim=target_actions.dim(), dtype=target_actions.dtype)

        noisy_trajectory = (1 - sigmas) * target_actions + sigmas * noise
        velocity_pred = self._predict_velocity(noisy_trajectory, timesteps, condition)
        target = noise - target_actions
        return F.mse_loss(velocity_pred.float(), target.float())

    @torch.no_grad()
    def forward(
        self,
        vl_features: torch.Tensor,
        num_inference_steps: Optional[int] = None,
        num_sample_trajs: Optional[int] = None,
        guidance_scale: Optional[float] = None,
    ) -> torch.Tensor:
        condition = self._encode_condition(vl_features)
        device = condition.device
        dtype = condition.dtype
        batch_size = condition.shape[0]
        num_inference_steps = int(num_inference_steps or self.num_inference_steps)
        num_sample_trajs = int(num_sample_trajs or self.num_sample_trajs)
        guidance_scale = float(self.guidance_scale if guidance_scale is None else guidance_scale)

        scheduler = FlowMatchEulerDiscreteScheduler()
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        scheduler.set_timesteps(num_inference_steps, device=device, sigmas=sigmas)

        hidden_states = torch.cat([torch.zeros_like(condition), condition], dim=0)
        hidden_states = hidden_states.repeat_interleave(num_sample_trajs, dim=0)
        latents = torch.randn(
            batch_size * num_sample_trajs,
            self.action_horizon,
            self.action_dim,
            device=device,
            dtype=dtype,
        )

        for timestep in scheduler.timesteps:
            latent_features = self._encode_actions(latents)
            latent_model_input = latent_features.repeat(2, 1, 1)
            if hasattr(scheduler, "scale_model_input"):
                latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)
            model_timesteps = timestep.reshape(1).expand(latent_model_input.shape[0]).to(device=device)
            velocity_pred = self.traj_dit(
                x=latent_model_input,
                timestep=model_timesteps,
                z_latents=hidden_states,
            )
            velocity_pred = self.action_decoder(velocity_pred)
            velocity_uncond, velocity_cond = velocity_pred.chunk(2)
            velocity_pred = velocity_uncond + guidance_scale * (velocity_cond - velocity_uncond)
            latents = scheduler.step(velocity_pred, timestep, latents).prev_sample

        latents = latents.view(batch_size, num_sample_trajs, self.action_horizon, self.action_dim)
        return latents.mean(dim=1)


class OmniHeadBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.mha = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.mha(q, k, v)
        out = self.norm1(q + attn_out)
        return self.norm2(out + self.ffn(out))


class OmniActionFormerHead(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4, num_blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([OmniHeadBlock(hidden_dim, num_heads) for _ in range(num_blocks)])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            q = block(q, k, v)
        return q


class OmniTimestepEncoder(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        return self.timestep_embedder(self.time_proj(timesteps).to(dtype))


class OmniAdaLayerNorm(nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1e-5):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim * 2)
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=eps)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(F.silu(temb)).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class OmniFlowBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = OmniAdaLayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, action_states: torch.Tensor, vl_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(action_states, temb)
        self_out, _ = self.self_attn(normed, normed, normed, need_weights=False)
        action_states = action_states + self_out
        normed = self.norm2(action_states)
        cross_out, _ = self.cross_attn(normed, vl_states, vl_states, need_weights=False)
        action_states = action_states + cross_out
        action_states = action_states + self.ffn(self.norm3(action_states))
        return action_states


class OmniFlowMatchingActionHead(nn.Module):
    """OmniNav-style flow matching over delta waypoints.

    The target is a normalized delta sequence of shape ``[B, 5, 4]`` containing
    ``dx, dy, d_sin(theta), d_cos(theta)``. The transformer conditions each noisy
    action token on the full VLM hidden-state sequence, matching the OmniNav
    action-head paradigm while staying native to the Qwen3.5/Omega stack.
    """

    def __init__(
        self,
        input_dim: int,
        action_horizon: int = 5,
        action_dim: int = 4,
        hidden_dim: Optional[int] = None,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_timestep_buckets: int = 100,
        num_inference_timesteps: int = 10,
        noise_beta_alpha: float = 1.5,
        noise_beta_beta: float = 1.0,
        noise_s: float = 0.999,
        add_pos_embed: bool = True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim or input_dim)
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.num_timestep_buckets = int(num_timestep_buckets)
        self.num_inference_timesteps = int(num_inference_timesteps)
        self.add_pos_embed = bool(add_pos_embed)

        self.condition_projector = nn.Linear(self.input_dim, self.hidden_dim)
        self.action_encoder = nn.Linear(self.action_dim, self.hidden_dim)
        self.time_encoder = OmniTimestepEncoder(self.hidden_dim)
        self.blocks = nn.ModuleList(
            [OmniFlowBlock(self.hidden_dim, int(num_heads), dropout=float(dropout)) for _ in range(int(num_layers))]
        )
        self.norm_out = nn.LayerNorm(self.hidden_dim, elementwise_affine=False, eps=1e-6)
        self.out_modulation = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.action_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )
        if self.add_pos_embed:
            self.position_embedding = nn.Embedding(self.action_horizon, self.hidden_dim)
        self.noise_beta_alpha = float(noise_beta_alpha)
        self.noise_beta_beta = float(noise_beta_beta)
        self.noise_s = float(noise_s)

    def sample_time(self, batch_size: int, device, dtype: torch.dtype) -> torch.Tensor:
        beta_dist = torch.distributions.Beta(
            torch.tensor(self.noise_beta_alpha, device=device, dtype=torch.float32),
            torch.tensor(self.noise_beta_beta, device=device, dtype=torch.float32),
        )
        sample = beta_dist.sample((batch_size,)).to(dtype=dtype)
        return ((self.noise_s - sample) / self.noise_s).clamp(0.0, 1.0)

    def _time_buckets(self, t: torch.Tensor) -> torch.Tensor:
        return (t.clamp(0.0, 1.0) * self.num_timestep_buckets).long().clamp(max=self.num_timestep_buckets - 1)

    def _model(self, noisy_actions: torch.Tensor, vl_features: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        vl_states = self.condition_projector(vl_features)
        action_states = self.action_encoder(noisy_actions)
        if self.add_pos_embed:
            pos_ids = torch.arange(action_states.shape[1], device=action_states.device)
            action_states = action_states + self.position_embedding(pos_ids).unsqueeze(0)

        temb = self.time_encoder(timesteps)
        for block in self.blocks:
            action_states = block(action_states, vl_states, temb)
        shift, scale = self.out_modulation(F.silu(temb)).chunk(2, dim=-1)
        action_states = self.norm_out(action_states) * (1 + scale[:, None]) + shift[:, None]
        return self.action_decoder(action_states)

    def flow_matching_loss(self, vl_features: torch.Tensor, target_actions: torch.Tensor) -> torch.Tensor:
        target_actions = target_actions.to(device=vl_features.device, dtype=vl_features.dtype)
        noise = torch.randn_like(target_actions)
        t = self.sample_time(target_actions.shape[0], device=target_actions.device, dtype=target_actions.dtype)
        noisy_actions = (1 - t[:, None, None]) * noise + t[:, None, None] * target_actions
        velocity = target_actions - noise
        pred = self._model(noisy_actions, vl_features, self._time_buckets(t))
        return F.mse_loss(pred.float(), velocity.float(), reduction="mean")

    @torch.no_grad()
    def forward(self, vl_features: torch.Tensor, num_inference_timesteps: Optional[int] = None) -> torch.Tensor:
        num_steps = int(num_inference_timesteps or self.num_inference_timesteps)
        actions = torch.randn(
            vl_features.shape[0],
            self.action_horizon,
            self.action_dim,
            device=vl_features.device,
            dtype=vl_features.dtype,
        )
        dt = 1.0 / max(num_steps, 1)
        for i in range(num_steps):
            t = torch.full((vl_features.shape[0],), i * dt, device=vl_features.device, dtype=vl_features.dtype)
            velocity = self._model(actions, vl_features, self._time_buckets(t))
            actions = actions + dt * velocity
        return actions


@dataclass
class Qwen35OmegaTrajectoryArguments(JoyNav_Qwen3_5OmegaSpatialForcingArguments):
    propagate_action_head_grad: bool = field(default=True)
    action_head_loss_weight: float = field(default=1.0)
    stop_head_loss_weight: float = field(default=1.0)
    stop_pos_weight: float = field(default=1.0)
    trajectory_horizon: int = field(default=8)
    trajectory_dim: int = field(default=3)
    action_head_hidden_dim: Optional[int] = field(default=None)
    action_latent_layers: int = field(default=8)
    action_latent_dim: int = field(default=1536)
    action_latent_heads: int = field(default=16)
    action_num_inference_timesteps: int = field(default=4)
    nextdit_dim: int = field(default=384)
    nextdit_layers: int = field(default=12)
    nextdit_heads: int = field(default=6)
    nextdit_kv_heads: int = field(default=6)
    nextdit_num_inference_steps: int = field(default=10)
    nextdit_num_sample_trajs: int = field(default=1)
    nextdit_guidance_scale: float = field(default=1.0)
    omni_waypoint_number: int = field(default=5)
    omni_action_dim: int = field(default=4)
    omni_step_scale: float = field(default=0.3)
    omni_norm_method: str = field(default="min_max_split_arrive")
    omni_coord_scale: float = field(default=8.0)
    omni_flow_hidden_dim: Optional[int] = field(default=None)
    omni_flow_layers: int = field(default=16)
    omni_flow_heads: int = field(default=32)
    omni_flow_dropout: float = field(default=0.2)
    omni_num_inference_timesteps: int = field(default=10)
    omni_num_timestep_buckets: int = field(default=100)
    omni_noise_beta_alpha: float = field(default=1.5)
    omni_noise_beta_beta: float = field(default=1.0)
    omni_noise_s: float = field(default=0.999)
    omni_query_action_layers: int = field(default=1)
    omni_use_arrive_list: bool = field(default=True)


class Qwen35OmegaTrajectoryMixin:
    def _build_stop_head(self, hidden_size: int):
        """Binary STOP/arrival classifier on the action-token hidden state.

        Shared by all trajectory heads so STOP supervision is decoupled from the
        (x, y, yaw) regression / flow-matching head.
        """
        self.stop_head = nn.Linear(int(hidden_size), 1)
        self._init_stop_head_weights()

    def _init_stop_head_weights(self, init_bias: float = -2.0):
        # Negative bias -> low initial stop probability so the agent does not stop
        # at step 0 before training shapes the head.
        nn.init.normal_(self.stop_head.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.stop_head.bias, float(init_bias))

    def _compute_stop_loss(self, selected_features: torch.Tensor, stop_targets: torch.Tensor) -> torch.Tensor:
        stop_logits = self.stop_head(selected_features).squeeze(-1).float()
        stop_targets = stop_targets.to(device=stop_logits.device, dtype=torch.float32).reshape(-1)
        if stop_targets.shape != stop_logits.shape:
            raise ValueError(
                f"stop_targets shape {tuple(stop_targets.shape)} does not match stop_logits {tuple(stop_logits.shape)}"
            )
        pos_weight_val = float(getattr(self._trajectory_args(), "stop_pos_weight", 1.0))
        pos_weight = torch.tensor(pos_weight_val, device=stop_logits.device) if pos_weight_val != 1.0 else None
        return F.binary_cross_entropy_with_logits(stop_logits, stop_targets, pos_weight=pos_weight)

    def _predict_stop_logit(self, last_hidden: torch.Tensor) -> torch.Tensor:
        return self.stop_head(last_hidden).squeeze(-1)

    def _apply_stop_loss(self, outputs, selected_features: torch.Tensor, stop_targets):
        stop_loss = None
        if stop_targets is not None:
            stop_loss = self._compute_stop_loss(selected_features, stop_targets)
            weighted = stop_loss * float(getattr(self._trajectory_args(), "stop_head_loss_weight", 1.0))
            outputs.loss = weighted if outputs.loss is None else outputs.loss + weighted
        outputs.stop_loss = stop_loss.detach() if stop_loss is not None else None
        return stop_loss

    def post_update_model(self):
        parent_post_update = getattr(super(), "post_update_model", None)
        if parent_post_update is not None:
            parent_post_update()

        loading_info = getattr(self, "_hf_loading_info", {}) or {}
        missing_keys = loading_info.get("missing_keys", []) or []

        if getattr(self, "stop_head", None) is not None and any(
            key.startswith("stop_head.") for key in missing_keys
        ):
            self._init_stop_head_weights()

        action_missing = any(
            key.startswith("action_head.") or key.startswith("action_latent.")
            for key in missing_keys
        )
        if not action_missing:
            return

        action_latent = getattr(self, "action_latent", None)
        if action_latent is not None and hasattr(action_latent, "init_weights"):
            action_latent.init_weights()
            return

        action_head = getattr(self, "action_head", None)
        if action_head is not None:
            action_head.apply(self._init_weights)

    def _trajectory_args(self):
        return getattr(self, "model_args", self.config)

    def _trajectory_hidden_states(self, outputs):
        if outputs.hidden_states is None:
            raise ValueError("Trajectory heads require output_hidden_states=True.")
        return outputs.hidden_states[-1]

    def _prepare_trajectory_forward(self, kwargs):
        labels = kwargs.pop("labels", None)
        kwargs["output_hidden_states"] = True
        return labels

    def _add_spatial_forcing_loss(self, outputs, kwargs, sf_image_tensors):
        input_ids = kwargs.get("input_ids")
        image_grid_thw = kwargs.get("image_grid_thw")
        use_spatial_forcing = (
            self.training
            and getattr(self, "sf_enabled", False)
            and sf_image_tensors is not None
            and image_grid_thw is not None
            and input_ids is not None
        )
        spatial_forcing_loss = None
        if use_spatial_forcing:
            spatial_forcing_loss = self._compute_spatial_forcing_loss(
                hidden_states=outputs.hidden_states,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                sf_image_tensors=sf_image_tensors,
            )
            sf_loss = spatial_forcing_loss * self.sf_alpha
            outputs.loss = sf_loss if outputs.loss is None else outputs.loss + sf_loss
            debug_loss = os.environ.get("JOYNAV_TRAJ_DEBUG") == "1" or not bool(torch.isfinite(spatial_forcing_loss).detach().cpu())
            if debug_loss:
                rank = os.environ.get("RANK", "?")
                print(
                    f"[rank {rank}] trajectory spatial_forcing_loss="
                    f"{float(spatial_forcing_loss.detach().float().cpu()):.6g} "
                    f"sf_alpha={float(self.sf_alpha):.6g}",
                    flush=True,
                )
        outputs.spatial_forcing_loss = spatial_forcing_loss.detach() if spatial_forcing_loss is not None else None
        return outputs

    def _normalize_target_actions(self, continuous_actions: torch.Tensor, selected_count: int) -> torch.Tensor:
        if continuous_actions.dim() == 4:
            continuous_actions = continuous_actions.reshape(-1, continuous_actions.shape[-2], continuous_actions.shape[-1])
        if continuous_actions.dim() != 3:
            raise ValueError(f"continuous_actions must be [B,N,3], got {tuple(continuous_actions.shape)}")
        if continuous_actions.shape[0] != selected_count:
            raise ValueError(
                "continuous_actions/select_mask mismatch: "
                f"{continuous_actions.shape[0]} targets for {selected_count} selected action tokens"
            )
        return continuous_actions


class JoyNav_Qwen3_5OmegaMLPForCausalLM(
    Qwen35OmegaTrajectoryMixin,
    JoyNav_Qwen3_5OmegaSpatialForcingForCausalLM,
):
    ARGUMENT_CLASS = Qwen35OmegaTrajectoryArguments

    def __init__(self, config):
        super().__init__(config)
        self.action_chunk_size = int(getattr(config, "trajectory_horizon", getattr(config, "action_chunk_size", 8)))
        self.action_dim = int(getattr(config, "trajectory_dim", getattr(config, "action_dim", 3)))
        self.config.trajectory_horizon = self.action_chunk_size
        self.config.trajectory_dim = self.action_dim
        hidden_size = config.text_config.hidden_size
        hidden_dim = int(getattr(config, "action_head_hidden_dim", None) or hidden_size)
        self.action_head = TrajectoryMLP(
            input_dim=hidden_size,
            hidden_dim=hidden_dim,
            num_points=self.action_chunk_size,
            point_dim=self.action_dim,
        )
        self._build_stop_head(hidden_size)

    def forward(self, *args, sf_image_tensors=None, continuous_actions=None, select_mask=None, stop_targets=None, **kwargs):
        self._prepare_trajectory_forward(kwargs)
        outputs = super().forward(*args, labels=None, sf_image_tensors=None, **kwargs)

        action_loss = None
        if continuous_actions is not None:
            hidden_states = self._trajectory_hidden_states(outputs)
            model_args = self._trajectory_args()
            features = hidden_states if getattr(model_args, "propagate_action_head_grad", True) else hidden_states.detach()
            selected_features = select_action_token_features(features, select_mask)
            pred_actions = self.action_head(selected_features)
            target_actions = self._normalize_target_actions(continuous_actions, selected_features.shape[0])
            action_loss = trajectory_mse_loss(pred_actions, target_actions)
            weighted = action_loss * float(getattr(model_args, "action_head_loss_weight", 1.0))
            outputs.loss = weighted if outputs.loss is None else outputs.loss + weighted
            outputs.action_pred = pred_actions
            self._apply_stop_loss(outputs, selected_features, stop_targets)

        outputs = self._add_spatial_forcing_loss(outputs, kwargs, sf_image_tensors)
        outputs.action_loss = action_loss.detach() if action_loss is not None else None
        return outputs

    @torch.no_grad()
    def predict_action(self, *args, **kwargs):
        kwargs.pop("labels", None)
        kwargs["output_hidden_states"] = True
        outputs = super().forward(*args, labels=None, sf_image_tensors=None, **kwargs)
        hidden_states = self._trajectory_hidden_states(outputs)
        outputs.action_pred = self.action_head(hidden_states[:, -1])
        outputs.stop_logit = self._predict_stop_logit(hidden_states[:, -1])
        return outputs


class JoyNav_Qwen3_5OmegaDiTForCausalLM(
    Qwen35OmegaTrajectoryMixin,
    JoyNav_Qwen3_5OmegaSpatialForcingForCausalLM,
):
    ARGUMENT_CLASS = Qwen35OmegaTrajectoryArguments

    def __init__(self, config):
        super().__init__(config)
        self.action_chunk_size = int(getattr(config, "trajectory_horizon", getattr(config, "action_chunk_size", 8)))
        self.action_dim = int(getattr(config, "trajectory_dim", getattr(config, "action_dim", 3)))
        self.config.trajectory_horizon = self.action_chunk_size
        self.config.trajectory_dim = self.action_dim
        hidden_size = config.text_config.hidden_size
        action_latent_config = getattr(config, "action_latent_config", None)
        if isinstance(action_latent_config, ActionLatent_Config):
            latent_config = action_latent_config
        elif isinstance(action_latent_config, dict):
            latent_config = ActionLatent_Config(**action_latent_config)
        else:
            latent_config = ActionLatent_Config(
                vl_input_dim=hidden_size,
                latent_dim=int(getattr(config, "action_latent_dim", 1536)),
                heads=int(getattr(config, "action_latent_heads", 16)),
                layers=int(getattr(config, "action_latent_layers", 8)),
                action_dim=self.action_dim,
                action_horizon=self.action_chunk_size,
                num_inference_timesteps=int(getattr(config, "action_num_inference_timesteps", 4)),
            )
        latent_config.vl_input_dim = hidden_size
        latent_config.action_dim = self.action_dim
        latent_config.action_horizon = self.action_chunk_size
        self.action_latent_config = latent_config
        self.config.action_latent_config = latent_config.to_dict()
        self.action_head = TrajectoryDiTHead(latent_config)
        self.action_latent = self.action_head.action_latent
        self._build_stop_head(hidden_size)

    def forward(self, *args, sf_image_tensors=None, continuous_actions=None, select_mask=None, stop_targets=None, **kwargs):
        self._prepare_trajectory_forward(kwargs)
        outputs = super().forward(*args, labels=None, sf_image_tensors=None, **kwargs)

        action_loss = None
        if continuous_actions is not None:
            hidden_states = self._trajectory_hidden_states(outputs)
            model_args = self._trajectory_args()
            features = hidden_states if getattr(model_args, "propagate_action_head_grad", True) else hidden_states.detach()
            selected_features = select_action_token_features(features, select_mask)
            target_actions = self._normalize_target_actions(continuous_actions, selected_features.shape[0])
            action_loss = self.action_head.flow_matching_loss(selected_features, target_actions)
            debug_loss = os.environ.get("JOYNAV_TRAJ_DEBUG") == "1" or not bool(torch.isfinite(action_loss).detach().cpu())
            if debug_loss:
                rank = os.environ.get("RANK", "?")
                selected_finite = int(torch.isfinite(selected_features).sum().item())
                target_finite = int(torch.isfinite(target_actions).sum().item())
                print(
                    f"[rank {rank}] DiT action_loss={float(action_loss.detach().float().cpu()):.6g} "
                    f"selected_features={tuple(selected_features.shape)} "
                    f"finite={selected_finite}/{selected_features.numel()} "
                    f"target_actions={tuple(target_actions.shape)} "
                    f"finite={target_finite}/{target_actions.numel()}",
                    flush=True,
                )
            weighted = action_loss * float(getattr(model_args, "action_head_loss_weight", 1.0))
            outputs.loss = weighted if outputs.loss is None else outputs.loss + weighted
            self._apply_stop_loss(outputs, selected_features, stop_targets)

        outputs = self._add_spatial_forcing_loss(outputs, kwargs, sf_image_tensors)
        outputs.action_loss = action_loss.detach() if action_loss is not None else None
        return outputs

    @torch.no_grad()
    def predict_action(self, *args, **kwargs):
        kwargs.pop("labels", None)
        kwargs["output_hidden_states"] = True
        outputs = super().forward(*args, labels=None, sf_image_tensors=None, **kwargs)
        hidden_states = self._trajectory_hidden_states(outputs)
        outputs.action_pred = self.action_head(hidden_states[:, -1])
        outputs.stop_logit = self._predict_stop_logit(hidden_states[:, -1])
        return outputs


class JoyNav_Qwen3_5OmegaNextDiTForCausalLM(
    Qwen35OmegaTrajectoryMixin,
    JoyNav_Qwen3_5OmegaSpatialForcingForCausalLM,
):
    ARGUMENT_CLASS = Qwen35OmegaTrajectoryArguments

    def __init__(self, config):
        super().__init__(config)
        self.action_chunk_size = int(getattr(config, "trajectory_horizon", getattr(config, "action_chunk_size", 8)))
        self.action_dim = int(getattr(config, "trajectory_dim", getattr(config, "action_dim", 3)))
        self.config.trajectory_horizon = self.action_chunk_size
        self.config.trajectory_dim = self.action_dim
        hidden_size = config.text_config.hidden_size
        self.action_head = TrajectoryNextDiTHead(
            input_dim=hidden_size,
            action_horizon=self.action_chunk_size,
            action_dim=self.action_dim,
            dim=int(getattr(config, "nextdit_dim", 384)),
            layers=int(getattr(config, "nextdit_layers", 12)),
            heads=int(getattr(config, "nextdit_heads", 6)),
            kv_heads=int(getattr(config, "nextdit_kv_heads", 6)),
            num_inference_steps=int(getattr(config, "nextdit_num_inference_steps", 10)),
            num_sample_trajs=int(getattr(config, "nextdit_num_sample_trajs", 1)),
            guidance_scale=float(getattr(config, "nextdit_guidance_scale", 1.0)),
        )
        self._build_stop_head(hidden_size)

    def forward(self, *args, sf_image_tensors=None, continuous_actions=None, select_mask=None, stop_targets=None, **kwargs):
        self._prepare_trajectory_forward(kwargs)
        outputs = super().forward(*args, labels=None, sf_image_tensors=None, **kwargs)

        action_loss = None
        if continuous_actions is not None:
            hidden_states = self._trajectory_hidden_states(outputs)
            model_args = self._trajectory_args()
            features = hidden_states if getattr(model_args, "propagate_action_head_grad", True) else hidden_states.detach()
            selected_features = select_action_token_features(features, select_mask)
            target_actions = self._normalize_target_actions(continuous_actions, selected_features.shape[0])
            action_loss = self.action_head.flow_matching_loss(selected_features, target_actions)
            weighted = action_loss * float(getattr(model_args, "action_head_loss_weight", 1.0))
            outputs.loss = weighted if outputs.loss is None else outputs.loss + weighted
            self._apply_stop_loss(outputs, selected_features, stop_targets)

        outputs = self._add_spatial_forcing_loss(outputs, kwargs, sf_image_tensors)
        outputs.action_loss = action_loss.detach() if action_loss is not None else None
        return outputs

    @torch.no_grad()
    def predict_action(self, *args, **kwargs):
        kwargs.pop("labels", None)
        kwargs["output_hidden_states"] = True
        outputs = super().forward(*args, labels=None, sf_image_tensors=None, **kwargs)
        hidden_states = self._trajectory_hidden_states(outputs)
        outputs.action_pred = self.action_head(hidden_states[:, -1])
        outputs.stop_logit = self._predict_stop_logit(hidden_states[:, -1])
        return outputs


class JoyNav_Qwen3_5OmegaOmniForCausalLM(
    Qwen35OmegaTrajectoryMixin,
    JoyNav_Qwen3_5OmegaSpatialForcingForCausalLM,
):
    ARGUMENT_CLASS = Qwen35OmegaTrajectoryArguments

    def __init__(self, config):
        super().__init__(config)
        self.waypoint_number = int(getattr(config, "omni_waypoint_number", getattr(config, "trajectory_horizon", 5)))
        self.omni_action_dim = int(getattr(config, "omni_action_dim", 4))
        self.omni_step_scale = float(getattr(config, "omni_step_scale", 0.3))
        self.omni_norm_method = str(getattr(config, "omni_norm_method", "min_max_split_arrive"))
        self.omni_coord_scale = float(getattr(config, "omni_coord_scale", 8.0))
        self.config.trajectory_horizon = self.waypoint_number
        self.config.trajectory_dim = 3
        self.config.omni_waypoint_number = self.waypoint_number
        self.config.omni_action_dim = self.omni_action_dim
        hidden_size = config.text_config.hidden_size
        self.action_head = OmniFlowMatchingActionHead(
            input_dim=hidden_size,
            action_horizon=self.waypoint_number,
            action_dim=self.omni_action_dim,
            hidden_dim=getattr(config, "omni_flow_hidden_dim", None),
            num_layers=int(getattr(config, "omni_flow_layers", 16)),
            num_heads=int(getattr(config, "omni_flow_heads", 32)),
            dropout=float(getattr(config, "omni_flow_dropout", 0.2)),
            num_timestep_buckets=int(getattr(config, "omni_num_timestep_buckets", 100)),
            num_inference_timesteps=int(getattr(config, "omni_num_inference_timesteps", 10)),
            noise_beta_alpha=float(getattr(config, "omni_noise_beta_alpha", 1.5)),
            noise_beta_beta=float(getattr(config, "omni_noise_beta_beta", 1.0)),
            noise_s=float(getattr(config, "omni_noise_s", 0.999)),
        )
        self.query_action = nn.Parameter(torch.empty(1, 1, hidden_size))
        query_layers = int(getattr(config, "omni_query_action_layers", 2))
        if query_layers <= 1:
            self.query_multihead_attn = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
            self.query_multihead_multi_attn = None
        else:
            self.query_multihead_attn = None
            self.query_multihead_multi_attn = OmniActionFormerHead(hidden_size, num_heads=4, num_blocks=query_layers)
        arrive_dim = self.waypoint_number if bool(getattr(config, "omni_use_arrive_list", True)) else 1
        self.arrive_predictor = nn.Linear(hidden_size, arrive_dim)
        self._init_omni_weights()

    def _init_omni_weights(self):
        nn.init.normal_(self.query_action, mean=0.0, std=0.02)
        nn.init.normal_(self.arrive_predictor.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.arrive_predictor.bias, -2.0)

    def post_update_model(self):
        parent_post_update = getattr(super(), "post_update_model", None)
        if parent_post_update is not None:
            parent_post_update()
        missing_keys = (getattr(self, "_hf_loading_info", {}) or {}).get("missing_keys", []) or []
        if any(key.startswith("action_head.") for key in missing_keys):
            self.action_head.apply(self._init_weights)
        if any(key.startswith("arrive_predictor.") or key == "query_action" for key in missing_keys):
            self._init_omni_weights()

    def _action_feature(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query_action = self.query_action.expand(hidden_states.shape[0], -1, -1)
        if self.query_multihead_attn is not None:
            action_feature, _ = self.query_multihead_attn(query_action, hidden_states, hidden_states)
        else:
            action_feature = self.query_multihead_multi_attn(query_action, hidden_states, hidden_states)
        return action_feature.squeeze(1)

    def _norm_from_config(self, device, dtype):
        norm = getattr(self.config, "omni_norm", None)
        if isinstance(norm, dict) and "min" in norm and "max" in norm:
            return {
                "min": torch.tensor(norm["min"], device=device, dtype=dtype),
                "max": torch.tensor(norm["max"], device=device, dtype=dtype),
            }
        return None

    def _coerce_norm(self, norm, device, dtype, batch_size: int):
        if norm is None:
            norm = self._norm_from_config(device, dtype)
        elif isinstance(norm, list) and norm:
            norm = norm[0]
        if isinstance(norm, dict) and "min" in norm and "max" in norm:
            min_vals = norm["min"]
            max_vals = norm["max"]
            if not torch.is_tensor(min_vals):
                min_vals = torch.tensor(min_vals, device=device, dtype=dtype)
            else:
                min_vals = min_vals.to(device=device, dtype=dtype)
            if not torch.is_tensor(max_vals):
                max_vals = torch.tensor(max_vals, device=device, dtype=dtype)
            else:
                max_vals = max_vals.to(device=device, dtype=dtype)
            if min_vals.dim() == 2:
                min_vals = min_vals.unsqueeze(0).expand(batch_size, -1, -1)
            if max_vals.dim() == 2:
                max_vals = max_vals.unsqueeze(0).expand(batch_size, -1, -1)
            return min_vals, max_vals
        return None, None

    def _action_norm(self, norm, action_dim: int, device, dtype, batch_size: int):
        min_vals, max_vals = self._coerce_norm(norm, device, dtype, batch_size)
        if min_vals is None or max_vals is None:
            return None, None
        return min_vals[..., :action_dim], max_vals[..., :action_dim]

    def _waypoints_to_delta_actions(self, gt_waypoints, gt_heading_angles=None):
        gt_waypoints = gt_waypoints[:, : self.waypoint_number, :]
        xy = gt_waypoints[..., :2]
        if gt_heading_angles is not None:
            angle = gt_heading_angles[:, : self.waypoint_number].to(device=gt_waypoints.device, dtype=gt_waypoints.dtype)
        elif gt_waypoints.shape[-1] >= 3:
            angle = gt_waypoints[..., 2]
        else:
            angle = torch.zeros_like(xy[..., 0])
        absolute = torch.cat([xy, torch.sin(angle).unsqueeze(-1), torch.cos(angle).unsqueeze(-1)], dim=-1)
        delta = torch.zeros_like(absolute)
        delta[:, 0] = absolute[:, 0]
        delta[:, 1:] = absolute[:, 1:] - absolute[:, :-1]
        return delta

    def _normalize_delta_actions(self, delta_actions: torch.Tensor, norm=None) -> torch.Tensor:
        min_vals, max_vals = self._action_norm(
            norm,
            delta_actions.shape[-1],
            delta_actions.device,
            delta_actions.dtype,
            delta_actions.shape[0],
        )
        if self.omni_norm_method in {"min_max", "min_max_split_arrive"}:
            if min_vals is None or max_vals is None:
                raise ValueError(f"omni_norm with min/max is required for {self.omni_norm_method}")
            return ((delta_actions - min_vals) / (max_vals - min_vals + 1e-8)) * 2 - 1
        if self.omni_norm_method != "coord_scale":
            raise ValueError(f"Unsupported omni_norm_method: {self.omni_norm_method}")
        scale = torch.tensor(
            [self.omni_coord_scale, self.omni_coord_scale, 1.0, 1.0],
            device=delta_actions.device,
            dtype=delta_actions.dtype,
        )
        return delta_actions / scale

    def _denormalize_delta_actions(self, delta_actions: torch.Tensor, norm=None) -> torch.Tensor:
        min_vals, max_vals = self._action_norm(
            norm,
            delta_actions.shape[-1],
            delta_actions.device,
            delta_actions.dtype,
            delta_actions.shape[0],
        )
        if self.omni_norm_method in {"min_max", "min_max_split_arrive"}:
            if min_vals is None or max_vals is None:
                raise ValueError(f"omni_norm with min/max is required for {self.omni_norm_method}")
            return ((delta_actions + 1) / 2) * (max_vals - min_vals + 1e-8) + min_vals
        if self.omni_norm_method != "coord_scale":
            raise ValueError(f"Unsupported omni_norm_method: {self.omni_norm_method}")
        scale = torch.tensor(
            [self.omni_coord_scale, self.omni_coord_scale, 1.0, 1.0],
            device=delta_actions.device,
            dtype=delta_actions.dtype,
        )
        return delta_actions * scale

    def _decode_waypoint_prediction(self, normalized_delta: torch.Tensor, norm=None):
        delta = self._denormalize_delta_actions(normalized_delta, norm=norm)
        absolute = torch.cumsum(delta, dim=1)
        return absolute[..., :2], absolute[..., 2], absolute[..., 3]

    def forward(
        self,
        *args,
        sf_image_tensors=None,
        gt_waypoints=None,
        gt_heading_angles=None,
        arrive=None,
        arrive_list=None,
        input_waypoints=None,
        step_scale=None,
        norm=None,
        action_former=True,
        drop_arrive_loss=None,
        train=True,
        train_branch=None,
        **kwargs,
    ):
        kwargs.pop("labels", None)
        kwargs["output_hidden_states"] = True
        outputs = super().forward(*args, labels=None, sf_image_tensors=None, **kwargs)
        hidden_states = self._trajectory_hidden_states(outputs)

        if gt_waypoints is not None and torch.is_tensor(gt_waypoints):
            delta_actions = self._waypoints_to_delta_actions(gt_waypoints, gt_heading_angles=gt_heading_angles)
            target_actions = self._normalize_delta_actions(delta_actions, norm=norm)
            action_loss = self.action_head.flow_matching_loss(hidden_states, target_actions)
            action_feature = self._action_feature(hidden_states)
            arrive_pred = self.arrive_predictor(action_feature)
            arrive_loss = None
            if arrive_list is not None and arrive_pred.shape[-1] == self.waypoint_number:
                arrive_loss = F.binary_cross_entropy_with_logits(
                    arrive_pred.float(),
                    arrive_list.to(device=arrive_pred.device, dtype=torch.float32),
                )
            elif arrive is not None:
                arrive_loss = F.binary_cross_entropy_with_logits(
                    arrive_pred.float().reshape(arrive.shape[0], -1)[:, :1],
                    arrive.to(device=arrive_pred.device, dtype=torch.float32).reshape(-1, 1),
                )
            total_loss = action_loss * float(getattr(self._trajectory_args(), "action_head_loss_weight", 1.0))
            if arrive_loss is not None:
                total_loss = total_loss + arrive_loss * float(getattr(self._trajectory_args(), "stop_head_loss_weight", 1.0))
            outputs.loss = total_loss if outputs.loss is None else outputs.loss + total_loss
            outputs.action_loss = action_loss.detach()
            outputs.arrive_loss = arrive_loss.detach() if arrive_loss is not None else None
            outputs.arrive_pred = arrive_pred
        else:
            outputs.action_loss = None
            outputs.arrive_loss = None

        outputs = self._add_spatial_forcing_loss(outputs, kwargs, sf_image_tensors)
        if train is False:
            wp_pred, sin_pred, cos_pred = self._decode_waypoint_prediction(self.action_head(hidden_states), norm=norm)
            arrive_pred = self.arrive_predictor(self._action_feature(hidden_states))
            return wp_pred, arrive_pred, sin_pred, cos_pred
        return outputs

    @torch.no_grad()
    def predict_waypoints(self, *args, norm=None, **kwargs):
        kwargs.pop("labels", None)
        kwargs["output_hidden_states"] = True
        outputs = super().forward(*args, labels=None, sf_image_tensors=None, **kwargs)
        hidden_states = self._trajectory_hidden_states(outputs)
        wp_pred, sin_pred, cos_pred = self._decode_waypoint_prediction(self.action_head(hidden_states), norm=norm)
        outputs.wp_pred = wp_pred
        outputs.arrive_pred = self.arrive_predictor(self._action_feature(hidden_states))
        outputs.sin_angle = sin_pred
        outputs.cos_angle = cos_pred
        return outputs
