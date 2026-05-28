from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def forward(self, vl_features: torch.Tensor, embodiment_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        if vl_features.dim() == 2:
            vl_features = vl_features.unsqueeze(1)
        if vl_features.dim() != 3:
            raise ValueError(f"vl_features must be [B,H] or [B,S,H], got {tuple(vl_features.shape)}")

        action_latent = self.action_latent
        action_latent.set_frozen_modules_to_eval_mode()
        batch_size = vl_features.shape[0]
        device = vl_features.device
        if embodiment_id is None:
            embodiment_id = torch.zeros(batch_size, device=device, dtype=torch.long)
        else:
            embodiment_id = embodiment_id.to(device=device, dtype=torch.long)

        vl_embeds = action_latent.vl_encoder(vl_features)
        timesteps = torch.zeros(batch_size, device=device, dtype=torch.long)
        action_seed = torch.zeros(
            batch_size,
            action_latent.action_horizon,
            action_latent.action_dim,
            device=device,
            dtype=vl_embeds.dtype,
        )
        action_features = action_latent.action_encoder(action_seed, timesteps, embodiment_id)
        pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
        action_features = action_features + action_latent.position_embedding(pos_ids).unsqueeze(0)

        model_output = action_latent.perceiver_net(
            latents=action_features,
            visual_language_states=vl_embeds,
            timestep=timesteps,
        )
        pred = action_latent.action_decoder(model_output, embodiment_id)
        return pred[:, -action_latent.action_horizon :]


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


@dataclass
class Qwen35OmegaTrajectoryArguments(JoyNav_Qwen3_5OmegaSpatialForcingArguments):
    propagate_action_head_grad: bool = field(default=True)
    action_head_loss_weight: float = field(default=1.0)
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


class Qwen35OmegaTrajectoryMixin:
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

    def forward(self, *args, sf_image_tensors=None, continuous_actions=None, select_mask=None, **kwargs):
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

    def forward(self, *args, sf_image_tensors=None, continuous_actions=None, select_mask=None, **kwargs):
        self._prepare_trajectory_forward(kwargs)
        outputs = super().forward(*args, labels=None, sf_image_tensors=None, **kwargs)

        action_loss = None
        if continuous_actions is not None:
            hidden_states = self._trajectory_hidden_states(outputs)
            model_args = self._trajectory_args()
            features = hidden_states if getattr(model_args, "propagate_action_head_grad", True) else hidden_states.detach()
            selected_features = select_action_token_features(features, select_mask)
            target_actions = self._normalize_target_actions(continuous_actions, selected_features.shape[0])
            target_actions = target_actions.to(device=features.device, dtype=features.dtype)
            pred_actions = self.action_head(selected_features)
            action_loss = trajectory_mse_loss(pred_actions, target_actions)
            weighted = action_loss * float(getattr(model_args, "action_head_loss_weight", 1.0))
            outputs.loss = weighted if outputs.loss is None else outputs.loss + weighted
            outputs.action_pred = pred_actions

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

    def forward(self, *args, sf_image_tensors=None, continuous_actions=None, select_mask=None, **kwargs):
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
        return outputs
