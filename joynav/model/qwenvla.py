"""Qwen-VLA: Qwen3.5-VL backbone + 1.15B single-stream DiT flow-matching action expert.

Faithful reproduction of the Qwen-VLA technical report (arXiv:2605.30280,
https://qwen.ai/blog?id=qwenvla, https://github.com/QwenLM/Qwen-VLA):

Architecture (paper Section 2.2):
  - The action expert is a single-stream DiT-style flow-matching policy. It
    concatenates (projected) VLM hidden states with a noisy action chunk into ONE
    sequence and processes them through joint self-attention with AdaLN(-Zero)
    timestep conditioning and multi-section RoPE aligned with the backbone.
  - The reference expert has ~1.15B parameters. The per-component counts in the
    paper pin down the geometry exactly (at the reference Qwen3.5-4B backbone,
    VLM hidden 2560):
        16 DiT blocks x 70.8M  = 1.13B   -> hidden 1536, AdaLN-Zero Linear(h,6h)
                                            (6h^2), attention qkv+o (4h^2), and a
                                            SwiGLU MLP with intermediate 10240
                                            (3*10240*1536): 70.79M per block.
        action projection MLPs   4.9M    -> K->h and h->K two-layer MLPs (hidden h).
        VLM->DiT linear          3.9M    -> Linear(2560, 1536).
        timestep embedding       2.8M    -> sinusoidal 256 -> Linear(256,h) -> Linear(h,h).
        output AdaLN modulation  4.7M    -> Linear(h, 2h) on the final norm.
    All sizes stay at the reference values by default; only the VLM->DiT linear
    input dim follows the actual backbone (Qwen3.5-0.8B here, hidden 1024,
    chosen because this environment has 24 GB GPUs; the paper uses Qwen3.5-4B).

Unified action representation (paper Section 2.4):
  - Targets are Y in R^{H x K} with a fixed horizon H and fixed channel dim K. A
    control mode uses the leading c <= K channels; the rest are zero padded. A
    binary mask M in {0,1}^{H x K} excludes padded channels/steps from the loss.
  - Navigation follows the VLN convention: (dx, dy, dtheta) per waypoint
    (c = 3), horizon 8 waypoints per chunk (paper Section 4.1).
  - Actions are normalized per dataset with 1st/99th quantile statistics mapped
    linearly to [-1, 1] and clipped (paper eq. 5).

Flow matching (paper Section 2.5):
  - Y_tau = (1 - tau) Y0 + tau Y1 with Y1 ~ N(0, I); the expert predicts the
    conditional velocity (Y1 - Y0). The loss uses two-level averaging: per-step
    masked MSE per active channel (eq. 1), then a uniform mean over the c active
    channels (eq. 2). Inference integrates Euler steps from tau=1 to tau=0.
  - Timestep distribution p(tau) (paper Section 5.2.1): Sigmoid-Normal
    (logit-normal, peaked at intermediate noise) for Stage I T2A; Beta
    (concentrated toward the clean end, pi0-style alpha=1.5, beta=1.0, s=0.999)
    for Stage II CPT and Stage III SFT.

Training stages (paper Section 3.1) are selected by the train script:
  Stage I  (T2A): freeze the VLM, train only the DiT, withhold images.
  Stage II (CPT): unfreeze both modules, mixed multimodal data.
  Stage III(SFT): joint fine-tuning; loss = lambda_act * L_act + lambda_vl * L_vl
                  with lambda_vl = 0.1, lambda_act = 1.0 (paper Section 4.1).
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import TimestepEmbedding, Timesteps

from .base_argument import BaseArguments
from .qwen3_5_lm_head import JoyNav_Qwen3_5ForCausalLM


@dataclass
class QwenVLAArguments(BaseArguments):
    # Unified action-and-trajectory representation (paper Section 2.4).
    qwenvla_action_horizon: int = field(default=8, metadata={"help": "Prediction horizon H (8 waypoints/chunk for VLN, 16 for manipulation)."})
    qwenvla_action_channels: int = field(default=32, metadata={"help": "Fixed channel dim K shared across control modes (zero-padding projection)."})
    # 1.15B DiT action expert (paper Section 2.2 parameter table).
    qwenvla_dit_hidden: int = field(default=1536)
    qwenvla_dit_layers: int = field(default=16)
    qwenvla_dit_heads: int = field(default=16)
    qwenvla_dit_mlp_dim: int = field(default=10240)
    qwenvla_dit_dropout: float = field(default=0.0)
    # Flow matching (paper Sections 2.5 and 5.2.1).
    qwenvla_num_inference_steps: int = field(default=10, metadata={"help": "Euler integration steps from tau=1 to tau=0."})
    qwenvla_time_dist: str = field(default="beta", metadata={"help": "'sigmoid_normal' at T2A, 'beta' at CPT/SFT."})
    qwenvla_beta_alpha: float = field(default=1.5)
    qwenvla_beta_beta: float = field(default=1.0)
    qwenvla_noise_s: float = field(default=0.999)
    # Joint objective L = lambda_act * L_act + lambda_vl * L_vl (paper eq. 4, Section 4.1).
    # The training stage itself is selected via the dataset's qwenvla_stage plus
    # the tune_mm_* freezing flags and qwenvla_time_dist (see train-qwenvla.sh).
    qwenvla_lambda_act: float = field(default=1.0)
    qwenvla_lambda_vl: float = field(default=0.1)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class QwenVLAMultiSectionRope(nn.Module):
    """Multi-section (t/h/w) RoPE aligned with the backbone's mRoPE layout.

    The backbone supplies 3-row position ids [3, B, S] (temporal/height/width,
    via Qwen3_5Model.get_rope_index). The expert's rotary half-dim is split into
    three sections proportional to the backbone's `mrope_section`, and section i
    reads its angles from position row i — the same scheme as Qwen mRoPE, scaled
    to the expert head dim.
    """

    def __init__(self, head_dim: int, mrope_section: Optional[List[int]] = None, theta: float = 10000.0):
        super().__init__()
        self.head_dim = int(head_dim)
        half = self.head_dim // 2
        mrope_section = list(mrope_section or [1, 1, 1])
        total = float(sum(mrope_section))
        sections = [int(round(part / total * half)) for part in mrope_section]
        sections[-1] = half - sum(sections[:-1])
        self.sections = sections
        self.half_dim = half
        self.theta = float(theta)

    def _inv_freq(self, device) -> torch.Tensor:
        # Computed on the fly (not a registered buffer): transformers 5.x builds
        # models on the meta device, and non-persistent buffers in custom modules
        # would be materialized as uninitialized memory after from_pretrained.
        half = max(self.half_dim, 1)
        return 1.0 / (self.theta ** (torch.arange(0, self.half_dim, dtype=torch.float32, device=device) / half))

    def forward(self, position_ids: torch.Tensor):
        # position_ids: [3, B, S] -> cos/sin [B, S, head_dim]
        freqs = position_ids[..., None].float() * self._inv_freq(position_ids.device)  # [3,B,S,half]
        chunks = torch.split(freqs, self.sections, dim=-1)
        merged = torch.cat([chunk[i % 3] for i, chunk in enumerate(chunks)], dim=-1)  # [B,S,half]
        emb = torch.cat([merged, merged], dim=-1)
        return emb.cos(), emb.sin()


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [B, heads, S, head_dim]; cos/sin: [B, S, head_dim]
    cos = cos[:, None, :, :].to(q.dtype)
    sin = sin[:, None, :, :].to(q.dtype)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class QwenVLASwiGLU(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class QwenVLADiTBlock(nn.Module):
    """Single-stream DiT block: joint self-attention over [VLM states; noisy
    action tokens] with AdaLN-Zero timestep conditioning and multi-section RoPE.

    Per-block parameters at h=1536, mlp=10240: 4h^2 (attn) + 6h^2 (AdaLN-Zero
    modulation) + 3*10240*1536 (SwiGLU) = 70.8M — the paper's reported size.
    """

    def __init__(self, hidden_dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}")
        self.num_heads = int(num_heads)
        self.head_dim = hidden_dim // num_heads
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.attn_out = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = QwenVLASwiGLU(hidden_dim, mlp_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, temb: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                attn_bias: Optional[torch.Tensor]) -> torch.Tensor:
        batch, seq_len, hidden = x.shape
        shift1, scale1, gate1, shift2, scale2, gate2 = (
            self.adaLN_modulation(temb)[:, None, :].chunk(6, dim=-1)
        )

        normed = modulate(self.norm1(x), shift1, scale1)
        qkv = self.qkv(normed).view(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))  # [B, heads, S, hd]
        q, k = apply_rope(q, k, cos, sin)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        attn = attn.transpose(1, 2).reshape(batch, seq_len, hidden)
        x = x + gate1 * self.dropout(self.attn_out(attn))

        x = x + gate2 * self.dropout(self.mlp(modulate(self.norm2(x), shift2, scale2)))
        return x


class QwenVLAActionExpert(nn.Module):
    """The ~1.15B Qwen-VLA DiT flow-matching action expert (paper Section 2.2)."""

    def __init__(
        self,
        vlm_hidden: int,
        action_horizon: int = 8,
        action_channels: int = 32,
        hidden_dim: int = 1536,
        num_layers: int = 16,
        num_heads: int = 16,
        mlp_dim: int = 10240,
        dropout: float = 0.0,
        num_inference_steps: int = 10,
        time_dist: str = "beta",
        beta_alpha: float = 1.5,
        beta_beta: float = 1.0,
        noise_s: float = 0.999,
        mrope_section: Optional[List[int]] = None,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.action_horizon = int(action_horizon)
        self.action_channels = int(action_channels)
        self.hidden_dim = int(hidden_dim)
        self.num_inference_steps = int(num_inference_steps)
        self.time_dist = str(time_dist)
        self.beta_alpha = float(beta_alpha)
        self.beta_beta = float(beta_beta)
        self.noise_s = float(noise_s)

        # Linear layer that transforms VLM hidden states to the DiT channel dim (3.9M @ 4B).
        self.vlm_projector = nn.Linear(int(vlm_hidden), self.hidden_dim)
        # Action projection MLPs between raw action dim and DiT latent space (4.9M),
        # shared zero-padding design (paper Section 5.2.2): one encoder/decoder for
        # all embodiments over the padded K-dim action vector.
        self.action_in_proj = nn.Sequential(
            nn.Linear(self.action_channels, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.action_out_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.action_channels),
        )
        # Timestep embedding (2.8M): sinusoidal 256 -> MLP(256->h->h).
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=self.hidden_dim)
        self.blocks = nn.ModuleList(
            [QwenVLADiTBlock(self.hidden_dim, num_heads, mlp_dim, dropout=dropout) for _ in range(int(num_layers))]
        )
        # Output AdaLN modulation (4.7M): Linear(h, 2h) on the final norm.
        self.norm_out = nn.LayerNorm(self.hidden_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_out = nn.Sequential(nn.SiLU(), nn.Linear(self.hidden_dim, self.hidden_dim * 2))
        self.rope = QwenVLAMultiSectionRope(
            self.hidden_dim // int(num_heads), mrope_section=mrope_section, theta=rope_theta
        )
        self.init_weights()

    def init_weights(self):
        def _basic(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic)
        # AdaLN-Zero: zero-init each block's modulation so every block starts as
        # identity, and zero-init the velocity output layer (DiT convention).
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.adaLN_out[-1].weight)
        nn.init.zeros_(self.adaLN_out[-1].bias)
        nn.init.zeros_(self.action_out_proj[-1].weight)
        nn.init.zeros_(self.action_out_proj[-1].bias)

    # ------------------------------------------------------------------ #
    # timestep sampling (paper Section 5.2.1)
    # ------------------------------------------------------------------ #
    def sample_tau(self, batch_size: int, device, dtype: torch.dtype) -> torch.Tensor:
        if self.time_dist == "sigmoid_normal":
            # Sigmoid-Normal (logit-normal): peaks at intermediate noise levels; T2A.
            tau = torch.sigmoid(torch.randn(batch_size, device=device))
        elif self.time_dist == "beta":
            # Beta concentrated toward the clean end (tau=0 is clean); CPT/SFT.
            beta = torch.distributions.Beta(
                torch.tensor(self.beta_alpha, device=device),
                torch.tensor(self.beta_beta, device=device),
            )
            tau = (1.0 - beta.sample((batch_size,))) * self.noise_s
        else:
            raise ValueError(f"Unsupported qwenvla_time_dist: {self.time_dist}")
        return tau.clamp(0.0, 1.0).to(dtype)

    # ------------------------------------------------------------------ #
    # joint single-stream forward
    # ------------------------------------------------------------------ #
    def _joint_positions(self, vl_position_ids: Optional[torch.Tensor], batch: int, vl_len: int, device):
        if vl_position_ids is not None and vl_position_ids.dim() == 3 and vl_position_ids.shape[0] >= 3:
            vl_pos = vl_position_ids[-3:].to(device)
        else:
            vl_pos = torch.arange(vl_len, device=device)[None, None, :].expand(3, batch, -1)
        next_pos = vl_pos.amax(dim=(0, 2), keepdim=False) + 1  # [B]
        action_pos = next_pos[None, :, None] + torch.arange(self.action_horizon, device=device)[None, None, :]
        action_pos = action_pos.expand(3, -1, -1)
        return torch.cat([vl_pos, action_pos], dim=-1)  # [3, B, S+H]

    def _velocity(
        self,
        noisy_actions: torch.Tensor,           # [B, H, K]
        vl_states: torch.Tensor,               # [B, S, vlm_hidden]
        tau: torch.Tensor,                     # [B]
        vl_attention_mask: Optional[torch.Tensor] = None,   # [B, S]
        vl_position_ids: Optional[torch.Tensor] = None,     # [3, B, S]
    ) -> torch.Tensor:
        param_dtype = self.vlm_projector.weight.dtype
        vl_states = vl_states.to(param_dtype)
        noisy_actions = noisy_actions.to(param_dtype)
        batch, vl_len = vl_states.shape[0], vl_states.shape[1]
        device = vl_states.device

        cond = self.vlm_projector(vl_states)
        act = self.action_in_proj(noisy_actions)
        x = torch.cat([cond, act], dim=1)  # [B, S+H, h]

        temb = self.timestep_embedder(self.time_proj(tau.float() * 1000.0).to(param_dtype))

        positions = self._joint_positions(vl_position_ids, batch, vl_len, device)
        cos, sin = self.rope(positions)

        deep_debug = os.environ.get("JOYNAV_QWENVLA_DEBUG") == "2"
        if deep_debug:
            def _fin(name, tensor):
                print(
                    f"[qwenvla.vel] {name}: finite={bool(torch.isfinite(tensor).all())} "
                    f"absmax={float(tensor.detach().float().abs().max()):.4g} dtype={tensor.dtype}",
                    flush=True,
                )
            _fin("vl_states", vl_states); _fin("cond", cond); _fin("act", act)
            _fin("temb", temb); _fin("cos", cos); _fin("sin", sin)

        attn_bias = None
        if vl_attention_mask is not None:
            valid = vl_attention_mask.to(device=device, dtype=torch.bool)
            joint_valid = torch.cat(
                [valid, torch.ones(batch, self.action_horizon, device=device, dtype=torch.bool)], dim=1
            )
            attn_bias = torch.zeros(batch, 1, 1, joint_valid.shape[1], device=device, dtype=param_dtype)
            attn_bias = attn_bias.masked_fill(~joint_valid[:, None, None, :], torch.finfo(param_dtype).min)

        for idx, block in enumerate(self.blocks):
            x = block(x, temb, cos, sin, attn_bias)
            if deep_debug:
                _fin(f"block{idx}", x)

        x_act = x[:, -self.action_horizon:]
        shift, scale = self.adaLN_out(temb)[:, None, :].chunk(2, dim=-1)
        x_act = modulate(self.norm_out(x_act), shift, scale)
        return self.action_out_proj(x_act)  # [B, H, K]

    # ------------------------------------------------------------------ #
    # training objective (paper eqs. 1-2)
    # ------------------------------------------------------------------ #
    def flow_matching_loss(
        self,
        vl_states: torch.Tensor,
        target_actions: torch.Tensor,          # [B, H, K], already normalized to [-1, 1]
        action_mask: torch.Tensor,             # [B, H, K] validity mask M
        vl_attention_mask: Optional[torch.Tensor] = None,
        vl_position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target_actions = target_actions.to(device=vl_states.device)
        action_mask = action_mask.to(device=vl_states.device, dtype=torch.float32)
        noise = torch.randn_like(target_actions)
        tau = self.sample_tau(target_actions.shape[0], target_actions.device, target_actions.dtype)
        noisy = (1.0 - tau[:, None, None]) * target_actions + tau[:, None, None] * noise
        pred = self._velocity(noisy, vl_states, tau, vl_attention_mask, vl_position_ids)
        velocity_target = noise - target_actions  # (Y1 - Y0)

        # Two-level averaging: per-channel masked MSE over time (eq. 1), then a
        # uniform mean over the active channels (eq. 2).
        sq_err = (pred.float() - velocity_target.float()).pow(2) * action_mask
        per_channel = sq_err.sum(dim=1) / action_mask.sum(dim=1).clamp_min(1.0)      # [B, K]
        active = (action_mask.sum(dim=1) > 0).float()                                # [B, K]
        per_sample = (per_channel * active).sum(dim=-1) / active.sum(dim=-1).clamp_min(1.0)
        return per_sample.mean()

    # ------------------------------------------------------------------ #
    # inference: Euler integration from tau=1 to tau=0 (paper Section 2.5)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(
        self,
        vl_states: torch.Tensor,
        num_inference_steps: Optional[int] = None,
        vl_attention_mask: Optional[torch.Tensor] = None,
        vl_position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        steps = int(num_inference_steps or self.num_inference_steps)
        batch = vl_states.shape[0]
        actions = torch.randn(
            batch, self.action_horizon, self.action_channels,
            device=vl_states.device, dtype=torch.float32,
        )
        dt = 1.0 / max(steps, 1)
        for i in range(steps):
            tau = torch.full((batch,), 1.0 - i * dt, device=vl_states.device, dtype=torch.float32)
            velocity = self._velocity(actions, vl_states, tau, vl_attention_mask, vl_position_ids)
            actions = actions - dt * velocity.float()  # dY/dtau = (Y1 - Y0)
        return actions


class JoyNav_QwenVLAForCausalLM(JoyNav_Qwen3_5ForCausalLM):
    """Qwen3.5 backbone + Qwen-VLA DiT flow-matching action expert."""

    ARGUMENT_CLASS = QwenVLAArguments

    def __init__(self, config):
        super().__init__(config)
        defaults = QwenVLAArguments()
        for name in (
            "qwenvla_action_horizon", "qwenvla_action_channels", "qwenvla_dit_hidden",
            "qwenvla_dit_layers", "qwenvla_dit_heads", "qwenvla_dit_mlp_dim",
            "qwenvla_dit_dropout", "qwenvla_num_inference_steps", "qwenvla_time_dist",
            "qwenvla_beta_alpha", "qwenvla_beta_beta", "qwenvla_noise_s",
            "qwenvla_lambda_act", "qwenvla_lambda_vl",
        ):
            setattr(self.config, name, getattr(config, name, getattr(defaults, name)))

        text_config = config.text_config
        rope_scaling = getattr(text_config, "rope_scaling", None) or {}
        mrope_section = rope_scaling.get("mrope_section") or [1, 1, 1]
        rope_theta = float(rope_scaling.get("rope_theta") or getattr(text_config, "rope_theta", 10000.0) or 10000.0)

        self.action_expert = QwenVLAActionExpert(
            vlm_hidden=text_config.hidden_size,
            action_horizon=int(self.config.qwenvla_action_horizon),
            action_channels=int(self.config.qwenvla_action_channels),
            hidden_dim=int(self.config.qwenvla_dit_hidden),
            num_layers=int(self.config.qwenvla_dit_layers),
            num_heads=int(self.config.qwenvla_dit_heads),
            mlp_dim=int(self.config.qwenvla_dit_mlp_dim),
            dropout=float(self.config.qwenvla_dit_dropout),
            num_inference_steps=int(self.config.qwenvla_num_inference_steps),
            time_dist=str(self.config.qwenvla_time_dist),
            beta_alpha=float(self.config.qwenvla_beta_alpha),
            beta_beta=float(self.config.qwenvla_beta_beta),
            noise_s=float(self.config.qwenvla_noise_s),
            mrope_section=mrope_section,
            rope_theta=rope_theta,
        )

    def post_update_model(self):
        parent_post_update = getattr(super(), "post_update_model", None)
        if parent_post_update is not None:
            parent_post_update()
        missing_keys = (getattr(self, "_hf_loading_info", {}) or {}).get("missing_keys", []) or []
        if any(key.startswith("action_expert.") for key in missing_keys):
            self.action_expert.init_weights()

    # ------------------------------------------------------------------ #
    # quantile normalization (paper eq. 5)
    # ------------------------------------------------------------------ #
    def _norm_from_config(self, device):
        norm = getattr(self.config, "qwenvla_norm", None)
        if isinstance(norm, dict) and "q01" in norm and "q99" in norm:
            return {
                "q01": torch.tensor(norm["q01"], device=device, dtype=torch.float32),
                "q99": torch.tensor(norm["q99"], device=device, dtype=torch.float32),
            }
        return None

    def _coerce_norm(self, norm, device, batch_size: int):
        if norm is None:
            norm = self._norm_from_config(device)
        if not (isinstance(norm, dict) and "q01" in norm and "q99" in norm):
            raise ValueError("Qwen-VLA quantile norm (q01/q99) is required; bake it via training or pass norm=...")
        q01, q99 = norm["q01"], norm["q99"]
        q01 = q01 if torch.is_tensor(q01) else torch.tensor(q01)
        q99 = q99 if torch.is_tensor(q99) else torch.tensor(q99)
        q01 = q01.to(device=device, dtype=torch.float32)
        q99 = q99.to(device=device, dtype=torch.float32)
        if q01.dim() == 1:
            q01 = q01[None, None, :].expand(batch_size, 1, -1)
        elif q01.dim() == 2:
            q01 = q01[:, None, :]
        if q99.dim() == 1:
            q99 = q99[None, None, :].expand(batch_size, 1, -1)
        elif q99.dim() == 2:
            q99 = q99[:, None, :]
        return q01, q99

    def _normalize_actions(self, actions: torch.Tensor, norm=None) -> torch.Tensor:
        q01, q99 = self._coerce_norm(norm, actions.device, actions.shape[0])
        scaled = 2.0 * (actions.float() - q01) / (q99 - q01).clamp_min(1e-8) - 1.0
        return scaled.clamp(-1.0, 1.0)

    def _denormalize_actions(self, actions: torch.Tensor, norm=None) -> torch.Tensor:
        q01, q99 = self._coerce_norm(norm, actions.device, actions.shape[0])
        return (actions.float().clamp(-1.0, 1.0) + 1.0) / 2.0 * (q99 - q01) + q01

    # ------------------------------------------------------------------ #
    # multi-section RoPE alignment with the backbone
    # ------------------------------------------------------------------ #
    _rope_fallback_warned = False

    def _warn_rope_fallback(self, reason):
        # Loud one-time warning: the expert's multi-section RoPE must use the
        # same position source at train and eval time; a silent fallback to
        # sequential positions on only one side would shift the input
        # distribution the expert was trained on.
        if not JoyNav_QwenVLAForCausalLM._rope_fallback_warned:
            JoyNav_QwenVLAForCausalLM._rope_fallback_warned = True
            print(f"[qwenvla] WARNING: falling back to sequential expert RoPE positions ({reason})", flush=True)

    def _expert_position_ids(self, input_ids, mm_token_type_ids, image_grid_thw, video_grid_thw, attention_mask):
        if input_ids is None:
            self._warn_rope_fallback("no input_ids")
            return None
        try:
            if mm_token_type_ids is None:
                image_token_id = getattr(self.config, "image_token_id", None)
                if image_token_id is None:
                    self._warn_rope_fallback("no mm_token_type_ids and no image_token_id")
                    return None
                mm_token_type_ids = (input_ids == int(image_token_id)).int()
            position_ids, _ = self.model.get_rope_index(
                input_ids,
                mm_token_type_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )
            if position_ids.dim() == 3 and position_ids.shape[0] >= 3:
                return position_ids[-3:]
            self._warn_rope_fallback(f"unexpected position_ids shape {tuple(position_ids.shape)}")
        except Exception as exc:  # fall back to sequential positions in the expert
            self._warn_rope_fallback(f"get_rope_index raised: {exc}")
        return None

    # ------------------------------------------------------------------ #
    # forward / inference
    # ------------------------------------------------------------------ #
    def forward(
        self,
        *args,
        qwenvla_actions=None,        # [B, H, K] raw (unnormalized) action targets
        qwenvla_action_mask=None,    # [B, H, K] validity mask M
        norm=None,                   # {"q01": [K] or [B,K], "q99": ...}
        labels=None,
        **kwargs,
    ):
        lambda_act = float(getattr(self.config, "qwenvla_lambda_act", 1.0))
        lambda_vl = float(getattr(self.config, "qwenvla_lambda_vl", 0.1))

        input_ids = kwargs.get("input_ids")
        attention_mask = kwargs.get("attention_mask")
        mm_token_type_ids = kwargs.get("mm_token_type_ids")
        image_grid_thw = kwargs.get("image_grid_thw")
        video_grid_thw = kwargs.get("video_grid_thw")

        # Guard against an all-masked label tensor (CE over zero targets is NaN).
        use_vl_loss = labels is not None and lambda_vl > 0 and bool((labels != -100).any())
        kwargs["output_hidden_states"] = True
        outputs = super().forward(*args, labels=labels if use_vl_loss else None, **kwargs)

        vl_loss = outputs.loss if use_vl_loss else None
        action_loss = None
        if qwenvla_actions is not None:
            if outputs.hidden_states is None:
                raise ValueError("Qwen-VLA requires output_hidden_states=True")
            hidden_states = outputs.hidden_states[-1]
            target = self._normalize_actions(qwenvla_actions.to(hidden_states.device), norm=norm)
            if qwenvla_action_mask is None:
                qwenvla_action_mask = torch.ones_like(target)
            vl_position_ids = self._expert_position_ids(
                input_ids, mm_token_type_ids, image_grid_thw, video_grid_thw, attention_mask
            )
            action_loss = self.action_expert.flow_matching_loss(
                hidden_states,
                target,
                qwenvla_action_mask,
                vl_attention_mask=attention_mask,
                vl_position_ids=vl_position_ids,
            )

        total = None
        if action_loss is not None:
            total = lambda_act * action_loss
        if vl_loss is not None:
            total = lambda_vl * vl_loss if total is None else total + lambda_vl * vl_loss
        if total is not None:
            outputs.loss = total
        outputs.action_loss = action_loss.detach() if action_loss is not None else None
        outputs.vl_loss = vl_loss.detach() if vl_loss is not None else None

        if (action_loss is not None) and (
            os.environ.get("JOYNAV_QWENVLA_DEBUG") == "1"
            or not bool(torch.isfinite(action_loss).detach().cpu())
        ):
            rank = os.environ.get("RANK", "?")
            vl_repr = f"{float(vl_loss.detach().float().cpu()):.6g}" if vl_loss is not None else "None"
            print(
                f"[rank {rank}] qwenvla action_loss={float(action_loss.detach().float().cpu()):.6g} "
                f"vl_loss={vl_repr} tau_dist={self.config.qwenvla_time_dist}",
                flush=True,
            )
        return outputs

    @torch.no_grad()
    def predict_actions(self, *args, norm=None, num_inference_steps=None, **kwargs):
        kwargs.pop("labels", None)
        kwargs["output_hidden_states"] = True

        input_ids = kwargs.get("input_ids")
        attention_mask = kwargs.get("attention_mask")
        mm_token_type_ids = kwargs.get("mm_token_type_ids")
        image_grid_thw = kwargs.get("image_grid_thw")
        video_grid_thw = kwargs.get("video_grid_thw")

        outputs = super().forward(*args, labels=None, **kwargs)
        hidden_states = outputs.hidden_states[-1]
        vl_position_ids = self._expert_position_ids(
            input_ids, mm_token_type_ids, image_grid_thw, video_grid_thw, attention_mask
        )
        normalized = self.action_expert.predict(
            hidden_states,
            num_inference_steps=num_inference_steps,
            vl_attention_mask=attention_mask,
            vl_position_ids=vl_position_ids,
        )
        outputs.action_pred = self._denormalize_actions(normalized, norm=norm)
        return outputs
