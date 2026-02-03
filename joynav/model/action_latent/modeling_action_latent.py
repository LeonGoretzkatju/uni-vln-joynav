import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature
from .module_utils import TimestepEncoder, CategorySpecificMLP, MultiEmbodimentActionEncoder, count_module_params
from .perceiver_module import PerceiverAttentionBlock
from .attention_utils import SelfAttentionTransformer


class ActionLatent_Config(PretrainedConfig):
    keys_to_ignore_at_inference = ["past_key_values"]
    def __init__(
        self,
        # PerceiverNet time embedding configuration
        time_channel: int = 320,               # Input channel dimension for timestep encoder
        time_embedding_dim: int = 768,         # Output dimension of timestep embedding

        # PerceiverNet core dimensions
        latent_dim: int = 1536,                # Dimension of latent representations in Perceiver blocks
        vl_input_dim: int = 2560,              # Input dimension of visual-language features, 2048 -> 2560 for qwen3-VL-4B
        heads: int = 16,                       # Number of attention heads in PerceiverAttentionBlock
        layers: int = 8,                       # Number of PerceiverAttentionBlock layers
        output_dim: int = 1024,                # Output dimension of the PerceiverNet

        # Weight initialization
        initializer_range=0.02,

        # Multi-embodiment (robot category) configuration
        action_dim: int = 5,                   # Dimension of raw action space
        action_hidden_dim: int = 1024,         # Hidden dimension for action/state MLP encoders/decoders

        # Visual-language self-attention configuration
        vl_heads: int = 32,                    # Number of attention heads in VL self-attention transformer
        vl_dropout: float = 0.2,               # Dropout rate in VL self-attention layers
        vl_final_dropout: bool = True,         # Whether to apply final dropout in VL transformer
        vl_self_layers: int = 4,               # Number of layers in VL self-attention transformer

        # Sequence and timestep configuration
        max_seq_len: int = 1024,               # Maximum sequence length for positional embeddings
        num_timestep_buckets: int = 1000,      # Number of buckets for timestep discretization

        # Noise distribution parameters (for flow-matching processes)
        noise_beta_alpha: float = 1.5,         # Alpha parameter for Beta distribution (noise generation)
        noise_beta_beta: float = 1.0,          # Beta parameter for Beta distribution (noise generation)
        noise_s: float = 0.999,                # Scaling factor for noise schedule

        # Fine-tuning flags
        tune_projector: bool = True,           # Whether to fine-tune action/state projection layers
        tune_perceiver: bool = True,           # Whether to fine-tune the PerceiverNet

        # Inference
        action_horizon: int = 5,
        num_inference_timesteps: int = 4,
        **kwargs,
    ):
        # Perceiver time embedding parameters
        # head_dim = embed_dim(action_dim) // num_heads
        self.time_channel = time_channel
        self.time_embedding_dim = time_embedding_dim

        # Perceiver core parameters
        self.latent_dim = latent_dim
        self.vl_input_dim = vl_input_dim
        self.heads = heads
        self.layers = layers
        self.output_dim = output_dim

        # Initialization parameter
        self.initializer_range = initializer_range

        # Multi-embodiment parameters
        self.action_dim = action_dim
        self.action_hidden_dim = action_hidden_dim

        # Visual-language transformer parameters
        self.vl_heads = vl_heads
        assert self.vl_input_dim % vl_heads == 0
        self.vl_head_dim = self.vl_input_dim // vl_heads
        self.vl_dropout = vl_dropout
        self.vl_final_dropout = vl_final_dropout
        self.vl_self_layers = vl_self_layers

        # Sequence and timestep parameters
        self.max_seq_len = max_seq_len
        self.num_timestep_buckets = num_timestep_buckets

        # Noise beta distribution parameters
        self.noise_beta_alpha = noise_beta_alpha
        self.noise_beta_beta = noise_beta_beta
        self.noise_s = noise_s

        # Fine-tuning configuration
        self.tune_projector = tune_projector
        self.tune_perceiver = tune_perceiver

        # Inference
        self.action_horizon = action_horizon
        self.num_inference_timesteps = num_inference_timesteps

        super().__init__(**kwargs)


class PerceiverNet(nn.Module):
    """
    Perceiver network for processing latent representations with visual-language conditioning
    and timestep awareness. Designed to fuse multi-modal inputs while incorporating temporal information.
    """
    config_class = ActionLatent_Config
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: ActionLatent_Config,
    ):
        super().__init__()
        self.vl_input_dim = config.vl_input_dim
        self.latent_dim = config.latent_dim
        self.output_dim = config.output_dim

        self.time_encoder = TimestepEncoder(
            time_channel=config.time_channel,
            time_embedding_dim=config.time_embedding_dim,
        )

        self.time_aware_linear = nn.Linear(
            config.time_embedding_dim, self.latent_dim, bias=True
        )

        if self.vl_input_dim is not None:
            self.proj_in = nn.Linear(self.vl_input_dim, self.latent_dim)
        else:
            self.proj_in = None

        # Stack of Perceiver attention blocks for multi-step feature fusion
        self.perceiver_blocks = nn.Sequential(
            *[
                PerceiverAttentionBlock(
                    d_model=self.latent_dim,
                    n_heads=config.heads,
                    time_embedding_dim=config.time_embedding_dim,
                )
                for _ in range(config.layers)
            ]
        )

        if self.output_dim is not None:
            self.proj_out = nn.Sequential(
                nn.Linear(self.latent_dim, self.output_dim),
                nn.LayerNorm(self.output_dim)
            )
        else:
            self.proj_out = None

        count_module_params(self, name="PerceiverNet")

    def forward(
        self,
        latents: torch.Tensor,  # Shape: (B, T, D)
        visual_language_states: torch.Tensor,  # Shape: (B, S, D)
        timestep: Optional[torch.LongTensor] = None,
    ):
        time_embedding = self.time_encoder(timestep)

        latents = latents + self.time_aware_linear(
            torch.nn.functional.silu(time_embedding)
        )

        if self.proj_in is not None:
            visual_language_states = self.proj_in(visual_language_states)

        for l_block in self.perceiver_blocks:
            latents = l_block(
                x=visual_language_states,
                latents=latents,
                timestep_embedding=time_embedding,
            )

        if self.proj_out is not None:
            latents = self.proj_out(latents)

        return latents


class ActionLatent(nn.Module):
    config_class = ActionLatent_Config
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: ActionLatent_Config,
    ):
        super().__init__()
        self.config = config
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        # Core perceiver network for fusion of latents, VL features, and timesteps
        self.perceiver_net = PerceiverNet(config)

        # Encoder for robot state with category-specific processing
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=config.action_dim,
            hidden_size=config.latent_dim,
            num_embodiments=1,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=1,
            input_dim=config.action_hidden_dim,
            hidden_dim=config.action_hidden_dim,
            output_dim=config.action_dim,
        )

        # count_module_params(self.state_encoder, name="state_encoder")
        count_module_params(self.action_encoder, name="action_encoder")
        count_module_params(self.action_decoder, name="action_decoder")

        vl_self_attn_configs = dict(
            attention_head_dim=config.vl_head_dim,
            dropout=config.vl_dropout,
            final_dropout=config.vl_final_dropout,
            num_attention_heads=config.vl_heads,
            num_layers=config.vl_self_layers,
            positional_embeddings=None,
        )

        # Encoder for visual-language features with layer normalization + self-attention
        self.vl_encoder = nn.Sequential(
            nn.LayerNorm(config.vl_input_dim),
            SelfAttentionTransformer(**vl_self_attn_configs)
        )

        # count_module_params(self.vl_encoder, name="vl_encoder")

        self.position_embedding = nn.Embedding(config.max_seq_len, config.latent_dim)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # Beta distribution for noise generation in flow-matching processes
        self.noise_s = config.noise_s
        # self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.beta_dist = Beta(
            torch.tensor(config.noise_beta_alpha, dtype=torch.float32), 
            torch.tensor(config.noise_beta_beta, dtype=torch.float32)
        )
        self.num_timestep_buckets = config.num_timestep_buckets

        # Set trainable parameters based on configuration
        self.set_trainable_parameters(
            tune_projector=config.tune_projector,
            tune_perceiver=config.tune_perceiver,
        )

        # count_module_params(self, name="ActionLatent")

        self.init_weights()

    def set_trainable_parameters(self, tune_projector, tune_perceiver):
        self.tune_projector = tune_projector
        self.tune_perceiver = tune_perceiver
        for p in self.parameters():
            p.requires_grad = True
        if not self.tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            self.position_embedding.requires_grad_(False)
        if not self.tune_perceiver:
            self.perceiver_net.requires_grad_(False)

        print("############# ActionLatent: Set Trainable Parameters #############")
        print(f"Tune sub-module robot projector: {self.tune_projector}")
        print(f"Tune sub-module perceiver-net: {self.tune_perceiver}")

        if not self.tune_projector and not self.tune_perceiver:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"ActionLatent trainable parameter: {name}")

        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No ActionLatent trainable parameters found.")

    def init_weights(self):
        for module in self.children():
            module.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        # elif isinstance(module, nn.MultiheadAttention):
        #     # This uses torch's original init
        #     module._reset_parameters()

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                self.position_embedding.eval()
            if not self.tune_perceiver:
                self.perceiver_net.eval()

    def sample_time(self, batch_size, device, dtype):
        """
        Beta distribution for noise generation in flow-matching processes.
        """
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        # Transform the Beta sample to timestep using the noise scaling factor (noise_s)
        # This maps the sample to [0, 1) range, where:
        # - When sample ≈ noise_s: t_sampled ≈ 0 (early in the flow process, high noise)
        # - When sample ≈ 0: t_sampled ≈ 1 (late in the flow process, low noise)
        t_sampled = (self.noise_s - sample) / self.noise_s
        return t_sampled

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def robot_encoder(self, action_input: BatchFeature, device, embodiment_id):
        # Embed state.
        # state_features = self.state_encoder(action_input.state, embodiment_id)
        state_features = None

        # Embed noised action trajectory.
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions    # [B, T, 3]
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Add position embedding.
        pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
        pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
        action_features = action_features + pos_embs

        return None, action_features, t_discretized, actions, velocity

    def forward(self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
    ) -> BatchFeature:

        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        # visual-language embeddings
        vl_embeds = backbone_output["backbone_features"]
        vl_embeds = self.vl_encoder(vl_embeds)
        device = vl_embeds.device
        vl_attn_mask = backbone_output.backbone_attention_mask

        # robot state, action encoder
        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id
        state_features, action_features, t_discretized, actions, velocity = self.robot_encoder(action_input, device, embodiment_id)

        # Join vision, language, state and action embedding along sequence dimension.
        # sa_embs = torch.cat((state_features, action_features), dim=1)
        sa_embeds = action_features
        model_output = self.perceiver_net(
            latents=sa_embeds,
            visual_language_states=vl_embeds,
            timestep=t_discretized,
        )

        # action prediction
        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        if action_mask.dim() == 2:
            action_mask = action_mask[...,None].expand_as(velocity)

        loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = loss.sum() / action_mask.sum()
        output_dict = {
            "loss": loss,
        }

        return BatchFeature(data=output_dict)

    @torch.no_grad()
    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
    ) -> BatchFeature:

        # Get vision and language embeddings.
        vl_embeds = backbone_output["backbone_features"]
        vl_embeds = self.vl_encoder(vl_embeds)

        # Get vision and language embeddings.
        embodiment_id = action_input.embodiment_id

        # Embed state.
        # state_features = self.state_encoder(action_input.state, embodiment_id)

        # Set initial actions as the sampled noise.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)

            # Maybe add position embedding.
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

            vl_embs = vl_embeds

            # Join vision, language, state and action embedding along sequence dimension.
            # sa_embs = torch.cat((state_features, action_features), dim=1)
            sa_embs = action_features

            # Run model forward.
            model_output = self.perceiver_net(
                latents=sa_embs,
                visual_language_states=vl_embs,
                timestep=timesteps_tensor,
            )

            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
        return BatchFeature(data={"action_pred": actions})


    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype