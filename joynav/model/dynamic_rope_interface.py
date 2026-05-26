"""
This model optimize the Qwen3-VL with rolling cache and dynamic rope.
"""
import types
from abc import ABC
from typing import Optional

import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    apply_rotary_pos_emb,
    Qwen3VLTextAttention,
    eager_attention_forward,
    rotate_half
)

from collections.abc import Callable
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs


def apply_rotary_pos_emb_single(k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    # q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return k_embed


def _cache_layer_seq_length(cache: Cache, layer_idx: int) -> int:
    if hasattr(cache, "layers"):
        return cache.layers[layer_idx].get_seq_length()
    return cache.get_seq_length(layer_idx)


def _first_cache_key(cache: Cache) -> torch.Tensor:
    if hasattr(cache, "layers"):
        return cache.layers[0].keys
    return cache[0][0]


def _update_dynamic_rope_cache_embeddings(
    cache: Cache,
    cos: torch.Tensor,
    sin: torch.Tensor,
    past_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cached = getattr(cache, "_dynamic_rope_position_embeddings", None)
    current_len = cos.shape[1]

    if cached is None:
        if past_len != 0:
            raise RuntimeError("Dynamic RoPE cache is missing past position embeddings.")
        full_cos, full_sin = cos, sin
    else:
        past_cos, past_sin = cached
        cached_len = past_cos.shape[1]
        full_len = past_len + current_len
        if cached_len == full_len:
            full_cos, full_sin = past_cos, past_sin
        elif cached_len == past_len:
            full_cos = torch.cat([past_cos, cos], dim=1)
            full_sin = torch.cat([past_sin, sin], dim=1)
        else:
            raise RuntimeError("Dynamic RoPE cache position length does not match KV length.")

    cache._dynamic_rope_position_embeddings = (full_cos, full_sin)
    return full_cos, full_sin


def _count_vision_segments(
    input_ids: torch.LongTensor,
    vision_start_token_id: Optional[int],
    media_token_id: Optional[int],
) -> int:
    if vision_start_token_id is None or media_token_id is None:
        return 0

    count = 0
    for row in input_ids:
        vision_start_indices = torch.argwhere(row == vision_start_token_id).squeeze(1)
        vision_start_indices = vision_start_indices[vision_start_indices + 1 < row.shape[0]]
        if vision_start_indices.numel() > 0:
            count += (row[vision_start_indices + 1] == media_token_id).sum().item()
    return count


def _expand_grid_thw(grid_thw: Optional[torch.LongTensor], count: int) -> Optional[torch.LongTensor]:
    if grid_thw is None or count == 0 or grid_thw.shape[0] == count:
        return grid_thw
    if grid_thw.shape[0] == 1:
        return grid_thw.repeat(count, 1)
    return grid_thw


def forward_with_dynamic_rope(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings  # [b, n_seq, d_head]

    if past_key_values is None:
        # Training / single-shot forward: standard RoPE on Q and K.
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    else:
        # Rolling-cache inference: apply RoPE to Q with current positions,
        # store un-rotated K in the cache, then re-apply RoPE over the full
        # (past || current) key sequence with dynamically concatenated cos/sin.
        query_states = apply_rotary_pos_emb_single(query_states, cos, sin)

        past_len = _cache_layer_seq_length(past_key_values, self.layer_idx)
        cos, sin = _update_dynamic_rope_cache_embeddings(past_key_values, cos, sin, past_len)

        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
        key_states = apply_rotary_pos_emb_single(key_states, cos, sin)

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


class Qwen3VLDynamicRopeInterface(ABC):

    def __init__(self, config):

        super().__init__(config)

        for _, module in self.model.named_modules():
            if isinstance(module, Qwen3VLTextAttention):
                module.forward = types.MethodType(forward_with_dynamic_rope, module)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            **kwargs,
        )

        model_inputs["position_ids"] = None
        if use_cache and model_inputs.get("input_ids") is not None:
            model_inputs["position_ids"] = self.prepare_position_ids(
                **model_inputs,
                total_input_ids=input_ids,
            )

        is_decoding_step = (
            (model_inputs["inputs_embeds"] is not None and model_inputs["inputs_embeds"].shape[1] == 1)
            or (model_inputs["input_ids"] is not None and model_inputs["input_ids"].shape[1] == 1)
        )
        if cache_position is not None and cache_position[0] != 0 and is_decoding_step:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
        else:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["pixel_values_videos"] = pixel_values_videos
        return model_inputs

    def prepare_position_ids(
        self, 
        input_ids,
        total_input_ids,
        attention_mask=None, 
        cache_position=None, 
        image_grid_thw=None, 
        video_grid_thw=None,
        past_key_values=None, 
        **kwargs,
    ):
        past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()

        if input_ids.shape[1] > 1:
            image_num = _count_vision_segments(
                total_input_ids,
                self.model.config.vision_start_token_id,
                self.model.config.image_token_id,
            )
            video_num = _count_vision_segments(
                total_input_ids,
                self.model.config.vision_start_token_id,
                self.model.config.video_token_id,
            )
            image_grid_thw = _expand_grid_thw(image_grid_thw, image_num)
            video_grid_thw = _expand_grid_thw(video_grid_thw, video_num)

            position_ids, rope_deltas = self.model.get_rope_index(
                total_input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask=attention_mask,
            )
            self.model.rope_deltas = rope_deltas

            if past_key_values_length > 0:
                past_position_ids = position_ids[:, :, :past_key_values_length]
                current_end = past_key_values_length + input_ids.shape[1]
                position_ids = position_ids[:, :, past_key_values_length:current_end]

                past_position_embeddings = self.model.language_model.rotary_emb(
                    _first_cache_key(past_key_values),
                    past_position_ids,
                )
                past_key_values._dynamic_rope_position_embeddings = past_position_embeddings

        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            batch_size, seq_length = input_ids.shape
            if self.model.rope_deltas is None:
                self.model.rope_deltas = torch.zeros(
                    batch_size,
                    1,
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )
            delta_base = cache_position[0] if cache_position is not None else past_key_values_length
            delta = (delta_base + self.model.rope_deltas).to(input_ids.device)
            position_ids = torch.arange(seq_length, device=input_ids.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        return position_ids
