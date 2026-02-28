"""
This model optimize the Qwen3-VL with rolling cache and dynamic rope.
"""
import types
from abc import ABC
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLCausalLMOutputWithPast,
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

from .base_model import BaseModel


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
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    else:
        query_states = apply_rotary_pos_emb_single(query_states, cos, sin)
        if past_key_values.layers[self.layer_idx].get_seq_length() > 0:
            past_cos, past_sin = self.past_position_embeddings
            cos = torch.cat([past_cos, cos], dim=1)
            sin = torch.cat([past_sin, sin], dim=1)
        self.past_position_embeddings = (cos, sin)

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
    

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.

        Example:
            TODO: Add example
        """

        if position_ids is None:
            past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()
            if self.model.rope_deltas is None or past_key_values_length == 0:
                position_ids, rope_deltas = self.model.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask,
                )
                self.model.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            elif input_ids.shape[1] > 1:
                part_attention_mask = attention_mask[:,-input_ids.shape[1]:]
                position_ids, rope_deltas = self.model.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=part_attention_mask,
                )
                delta = (past_key_values_length + self.model.rope_deltas).to(input_ids.device)
                position_ids = position_ids.add(delta)
                self.model.rope_deltas = self.model.rope_deltas + rope_deltas
            else:
                batch_size, seq_length = input_ids.shape
                delta = (past_key_values_length + self.model.rope_deltas).to(input_ids.device)
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)


        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )

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

        # Qwen3VL position_ids are prepareed with rope_deltas in forward
        # model_inputs["position_ids"] = None
        model_inputs["position_ids"] = self.prepare_position_ids(**model_inputs, total_input_ids=input_ids)

        is_decoding_step = (model_inputs["inputs_embeds"] is not None and model_inputs["inputs_embeds"].shape[1] == 1) or (model_inputs["input_ids"] is not None and model_inputs["input_ids"].shape[1] == 1)
        if cache_position[0] != 0 and is_decoding_step:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
        else:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["pixel_values_videos"] = pixel_values_videos
        return model_inputs

    def prepare_position_ids(
        self, 
        input_ids,
        attention_mask, 
        cache_position, 
        image_grid_thw, 
        video_grid_thw,
        total_input_ids,
        past_key_values=None, 
        **kwargs,
    ):
        # # Qwen3VL position_ids are prepareed with rope_deltas in forward
        past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()
        # Note: this implementation only support batch_size = 1 currently.
        if input_ids.shape[1] > 1:
            # all_image_grid_thw = torch.cat([*past_image_grid_thw, image_grid_thw], dim=0) if past_image_grid_thw is not None else image_grid_thw
            # count the number of special tokens <|i|>
            vision_end_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
            image_num = (total_input_ids == vision_end_token_id).sum().item()
            image_grid_thw = image_grid_thw[0:1].repeat(image_num, 3)

            position_ids, rope_deltas = self.model.get_rope_index(
                total_input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask=attention_mask,
            )
            self.model.rope_deltas = rope_deltas

            if past_key_values_length > 0:
                past_position_ids = position_ids[:, :, :past_key_values_length]
                position_ids = position_ids[:, :, past_key_values_length:]

                past_position_embeddings = self.model.language_model.rotary_emb(past_key_values[0][0], past_position_ids)
                for _, module in self.model.named_modules():
                    if isinstance(module, Qwen3VLTextAttention):
                        module.past_position_embeddings = past_position_embeddings

        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            batch_size, seq_length = input_ids.shape
            delta = (past_key_values_length + self.model.rope_deltas).to(input_ids.device)
            position_ids = torch.arange(seq_length, device=input_ids.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        return position_ids