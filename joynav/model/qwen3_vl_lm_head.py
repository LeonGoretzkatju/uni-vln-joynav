from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLCausalLMOutputWithPast,
)
from torch.nn import CrossEntropyLoss

from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention

from .base_model import BaseModel
from .dynamic_rope_interface import Qwen3VLDynamicRopeInterface

class JoyNavModelConfig(Qwen3VLConfig):
    model_type = "joynav_qwen3_vl"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_cfg = kwargs.get('model_cfg', None)


class JoyNav_Qwen3VLForCausalLM(BaseModel, Qwen3VLForConditionalGeneration):
    config_class = JoyNavModelConfig

    def __init__(self, config):
        Qwen3VLForConditionalGeneration.__init__(self, config)

    def get_model(self):
        return self.model
    
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
        mm_token_type_ids: Optional[torch.Tensor] = None,
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
                    mm_token_type_ids=mm_token_type_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    attention_mask=attention_mask,
                )
                self.model.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            elif input_ids.shape[1] > 1:
                part_attention_mask = attention_mask[:,-input_ids.shape[1]:]
                position_ids, rope_deltas = self.model.get_rope_index(
                    input_ids,
                    mm_token_type_ids=mm_token_type_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
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
            mm_token_type_ids=mm_token_type_ids,
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
        model_inputs["position_ids"] = None

        is_decoding_step = (model_inputs["inputs_embeds"] is not None and model_inputs["inputs_embeds"].shape[1] == 1) or (model_inputs["input_ids"] is not None and model_inputs["input_ids"].shape[1] == 1)
        if cache_position[0] != 0 and is_decoding_step:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
        else:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["pixel_values_videos"] = pixel_values_videos
        return model_inputs


class JoyNav_Qwen3VLForCausalLMWithDynamicRope(Qwen3VLDynamicRopeInterface, JoyNav_Qwen3VLForCausalLM):
    """Qwen3-VL LM-head model with dynamic RoPE for rolling KV-cache inference.

    During training (``past_key_values is None``) behavior is identical to the
    base :class:`JoyNav_Qwen3VLForCausalLM`. The dynamic-rope path activates at
    inference time when a cache is present — see
    :func:`dynamic_rope_interface.forward_with_dynamic_rope`.
    """

    config_class = JoyNavModelConfig

    def __init__(self, config):
        # MRO: Interface.__init__ -> ABC (no __init__) -> JoyNav.__init__ -> Qwen3VLForConditionalGeneration.__init__
        super().__init__(config)
        # Ensure every patched attention starts with a clean rolling-rope buffer.
        self._reset_dynamic_rope_state()

    def _reset_dynamic_rope_state(self):
        for _, module in self.model.named_modules():
            if isinstance(module, Qwen3VLTextAttention):
                module.past_position_embeddings = None

    def forward(self, *args, **kwargs):
        # A stale rolling buffer from a previous generation would corrupt
        # training gradients, so clear it whenever we enter a fresh forward
        # without an input cache.
        if kwargs.get("past_key_values", None) is None:
            self._reset_dynamic_rope_state()
        return super().forward(*args, **kwargs)
