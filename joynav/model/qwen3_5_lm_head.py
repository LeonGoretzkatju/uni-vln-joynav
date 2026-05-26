from typing import Optional

import torch

from .base_argument import BaseArguments
from .base_model import BaseModel

try:
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
except ImportError:
    Qwen3_5Config = None
    Qwen3_5ForConditionalGeneration = None


QWEN3_5_AVAILABLE = Qwen3_5ForConditionalGeneration is not None
QWEN3_5_REQUIREMENT = "Qwen3.5 support requires Python >=3.10 and transformers >=5.0."


def _expand_qwen3_5_position_ids(
    input_ids: Optional[torch.LongTensor],
    position_ids: Optional[torch.LongTensor],
) -> Optional[torch.LongTensor]:
    if position_ids is None or position_ids.dim() != 3 or position_ids.shape[0] != 3:
        return position_ids

    batch_size = position_ids.shape[1] if input_ids is None else input_ids.shape[0]
    seq_len = position_ids.shape[2] if input_ids is None else input_ids.shape[1]
    text_position_ids = torch.arange(
        seq_len,
        device=position_ids.device,
        dtype=position_ids.dtype,
    ).view(1, 1, -1).expand(1, batch_size, -1)
    return torch.cat([text_position_ids, position_ids], dim=0)


if not QWEN3_5_AVAILABLE:

    class JoyNav_Qwen3_5ForCausalLM(BaseModel):
        ARGUMENT_CLASS = BaseArguments

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise RuntimeError(QWEN3_5_REQUIREMENT)

        def forward(self, *args, **kwargs):
            raise RuntimeError(QWEN3_5_REQUIREMENT)

else:

    class JoyNav_Qwen3_5ForCausalLM(BaseModel, Qwen3_5ForConditionalGeneration):
        config_class = Qwen3_5Config

        def __init__(self, config):
            Qwen3_5ForConditionalGeneration.__init__(self, config)

        @property
        def visual(self):
            return self.model.visual

        @property
        def language_model(self):
            return self.model.language_model

        def get_model(self):
            return self.model

        def forward(self, *args, **kwargs):
            if kwargs.get("mm_token_type_ids") is not None:
                kwargs["position_ids"] = None
            else:
                kwargs["position_ids"] = _expand_qwen3_5_position_ids(
                    kwargs.get("input_ids"),
                    kwargs.get("position_ids"),
                )
            return Qwen3_5ForConditionalGeneration.forward(self, *args, **kwargs)

        def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            position_ids=None,
            use_cache=True,
            pixel_values=None,
            pixel_values_videos=None,
            image_grid_thw=None,
            video_grid_thw=None,
            mm_token_type_ids=None,
            is_first_iteration=False,
            **kwargs,
        ):
            model_inputs = super().prepare_inputs_for_generation(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                use_cache=use_cache,
                is_first_iteration=is_first_iteration,
                **kwargs,
            )
            if mm_token_type_ids is not None and model_inputs.get("pixel_values") is not None:
                model_inputs["mm_token_type_ids"] = mm_token_type_ids
            return model_inputs
