from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import os 
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
from transformers import PretrainedConfig
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel, GenerationMixin, PreTrainedTokenizer

from transformers.feature_extraction_utils import BatchFeature
from .base_model import BaseModel
from .base_argument import BaseArguments



def change_tokenizer_and_embedding_resize(
    special_tokens_dict: dict,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
):
    """
    1. add special tokens to tokenizer
    2. change the embedding layer size of model
    3. (optional) init special-token embedding
    """
    # import ipdb;ipdb.set_trace()
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    
    if num_new_tokens > 0:
        print(f"Adding {num_new_tokens} new tokens: {special_tokens_dict}")
        
        # Resize model embeddings to match new tokenizer size
        model.resize_token_embeddings(len(tokenizer))
        
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
        
        print("Resized token embeddings and initialized new tokens with average embeddings.")
    else:
        print("No new tokens added (token might already exist).")


class ActionHead_Config(PretrainedConfig):
    keys_to_ignore_at_inference = ["past_key_values"]
    def __init__(
        self,
        input_dim = None,
        hidden_dim = None,
        num_points: int = 8,   
        point_dim: int = 5,   
        **kwargs,
    ):
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim

        self.num_points = num_points
        self.point_dim = point_dim

        super().__init__(**kwargs)


class ActionMLP(nn.Module):
    """
    N-layer MLP to decode continuous trajectory from the action token's hidden state.
    Output: M waypoints * Dim = M * D dimensions.
    """
    def __init__(self, config: ActionHead_Config):
        super().__init__()
        input_dim = config.input_dim
        hidden_dim = config.hidden_dim
        self.num_points = config.num_points
        self.point_dim = config.point_dim
        
        output_dim = self.num_points * self.point_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        out = self.net(x)
        return out.view(*x.shape[:-1], self.num_points, self.point_dim)
    

class JoyNavModelConfig(Qwen3VLConfig):
    model_type = "joynav_qwen3vl_action"
    sub_configs = Qwen3VLConfig.sub_configs.copy()
    sub_configs["action_head_config"] = ActionHead_Config
    def __init__(self, 
        action_head_config: ActionHead_Config = None,
        **kwargs
    ):
        if isinstance(action_head_config, ActionHead_Config):
            self.action_head_config = action_head_config
        elif isinstance(action_head_config, dict):
            self.action_head_config = ActionHead_Config(**action_head_config)
        elif action_head_config is None:
            self.action_head_config = self.sub_configs["action_head_config"]()
        super().__init__(**kwargs)
        self.model_cfg = kwargs.get('model_cfg', None)
        
    @classmethod
    def initialize_config(cls):
        vln_config = cls(
            action_head_config=ActionHead_Config()
        )
        return vln_config


@dataclass
class JoyNav_Qwen3VLActionForCauslaLMArgruments(BaseArguments):

    propagate_action_head_grad: bool = field(default=True, metadata={"help": "Whether to propagate the gradients from action latent module to backbone LLM"})
    action_head_loss_weight: float = field(default=1.0, metadata={"help": "Weight for action latent loss"})



class JoyNav_Qwen3VLActionForCausalLM(BaseModel, Qwen3VLForConditionalGeneration):
    config_class = JoyNavModelConfig
    ARGUMENT_CLASS = JoyNav_Qwen3VLActionForCauslaLMArgruments

    def __init__(self, config:JoyNavModelConfig):
        Qwen3VLForConditionalGeneration.__init__(self, config)
        config.model_type = "joynav_qwen3_vl"
        
        self.model = Qwen3VLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        
        self.post_init()
        
        config.action_head_config.input_dim = config.text_config.hidden_size
        config.action_head_config.hidden_dim = config.text_config.hidden_size

        self.action_head_config = config.action_head_config

        self.action_head = ActionMLP(self.action_head_config)
        
        print(f"******************* init model with Action MLP Head *******************")

    def post_update_model(self):
        self.action_head = ActionMLP(self.action_head_config)

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
        # import ipdb;ipdb.set_trace()
        
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

        if "select_mask" in kwargs and kwargs.get("continuous_actions") is not None:
            # select_mask: [Batch, Seq_Len] boolean mask, True where the action token is
            select_mask = kwargs["select_mask"]
            gt_actions = kwargs["continuous_actions"]  

            if not getattr(self.model_args, "propagate_action_head_grad", True):
                features_for_action = hidden_states.detach()
            else:
                features_for_action = hidden_states

            # action_features shape: [Num_Actions_In_Batch, Hidden_Size]
            action_features = features_for_action[select_mask]

            # pred_actions shape: [Num_Actions_In_Batch, 8]
            pred_actions = self.action_head(action_features)

            action_loss_fct = nn.MSELoss()
            action_loss = action_loss_fct(pred_actions, gt_actions.to(pred_actions.dtype))

            loss += action_loss * self.model_args.action_head_loss_weight

        else:
            action_loss = None

        final_outputs = Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )
        
        final_outputs.action_loss = action_loss.detach() if action_loss is not None else None

        return final_outputs


    def predict_action(
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
        backbone_output = BatchFeature(
            data={"backbone_features": hidden_states, "backbone_attention_mask": attention_mask}
        )
        action_input = BatchFeature(
            data={"embodiment_id": 0}
        )
        action_output = self.action_head.get_action(backbone_output, action_input)

        return action_output


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


