import joynav.utils.registry
from joynav.utils.registry import register_component

from .qwen2_5_vl_lm_head import JoyNav_Qwen2_5_VLForCausalLM
from .qwen3_vl_lm_head import JoyNav_Qwen3VLForCausalLM
from .qwen3_vl_dit_head import JoyNav_Qwen3VLDiTForCausalLM
from .qwen3_vl_mlp_head import JoyNav_Qwen3VLMLPForCausalLM

register_component("model", "qwen2_5_vl_lm_head", JoyNav_Qwen2_5_VLForCausalLM)
register_component("model", "qwen3_vl_lm_head", JoyNav_Qwen3VLForCausalLM)
register_component("model", "qwen3_vl_dit_head", JoyNav_Qwen3VLDiTForCausalLM)
register_component("model", "qwen3_vl_mlp_head", JoyNav_Qwen3VLMLPForCausalLM)