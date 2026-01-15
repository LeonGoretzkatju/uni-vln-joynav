import joynav.utils.registry
from joynav.utils.registry import register_component

from .qwen2_5_vl_discrete import JoyNav_Qwen2_5_VLForCausalLM
from .qwen3_vl_discrete import JoyNav_Qwen3VLForCausalLM

register_component("model", "qwen2_5_vl_discrete", JoyNav_Qwen2_5_VLForCausalLM)
register_component("model", "qwen3_vl_discrete", JoyNav_Qwen3VLForCausalLM)