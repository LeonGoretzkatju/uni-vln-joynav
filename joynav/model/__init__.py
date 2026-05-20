import joynav.utils.registry
from joynav.utils.registry import register_component

from .qwen2_5_vl_lm_head import JoyNav_Qwen2_5_VLForCausalLM
from .qwen3_vl_lm_head import (
    JoyNav_Qwen3VLForCausalLM, 
    JoyNav_Qwen3VLForCausalLMWithDynamicRope,
)
from .qwen3_vl_lm_head_sf import (
    JoyNav_Qwen3VLSpatialForcingForCausalLM,
    JoyNav_Qwen3VLSpatialForcingForCausalLMWithDynamicRope,
)
from .qwen3_5_lm_head import JoyNav_Qwen3_5ForCausalLM
from .qwen3_5_lm_head_sf import JoyNav_Qwen3_5SpatialForcingForCausalLM
from .qwen3_5_lm_head_sf_omega import JoyNav_Qwen3_5OmegaSpatialForcingForCausalLM
# from .qwen3_vl_lm_head_geo import (
#     JoyNav_Qwen3VLGeoForCausalLM,
#     JoyNav_Qwen3VLGeoForCausalLMWithDynamicRope
# )
from .qwen3_vl_dit_head import JoyNav_Qwen3VLDiTForCausalLM
from .qwen3_vl_mlp_head import JoyNav_Qwen3VLMLPForCausalLM

register_component("model", "qwen2_5_vl_lm_head", JoyNav_Qwen2_5_VLForCausalLM)
register_component("model", "qwen3_vl_lm_head", JoyNav_Qwen3VLForCausalLM)
register_component("model", "qwen3_vl_lm_head_dyna", JoyNav_Qwen3VLForCausalLMWithDynamicRope)
register_component("model", "qwen3_vl_lm_head_sf", JoyNav_Qwen3VLSpatialForcingForCausalLM)
register_component("model", "qwen3_vl_lm_head_sf_dyna", JoyNav_Qwen3VLSpatialForcingForCausalLMWithDynamicRope)
register_component("model", "qwen3_5_lm_head", JoyNav_Qwen3_5ForCausalLM)
register_component("model", "qwen3_5_lm_head_sf", JoyNav_Qwen3_5SpatialForcingForCausalLM)
register_component("model", "qwen3_5_lm_head_sf_omega", JoyNav_Qwen3_5OmegaSpatialForcingForCausalLM)
# register_component("model", "qwen3_vl_lm_head_geo", JoyNav_Qwen3VLGeoForCausalLM)
# register_component("model", "qwen3_vl_lm_head_geo_dyna", JoyNav_Qwen3VLGeoForCausalLMWithDynamicRope)

register_component("model", "qwen3_vl_dit_head", JoyNav_Qwen3VLDiTForCausalLM)
register_component("model", "qwen3_vl_mlp_head", JoyNav_Qwen3VLMLPForCausalLM)
