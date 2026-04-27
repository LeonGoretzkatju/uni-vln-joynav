from dataclasses import dataclass, field

from joynav.eval.qwen3_vl_lm_head_dyna_evaluator import (
    Qwen3VLLMDynamicRopeEvaluator,
    Qwen3VLLMHeadEvaluatorArguments,
)


@dataclass
class Qwen3VLSpatialForcingDynamicRopeEvaluatorArguments(Qwen3VLLMHeadEvaluatorArguments):
    evaluator_type: str = field(
        default="qwen3_vl_lm_head_sf_dyna",
        metadata={"help": "Spatial-Forcing Qwen3-VL dynamic-rope evaluator"},
    )
    model_type: str = field(
        default="qwen3_vl_lm_head_sf_dyna",
        metadata={"help": "Spatial-Forcing Qwen3-VL dynamic-rope model type"},
    )


class Qwen3VLSpatialForcingDynamicRopeEvaluator(Qwen3VLLMDynamicRopeEvaluator):
    """Spatial Forcing uses normal dynamic-rope generation at eval time."""

    ARGUMENT_CLASS = Qwen3VLSpatialForcingDynamicRopeEvaluatorArguments
