from dataclasses import dataclass, field

from joynav.eval.qwen3_vl_lm_head_evaluator import (
    Qwen3VLLMHeadEvaluator,
    Qwen3VLLMHeadEvaluatorArguments,
)


@dataclass
class Qwen3VLSpatialForcingEvaluatorArguments(Qwen3VLLMHeadEvaluatorArguments):
    evaluator_type: str = field(default="qwen3_vl_lm_head_sf", metadata={"help": "Spatial Forcing Qwen3-VL evaluator"})
    model_type: str = field(default="qwen3_vl_lm_head_sf", metadata={"help": "Spatial Forcing Qwen3-VL model type"})


class Qwen3VLSpatialForcingEvaluator(Qwen3VLLMHeadEvaluator):
    """Spatial Forcing uses the normal LM-head policy at inference time."""

    ARGUMENT_CLASS = Qwen3VLSpatialForcingEvaluatorArguments
