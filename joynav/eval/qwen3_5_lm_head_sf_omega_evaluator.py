import copy
from dataclasses import dataclass, field

from joynav.dataset.vln_action_omega_spatial_forcing_dataset import (
    prepare_qwen_images_for_omega_direct,
)
from joynav.eval.qwen3_vl_lm_head_evaluator import (
    PROMPT_MODE_QWEN_FINAL,
    PROMPT_MODE_TRAINING_INTERMEDIATE,
    Qwen3VLLMHeadEvaluatorArguments,
    QwenVLLMHeadEvaluator,
)
from joynav.model.geometry_encoder.vggt_omega_encoder import (
    DEFAULT_VGGT_OMEGA_TEXT_ALIGN_CHECKPOINT,
)


@dataclass
class Qwen3_5OmegaSpatialForcingEvaluatorArguments(Qwen3VLLMHeadEvaluatorArguments):
    evaluator_type: str = field(
        default="qwen3_5_lm_head_sf_omega",
        metadata={"help": "Qwen3.5-VL VGGT-Omega Spatial Forcing evaluator"},
    )
    model_type: str = field(
        default="qwen3_5_lm_head_sf_omega",
        metadata={"help": "Qwen3.5-VL VGGT-Omega Spatial Forcing model type"},
    )
    omega_mode: str = field(default="text_align")
    num_frames: int = field(default=24)
    num_history: int = field(default=6)
    action_chunk_num: int = field(default=4)
    spatial_forcing_teacher_patch_size: int = field(default=16)
    spatial_forcing_image_resolution: int = field(default=256)
    sf_geometry_encoder_path: str = field(default=DEFAULT_VGGT_OMEGA_TEXT_ALIGN_CHECKPOINT)
    sf_target_dim: int = field(default=2048)
    sf_teacher_layers: str = field(default="23")
    sf_align_layers: str = field(default="18")
    sf_alpha: float = field(default=0.1)
    sf_add_pos_embed: bool = field(default=False)


class Qwen3_5OmegaSpatialForcingEvaluator(QwenVLLMHeadEvaluator):
    """VGGT-Omega Spatial Forcing uses the normal LM-head policy at inference time."""

    ARGUMENT_CLASS = Qwen3_5OmegaSpatialForcingEvaluatorArguments

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_cache = False
        if hasattr(self.args, "use_cache"):
            self.args.use_cache = False

    def get_generation_prompt_mode(self, step_id):
        num_frames = max(int(self.num_frames), 1)
        action_chunk_num = max(int(self.action_chunk_num), 1)
        final_chunk_start = max(num_frames - action_chunk_num, 0)
        if int(step_id) % num_frames >= final_chunk_start:
            return PROMPT_MODE_QWEN_FINAL
        return PROMPT_MODE_TRAINING_INTERMEDIATE

    def prepare_input_images(self, input_images):
        if self.args.omega_mode != "text_align_force_qwen":
            return input_images
        spatial_merge_size = getattr(self.processor.image_processor, "merge_size", 2)
        return prepare_qwen_images_for_omega_direct(
            input_images,
            spatial_merge_size=spatial_merge_size,
            image_resolution=self.args.spatial_forcing_image_resolution,
            patch_size=self.args.spatial_forcing_teacher_patch_size,
        )

    def prepare_processor_source(self, source):
        if self.args.omega_mode != "text_align_force_qwen":
            return source
        processor_source = {
            "image": self.prepare_input_images(source["image"]),
            "conversations": copy.deepcopy(source["conversations"]),
        }
        return processor_source
