from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BaseArguments:
    """
    Base arguments for all JoyNav models.
    """
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False, metadata={"help": "Whether to fine-tune the multi-modal LLM"})
    tune_mm_mlp: bool = field(default=False, metadata={"help": "Whether to fine-tune the multi-modal MLP"})
    tune_mm_vision: bool = field(default=False, metadata={"help": "Whether to fine-tune the multi-modal vision encoder"})
    model_load_dtype: str = field(
        default="auto",
        metadata={"help": "Model weight loading dtype: auto, float32, bfloat16, or float16."},
    )

    # LoRA arguments
    use_lora: bool = field(default=False, metadata={"help": "Whether to use LoRA for parameter-efficient fine-tuning"})
    lora_r: int = field(default=64, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha scaling factor"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout rate"})
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated list of target modules for LoRA"}
    )
    lora_modules_to_save: str = field(
        default="",
        metadata={"help": "Comma-separated list of modules to save fully (e.g. action_head,lm_head)"}
    )
    lora_merge_and_save: bool = field(
        default=False,
        metadata={
            "help": "After LoRA training, merge adapters into the base weights and save a full "
            "standalone checkpoint under <output_dir>/merged (the eval path cannot load bare adapters)."
        },
    )
