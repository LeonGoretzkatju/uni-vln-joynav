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
