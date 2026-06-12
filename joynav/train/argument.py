import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    # Qwen-VLA: separate group-wise LR for the DiT action expert (the paper uses
    # cosine-decayed schedules with separate groups for the VLM backbone and the
    # action decoder). None -> the expert shares the base learning_rate.
    action_expert_lr: Optional[float] = None
