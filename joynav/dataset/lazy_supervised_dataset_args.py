"""
Arguments for LazySupervisedDataset.
Inherits from BaseDatasetArguments and adds supervised learning specific parameters.
"""
from dataclasses import dataclass, field
from typing import Optional
from .base_dataset_args import BaseDatasetArguments


@dataclass
class LazySupervisedDatasetArguments(BaseDatasetArguments):
    """Arguments for supervised fine-tuning datasets."""

    # Dataset selection
    dataset_use: str = field(
        default="",
        metadata={"help": "Comma-separated list of datasets to use (e.g., 'cambrian_737k,demo')"}
    )
    
    # Model configuration
    model_type: str = field(
        default="qwen2.5vl",
        metadata={"help": "Model type: qwen2vl, qwen2.5vl, or qwen3vl"}
    )
    
    # Image processing parameters
    max_pixels: int = field(
        default=28 * 28 * 576,
        metadata={"help": "Maximum pixels for images"}
    )
    min_pixels: int = field(
        default=28 * 28 * 16,
        metadata={"help": "Minimum pixels for images"}
    )
    
    # Video processing parameters
    video_max_frames: Optional[int] = field(
        default=8,
        metadata={"help": "Maximum number of frames for videos"}
    )
    video_min_frames: Optional[int] = field(
        default=4,
        metadata={"help": "Minimum number of frames for videos"}
    )
    video_max_pixels: int = field(
        default=1024 * 28 * 28,
        metadata={"help": "Maximum total pixels for videos"}
    )
    video_min_pixels: int = field(
        default=256 * 28 * 28,
        metadata={"help": "Minimum total pixels for videos"}
    )
    video_fps: float = field(
        default=2.0,
        metadata={"help": "Video frames per second"}
    )
    
    # Dataset configuration
    dataset_seed: int = field(
        default=42,
        metadata={"help": "Random seed for dataset shuffling"}
    )
    
    # Data packing/flattening
    data_flatten: bool = field(
        default=False,
        metadata={"help": "Whether to flatten the data"}
    )
    data_packing: bool = field(
        default=False,
        metadata={"help": "Whether to pack the data"}
    )
    
    # Interval for base processing
    base_interval: int = field(
        default=2,
        metadata={"help": "Base interval for data processing"}
    )
    