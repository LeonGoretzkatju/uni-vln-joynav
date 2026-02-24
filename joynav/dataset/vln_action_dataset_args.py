"""
VLN Action Dataset specific arguments.
Inherits from LazySupervisedDatasetArguments and adds VLN-specific parameters.
"""
from dataclasses import dataclass, field
from typing import Optional
from .lazy_supervised_dataset_args import LazySupervisedDatasetArguments


@dataclass
class VLNActionDatasetArguments(LazySupervisedDatasetArguments):
    """Arguments specific to VLN Action Dataset."""

    # VLN-specific data paths
    video_folder: str = field(
        default="",
        metadata={"help": "Path to video folders, comma-separated for multiple folders"}
    )

    min_window_size: Optional[int] = field(
        default=8,
        metadata={"help": "Minimum window size for sampling frames"}
    )
    max_window_size: Optional[int] = field(
        default=16,
        metadata={"help": "Maximum window size for sampling frames"}
    )
    action_chunk_num: Optional[int] = field(
        default=8,
        metadata={"help": "Number of future action steps to predict"}
    )
    sampling_stride: Optional[int] = field(
        default=4,
        metadata={"help": "Stride for sampling frames from videos"}
    )
    
    add_continuous_action: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to add continuous action representation"}
    )

    x_norm_factor: Optional[float] = field(
        default=1.0,
        metadata={"help": "Normalization factor for x-coordinate"}
    )

    y_norm_factor: Optional[float] = field(
        default=0.433,
        metadata={"help": "Normalization factor for y-coordinate"}
    )

    history_sampling_mode: Optional[str] = field(
        default="uniform",
        metadata={"help": "Sampling mode for historical frames, e.g., 'recent' or 'uniform'"}
    )

    split_forward: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to split forward and backward navigation data into separate samples"}
    )
    
