"""
VLN Action Dataset specific arguments.
Inherits from LazySupervisedDatasetArguments and adds VLN-specific parameters.
"""
from dataclasses import dataclass, field
from typing import Optional
from .lazy_supervised_dataset_args import LazySupervisedDatasetArguments


@dataclass
class StreamVLNDatasetArguments(LazySupervisedDatasetArguments):
    """Arguments specific to VLN Action Dataset."""

    # VLN-specific data paths
    video_folder: str = field(
        default="",
        metadata={"help": "Path to video folders, comma-separated for multiple folders"}
    )
    
    # VLN-specific sampling parameters
    num_frames: Optional[int] = field(
        default=32,
        metadata={"help": "Number of frames to sample from the trajectory"}
    )
    num_history: Optional[int] = field(
        default=8,
        metadata={"help": "Number of historical frames to include"}
    )
    num_future_steps: Optional[int] = field(
        default=4,
        metadata={"help": "Number of future action steps to predict"}
    )
    
    # VLN-specific data processing
    remove_init_turns: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to remove initial rotation actions"}
    )
    
    def validate(self):
        """Validate VLN-specific arguments."""
        # Call parent validation
        super().validate()
        
        # VLN-specific validation
        if not self.video_folder:
            raise ValueError("video_folder must be specified for VLNActionDataset")
        
        if self.num_frames is not None and self.num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {self.num_frames}")
        
        if self.num_history is not None and self.num_history < 0:
            raise ValueError(f"num_history must be non-negative, got {self.num_history}")
        
        if self.num_future_steps is not None and self.num_future_steps <= 0:
            raise ValueError(
                f"num_future_steps must be positive, got {self.num_future_steps}"
            )
