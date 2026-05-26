"""
Continuous Action Dataset specific arguments.
Inherits from LazySupervisedDatasetArguments and adds continuous action prediction parameters.
"""
from dataclasses import dataclass, field
from typing import Optional
from .lazy_supervised_dataset_args import LazySupervisedDatasetArguments


@dataclass
class ContinuousActionDatasetArguments(LazySupervisedDatasetArguments):
    """Arguments specific to Continuous Action Prediction Dataset."""

    # Data paths
    video_folder: str = field(
        default="",
        metadata={"help": "Path to video folders, comma-separated for multiple folders"}
    )
    
    # Sampling parameters
    num_history_frames: Optional[int] = field(
        default=8,
        metadata={"help": "Number of historical frames to uniformly sample as observations"}
    )

    num_special_tokens: Optional[int] = field(
        default=8,
        metadata={"help": "Number of special tokens to represent the continuous action prediction"}
    )
    
    action_chunk_size: Optional[int] = field(
        default=8,
        metadata={"help": "Number of future continuous actions to predict"}
    )
    
    action_dim: Optional[int] = field(
        default=3,
        metadata={"help": "Dimension of each action (x, y, yaw)"}
    )
    
    def validate(self):
        """Validate continuous action dataset arguments."""
        # Call parent validation
        super().validate()
        
        # Continuous action specific validation
        if not self.video_folder:
            raise ValueError("video_folder must be specified for ContinuousActionDataset")
        