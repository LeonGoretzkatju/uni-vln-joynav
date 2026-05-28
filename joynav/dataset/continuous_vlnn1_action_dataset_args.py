from dataclasses import dataclass, field
from typing import Optional

from .lazy_supervised_dataset_args import LazySupervisedDatasetArguments


@dataclass
class ContinuousVLNN1ActionDatasetArguments(LazySupervisedDatasetArguments):
    video_folder: str = field(
        default="",
        metadata={"help": "InternData-N1 traj_data root, or comma-separated roots."},
    )
    annotations_file: str = field(
        default="annotations.json",
        metadata={"help": "Annotation JSON name under each video_folder root."},
    )
    num_history_frames: int = field(default=8)
    action_chunk_size: int = field(default=8)
    action_dim: int = field(default=3)
    trajectory_stride: int = field(default=3)
    image_type: str = field(default="rgb")
    episodes_per_chunk: int = field(default=1000)
    include_current_frame: bool = field(default=True)
    action_token: str = field(default="<|action|>")
    interleaved_num_chunks: int = field(default=4)

    spatial_forcing_teacher_patch_size: int = field(default=16)
    spatial_forcing_image_resolution: int = field(default=256)

    def validate(self):
        super().validate()
        if not self.video_folder:
            raise ValueError("video_folder must be specified for ContinuousVLNN1ActionDataset")
        if self.action_chunk_size <= 0:
            raise ValueError("action_chunk_size must be positive")
        if self.action_dim != 3:
            raise ValueError("ContinuousVLNN1ActionDataset uses action_dim=3 for x,y,yaw")
        if self.trajectory_stride <= 0:
            raise ValueError("trajectory_stride must be positive")
        if self.num_history_frames <= 0:
            raise ValueError("num_history_frames must be positive")
        if self.image_type not in ("rgb", "depth"):
            raise ValueError("image_type must be 'rgb' or 'depth'")
        if self.episodes_per_chunk <= 0:
            raise ValueError("episodes_per_chunk must be positive")
        if self.interleaved_num_chunks <= 0:
            raise ValueError("interleaved_num_chunks must be positive")
