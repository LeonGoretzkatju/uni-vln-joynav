import copy
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class BaseEvaluatorArguments:
    """Base evaluator arguments."""
    evaluator_type: str = field(
        default="streamvln",
        metadata={"help": "Evaluator type: streamvln, etc."}
    )


class BaseEvaluator(ABC):
    """Abstract base class for evaluators."""
    
    ARGUMENT_CLASS = BaseEvaluatorArguments

    @classmethod
    def get_argument_class(cls):
        """Return the argument class for the evaluator."""
        return cls.ARGUMENT_CLASS

    def get_intrinsic_matrix(self, sensor_cfg) -> np.ndarray:
        width = sensor_cfg.width
        height = sensor_cfg.height
        fov = sensor_cfg.hfov
        fx = (width / 2.0) / np.tan(np.deg2rad(fov / 2.0))
        fy = fx  # Assuming square pixels (fx = fy)
        cx = (width - 1.0) / 2.0
        cy = (height - 1.0) / 2.0

        intrinsic_matrix = np.array(
            [[fx, 0.0, cx, 0.0], [0.0, fy, cy, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        return intrinsic_matrix

    def preprocess_instrinsic(self, intrinsic, ori_size, target_size):  # (V, 4, 4) (resize_shape) (h, w)
        intrinsic = copy.deepcopy(intrinsic)
        if len(intrinsic.shape) == 2:
            intrinsic = intrinsic[None, :, :]  # (1, 4, 4) or (B, 4, 4)

        intrinsic[:, 0] /= ori_size[0] / target_size[0]  # width
        intrinsic[:, 1] /= ori_size[1] / target_size[1]  # height

        # for crop transform
        intrinsic[:, 0, 2] -= (target_size[0] - target_size[1]) / 2

        if intrinsic.shape[0] == 1:
            intrinsic = intrinsic.squeeze(0)

        return intrinsic

    def get_axis_align_matrix(self):
        ma = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        return ma

    def xyz_yaw_to_tf_matrix(self, xyz: np.ndarray, yaw: float) -> np.ndarray:
        x, y, z = xyz
        transformation_matrix = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0, x],
                [np.sin(yaw), np.cos(yaw), 0, y],
                [0, 0, 1, z],
                [0, 0, 0, 1],
            ]
        )
        return transformation_matrix

    def xyz_pitch_to_tf_matrix(self, xyz: np.ndarray, pitch: float) -> np.ndarray:
        """Converts a given position and pitch angle to a 4x4 transformation matrix.

        Args:
            xyz (np.ndarray): A 3D vector representing the position.
            pitch (float): The pitch angle in radians for y axis.
        Returns:
            np.ndarray: A 4x4 transformation matrix.
        """

        x, y, z = xyz
        transformation_matrix = np.array(
            [
                [np.cos(pitch), 0, np.sin(pitch), x],
                [0, 1, 0, y],
                [-np.sin(pitch), 0, np.cos(pitch), z],
                [0, 0, 0, 1],
            ]
        )
        return transformation_matrix

    def xyz_yaw_pitch_to_tf_matrix(self, xyz: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
        """Converts a given position and yaw, pitch angles to a 4x4 transformation matrix.

        Args:
            xyz (np.ndarray): A 3D vector representing the position.
            yaw (float): The yaw angle in radians.
            pitch (float): The pitch angle in radians for y axis.
        Returns:
            np.ndarray: A 4x4 transformation matrix.
        """
        x, y, z = xyz
        rot1 = self.xyz_yaw_to_tf_matrix(xyz, yaw)[:3, :3]
        rot2 = self.xyz_pitch_to_tf_matrix(xyz, pitch)[:3, :3]
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rot1 @ rot2
        transformation_matrix[:3, 3] = xyz
        return transformation_matrix
