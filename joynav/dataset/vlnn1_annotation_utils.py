"""
Utilities for converting InternData-N1 VLN poses into ego-centric waypoints.

InternData-N1 stores per-frame camera/agent transforms in Blender coordinates.
The model target used here is a future trajectory in the current agent frame:
``[x_forward, y_left, yaw]`` in ROS-style planar coordinates.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EpisodeData:
    transforms: np.ndarray
    frame_indices: np.ndarray


@dataclass
class ActionChunkResult:
    step_index: int
    original_frame_idx: int
    actions: List[List[float]]
    future_original_indices: List[int]


def resolve_traj_data_root(input_path: str) -> Optional[str]:
    candidates = [
        os.path.join(input_path, "vln_n1", "traj_data"),
        os.path.join(input_path, "traj_data"),
        input_path,
    ]
    for candidate in candidates:
        if os.path.isdir(candidate) and os.path.basename(candidate) == "traj_data":
            return candidate
    return None


def find_all_scenes(traj_data_path: str) -> List[Dict[str, str]]:
    scenes: List[Dict[str, str]] = []
    if not os.path.isdir(traj_data_path):
        return scenes

    for dataset_source in sorted(os.listdir(traj_data_path)):
        dataset_path = os.path.join(traj_data_path, dataset_source)
        if not os.path.isdir(dataset_path):
            continue
        for scene_name in sorted(os.listdir(dataset_path)):
            scene_path = os.path.join(dataset_path, scene_name)
            data_path = os.path.join(scene_path, "data")
            meta_path = os.path.join(scene_path, "meta")
            videos_path = os.path.join(scene_path, "videos")
            if os.path.isdir(data_path) and os.path.isdir(meta_path):
                scenes.append(
                    {
                        "dataset_source": dataset_source,
                        "scene_name": scene_name,
                        "scene_path": scene_path,
                        "data_path": data_path,
                        "meta_path": meta_path,
                        "videos_path": videos_path,
                    }
                )
    return scenes


def get_all_parquet_files(data_path: str) -> Dict[int, Tuple[str, str]]:
    parquet_files: Dict[int, Tuple[str, str]] = {}
    for chunk_dir in sorted(glob.glob(os.path.join(data_path, "chunk-*"))):
        chunk_id = os.path.basename(chunk_dir)
        for parquet_file in glob.glob(os.path.join(chunk_dir, "episode_*.parquet")):
            episode_idx_str = os.path.basename(parquet_file).replace("episode_", "").replace(".parquet", "")
            try:
                parquet_files[int(episode_idx_str)] = (parquet_file, chunk_id)
            except ValueError:
                logger.warning("Invalid episode parquet name: %s", parquet_file)
    return parquet_files


def load_instructions_from_jsonl(episodes_jsonl_path: str) -> Dict[int, List[str]]:
    instructions: Dict[int, List[str]] = {}
    with open(episodes_jsonl_path, "r", encoding="utf-8") as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
            try:
                episode = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Could not parse %s line %s: %s", episodes_jsonl_path, line_num, exc)
                continue
            episode_index = episode.get("episode_index")
            if episode_index is None:
                continue
            tasks = episode.get("tasks", [])
            instructions[int(episode_index)] = [
                task["sub_instruction"]
                for task in tasks
                if isinstance(task, dict) and task.get("sub_instruction")
            ]
    return instructions


def load_episode_data(file_path: str, trajectory_stride: int) -> EpisodeData:
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(file_path, columns=["action"])
        action_column = table.column("action").to_pylist()
    except ImportError:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("VLNN1 annotation conversion requires pyarrow or pandas to read parquet files.") from exc

        df = pd.read_parquet(file_path, columns=["action"])
        if "action" not in df.columns:
            raise ValueError(f"Missing 'action' column in parquet: {file_path}")
        action_column = df["action"]
    except Exception as exc:
        if "No match for FieldRef.Name(action)" in str(exc) or "action" in str(exc):
            raise ValueError(f"Missing 'action' column in parquet: {file_path}") from exc
        raise

    matrices = []
    for idx, frame in enumerate(action_column):
        mat = np.array(frame, dtype=np.float64)
        if mat.shape == (4, 4):
            matrices.append(mat)
        elif mat.size == 16:
            matrices.append(mat.reshape(4, 4))
        else:
            raise ValueError(f"Invalid action matrix shape {mat.shape} at row {idx} in {file_path}")

    transforms = np.stack(matrices, axis=0)
    if trajectory_stride <= 0:
        raise ValueError(f"trajectory_stride must be positive, got {trajectory_stride}")
    if trajectory_stride > len(transforms):
        trajectory_stride = 1

    frame_indices = np.arange(len(transforms), dtype=np.int64)[::trajectory_stride]
    return EpisodeData(transforms=transforms[frame_indices], frame_indices=frame_indices)


def blender_to_ros_coordinates(xyz_blender: np.ndarray) -> np.ndarray:
    xyz_ros = np.empty_like(xyz_blender)
    xyz_ros[..., 0] = xyz_blender[..., 1]
    xyz_ros[..., 1] = -xyz_blender[..., 0]
    xyz_ros[..., 2] = xyz_blender[..., 2]
    return xyz_ros


def relative_pose_batch(
    r_base: np.ndarray,
    t_base: np.ndarray,
    r_world: np.ndarray,
    t_world: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    r_rel = np.matmul(r_base.T, r_world)
    t_rel_blender = (t_world - t_base) @ r_base
    t_rel_ros = blender_to_ros_coordinates(t_rel_blender)
    yaws = np.arctan2(r_rel[:, 1, 0], r_rel[:, 0, 0])
    return t_rel_ros[:, :2], yaws


def build_continuous_actions(
    transforms: np.ndarray,
    frame_indices: np.ndarray,
    step_stride: int,
    action_chunk_size: int,
) -> Tuple[Dict[str, List[List[float]]], List[ActionChunkResult]]:
    """Build ego-centric future waypoint chunks.

    ``action_chunk_size`` includes the leading zero reference pose. For an
    eight-waypoint target, pass ``action_chunk_size=9`` and drop the first entry
    in the training dataset.
    """
    if step_stride <= 0:
        raise ValueError(f"step_stride must be positive, got {step_stride}")
    if action_chunk_size < 1:
        raise ValueError(f"action_chunk_size must be at least 1, got {action_chunk_size}")

    continuous_actions: Dict[str, List[List[float]]] = {}
    action_results: List[ActionChunkResult] = []
    future_len = action_chunk_size - 1
    max_start = transforms.shape[0] - (future_len + 1)
    if max_start < 0:
        return continuous_actions, action_results

    for step in range(0, max_start + 1, step_stride):
        ref_t = transforms[step]
        future_t = transforms[step + 1 : step + 1 + future_len]
        future_steps = list(range(step + 1, step + 1 + future_len))
        future_original_indices = [int(frame_indices[s]) for s in future_steps]

        if len(future_t) > 0:
            rel_xy, rel_yaw = relative_pose_batch(
                ref_t[:3, :3],
                ref_t[:3, 3],
                future_t[:, :3, :3],
                future_t[:, :3, 3],
            )
            future_actions = np.column_stack([rel_xy, rel_yaw]).astype(float).tolist()
        else:
            future_actions = []

        actions = [[0.0, 0.0, 0.0]] + future_actions
        continuous_actions[str(step)] = actions
        action_results.append(
            ActionChunkResult(
                step_index=step,
                original_frame_idx=int(frame_indices[step]),
                actions=actions,
                future_original_indices=future_original_indices,
            )
        )

    return continuous_actions, action_results
