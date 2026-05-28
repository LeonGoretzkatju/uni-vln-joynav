#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from joynav.dataset.vlnn1_annotation_utils import (
    build_continuous_actions,
    find_all_scenes,
    get_all_parquet_files,
    load_episode_data,
    load_instructions_from_jsonl,
    resolve_traj_data_root,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def convert_to_annotations(input_path, output_path, future_action_chunk_size, step_stride, trajectory_stride):
    traj_data_root = resolve_traj_data_root(input_path)
    if traj_data_root is None:
        raise ValueError("input_path must be InternData-N1, vln_n1, or traj_data")

    annotations = []
    scenes = find_all_scenes(traj_data_root)
    logger.info("Found %s scenes under %s", len(scenes), traj_data_root)
    stored_action_chunk_size = int(future_action_chunk_size) + 1

    for scene in scenes:
        episodes_path = os.path.join(scene["meta_path"], "episodes.jsonl")
        if not os.path.exists(episodes_path):
            continue
        instructions = load_instructions_from_jsonl(episodes_path)
        parquet_files = get_all_parquet_files(scene["data_path"])
        scene_rel_path = f"{scene['dataset_source']}/{scene['scene_name']}"

        for episode_idx, (parquet_path, chunk_id) in sorted(parquet_files.items()):
            episode_data = load_episode_data(parquet_path, trajectory_stride=trajectory_stride)
            continuous_actions, _ = build_continuous_actions(
                episode_data.transforms,
                episode_data.frame_indices,
                step_stride=step_stride,
                action_chunk_size=stored_action_chunk_size,
            )
            if not continuous_actions:
                continue
            annotations.append(
                {
                    "path": scene_rel_path,
                    "id": int(episode_idx),
                    "chunk_id": chunk_id,
                    "instructions": instructions.get(int(episode_idx), []),
                    "continuous_actions": continuous_actions,
                }
            )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(annotations, file, indent=2, ensure_ascii=False)

    meta = {
        "schema_version": "vlnn1_ego_xyz_yaw_v1",
        "trajectory_stride": trajectory_stride,
        "step_stride": step_stride,
        "future_action_chunk_size": future_action_chunk_size,
        "stored_action_chunk_size": stored_action_chunk_size,
        "action_dim": 3,
        "coordinate_frame": "ego_ros_xy_yaw",
    }
    with output_path.with_name("annotations_meta.json").open("w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)
    logger.info("Wrote %s episodes to %s", len(annotations), output_path)


def main():
    parser = argparse.ArgumentParser(description="Create InternData-N1 VLNN1 continuous trajectory annotations")
    parser.add_argument("input_path")
    parser.add_argument("-o", "--output", default="annotations.json")
    parser.add_argument("--future-action-chunk-size", type=int, default=8)
    parser.add_argument("--step-stride", type=int, default=4)
    parser.add_argument("--trajectory-stride", type=int, default=3)
    args = parser.parse_args()
    convert_to_annotations(
        input_path=args.input_path,
        output_path=args.output,
        future_action_chunk_size=args.future_action_chunk_size,
        step_stride=args.step_stride,
        trajectory_stride=args.trajectory_stride,
    )


if __name__ == "__main__":
    main()
