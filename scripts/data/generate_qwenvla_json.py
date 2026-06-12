#!/usr/bin/env python3
"""Generate Qwen-VLA unified action-and-trajectory supervision from JD-VLN VLN data.

Follows the Qwen-VLA technical report (arXiv:2605.30280):
  - Embodiment-aware prompt conditioning (Section 2.3): every sample is prefixed
    with a textual prompt naming the platform, control convention, control
    frequency, and prediction horizon.
  - Camera view representation (Section 3.2.1): each image is wrapped in
    view-specific boundary tokens ``<|ego_start|> <image> <|ego_end|>``.
  - Unified action representation (Section 2.4): navigation targets follow the
    VLN convention — (dx, dy, dtheta) per waypoint over a fixed horizon (8
    waypoints per chunk for navigation, Section 4.1); the channel layout is
    zero-padded to a fixed K at load time, with a validity mask.
  - Per-dataset quantile normalization (eq. 5): 1st/99th percentiles per action
    channel over the generated corpus, mapped linearly to [-1, 1]. Actions are
    stored RAW here; the model applies the normalization (and the train script
    bakes the stats into the checkpoint config for evaluation).
  - Stage I T2A (Section 3.1): a text-only ``t2a_content`` prompt rides along in
    each record so T2A can withhold images while training the DiT as a pure
    language-to-action decompressor.
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from joynav.dataset.qwenvla_dataset import build_qwenvla_prompt, compute_qwenvla_norm
from joynav.dataset.vlnn1_annotation_utils import discrete_actions_to_ego_trajectory


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-folder", required=True, help="Comma-separated VLNN1/R2R/RxR roots.")
    parser.add_argument("--annotations-file", default="annotations.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-history-images", type=int, default=8)
    parser.add_argument("--action-horizon", type=int, default=8, help="Waypoints per chunk (paper: 8 for VLN).")
    parser.add_argument("--action-channels", type=int, default=32, help="Fixed channel dim K for the norm sidecar.")
    parser.add_argument("--control-frequency", type=float, default=2.0, help="Navigation video FPS (paper Section 3.2.4).")
    parser.add_argument("--trajectory-stride", type=int, default=3)
    parser.add_argument("--episodes-per-chunk", type=int, default=1000)
    parser.add_argument("--r2r-forward-step", type=float, default=0.25)
    parser.add_argument("--r2r-turn-angle", type=float, default=15.0)
    parser.add_argument("--r2r-step-stride", type=int, default=4)
    return parser.parse_args()


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def waypoints_to_deltas(waypoints: np.ndarray) -> np.ndarray:
    """Absolute ego waypoints [H,3] -> per-waypoint (dx, dy, dtheta) [H,3].

    The first delta is relative to the current pose (the ego origin), later
    deltas are relative to the previous waypoint — relative displacements, as
    used by the Qwen-VLA flow-matching decoder.
    """
    deltas = np.zeros_like(waypoints, dtype=np.float32)
    deltas[0] = waypoints[0]
    deltas[1:] = waypoints[1:] - waypoints[:-1]
    deltas[:, 2] = wrap_angle(deltas[:, 2])
    return deltas


def detect_source(annotations: List[Dict]) -> str:
    first = annotations[0] if annotations else {}
    if "continuous_actions" in first:
        return "vlnn1"
    if "actions" in first:
        return "r2r"
    return "vlnn1"


def get_instructions(item: Dict) -> List[str]:
    instructions = item.get("instructions") or ["Navigate to the goal."]
    if not isinstance(instructions, list):
        instructions = [instructions]
    return instructions or ["Navigate to the goal."]


def chunk_id(item: Dict, episodes_per_chunk: int) -> str:
    if item.get("chunk_id"):
        return str(item["chunk_id"])
    return f"chunk-{int(item['id']) // int(episodes_per_chunk):03d}"


def vlnn1_frame(root: str, item: Dict, step: int, trajectory_stride: int, episodes_per_chunk: int) -> str:
    episode_id = int(item["id"])
    original_frame_idx = int(step) * int(trajectory_stride)
    filename = f"episode_{episode_id:06d}_{original_frame_idx:03d}.jpg"
    return os.path.join(root, item["path"], "videos", chunk_id(item, episodes_per_chunk),
                        "observation.images.rgb", filename)


def sample_history_indices(step: int, num_history: int) -> List[int]:
    if step <= 0:
        return [0] * num_history
    if step < num_history:
        indices = list(range(step)) + [step - 1] * (num_history - step)
    else:
        indices = np.linspace(0, step - 1, num_history, dtype=np.int32).tolist()
    return [int(idx) for idx in indices[:num_history]]


def r2r_frame_map(item: Dict):
    frame_map = item.get("_frame_map")
    if frame_map is None:
        rgb_dir = os.path.join(item["video"], "rgb")
        names = sorted(name for name in os.listdir(rgb_dir) if name.lower().endswith((".jpg", ".jpeg", ".png")))
        frame_map = {int(Path(name).stem) - 1: name for name in names}
        item["_frame_map"] = frame_map
    return frame_map


def r2r_frame(item: Dict, frame_idx: int) -> str:
    frame_map = r2r_frame_map(item)
    if frame_idx not in frame_map:
        available = sorted(frame_map)
        below = [key for key in available if key <= frame_idx]
        frame_idx = below[-1] if below else available[0]
    return os.path.join(item["video"], "rgb", frame_map[frame_idx])


def build_record(instruction: str, images: List[str], waypoints_abs: np.ndarray, args) -> Dict:
    deltas = waypoints_to_deltas(waypoints_abs[: args.action_horizon].astype(np.float32))
    content, t2a_content = build_qwenvla_prompt(
        instruction,
        num_history=int(args.num_history_images),
        chunk_size=int(args.action_horizon),
        fps=float(args.control_frequency),
    )
    return {
        "messages": [
            {"role": "user", "content": content, "t2a_content": t2a_content},
            {
                "role": "assistant",
                "content": "",
                "qwenvla_actions": deltas.astype(float).tolist(),     # [H, 3] (dx, dy, dtheta)
                "qwenvla_mask": np.ones_like(deltas).astype(float).tolist(),
                "active_dims": 3,
            },
        ],
        "images": images,
    }


def generate_vlnn1_records(root: str, annotations: List[Dict], args, per_source_limit: int) -> List[Dict]:
    item_indices = list(range(len(annotations)))
    random.shuffle(item_indices)
    records = []
    for item_idx in item_indices:
        if len(records) >= per_source_limit:
            break
        item = annotations[item_idx]
        actions = item.get("continuous_actions") or {}
        steps = []
        for step_str, value in actions.items():
            try:
                step = int(step_str)
            except ValueError:
                continue
            arr = np.asarray(value, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[0] >= args.action_horizon + 1 and arr.shape[1] >= 3:
                steps.append(step)
        random.shuffle(steps)
        instructions = get_instructions(item)
        for step in steps[: max(len(instructions), 1)]:
            if len(records) >= per_source_limit:
                break
            current = vlnn1_frame(root, item, step, args.trajectory_stride, args.episodes_per_chunk)
            history = [
                vlnn1_frame(root, item, hist_step, args.trajectory_stride, args.episodes_per_chunk)
                for hist_step in sample_history_indices(step, args.num_history_images)
            ]
            if not os.path.exists(current) or any(not os.path.exists(path) for path in history):
                continue
            all_actions = np.asarray(item["continuous_actions"][str(step)], dtype=np.float32)
            waypoints = all_actions[1 : 1 + args.action_horizon, :3]
            ins_id = random.randrange(len(instructions))
            records.append(build_record(instructions[ins_id], history + [current], waypoints, args))
    return records


def generate_r2r_records(root: str, annotations: List[Dict], args, per_source_limit: int) -> List[Dict]:
    for item in annotations:
        item["video"] = os.path.join(root, item["video"])
        item["_actions"] = list(item.get("actions") or [])[1:] + [0]
    item_indices = list(range(len(annotations)))
    random.shuffle(item_indices)
    records = []
    for item_idx in item_indices:
        if len(records) >= per_source_limit:
            break
        item = annotations[item_idx]
        actions = item.get("_actions") or []
        if len(actions) < 2 or not os.path.isdir(os.path.join(item["video"], "rgb")):
            continue
        steps = list(range(0, len(actions), max(int(args.r2r_step_stride), 1)))
        random.shuffle(steps)
        instructions = get_instructions(item)
        for step in steps[: max(len(instructions), 1)]:
            if len(records) >= per_source_limit:
                break
            current = r2r_frame(item, step)
            history = [r2r_frame(item, hist_step) for hist_step in sample_history_indices(step, args.num_history_images)]
            if not os.path.exists(current) or any(not os.path.exists(path) for path in history):
                continue
            waypoints, _ = discrete_actions_to_ego_trajectory(
                item["_actions"],
                start=step,
                horizon=int(args.action_horizon),
                forward_step=float(args.r2r_forward_step),
                turn_angle_deg=float(args.r2r_turn_angle),
            )
            ins_id = random.randrange(len(instructions))
            records.append(build_record(instructions[ins_id], history + [current], waypoints, args))
    return records


def main():
    args = parse_args()
    random.seed(args.seed)
    roots = [root.strip() for root in args.video_folder.split(",") if root.strip()]
    per_source_limit = max(math.ceil(args.max_samples / max(len(roots), 1)), 1)

    records = []
    source_counts = {}
    for raw_root in roots:
        ratio = None
        if "%" in raw_root:
            raw_root, ratio_text = raw_root.split("%", 1)
            ratio = max(min(float(ratio_text) / 100.0, 1.0), 0.0)
        annotations_path = os.path.join(raw_root, args.annotations_file)
        if not os.path.exists(annotations_path):
            print(f"warning: missing {annotations_path}, skipping")
            continue
        with open(annotations_path, "r", encoding="utf-8") as file:
            annotations = json.load(file)
        if ratio is not None and annotations:
            annotations = random.sample(annotations, max(int(len(annotations) * ratio), 1))
        source = detect_source(annotations)
        if source == "r2r":
            new_records = generate_r2r_records(raw_root, annotations, args, per_source_limit)
        else:
            new_records = generate_vlnn1_records(raw_root, annotations, args, per_source_limit)
        source_counts[raw_root] = len(new_records)
        records.extend(new_records)

    random.shuffle(records)
    if args.max_samples > 0:
        records = records[: args.max_samples]

    # One navigation "dataset" in the paper's per-dataset quantile sense — all
    # three roots share the same VLN control convention (dx, dy, dtheta).
    norm = compute_qwenvla_norm(records, int(args.action_channels))
    if norm is not None:
        for record in records:
            record["messages"][1]["norm"] = norm

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as file:
        json.dump(records, file, indent=2)
    with open(output.with_suffix(".norm.json"), "w", encoding="utf-8") as file:
        json.dump(norm or {}, file, indent=2)
    print(json.dumps({"output": str(output), "records": len(records), "sources": source_counts}, indent=2))


if __name__ == "__main__":
    main()
