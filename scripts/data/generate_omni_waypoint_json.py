#!/usr/bin/env python3
"""Generate OmniNav-format waypoint supervision from JD-VLN mixed VLN data."""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from joynav.dataset.omni_waypoint_dataset import compute_omni_norm
from joynav.dataset.vlnn1_annotation_utils import discrete_actions_to_ego_trajectory


PROMPT_TEMPLATE = """You are an autonomous navigation robot. You will get a task with historical pictures and current pictures you see.
Based on these information, you need to decide your next {num_action_trunck} actions, which could involve <|left|>,<|right|>,<|forward|>. If you finish your mission, output <|stop|>. Here are some examples: <|left|><|forward|><|forward|><|stop|>, <|forward|><|forward|><|forward|><|left|><|forward|> or <|stop|>
# Your historical pictures are: {history_img_string}
# Your current observations is leftside: <image>, frontside: <image>, rightside: <image>
# Your mission is: {instruction}<|NAV|>
Output the waypoint"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-folder", required=True, help="Comma-separated VLNN1/R2R/RxR roots.")
    parser.add_argument("--annotations-file", default="annotations.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-history-images", type=int, default=20)
    parser.add_argument("--waypoint-number", type=int, default=5)
    parser.add_argument("--trajectory-stride", type=int, default=3)
    parser.add_argument("--episodes-per-chunk", type=int, default=1000)
    parser.add_argument("--r2r-forward-step", type=float, default=0.25)
    parser.add_argument("--r2r-turn-angle", type=float, default=15.0)
    parser.add_argument("--r2r-step-stride", type=int, default=4)
    parser.add_argument("--step-scale", type=float, default=0.3)
    return parser.parse_args()


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
    return os.path.join(
        root,
        item["path"],
        "videos",
        chunk_id(item, episodes_per_chunk),
        "observation.images.rgb",
        filename,
    )


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


def waypoint_payload(waypoints_m: np.ndarray, stop: float, waypoint_number: int, step_scale: float) -> Dict:
    waypoints_scaled = waypoints_m[:waypoint_number].copy()
    waypoints_scaled[:, :2] = waypoints_scaled[:, :2] / float(step_scale)
    heading = waypoints_scaled[:, 2].astype(float).tolist()
    arrive_list = [0.0] * waypoint_number
    if float(stop) > 0:
        arrive_list[-1] = 1.0
    return {
        "content": "",
        "role": "assistant",
        "input_waypoints": [[0.0, 0.0] for _ in range(21)],
        "gt_waypoints": waypoints_scaled.astype(float).tolist(),
        "gt_heading_angles": heading,
        "arrive": [float(stop)],
        "step_scale": float(step_scale),
        "train_branch": "continue",
        "arrive_list": arrive_list,
    }


def build_record(instruction: str, images: List[str], waypoints_m: np.ndarray, stop: float, args) -> Dict:
    history_placeholders = "<image>" * int(args.num_history_images)
    prompt = PROMPT_TEMPLATE.format(
        num_action_trunck=int(args.waypoint_number),
        history_img_string=history_placeholders,
        instruction=instruction,
    )
    return {
        "messages": [
            {"content": prompt, "role": "user"},
            waypoint_payload(waypoints_m, stop, int(args.waypoint_number), float(args.step_scale)),
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
            if arr.ndim == 2 and arr.shape[0] >= args.waypoint_number + 1 and arr.shape[1] >= 3:
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
            waypoints = all_actions[1 : 1 + args.waypoint_number, :3]
            stop_flags = item.get("stop_flags") or {}
            stop = float(stop_flags.get(str(step), 0.0))
            ins_id = random.randrange(len(instructions))
            records.append(build_record(instructions[ins_id], history + [current, current, current], waypoints, stop, args))
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
            waypoints, stop = discrete_actions_to_ego_trajectory(
                item["_actions"],
                start=step,
                horizon=int(args.waypoint_number),
                forward_step=float(args.r2r_forward_step),
                turn_angle_deg=float(args.r2r_turn_angle),
            )
            ins_id = random.randrange(len(instructions))
            records.append(build_record(instructions[ins_id], history + [current, current, current], waypoints, stop, args))
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
    norm = compute_omni_norm(records, int(args.waypoint_number))
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
