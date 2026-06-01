import copy
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch

from .lazy_supervised_dataset import DataCollatorForSupervisedDataset, LazySupervisedDataset
from .lazy_supervised_dataset_args import LazySupervisedDatasetArguments
from .vln_action_omega_spatial_forcing_dataset import load_qwen_images_for_omega_direct
from vggt_omega.utils.load_fn import load_and_preprocess_images


@dataclass
class OmniWaypointDatasetArguments(LazySupervisedDatasetArguments):
    omni_json_path: str = field(default="", metadata={"help": "OmniNav-format waypoint JSON file."})
    nav_token: str = field(default="<|NAV|>")
    waypoint_number: int = field(default=5)
    spatial_forcing_teacher_patch_size: int = field(default=16)
    spatial_forcing_image_resolution: int = field(default=256)

    def validate(self):
        super().validate()
        if not self.omni_json_path:
            raise ValueError("omni_json_path must be specified for OmniWaypointDataset")


def _load_omni_records(path: str):
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    global_norm = None
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and "messages" in payload:
        records = [payload]
        global_norm = payload.get("norm")
    elif isinstance(payload, dict):
        global_norm = payload.get("norm")
        for key in ("data", "records", "samples"):
            if key in payload:
                records = payload[key]
                break
        else:
            raise ValueError(f"Unsupported Omni JSON object keys: {sorted(payload)}")
    else:
        raise ValueError(f"Unsupported Omni JSON payload type: {type(payload)}")
    return records, global_norm


def _assistant_payload(record: Dict) -> Dict:
    messages = record.get("messages") or []
    if len(messages) < 2:
        raise ValueError("Omni record must contain user and assistant messages")
    return messages[1]


def _messages_to_conversations(record: Dict):
    conversations = []
    for message in record["messages"]:
        role = message.get("role")
        if role == "user":
            conversations.append({"from": "human", "value": message.get("content", "")})
        elif role == "assistant":
            conversations.append({"from": "gpt", "value": message.get("content", "")})
        else:
            raise ValueError(f"Unsupported Omni message role: {role}")
    return conversations


def _delta_from_waypoints(gt_waypoints, gt_heading_angles=None, arrive_list=None):
    gt_waypoints = np.asarray(gt_waypoints, dtype=np.float32)
    if gt_heading_angles is None:
        angles = gt_waypoints[:, 2] if gt_waypoints.shape[-1] >= 3 else np.zeros(gt_waypoints.shape[0], dtype=np.float32)
    else:
        angles = np.asarray(gt_heading_angles, dtype=np.float32)
    absolute = np.concatenate(
        [gt_waypoints[:, :2], np.sin(angles)[:, None], np.cos(angles)[:, None]],
        axis=-1,
    )
    delta = np.zeros_like(absolute, dtype=np.float32)
    delta[0] = absolute[0]
    delta[1:] = absolute[1:] - absolute[:-1]
    if arrive_list is not None:
        arrive = np.asarray(arrive_list, dtype=np.float32)[: delta.shape[0], None]
        delta = np.concatenate([delta, arrive], axis=-1)
    return delta


def compute_omni_norm(records, waypoint_number: int):
    deltas = []
    for record in records:
        assistant = _assistant_payload(record)
        gt_waypoints = assistant.get("gt_waypoints")
        if gt_waypoints is None:
            continue
        arrive_list = assistant.get("arrive_list") or [0.0] * waypoint_number
        delta = _delta_from_waypoints(
            gt_waypoints[:waypoint_number],
            assistant.get("gt_heading_angles"),
            arrive_list=arrive_list,
        )
        if delta.shape == (waypoint_number, 5):
            deltas.append(delta)
    if not deltas:
        return None
    stacked = np.stack(deltas, axis=0)
    min_vals = stacked.min(axis=0)
    max_vals = stacked.max(axis=0)
    mean_vals = stacked.mean(axis=0)
    std_vals = stacked.std(axis=0)
    # Expand only degenerate (near-constant) columns so normalization stays
    # well-defined on tiny smoke subsets, while preserving OmniNav's tight
    # data-driven min/max on real data. This matters for the sin/cos channels:
    # blanket-widening them to [-1, 1] would compress heading resolution
    # relative to the reference, which uses the observed delta range.
    same = np.isclose(min_vals, max_vals)
    min_vals[same] -= 1.0
    max_vals[same] += 1.0
    return {
        "min": min_vals.tolist(),
        "max": max_vals.tolist(),
        "cliped_min": min_vals.tolist(),
        "cliped_max": max_vals.tolist(),
        "mean": mean_vals.tolist(),
        "std": np.maximum(std_vals, 1e-6).tolist(),
    }


class OmniWaypointDataset(LazySupervisedDataset):
    ARGUMENT_CLASS = OmniWaypointDatasetArguments

    def __init__(self, processor, data_args):
        if not isinstance(data_args, OmniWaypointDatasetArguments):
            raise TypeError(f"data_args must be OmniWaypointDatasetArguments, got {type(data_args)}")
        self.omni_json_path = str(data_args.omni_json_path)
        self.nav_token = str(data_args.nav_token)
        self.waypoint_number = int(data_args.waypoint_number)
        processor.tokenizer.add_special_tokens({"additional_special_tokens": [self.nav_token]})
        super().__init__(processor, data_args)

    def load_data(self):
        records, global_norm = _load_omni_records(self.omni_json_path)
        if global_norm is None:
            global_norm = compute_omni_norm(records, self.waypoint_number)
        self.omni_norm = global_norm
        self.list_data_dict = records
        self.data_path = str(Path(self.omni_json_path).resolve().parent)

    def prepare_sources(self, i):
        record = copy.deepcopy(self.list_data_dict[i])
        assistant = _assistant_payload(record)
        if self.omni_norm is not None and "norm" not in assistant:
            assistant["norm"] = self.omni_norm
        return {
            "data_path": self.data_path,
            "image": record.get("images") or record.get("image") or [],
            "conversations": _messages_to_conversations(record),
            "omni_payload": assistant,
        }

    def _get_item(self, sources) -> Dict[str, torch.Tensor]:
        source = copy.deepcopy(sources[0] if isinstance(sources, list) else sources)
        payload = source.pop("omni_payload")
        data_dict = super()._get_item([source])

        data_dict["gt_waypoints"] = torch.as_tensor(payload["gt_waypoints"], dtype=torch.float32)[: self.waypoint_number]
        if "gt_heading_angles" in payload:
            data_dict["gt_heading_angles"] = torch.as_tensor(payload["gt_heading_angles"], dtype=torch.float32)[
                : self.waypoint_number
            ]
        if "arrive" in payload:
            data_dict["arrive"] = torch.as_tensor(payload["arrive"], dtype=torch.float32).reshape(-1)
        if "arrive_list" in payload:
            data_dict["arrive_list"] = torch.as_tensor(payload["arrive_list"], dtype=torch.float32)[
                : self.waypoint_number
            ]
        if "input_waypoints" in payload:
            data_dict["input_waypoints"] = torch.as_tensor(payload["input_waypoints"], dtype=torch.float32)
        if "norm" in payload:
            data_dict["norm"] = {
                "min": torch.as_tensor(payload["norm"]["min"], dtype=torch.float32),
                "max": torch.as_tensor(payload["norm"]["max"], dtype=torch.float32),
            }
        data_dict["train_branch"] = payload.get("train_branch", "continue")
        data_dict["step_scale"] = torch.tensor(float(payload.get("step_scale", 0.3)), dtype=torch.float32)
        return data_dict


class OmniWaypointOmegaSpatialForcingDataset(OmniWaypointDataset):
    def __init__(self, processor, data_args):
        self.teacher_patch_size = int(data_args.spatial_forcing_teacher_patch_size)
        self.image_resolution = int(data_args.spatial_forcing_image_resolution)
        self.omega_mode = str(getattr(data_args, "omega_mode", ""))
        self.spatial_merge_size = getattr(processor.image_processor, "merge_size", 2)
        super().__init__(processor, data_args)

    def _get_item(self, sources) -> Dict[str, torch.Tensor]:
        source = copy.deepcopy(sources[0] if isinstance(sources, list) else sources)
        image_files = source.get("image") or []
        if isinstance(image_files, str):
            image_files = [image_files]

        sf_image_tensors = None
        if image_files:
            resolved = [
                image if os.path.isabs(str(image)) else os.path.join(source.get("data_path", ""), str(image))
                for image in image_files
            ]
            sf_image_tensors = load_and_preprocess_images(
                resolved,
                image_resolution=self.image_resolution,
                patch_size=self.teacher_patch_size,
            )
            if self.omega_mode == "text_align_force_qwen":
                source["image"] = load_qwen_images_for_omega_direct(
                    resolved,
                    sf_image_tensors.shape[-2:],
                    self.spatial_merge_size,
                )

        item = super()._get_item([source])
        if sf_image_tensors is not None:
            item["sf_image_tensors"] = sf_image_tensors
        return item


class OmniWaypointCollator(DataCollatorForSupervisedDataset):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        gt_waypoints = [instance.pop("gt_waypoints") for instance in instances]
        gt_heading_angles = [instance.pop("gt_heading_angles") for instance in instances if "gt_heading_angles" in instance]
        arrive = [instance.pop("arrive") for instance in instances if "arrive" in instance]
        arrive_list = [instance.pop("arrive_list") for instance in instances if "arrive_list" in instance]
        input_waypoints = [instance.pop("input_waypoints") for instance in instances if "input_waypoints" in instance]
        norms = [instance.pop("norm") for instance in instances if "norm" in instance]
        step_scales = [instance.pop("step_scale") for instance in instances if "step_scale" in instance]
        for instance in instances:
            instance.pop("train_branch", None)
        sf_image_tensors = [instance.pop("sf_image_tensors") for instance in instances if "sf_image_tensors" in instance]

        batch = super().__call__([{key: value for key, value in instance.items() if value is not None} for instance in instances])
        batch["gt_waypoints"] = torch.stack(gt_waypoints, dim=0)
        if len(gt_heading_angles) == len(gt_waypoints):
            batch["gt_heading_angles"] = torch.stack(gt_heading_angles, dim=0)
        if len(arrive) == len(gt_waypoints):
            batch["arrive"] = torch.stack(arrive, dim=0)
        if len(arrive_list) == len(gt_waypoints):
            batch["arrive_list"] = torch.stack(arrive_list, dim=0)
        if len(input_waypoints) == len(gt_waypoints):
            batch["input_waypoints"] = torch.stack(input_waypoints, dim=0)
        if len(norms) == len(gt_waypoints):
            batch["norm"] = {
                "min": torch.stack([norm["min"] for norm in norms], dim=0),
                "max": torch.stack([norm["max"] for norm in norms], dim=0),
            }
        if len(step_scales) == len(gt_waypoints):
            batch["step_scale"] = torch.stack(step_scales, dim=0)
        if len(sf_image_tensors) == len(gt_waypoints):
            batch["sf_image_tensors"] = sf_image_tensors
        batch["train_branch"] = ["continue"]
        return batch


OmniWaypointDataset.COLLATOR_CLASS = OmniWaypointCollator
OmniWaypointOmegaSpatialForcingDataset.COLLATOR_CLASS = OmniWaypointCollator
