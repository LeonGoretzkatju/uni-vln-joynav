import copy
import json
import os
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .continuous_vlnn1_action_dataset_args import ContinuousVLNN1ActionDatasetArguments
from .lazy_supervised_dataset import (
    DEFAULT_IMAGE_TOKEN,
    DataCollatorForSupervisedDataset,
    LazySupervisedDataset,
    rank0_print,
)
from .vln_action_omega_spatial_forcing_dataset import (
    load_qwen_images_for_omega_direct,
)
from vggt_omega.utils.load_fn import load_and_preprocess_images


class ContinuousVLNN1ActionDataset(LazySupervisedDataset):
    ARGUMENT_CLASS = ContinuousVLNN1ActionDatasetArguments

    def __init__(self, processor, data_args):
        if not isinstance(data_args, ContinuousVLNN1ActionDatasetArguments):
            raise TypeError(f"data_args must be ContinuousVLNN1ActionDatasetArguments, got {type(data_args)}")

        self.num_history_frames = int(data_args.num_history_frames)
        self.action_chunk_size = int(data_args.action_chunk_size)
        self.action_dim = int(data_args.action_dim)
        self.trajectory_stride = int(data_args.trajectory_stride)
        self.image_type = str(data_args.image_type)
        self.episodes_per_chunk = int(data_args.episodes_per_chunk)
        self.include_current_frame = bool(data_args.include_current_frame)
        self.action_token = str(data_args.action_token)
        self.interleaved_num_chunks = int(getattr(data_args, "interleaved_num_chunks", 4))

        processor.tokenizer.add_special_tokens({"additional_special_tokens": [self.action_token]})
        super().__init__(processor, data_args)
        self.action_token_id = self.tokenizer.convert_tokens_to_ids(self.action_token)

        prompt = (
            "You are an autonomous navigation assistant. Your task is to <instruction>. "
            "Based on the historical observations, predict the next ego-centric trajectory waypoints."
        )
        self.conversations = [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": self.action_token},
        ]
        rank0_print(
            "ContinuousVLNN1ActionDataset: "
            f"samples={len(self.list_data_dict)}, horizon={self.action_chunk_size}, action_dim={self.action_dim}"
        )

    def _get_chunk_id(self, episode_id: int, item: Dict | None = None) -> str:
        if item is not None and item.get("chunk_id"):
            return str(item["chunk_id"])
        return f"chunk-{int(episode_id) // self.episodes_per_chunk:03d}"

    def _get_image_subdir_and_ext(self) -> Tuple[str, str]:
        if self.image_type == "rgb":
            return "observation.images.rgb", "jpg"
        return "observation.images.depth", "png"

    def _step_to_original_frame(self, step_idx: int) -> int:
        return int(step_idx) * self.trajectory_stride

    def _get_frame_path(self, video_folder: str, item: Dict, step_idx: int) -> str:
        subdir, ext = self._get_image_subdir_and_ext()
        episode_id = int(item["id"])
        original_frame_idx = self._step_to_original_frame(step_idx)
        filename = f"episode_{episode_id:06d}_{original_frame_idx:03d}.{ext}"
        return os.path.join(
            video_folder,
            item["path"],
            "videos",
            self._get_chunk_id(episode_id, item),
            subdir,
            filename,
        )

    def _get_instructions(self, item: Dict) -> List[str]:
        instructions = item.get("instructions") or ["Navigate to the goal."]
        if not isinstance(instructions, list):
            instructions = [instructions]
        return instructions or ["Navigate to the goal."]

    def _get_action_array(self, item: Dict, step: int) -> np.ndarray:
        return np.asarray(item["continuous_actions"][str(step)], dtype=np.float32)

    def _get_stop_flags(self, item: Dict, steps: List[int]) -> np.ndarray | None:
        """Per-chunk stop labels (1.0 near the goal) aligned with ``steps``.

        Returns ``None`` when the annotation predates the stop-flag schema so the
        caller can fall back to a geometry-derived label.
        """
        stop_flags = item.get("stop_flags")
        if not stop_flags:
            return None
        return np.asarray([float(stop_flags.get(str(step), 0.0)) for step in steps], dtype=np.float32)

    @staticmethod
    def _derive_stop_targets(continuous_actions: torch.Tensor, eps: float = 0.05) -> torch.Tensor:
        """Fallback stop label: a chunk with ~zero future translation is a stop."""
        actions = continuous_actions if continuous_actions.dim() == 3 else continuous_actions.unsqueeze(0)
        max_disp = torch.linalg.norm(actions[..., :2], dim=-1).amax(dim=-1)
        return (max_disp < eps).float()

    def _sorted_valid_steps(self, item: Dict) -> List[int]:
        valid_steps = []
        expected_shape = (self.action_chunk_size + 1, self.action_dim)
        for step_str, actions in item["continuous_actions"].items():
            try:
                step = int(step_str)
            except ValueError:
                continue
            if np.asarray(actions).shape == expected_shape:
                valid_steps.append(step)
        return sorted(valid_steps)

    def _sample_history_step_indices(self, step_idx: int) -> List[int]:
        if self.include_current_frame:
            num_history = max(self.num_history_frames - 1, 0)
            if step_idx == 0:
                return [0]
            if step_idx <= num_history:
                indices = list(range(step_idx + 1))
            else:
                indices = np.linspace(0, step_idx - 1, num_history, dtype=np.int32).tolist()
                indices.append(step_idx)
        else:
            if step_idx == 0:
                indices = []
            elif step_idx <= self.num_history_frames:
                indices = list(range(step_idx))
            else:
                indices = np.linspace(0, step_idx - 1, self.num_history_frames, dtype=np.int32).tolist()
        return [int(idx) for idx in indices]

    def _sample_history_before_step_indices(self, step_idx: int) -> List[int]:
        if step_idx <= 0:
            return []
        if step_idx <= self.num_history_frames:
            indices = list(range(step_idx))
        else:
            indices = np.linspace(0, step_idx - 1, self.num_history_frames, dtype=np.int32).tolist()
        return [int(idx) for idx in indices]

    def load_data(self):
        self.nav_data = []
        for video_folder in self.data_args.video_folder.split(","):
            video_folder = video_folder.strip()
            if not video_folder:
                continue
            annotations_path = os.path.join(video_folder, self.data_args.annotations_file)
            if not os.path.exists(annotations_path):
                rank0_print(f"Warning: {annotations_path} not found, skipping")
                continue
            with open(annotations_path, "r", encoding="utf-8") as file:
                annotations = json.load(file)
            for item in annotations:
                item["video_folder"] = video_folder
            self.nav_data.extend(annotations)

        list_data_dict = []
        expected_shape = (self.action_chunk_size + 1, self.action_dim)
        skipped = 0
        for ep_id, item in enumerate(self.nav_data):
            continuous_actions = item.get("continuous_actions") or {}
            if not item.get("path") or not continuous_actions:
                skipped += 1
                continue
            instructions = item.get("instructions") or ["Navigate to the goal."]
            if not isinstance(instructions, list):
                instructions = [instructions]
            for step_str, actions in continuous_actions.items():
                try:
                    step = int(step_str)
                except ValueError:
                    continue
                if np.asarray(actions).shape != expected_shape:
                    continue
                for ins_id in range(max(len(instructions), 1)):
                    list_data_dict.append((ep_id, ins_id, step))

        rank0_print(f"Loaded {len(list_data_dict)} VLNN1 continuous samples from {len(self.nav_data)} episodes.")
        if skipped:
            rank0_print(f"Skipped {skipped} VLNN1 episodes with missing path/actions.")
        random.shuffle(list_data_dict)
        self.list_data_dict = list_data_dict

    def prepare_sources(self, i):
        ep_id, ins_id, step = self.list_data_dict[i]
        item = self.nav_data[ep_id]
        instructions = self._get_instructions(item)

        all_actions = self._get_action_array(item, step)
        continuous_actions = all_actions[1:]
        stop_targets = self._get_stop_flags(item, [step])

        history_frames = [
            self._get_frame_path(item["video_folder"], item, step_idx)
            for step_idx in self._sample_history_step_indices(step)
        ]
        if not history_frames:
            history_frames = [self._get_frame_path(item["video_folder"], item, step)]

        conversations = copy.deepcopy(self.conversations)
        history_str = (DEFAULT_IMAGE_TOKEN + "\n") * len(history_frames)
        conversations[0]["value"] += f" These are your historical observations:\n{history_str}"
        conversations[0]["value"] = conversations[0]["value"].replace("<instruction>", instructions[ins_id])

        return {
            "image": history_frames,
            "conversations": conversations,
            "continuous_actions": continuous_actions,
            "stop_targets": stop_targets,
        }

    def _get_item(self, sources) -> Dict[str, torch.Tensor]:
        source = copy.deepcopy(sources[0] if isinstance(sources, list) else sources)
        continuous_actions = torch.as_tensor(source.pop("continuous_actions"), dtype=torch.float32)
        stop_targets = source.pop("stop_targets", None)
        data_dict = super()._get_item([source])

        input_ids = data_dict["input_ids"][0]
        select_mask = input_ids == self.action_token_id
        expected_tokens = continuous_actions.shape[0] if continuous_actions.dim() == 3 else 1
        if int(select_mask.sum().item()) != expected_tokens:
            rank0_print(
                f"Warning: expected {expected_tokens} {self.action_token} token(s), "
                f"found {int(select_mask.sum().item())}"
            )

        if stop_targets is None:
            stop_targets = self._derive_stop_targets(continuous_actions)
        else:
            stop_targets = torch.as_tensor(stop_targets, dtype=torch.float32).reshape(-1)

        data_dict["continuous_actions"] = continuous_actions
        data_dict["select_mask"] = select_mask
        data_dict["stop_targets"] = stop_targets
        return data_dict


class ContinuousVLNN1ActionInterleavedDataset(ContinuousVLNN1ActionDataset):
    def prepare_sources(self, i):
        ep_id, ins_id, step = self.list_data_dict[i]
        item = self.nav_data[ep_id]
        instructions = self._get_instructions(item)

        candidate_steps = [valid_step for valid_step in self._sorted_valid_steps(item) if valid_step >= step]
        selected_steps = candidate_steps[: self.interleaved_num_chunks] or [step]
        continuous_actions = np.stack(
            [self._get_action_array(item, selected_step)[1:] for selected_step in selected_steps],
            axis=0,
        )
        stop_targets = self._get_stop_flags(item, selected_steps)

        history_frames = [
            self._get_frame_path(item["video_folder"], item, step_idx)
            for step_idx in self._sample_history_before_step_indices(selected_steps[0])
        ]
        sample_frames = [
            self._get_frame_path(item["video_folder"], item, selected_step)
            for selected_step in selected_steps
        ]
        image_files = history_frames + sample_frames

        prompt = self.conversations[0]["value"].replace("<instruction>", instructions[ins_id])
        if history_frames:
            history_str = (DEFAULT_IMAGE_TOKEN + "\n") * len(history_frames)
            prompt += f" These are your historical observations:\n{history_str}"
        prompt += f" Current observation:\n{DEFAULT_IMAGE_TOKEN}"

        conversations = [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": self.action_token},
        ]
        for _ in selected_steps[1:]:
            conversations.extend(
                [
                    {"from": "human", "value": f"Next observation:\n{DEFAULT_IMAGE_TOKEN}"},
                    {"from": "gpt", "value": self.action_token},
                ]
            )

        return {
            "image": image_files,
            "conversations": conversations,
            "continuous_actions": continuous_actions,
            "stop_targets": stop_targets,
        }


class ContinuousVLNN1ActionOmegaSpatialForcingMixin:
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
            sf_image_tensors = load_and_preprocess_images(
                image_files,
                image_resolution=self.image_resolution,
                patch_size=self.teacher_patch_size,
            )
            if self.omega_mode == "text_align_force_qwen":
                source["image"] = load_qwen_images_for_omega_direct(
                    image_files,
                    sf_image_tensors.shape[-2:],
                    self.spatial_merge_size,
                )

        item = super()._get_item([source])
        if sf_image_tensors is not None:
            item["sf_image_tensors"] = sf_image_tensors
        return item


class ContinuousVLNN1ActionOmegaSpatialForcingDataset(
    ContinuousVLNN1ActionOmegaSpatialForcingMixin,
    ContinuousVLNN1ActionDataset,
):
    pass


class ContinuousVLNN1ActionInterleavedOmegaSpatialForcingDataset(
    ContinuousVLNN1ActionOmegaSpatialForcingMixin,
    ContinuousVLNN1ActionInterleavedDataset,
):
    pass


class ContinuousVLNN1ActionCollator(DataCollatorForSupervisedDataset):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        continuous_actions = [instance.pop("continuous_actions") for instance in instances]
        select_masks = [instance.pop("select_mask") for instance in instances]
        stop_targets = [instance.pop("stop_targets") for instance in instances if "stop_targets" in instance]
        sf_image_tensors = [instance.pop("sf_image_tensors") for instance in instances if "sf_image_tensors" in instance]

        cleaned_instances = []
        for instance in instances:
            cleaned_instances.append({key: value for key, value in instance.items() if value is not None})

        batch = super().__call__(cleaned_instances)
        max_len = batch["input_ids"].shape[1]
        padded_select_masks = []
        for mask in select_masks:
            mask = mask[:max_len]
            if mask.shape[0] < max_len:
                mask = torch.cat([mask, torch.zeros(max_len - mask.shape[0], dtype=torch.bool)], dim=0)
            padded_select_masks.append(mask)

        if all(actions.dim() == 2 for actions in continuous_actions):
            batch["continuous_actions"] = torch.stack(continuous_actions, dim=0)
        else:
            normalized_actions = [
                actions.unsqueeze(0) if actions.dim() == 2 else actions
                for actions in continuous_actions
            ]
            batch["continuous_actions"] = torch.cat(normalized_actions, dim=0)
        batch["select_mask"] = torch.stack(padded_select_masks, dim=0)
        if len(stop_targets) == len(continuous_actions):
            # Flatten in the same (sample, chunk) order as the selected action tokens
            # so stop_targets[k] aligns with continuous_actions row k.
            batch["stop_targets"] = torch.cat([target.reshape(-1) for target in stop_targets], dim=0)
        if len(sf_image_tensors) == len(continuous_actions):
            batch["sf_image_tensors"] = sf_image_tensors
        return batch


ContinuousVLNN1ActionDataset.COLLATOR_CLASS = ContinuousVLNN1ActionCollator
ContinuousVLNN1ActionOmegaSpatialForcingDataset.COLLATOR_CLASS = ContinuousVLNN1ActionCollator
ContinuousVLNN1ActionInterleavedDataset.COLLATOR_CLASS = ContinuousVLNN1ActionCollator
ContinuousVLNN1ActionInterleavedOmegaSpatialForcingDataset.COLLATOR_CLASS = ContinuousVLNN1ActionCollator
