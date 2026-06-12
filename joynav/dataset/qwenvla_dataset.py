"""Qwen-VLA dataset: unified action-and-trajectory supervision (arXiv:2605.30280).

Consumes the JSON produced by ``scripts/data/generate_qwenvla_json.py``. Each
record carries the embodiment-aware prompt (paper Section 2.3), view-tagged
images (``<|ego_start|><image><|ego_end|>``, Section 3.2.1), and a raw action
target in the unified representation (Section 2.4):

    Y in R^{H x K}: leading c channels valid (VLN: c=3, (dx, dy, dtheta)),
    rest zero-padded; mask M in {0,1}^{H x K}.

Per-dataset 1%/99% quantile normalization statistics (eq. 5) ride along in each
record's ``norm`` field; the model normalizes targets with them and the train
entrypoint bakes them into ``config.qwenvla_norm`` for evaluation.

Stage handling (paper Section 3.1): Stage I T2A deliberately withholds images —
with ``qwenvla_stage=t2a`` this dataset drops the images and uses the record's
text-only T2A prompt, so the DiT learns a language-conditioned action prior
without any visual shortcut.
"""

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch

from .lazy_supervised_dataset import DataCollatorForSupervisedDataset, LazySupervisedDataset
from .lazy_supervised_dataset_args import LazySupervisedDatasetArguments

QWENVLA_VIEW_TOKENS = ["<|ego_start|>", "<|ego_end|>"]

# Single source of truth for the Qwen-VLA prompt. The generator bakes it into
# the training JSON and the evaluator rebuilds it at rollout time — they must
# stay byte-identical or the eval-time context distribution drifts from
# training. Embodiment-aware prompt per paper Section 2.3 (VLN instantiation);
# view boundary tokens per Section 3.2.1.
QWENVLA_EMBODIMENT_PROMPT = (
    "The robot is a wheeled navigation robot with mobile base. The control "
    "convention is planar waypoints (delta x, delta y, delta yaw) in meters and "
    "radians. The control frequency is {fps} Hz. Please predict the next "
    "{chunk_size} control actions to execute the following task: {instruction}"
)

QWENVLA_VISION_BLOCK = (
    "\n# Historical visual observations: {history_img_string}"
    "\n# Current visual observation: <|ego_start|><image><|ego_end|>"
)


def build_qwenvla_prompt(instruction: str, num_history: int, chunk_size: int, fps: float = 2.0):
    """Returns (full multimodal prompt, text-only T2A prompt)."""
    t2a_content = QWENVLA_EMBODIMENT_PROMPT.format(
        fps=float(fps), chunk_size=int(chunk_size), instruction=instruction
    )
    history_placeholders = "<|ego_start|><image><|ego_end|>" * int(num_history)
    content = t2a_content + QWENVLA_VISION_BLOCK.format(history_img_string=history_placeholders)
    return content, t2a_content


def strip_qwenvla_vision_block(content: str) -> str:
    """Fallback for records without a stored t2a_content: drop the image lines."""
    return "\n".join(
        line for line in content.split("\n") if "<image>" not in line
    ).rstrip()


@dataclass
class QwenVLADatasetArguments(LazySupervisedDatasetArguments):
    # The action horizon H and channel dim K are carried by the JSON itself
    # (action rows / norm length) and by the model's qwenvla_* arguments, so the
    # dataset does not re-declare them (HfArgumentParser parses dataset and
    # model args into one CLI namespace).
    qwenvla_json_path: str = field(default="", metadata={"help": "Qwen-VLA unified action JSON file."})
    qwenvla_stage: str = field(default="sft", metadata={"help": "t2a | cpt | sft. t2a withholds images."})

    def validate(self):
        super().validate()
        if not self.qwenvla_json_path:
            raise ValueError("qwenvla_json_path must be specified for QwenVLADataset")


def compute_qwenvla_norm(records, action_channels: int):
    """Per-dataset 1st/99th percentile statistics over all action deltas (paper eq. 5)."""
    rows = []
    for record in records:
        assistant = record["messages"][1]
        actions = assistant.get("qwenvla_actions")
        if actions is None:
            continue
        arr = np.asarray(actions, dtype=np.float32)
        mask = np.asarray(assistant.get("qwenvla_mask") or np.ones_like(arr), dtype=np.float32)
        for channel in range(arr.shape[1]):
            valid = mask[:, channel] > 0
            if valid.any():
                rows.append((channel, arr[valid, channel]))
    if not rows:
        return None
    q01 = np.full(action_channels, -1.0, dtype=np.float64)
    q99 = np.full(action_channels, 1.0, dtype=np.float64)
    per_channel = {}
    for channel, values in rows:
        per_channel.setdefault(channel, []).append(values)
    for channel, chunks in per_channel.items():
        values = np.concatenate(chunks)
        lo, hi = np.percentile(values, [1.0, 99.0])
        if np.isclose(lo, hi):
            lo, hi = lo - 1.0, hi + 1.0
        q01[channel], q99[channel] = float(lo), float(hi)
    return {"q01": q01.tolist(), "q99": q99.tolist()}


class QwenVLADataset(LazySupervisedDataset):
    ARGUMENT_CLASS = QwenVLADatasetArguments

    def __init__(self, processor, data_args):
        if not isinstance(data_args, QwenVLADatasetArguments):
            raise TypeError(f"data_args must be QwenVLADatasetArguments, got {type(data_args)}")
        self.qwenvla_json_path = str(data_args.qwenvla_json_path)
        self.stage = str(data_args.qwenvla_stage).lower()
        processor.tokenizer.add_special_tokens({"additional_special_tokens": QWENVLA_VIEW_TOKENS})
        super().__init__(processor, data_args)

    def load_data(self):
        with open(self.qwenvla_json_path, "r", encoding="utf-8") as file:
            records = json.load(file)
        if not isinstance(records, list):
            raise ValueError(f"Qwen-VLA JSON must be a list of records, got {type(records)}")
        norm = None
        for record in records:
            norm = record["messages"][1].get("norm")
            if norm is not None:
                break
        first_actions = np.asarray(records[0]["messages"][1]["qwenvla_actions"], dtype=np.float32)
        self.action_horizon = int(first_actions.shape[0])
        self.action_channels = int(len(norm["q01"])) if norm else max(int(first_actions.shape[1]), 32)
        if norm is None:
            norm = compute_qwenvla_norm(records, self.action_channels)
        self.qwenvla_norm = norm
        self.list_data_dict = records
        self.data_path = str(Path(self.qwenvla_json_path).resolve().parent)

    def prepare_sources(self, i):
        record = copy.deepcopy(self.list_data_dict[i])
        user, assistant = record["messages"][0], record["messages"][1]
        if self.stage == "t2a":
            # Stage I T2A: condition on text and the embodiment prompt only —
            # image tokens are fully suppressed (paper Sections 3.1 and 5.2.1).
            prompt = user.get("t2a_content") or strip_qwenvla_vision_block(user.get("content", ""))
            images = []
        else:
            prompt = user.get("content", "")
            images = record.get("images") or []
        return {
            "data_path": self.data_path,
            "image": images,
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": assistant.get("content", "")},
            ],
            "qwenvla_payload": assistant,
        }

    def _pad_actions(self, actions: np.ndarray) -> np.ndarray:
        padded = np.zeros((self.action_horizon, self.action_channels), dtype=np.float32)
        horizon = min(actions.shape[0], self.action_horizon)
        channels = min(actions.shape[1], self.action_channels)
        padded[:horizon, :channels] = actions[:horizon, :channels]
        return padded

    def _get_item(self, sources) -> Dict[str, torch.Tensor]:
        source = copy.deepcopy(sources[0] if isinstance(sources, list) else sources)
        payload = source.pop("qwenvla_payload")
        data_dict = super()._get_item([source])

        actions = np.asarray(payload["qwenvla_actions"], dtype=np.float32)
        mask = payload.get("qwenvla_mask")
        mask = np.asarray(mask, dtype=np.float32) if mask is not None else np.ones_like(actions)
        data_dict["qwenvla_actions"] = torch.from_numpy(self._pad_actions(actions))
        data_dict["qwenvla_action_mask"] = torch.from_numpy(self._pad_actions(mask))

        norm = payload.get("norm") or self.qwenvla_norm
        if norm is not None:
            data_dict["norm"] = {
                "q01": torch.as_tensor(norm["q01"], dtype=torch.float32),
                "q99": torch.as_tensor(norm["q99"], dtype=torch.float32),
            }
        return data_dict


class QwenVLACollator(DataCollatorForSupervisedDataset):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        actions = [instance.pop("qwenvla_actions") for instance in instances]
        masks = [instance.pop("qwenvla_action_mask") for instance in instances]
        norms = [instance.pop("norm") for instance in instances if "norm" in instance]

        batch = super().__call__(
            [{key: value for key, value in instance.items() if value is not None} for instance in instances]
        )
        batch["qwenvla_actions"] = torch.stack(actions, dim=0)
        batch["qwenvla_action_mask"] = torch.stack(masks, dim=0)
        if len(norms) == len(actions):
            batch["norm"] = {
                "q01": torch.stack([norm["q01"] for norm in norms], dim=0),
                "q99": torch.stack([norm["q99"] for norm in norms], dim=0),
            }
        return batch


QwenVLADataset.COLLATOR_CLASS = QwenVLACollator
