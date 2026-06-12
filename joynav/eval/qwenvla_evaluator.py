"""Qwen-VLA Habitat evaluator (arXiv:2605.30280, Section 5.1.3).

Mirrors the paper's VLN-CE evaluation: the model predicts a chunk of 8
navigation waypoints — (dx, dy, dtheta) relative displacements — and we execute
them with a sliding-window waypoint action ("we implement a sliding-window
waypoint action compatible with the predicted trajectory of Qwen-VLA"): the
predicted deltas are integrated into ego waypoints, converted to discrete
VLN-CE actions, only the first ``replan_every`` are executed, then the model
re-plans from the new observation. STOP is emitted when the predicted
trajectory stalls (total displacement below ``stall_distance`` — the trained
delta targets go to zero at the goal because episodes end with stop frames).

The prompt is identical to training: embodiment-aware prompt (Section 2.3) +
``<|ego_start|><image><|ego_end|>`` view-tagged sparse history and current
frame (Section 3.2.1).
"""

import copy
import json
import os
import random
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import tqdm
from habitat.utils.visualizations.utils import images_to_video, observations_to_image
from PIL import Image

from joynav.eval.qwen3_vl_lm_head_evaluator import (
    Qwen3VLLMHeadEvaluatorArguments,
    QwenVLLMHeadEvaluator,
    build_messages,
)
from joynav.dataset.qwenvla_dataset import build_qwenvla_prompt
from joynav.eval.qwen3_5_omega_trajectory_head_evaluator import trajectory_to_discrete_actions_3d
from joynav.utils.dist import get_rank


@dataclass
class QwenVLAEvaluatorArguments(Qwen3VLLMHeadEvaluatorArguments):
    evaluator_type: str = field(default="qwenvla")
    model_type: str = field(default="qwenvla")
    num_history: int = field(default=8)
    control_frequency: float = field(default=2.0)
    replan_every: int = field(default=1, metadata={"help": "Sliding window: actions executed before re-planning."})
    stall_distance: float = field(default=0.05, metadata={"help": "STOP when the predicted trajectory moves less than this."})
    # Qwen-VLA action expert config (architecture fields are restored from the
    # checkpoint config by eval_habitat.build_model_config; CLI only overrides
    # inference-time knobs).
    qwenvla_action_horizon: int = field(default=8)
    qwenvla_action_channels: int = field(default=32)
    qwenvla_active_dims: int = field(default=3)
    qwenvla_dit_hidden: int = field(default=1536)
    qwenvla_dit_layers: int = field(default=16)
    qwenvla_dit_heads: int = field(default=16)
    qwenvla_dit_mlp_dim: int = field(default=10240)
    qwenvla_dit_dropout: float = field(default=0.0)
    qwenvla_num_inference_steps: int = field(default=10)
    qwenvla_time_dist: str = field(default="beta")
    qwenvla_beta_alpha: float = field(default=1.5)
    qwenvla_beta_beta: float = field(default=1.0)
    qwenvla_noise_s: float = field(default=0.999)
    qwenvla_lambda_act: float = field(default=1.0)
    qwenvla_lambda_vl: float = field(default=0.1)


class QwenVLAEvaluator(QwenVLLMHeadEvaluator):
    ARGUMENT_CLASS = QwenVLAEvaluatorArguments

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_cache = False
        self.action_horizon = int(getattr(self.args, "qwenvla_action_horizon", 8))
        self.active_dims = int(getattr(self.args, "qwenvla_active_dims", 3))

    def _history_indices(self, rgb_count: int) -> List[int]:
        num_history = max(int(self.num_history), 1)
        if rgb_count <= 1:
            return [0] * num_history
        latest = rgb_count - 1
        if latest < num_history:
            indices = list(range(latest)) + [max(latest - 1, 0)] * (num_history - latest)
        else:
            indices = np.linspace(0, latest - 1, num_history, dtype=np.int32).tolist()
        return [int(max(min(idx, latest), 0)) for idx in indices[:num_history]]

    def prepare_qwenvla_inputs(self, step_id, episode, rgb_list):
        history_id = self._history_indices(len(rgb_list))
        # Same prompt builder as the training-data generator (byte-identical).
        prompt, _ = build_qwenvla_prompt(
            episode.instruction.instruction_text,
            num_history=len(history_id),
            chunk_size=self.action_horizon,
            fps=float(getattr(self.args, "control_frequency", 2.0)),
        )
        source = {
            "image": [rgb_list[idx] for idx in history_id] + [rgb_list[-1]],
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": ""},
            ],
        }
        messages = build_messages(self.prepare_processor_source(source))
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        if os.environ.get("JOYNAV_QWENVLA_EVAL_DEBUG", "0") == "1":
            print(
                f"episode_id-{episode.episode_id} step_id-{step_id} === history_id: {history_id} === "
                f"decoded input_ids: ```{self.decode_input_ids(inputs['input_ids'])}```"
            )
        return inputs

    def _deltas_to_actions(self, action_pred: torch.Tensor) -> List[int]:
        """Integrate predicted (dx, dy, dtheta) deltas into ego waypoints and
        convert them to discrete VLN-CE actions (sliding-window execution)."""
        deltas = action_pred.detach().float().cpu()
        if deltas.dim() == 3:
            deltas = deltas[0]
        deltas = deltas[: self.action_horizon, : self.active_dims]
        trajectory = torch.cumsum(deltas, dim=0)
        trajectory[:, 2] = torch.atan2(torch.sin(trajectory[:, 2]), torch.cos(trajectory[:, 2]))
        actions = trajectory_to_discrete_actions_3d(
            trajectory,
            forward_step=float(self.config.habitat.simulator.forward_step_size),
            turn_angle_deg=float(self.config.habitat.simulator.turn_angle),
            max_actions=int(self.action_chunk_num),
            stall_distance=float(self.args.stall_distance),
        )[0]
        replan_every = max(int(self.args.replan_every), 1)
        return (actions[:replan_every] if len(actions) > replan_every else actions) or [0]

    def eval_action(self, idx) -> None:
        self.model.eval()
        env = self.config_env()
        scene_episode_dict = {}
        episodes = copy.deepcopy(env.episodes)
        if self.args.limit > 0:
            random.seed(42)
            random.shuffle(episodes)
            episodes = episodes[: self.args.limit]
        for episode in episodes:
            scene_episode_dict.setdefault(episode.scene_id, []).append(episode)

        sucs, spls, oss, nes, ndtws = [], [], [], [], []
        done_res = []
        result_path = os.path.join(self.output_path, "result.json")
        if os.path.exists(result_path):
            with open(result_path, "r") as file:
                for line in file:
                    res = json.loads(line)
                    done_res.append([res["scene_id"], res["episode_id"], res["episode_instruction"]])
                    if get_rank() == 0:
                        sucs.append(res["success"])
                        spls.append(res["spl"])
                        oss.append(res["os"])
                        nes.append(res["ne"])
                        ndtws.append(res.get("ndtw", 0))

        for scene in sorted(scene_episode_dict.keys()):
            scene_id = scene.split("/")[-2]
            episodes = scene_episode_dict[scene]
            process_bar = tqdm.tqdm(range(len(episodes[idx :: self.env_num])), desc=f"scene {scene_id}")
            for episode in episodes[idx :: self.env_num]:
                episode_instruction = (
                    episode.instruction.instruction_text
                    if "objectnav" not in self.config_path
                    else episode.object_category
                )
                episode_id = int(episode.episode_id)
                if [scene_id, episode_id, episode_instruction] in done_res:
                    continue

                env.current_episode = episode
                observations = env.reset()
                vis_frames = []
                step_id = 0
                rgb_list = []
                action_seq = []

                while not env.episode_over and step_id <= 500:
                    image = Image.fromarray(observations["rgb"]).convert("RGB")
                    rgb_list.append(image)
                    info = env.get_metrics()
                    if info["top_down_map"] is not None:
                        vis_frames.append(observations_to_image({"rgb": np.asarray(image)}, info))

                    if len(action_seq) == 0:
                        inputs = self.prepare_qwenvla_inputs(step_id, episode, rgb_list)
                        with torch.no_grad():
                            outputs = self.model.predict_actions(**inputs)
                        action_seq = self._deltas_to_actions(outputs.action_pred)
                        if os.environ.get("JOYNAV_QWENVLA_EVAL_DEBUG", "0") == "1":
                            print(
                                f"episode_id-{episode.episode_id} step_id-{step_id} === "
                                f"action_pred={outputs.action_pred[0, :, :self.active_dims].detach().float().cpu().tolist()} "
                                f"action_seq={action_seq}"
                            )

                    observations = env.step(action_seq.pop(0))
                    step_id += 1

                process_bar.update(1)
                metrics = env.get_metrics()
                if self.should_save_navigation_video(metrics):
                    video_dir = os.path.join(self.output_path, f"vis_{self.epoch}", f"{scene_id}")
                    os.makedirs(video_dir, exist_ok=True)
                    images_to_video(vis_frames, video_dir, f"{episode_id:04d}", fps=6, quality=9)
                vis_frames.clear()

                sucs.append(metrics["success"])
                spls.append(metrics["spl"])
                oss.append(metrics["oracle_success"])
                nes.append(metrics["distance_to_goal"])
                ndtws.append(metrics.get("ndtw", 0))
                result = {
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "success": metrics["success"],
                    "spl": metrics["spl"],
                    "os": metrics["oracle_success"],
                    "ne": metrics["distance_to_goal"],
                    "ndtw": metrics.get("ndtw", 0),
                    "steps": step_id,
                    "episode_instruction": episode_instruction,
                }
                with open(result_path, "a") as file:
                    file.write(json.dumps(result) + "\n")
        env.close()
        return (
            torch.tensor(sucs).to(self.device),
            torch.tensor(spls).to(self.device),
            torch.tensor(oss).to(self.device),
            torch.tensor(nes).to(self.device),
            torch.tensor(ndtws).to(self.device),
            torch.tensor(len(sucs)).to(self.device),
        )
