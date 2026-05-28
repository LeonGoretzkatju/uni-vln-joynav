import copy
import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, List, Optional

import habitat
import numpy as np
import torch
import tqdm
from depth_camera_filtering import filter_depth
from habitat import Env
from habitat.utils.visualizations.utils import images_to_video, observations_to_image
from PIL import Image

from joynav.eval.qwen3_5_lm_head_sf_omega_evaluator import (
    Qwen3_5OmegaSpatialForcingEvaluator,
    Qwen3_5OmegaSpatialForcingEvaluatorArguments,
)
from joynav.eval.qwen3_vl_lm_head_evaluator import PROMPT_MODE_TRAINING_INTERMEDIATE
from joynav.utils.dist import get_rank


def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def trajectory_to_discrete_actions_3d(
    trajectory: torch.Tensor,
    forward_step: float,
    turn_angle_deg: float,
    max_actions: int = 8,
    stall_distance: float = 0.05,
) -> List[List[int]]:
    if trajectory.dim() == 2:
        trajectory = trajectory.unsqueeze(0)
    if trajectory.shape[-1] != 3:
        raise ValueError(f"Expected trajectory dim 3, got {trajectory.shape[-1]}")

    turn_angle = math.radians(float(turn_angle_deg))
    all_actions: List[List[int]] = []
    for batch_idx in range(trajectory.shape[0]):
        traj = trajectory[batch_idx].detach().float().cpu()
        if not torch.isfinite(traj).all():
            traj = torch.nan_to_num(traj, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.linalg.norm(traj[:, :2], dim=-1).max().item() <= stall_distance:
            all_actions.append([0])
            continue

        actions: List[int] = []
        agent_x, agent_y, agent_yaw = 0.0, 0.0, 0.0
        for point in traj:
            if len(actions) >= max_actions:
                break
            target_x, target_y, target_yaw = point.tolist()
            dx = target_x - agent_x
            dy = target_y - agent_y
            distance = math.hypot(dx, dy)
            if distance > forward_step * 0.1:
                angle_to_target = math.atan2(dy, dx)
                turn_steps = int(round(normalize_angle(angle_to_target - agent_yaw) / turn_angle))
                if turn_steps > 0:
                    actions.extend([2] * min(turn_steps, max_actions - len(actions)))
                elif turn_steps < 0:
                    actions.extend([3] * min(-turn_steps, max_actions - len(actions)))
                agent_yaw = normalize_angle(agent_yaw + turn_steps * turn_angle)

                forward_steps = max(int(round(distance / forward_step)), 0)
                actions.extend([1] * min(forward_steps, max_actions - len(actions)))
                agent_x += forward_steps * forward_step * math.cos(agent_yaw)
                agent_y += forward_steps * forward_step * math.sin(agent_yaw)

            final_turn = int(round(normalize_angle(target_yaw - agent_yaw) / turn_angle))
            if final_turn > 0:
                actions.extend([2] * min(final_turn, max_actions - len(actions)))
            elif final_turn < 0:
                actions.extend([3] * min(-final_turn, max_actions - len(actions)))
            agent_yaw = normalize_angle(agent_yaw + final_turn * turn_angle)

        all_actions.append(actions[:max_actions] or [0])
    return all_actions


@dataclass
class Qwen3_5OmegaTrajectoryEvaluatorArguments(Qwen3_5OmegaSpatialForcingEvaluatorArguments):
    action_chunk_num: int = field(default=8)
    stall_distance: float = field(default=0.05)
    trajectory_horizon: int = field(default=8)
    trajectory_dim: int = field(default=3)
    action_head_hidden_dim: Optional[int] = field(default=None)
    action_head_loss_weight: float = field(default=1.0)
    propagate_action_head_grad: bool = field(default=True)
    action_latent_layers: int = field(default=8)
    action_latent_dim: int = field(default=1536)
    action_latent_heads: int = field(default=16)
    action_num_inference_timesteps: int = field(default=4)
    nextdit_dim: int = field(default=384)
    nextdit_layers: int = field(default=12)
    nextdit_heads: int = field(default=6)
    nextdit_kv_heads: int = field(default=6)
    nextdit_num_inference_steps: int = field(default=10)
    nextdit_num_sample_trajs: int = field(default=1)
    nextdit_guidance_scale: float = field(default=1.0)


@dataclass
class Qwen3_5OmegaMLPTrajectoryEvaluatorArguments(Qwen3_5OmegaTrajectoryEvaluatorArguments):
    evaluator_type: str = field(default="qwen3_5_mlp_head_sf_omega")
    model_type: str = field(default="qwen3_5_mlp_head_sf_omega")


@dataclass
class Qwen3_5OmegaDiTTrajectoryEvaluatorArguments(Qwen3_5OmegaTrajectoryEvaluatorArguments):
    evaluator_type: str = field(default="qwen3_5_dit_head_sf_omega")
    model_type: str = field(default="qwen3_5_dit_head_sf_omega")


@dataclass
class Qwen3_5OmegaNextDiTTrajectoryEvaluatorArguments(Qwen3_5OmegaTrajectoryEvaluatorArguments):
    evaluator_type: str = field(default="qwen3_5_nextdit_head_sf_omega")
    model_type: str = field(default="qwen3_5_nextdit_head_sf_omega")


class Qwen3_5OmegaTrajectoryEvaluator(Qwen3_5OmegaSpatialForcingEvaluator):
    ARGUMENT_CLASS = Qwen3_5OmegaTrajectoryEvaluatorArguments

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conversation = [
            {
                "from": "human",
                "value": (
                    "You are an autonomous navigation assistant. Your task is to <instruction>. "
                    "Predict the next ego-centric trajectory waypoints."
                ),
            }
        ]
        self.action_token = "<|action|>"

    def get_generation_prompt_mode(self, step_id):
        return PROMPT_MODE_TRAINING_INTERMEDIATE

    def append_action_token(self, inputs):
        token_id = self.processor.tokenizer.convert_tokens_to_ids(self.action_token)
        if token_id is None or token_id < 0:
            raise ValueError(f"Tokenizer does not contain {self.action_token}")
        batch_size = inputs["input_ids"].shape[0]
        token_tensor = torch.full(
            (batch_size, 1),
            int(token_id),
            dtype=inputs["input_ids"].dtype,
            device=inputs["input_ids"].device,
        )
        inputs["input_ids"] = torch.cat([inputs["input_ids"], token_tensor], dim=1)
        if "attention_mask" in inputs:
            inputs["attention_mask"] = torch.cat(
                [
                    inputs["attention_mask"],
                    torch.ones((batch_size, 1), dtype=inputs["attention_mask"].dtype, device=inputs["attention_mask"].device),
                ],
                dim=1,
            )
        if "mm_token_type_ids" in inputs:
            inputs["mm_token_type_ids"] = torch.cat(
                [
                    inputs["mm_token_type_ids"],
                    torch.zeros((batch_size, 1), dtype=inputs["mm_token_type_ids"].dtype, device=inputs["mm_token_type_ids"].device),
                ],
                dim=1,
            )
        return inputs

    def parse_trajectory_actions(self, action_pred: torch.Tensor) -> List[int]:
        return trajectory_to_discrete_actions_3d(
            action_pred,
            forward_step=float(self.config.habitat.simulator.forward_step_size),
            turn_angle_deg=float(self.config.habitat.simulator.turn_angle),
            max_actions=int(self.action_chunk_num),
            stall_distance=float(self.args.stall_distance),
        )[0]

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
                initial_height = env.sim.get_agent_state().position[1]
                rgb_list = []
                action_seq = []
                previous_assistant_text = None
                source = {"image": [], "conversations": []}

                while not env.episode_over and step_id <= 500:
                    rgb = observations["rgb"]
                    depth = observations["depth"]
                    x, y = observations["gps"]
                    camera_yaw = observations["compass"][0]
                    depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                    depth = depth * (self._max_depth - self._min_depth) + self._min_depth

                    agent_state = env.sim.get_agent_state()
                    height = agent_state.position[1] - initial_height
                    camera_position = np.array([x, -y, self._camera_height + height])
                    _ = (
                        self.xyz_yaw_pitch_to_tf_matrix(camera_position, camera_yaw, np.deg2rad(30))
                        @ self.get_axis_align_matrix()
                    )

                    image = Image.fromarray(rgb).convert("RGB")
                    rgb_list.append(image)
                    info = env.get_metrics()
                    if info["top_down_map"] is not None:
                        vis_frames.append(observations_to_image({"rgb": np.asarray(image)}, info))

                    if len(action_seq) == 0:
                        inputs, source = self.prepare_inputs_no_cache(
                            source,
                            previous_assistant_text,
                            step_id,
                            episode,
                            rgb_list,
                        )
                        inputs = self.append_action_token(inputs)
                        with torch.no_grad():
                            outputs = self.model.predict_action(**inputs)
                        action_pred = getattr(outputs, "action_pred", None)
                        if action_pred is None and isinstance(outputs, dict):
                            action_pred = outputs["action_pred"]
                        action_seq = self.parse_trajectory_actions(action_pred)
                        previous_assistant_text = self.action_token
                        if len(action_seq) > self.action_chunk_num:
                            action_seq = action_seq[: self.action_chunk_num]
                        if len(action_seq) == 0:
                            action_seq = [0]

                    observations = env.step(action_seq.pop(0))
                    step_id += 1
                    if self.should_reset_interleaved_source(step_id):
                        previous_assistant_text = None
                        source = {"image": [], "conversations": []}

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


class Qwen3_5OmegaMLPTrajectoryEvaluator(Qwen3_5OmegaTrajectoryEvaluator):
    ARGUMENT_CLASS = Qwen3_5OmegaMLPTrajectoryEvaluatorArguments


class Qwen3_5OmegaDiTTrajectoryEvaluator(Qwen3_5OmegaTrajectoryEvaluator):
    ARGUMENT_CLASS = Qwen3_5OmegaDiTTrajectoryEvaluatorArguments


class Qwen3_5OmegaNextDiTTrajectoryEvaluator(Qwen3_5OmegaTrajectoryEvaluator):
    ARGUMENT_CLASS = Qwen3_5OmegaNextDiTTrajectoryEvaluatorArguments
