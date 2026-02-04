import argparse
import copy
import itertools
import json
import os
import random
import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

import habitat
import numpy as np
import quaternion
import torch
import tqdm
from depth_camera_filtering import filter_depth
from habitat import Env
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations.utils import images_to_video, observations_to_image
from habitat_baselines.config.default import get_config as get_habitat_config
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor
from transformers.image_utils import to_numpy_array
from qwen_vl_utils import process_vision_info
from joynav.utils.dist import *
from joynav.eval.base_evaluator import BaseEvaluator

DEFAULT_IMAGE_TOKEN = "<image>"

@dataclass
class Qwen3VLContEvaluatorArguments:
    """Arguments for VLN Evaluator - includes all parameters."""
    # Evaluator selection
    evaluator_type: str = field(default="vln", metadata={"help": "Type of evaluator: vln, etc."})
    
    # Model selection and loading
    model_type: str = field(default="qwen3_vl_dit", metadata={"help": "Model type: qwen2_5_vl_discrete, qwen3_vl_discrete, qwen3_vl_dit"})
    model_path: str = field(default="", metadata={"help": "Path to pretrained model"})
    
    # Habitat configuration
    habitat_config_path: str = field(default='configs/vln_r2r_cfg2.yaml', metadata={"help": "Path to Habitat config file"})
    eval_split: str = field(default='val_unseen', metadata={"help": "Evaluation split: val_seen, val_unseen, test"})
    output_path: str = field(default='./results/r2r/val_unseen', metadata={"help": "Output path for evaluation results"})
    
    # Model parameters
    min_pixels: Optional[int] = field(default=None, metadata={"help": "Minimum number of pixels for image processor"})
    max_pixels: int = field(default=392*392, metadata={"help": "Maximum number of pixels for image processor"})
    max_new_tokens: int = field(default=128, metadata={"help": "Maximum number of new tokens to generate"})
    
    # Evaluation parameters
    save_video: bool = field(default=False, metadata={"help": "Whether to save video of trajectories"})
    use_cache: bool = field(default=False, metadata={"help": "Whether to use KV cache during generation"})
    limit: int = field(default=-1, metadata={"help": "Limit number of evaluation episodes (0 for no limit)"})

    predict_type: str = field(default="discrtete", metadata={"help": "Type of prediction: discrete"})
    action_chunk_num: int = field(default=8, metadata={"help": "Number of actions to generate per chunk"})
    min_window_size: int = field(default=8, metadata={"help": "Minimum window size for action generation"})
    max_window_size: int = field(default=16, metadata={"help": "Maximum window size for action generation"})
    temporal_interval: int = field(default=8, metadata={"help": "Temporal interval for action generation"})

    do_smoothing: bool = field(default=True, metadata={"help": "Whether to smooth the pred traj"})
    
    # Distributed training parameters
    local_rank: int = field(default=0, metadata={"help": "Local rank for distributed training"})
    world_size: int = field(default=1, metadata={"help": "Number of distributed processes"})
    rank: int = field(default=0, metadata={"help": "Global rank"})
    gpu: int = field(default=0, metadata={"help": "GPU id"})
    port: str = field(default='2333', metadata={"help": "Port for distributed training"})
    dist_url: str = field(default='env://', metadata={"help": "URL for distributed training setup"})
    device: str = field(default='cuda', metadata={"help": "Device to use"})


def build_messages(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Extract and normalize images and videos
    images = item.get("image") or []
    if isinstance(images, str):
        images = [images]

    videos = item.get("video") or []
    if isinstance(videos, str):
        videos = [videos]

    # Build media pools with absolute paths
    image_pool = [
        {"type": "image", "image": img} for img in images
    ]
    video_pool = [
        {"type": "video", "video": vid} for vid in videos
    ]

    messages = []
    for turn in item["conversations"]:
        role = "user" if turn["from"] == "human" else "assistant"
        text: str = turn["value"]

        if role == "user":
            content = []
            # Split text by <image> or <video> placeholders while keeping delimiters
            text_parts = re.split(r"(<image>|<video>)", text)
            for seg in text_parts:
                if seg == "<image>":
                    if not image_pool:
                        raise ValueError(
                            "Number of <image> placeholders exceeds the number of provided images"
                        )
                    content.append(image_pool.pop(0))
                elif seg == "<video>":
                    if not video_pool:
                        raise ValueError(
                            "Number of <video> placeholders exceeds the number of provided videos"
                        )
                    content.append(video_pool.pop(0))
                elif seg.strip():
                    content.append({"type": "text", "text": seg.strip()})
            messages.append({"role": role, "content": content})
        else:
            # Assistant messages contain only text
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    # Check for unused media files
    if image_pool:
        raise ValueError(
            f"{len(image_pool)} image(s) remain unused (not consumed by placeholders)"
        )
    if video_pool:
        raise ValueError(
            f"{len(video_pool)} video(s) remain unused (not consumed by placeholders)"
        )

    return messages


def traj_to_actions(dp_actions, use_discrate_action=True):
    def reconstruct_xy_from_delta(delta_xyt):
        """
        Input:
            delta_xyt: [B, T, 3], dx, dy are position increments in global coordinates, dθ is heading difference (not used for position)
            start_xy: [B, 2] starting point
        Output:
            xy: [B, T+1, 2] reconstructed global trajectory
        """
        start_xy = np.zeros((len(delta_xyt), 2))
        delta_xy = delta_xyt[:, :, :2]  # Take dx, dy parts
        cumsum_xy = np.cumsum(delta_xy, axis=1)  # [B, T, 2]

        B = delta_xyt.shape[0]
        T = delta_xyt.shape[1]
        xy = np.zeros((B, T + 1, 2))
        xy[:, 0] = start_xy
        xy[:, 1:] = start_xy[:, None, :] + cumsum_xy

        return xy

    def trajectory_to_discrete_actions_close_to_goal(trajectory, step_size=0.125, turn_angle_deg=15, lookahead=4):
        actions = []
        yaw = 0.0
        pos = trajectory[0]
        turn_angle_rad = np.deg2rad(turn_angle_deg)
        traj = trajectory
        goal = trajectory[-1]

        def normalize_angle(angle):
            return (angle + np.pi) % (2 * np.pi) - np.pi

        while np.linalg.norm(pos - goal) > 0.2:
            # Find the nearest trajectory point index to current position
            dists = np.linalg.norm(traj - pos, axis=1)
            nearest_idx = np.argmin(dists)
            # Look ahead a bit (not exceeding trajectory end)
            target_idx = min(nearest_idx + lookahead, len(traj) - 1)
            target = traj[target_idx]
            # Target direction
            target_dir = target - pos
            if np.linalg.norm(target_dir) < 1e-6:
                break
            target_yaw = np.arctan2(target_dir[1], target_dir[0])
            # Difference between current yaw and target yaw
            delta_yaw = normalize_angle(target_yaw - yaw)
            n_turns = int(round(delta_yaw / turn_angle_rad))
            if n_turns > 0:
                actions += [2] * n_turns
            elif n_turns < 0:
                actions += [3] * (-n_turns)
            yaw = normalize_angle(yaw + n_turns * turn_angle_rad)

            # Move forward one step
            next_pos = pos + step_size * np.array([np.cos(yaw), np.sin(yaw)])

            # If moving forward one step makes us farther from goal, stop
            if np.linalg.norm(next_pos - goal) > np.linalg.norm(pos - goal):
                break

            actions.append(1)
            pos = next_pos

        return actions

    # unnormalize
    # dp_actions[:, :, :2] /= 4.0
    all_trajectory = reconstruct_xy_from_delta(dp_actions.float().cpu().numpy())
    trajectory = np.mean(all_trajectory, axis=0)
    if use_discrate_action:
        actions = trajectory_to_discrete_actions_close_to_goal(trajectory)
        return actions
    else:
        return trajectory
    

class Qwen3VLContEvaluator(BaseEvaluator):
    """VLN evaluator for discrete actions."""
    
    ARGUMENT_CLASS = Qwen3VLContEvaluatorArguments
    
    def __init__(
        self,
        config_path: str,
        split: str = "val_seen",
        env_num: int = 1,
        output_path: str = None,
        model: Any = None,
        processor: Any = None,
        epoch: int = 0,
        args = None,
    ):
        self.args = args
        self.device = torch.device('cuda')
        self.split = split
        self.env_num = env_num
        self.save_video = args.save_video
        self.output_path = output_path
        self.epoch = epoch
        self.config_path = config_path
        self.config = get_habitat_config(config_path)
        self.agent_config = get_agent_config(self.config.habitat.simulator)
        self.sim_sensors_config = self.config.habitat.simulator.agents.main_agent.sim_sensors
        self.use_cache = args.use_cache if hasattr(args, 'use_cache') else False

        with habitat.config.read_write(self.config):
            # self.config.habitat.task.measurements.success.success_distance=3.0
            self.config.habitat.dataset.split = self.split
            self.config.habitat.task.measurements.update(
                {
                    "top_down_map": TopDownMapMeasurementConfig(
                        map_padding=3,
                        map_resolution=1024,
                        draw_source=True,
                        draw_border=True,
                        draw_shortest_path=True,
                        draw_view_points=True,
                        draw_goal_positions=True,
                        draw_goal_aabbs=True,
                        fog_of_war=FogOfWarConfig(
                            draw=True,
                            visibility_dist=5.0,
                            fov=90,
                        ),
                    ),
                    "collisions": CollisionsMeasurementConfig(),
                }
            )

        print(f"config = {type(self.config)}")
        print(OmegaConf.to_yaml(self.config))

        self._camera_height = self.sim_sensors_config.rgb_sensor.position[1]
        self._min_depth = self.sim_sensors_config.depth_sensor.min_depth
        self._max_depth = self.sim_sensors_config.depth_sensor.max_depth

        camera_fov_rad = np.deg2rad(self.sim_sensors_config.depth_sensor.hfov)
        self._camera_fov = camera_fov_rad
        self._fx = self._fy = self.sim_sensors_config.depth_sensor.width / (2 * np.tan(camera_fov_rad / 2))

        self.model = model
        self.processor = processor

        prompt = f"You are an autonomous navigation assistant. Your task is to <instruction>. Devise an action sequence to follow the instruction using the four actions: TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE FORWARD (↑) by 25 centimeters, or STOP."
        self.conversation = [{"from": "human", "value": prompt}]
        self.actions2idx = OrderedDict({
            'STOP': [0],
            "↑": [1],
            "←": [2],
            "→": [3]
        })

        self.action_chunk_num = args.action_chunk_num
        self.min_window_size = args.min_window_size
        self.max_window_size = args.max_window_size
        self.temporal_interval = args.temporal_interval

        self.do_smoothing = args.do_smoothing

    def config_env(self) -> Env:
        env = Env(config=self.config)
        # env.episodes = env.episodes[0:1]
        return env

    def eval_action(self, idx) -> None:  # noqa: C901
        self.model.eval()
        env = self.config_env()
        scene_episode_dict = {}

        episodes = copy.deepcopy(env.episodes)
        if self.args.limit > 0:
            random.seed(42)
            random.shuffle(episodes)
            episodes = episodes[:self.args.limit]

        for episode in episodes:
            if episode.scene_id not in scene_episode_dict:
                scene_episode_dict[episode.scene_id] = []
            scene_episode_dict[episode.scene_id].append(episode)

        intrinsic_matrix = self.get_intrinsic_matrix(
            self.config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor
        )
        sucs, spls, oss, nes = [], [], [], []
        done_res = []

        if os.path.exists(os.path.join(self.output_path, 'result.json')):
            with open(os.path.join(self.output_path, 'result.json'), 'r') as f:
                for line in f.readlines():
                    res = json.loads(line)
                    done_res.append([res["scene_id"], res["episode_id"], res["episode_instruction"]])
                    if get_rank() == 0:  # noqa: F405
                        sucs.append(res['success'])
                        spls.append(res['spl'])
                        oss.append(res['os'])
                        nes.append(res['ne'])

        for scene in sorted(scene_episode_dict.keys()):
            episodes = scene_episode_dict[scene]
            scene_id = scene.split('/')[-2]
            print(f"scene_id = {scene_id}")
            process_bar = tqdm.tqdm(range(len(episodes[idx :: self.env_num])), desc=f"scene {scene_id}")
            for episode in episodes[idx :: self.env_num]:
                episode_instruction = (
                    episode.instruction.instruction_text
                    if 'objectnav' not in self.config_path
                    else episode.object_category
                )
                print("episode start", episode_instruction)
                episode_id = int(episode.episode_id)
                if [scene_id, episode_id, episode_instruction] in done_res:
                    continue

                env.current_episode = episode
                observations = env.reset()

                agent_state = env.sim.get_agent_state()
                rotation = agent_state.rotation
                translation = agent_state.position
                rotation_matrix = quaternion.as_rotation_matrix(rotation)
                transformation_matrix = np.eye(4)
                transformation_matrix[:3, :3] = rotation_matrix
                transformation_matrix[:3, 3] = translation

                os.makedirs(os.path.join(self.output_path, f'check_sim_{self.epoch}'), exist_ok=True)
                Image.fromarray(observations['rgb']).save(
                    os.path.join(self.output_path, f'check_sim_{self.epoch}', f'rgb_{idx}.jpg')
                )

                vis_frames = []
                step_id = 0

                if self.save_video:
                    os.makedirs(os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'), exist_ok=True)
                initial_height = env.sim.get_agent_state().position[1]

                rgb_list = []
                action_seq = []
                past_key_values = None
                output_ids = None
                llm_outputs = None
                source = {
                    "image": [],
                    "conversations": [],
                }

                action = None

                while not env.episode_over and step_id <= 500:
                    rgb = observations["rgb"]
                    depth = observations["depth"]
                    x, y = observations["gps"]
                    camera_yaw = observations["compass"][0]
                    depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                    depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                    depth = depth * 1000

                    agent_state = env.sim.get_agent_state()
                    height = agent_state.position[1] - initial_height
                    camera_position = np.array([x, -y, self._camera_height + height])
                    tf_camera_to_episodic = (
                        self.xyz_yaw_pitch_to_tf_matrix(camera_position, camera_yaw, np.deg2rad(30))
                        @ self.get_axis_align_matrix()
                    )

                    image = Image.fromarray(rgb).convert('RGB')
                    save_raw_image = image.copy()

                    rgb_list.append(image)

                    info = env.get_metrics()
                    if info['top_down_map'] is not None:
                        frame = observations_to_image({'rgb': np.asarray(save_raw_image)}, info)
                        vis_frames.append(frame)
                    
                    

                    if len(action_seq) == 0:
                        if self.use_cache:
                            inputs, source = self.prepare_inputs_use_cache(source, output_ids, llm_outputs, step_id, episode, rgb_list)
                        else:
                            inputs, source = self.prepare_inputs_no_cache(source, output_ids, llm_outputs, step_id, episode, rgb_list)
                        
                        input_len = inputs.input_ids.shape[1]
                        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                        cache_position = torch.arange(past_seen_tokens, input_len, device=inputs.input_ids.device) if past_key_values is not None else None

                        with torch.no_grad():
                            outputs = self.model.predict_action(**inputs)
                        

                        # import ipdb;ipdb.set_trace()
                        action_seq = self.parse_actions(outputs['action_pred'])


                        print(f"episode_id-{episode.episode_id} step_id-{step_id} === llm_outputs: {llm_outputs} === action_seq: {action_seq}")

                        if len(action_seq) > self.temporal_interval:  
                            action_seq = action_seq[:self.temporal_interval]  
                        if len(action_seq) == 0: ## if generated llm without Specific values
                            action_seq = [0]

                    action = action_seq.pop(0)
                    # print("step_id", step_id, "action", action)
                    
                    observations = env.step(action)
                    step_id += 1
                    # if step_id % self.num_frames == 0:
                    #     output_ids = None
                    #     past_key_values = None
                    #     llm_outputs = None
                    #     source = {
                    #         "image": [],
                    #         "conversations": [],
                    #     }

                process_bar.update(1)

                metrics = env.get_metrics()
                if self.save_video:
                    images_to_video(
                        vis_frames,
                        os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'),
                        f'{episode_id:04d}',
                        fps=6,
                        quality=9,
                    )
                vis_frames.clear()
                sucs.append(metrics['success'])
                spls.append(metrics['spl'])
                oss.append(metrics['oracle_success'])
                nes.append(metrics["distance_to_goal"])
                print(
                    f"scene_episode {scene_id}_{episode_id:04d} success: {metrics['success']}, spl: {metrics['spl']}, os: {metrics['oracle_success']}, ne: {metrics['distance_to_goal']}"
                )

                result = {
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "success": metrics["success"],
                    "spl": metrics["spl"],
                    "os": metrics['oracle_success'],
                    "ne": metrics["distance_to_goal"],
                    "steps": step_id,
                    "episode_instruction": episode_instruction,
                }

                with open(os.path.join(self.output_path, 'result.json'), 'a') as f:
                    f.write(json.dumps(result) + "\n")
        env.close()
        return (
            torch.tensor(sucs).to(self.device),
            torch.tensor(spls).to(self.device),
            torch.tensor(oss).to(self.device),
            torch.tensor(nes).to(self.device),
            torch.tensor(len(sucs)).to(self.device),
        )

    def parse_actions(self, action_pred, turn_angle_deg=15):
        if self.do_smoothing:
            target_pose = action_pred[0].sum(dim=0).cpu().float().numpy()
            actions = traj_to_actions(action_pred)

            def normalize_angle(angle):
                return (angle + np.pi) % (2 * np.pi) - np.pi

            res = 0
            turn_angle_rad = np.deg2rad(turn_angle_deg)
            for action in actions:
                if action == 2:
                    res += turn_angle_rad
                elif action == 3:
                    res -= turn_angle_rad
                res = normalize_angle(res)
            
            delta_yaw = normalize_angle(target_pose[-1] - res)

            if abs(delta_yaw) > turn_angle_rad:
                n_turns = int(round(abs(delta_yaw) / turn_angle_rad))
                for _ in range(n_turns):
                    actions.append(2 if delta_yaw > 0 else 3)
        else:
            atom_actions = torch.tensor([
                [0, 0, 0],
                [0.25, 0, 0],
                [0, 0, np.deg2rad(turn_angle_deg)],
                [0, 0, np.deg2rad(turn_angle_deg)],
            ], dtype=torch.float32)
            distances = (action_pred[0][:, None, :].cpu() - atom_actions).abs().sum(dim=-1)
            actions = distances.argmin(dim=-1).tolist()

        return actions

    def prepare_inputs_no_cache(self, source, output_ids, llm_outputs, step_id, episode, rgb_list):
        history_id = []
        conversation = copy.deepcopy(self.conversation)
        conversation[0]["value"] = conversation[0]["value"].replace(
            '<instruction>.', episode.instruction.instruction_text[:-1]
        )
        
        # iterate from min_window_size+1 to max_window_size
        image_num = self.min_window_size + 1 + \
            (step_id // self.temporal_interval)%(self.max_window_size - self.min_window_size)

        history_ids = list(range(step_id, -1, -self.temporal_interval))[:image_num][::-1]
        input_images = [rgb_list[i] for i in history_ids]
        history_str = (DEFAULT_IMAGE_TOKEN + '\n') * len(history_ids)
        conversation[0]["value"] += history_str

        source = {
            "image": input_images,
            "conversations": conversation,
        }
        messages = build_messages(source)

        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(self.model.device)
        print(f"episode_id-{episode.episode_id} step_id-{step_id} === history_id: {history_id} === decoded input_ids: ```{self.decode_input_ids(inputs['input_ids'])}```")
        return inputs, source


    def prepare_inputs_use_cache(self, source, output_ids, llm_outputs, step_id, episode, rgb_list):
        
        raise NotImplementedError("prepare_inputs_use_cache is not implemented yet.")


    def decode_input_ids(self, input_ids):
        """
        Replace <|image_pad|><|image_pad|>...<|image_pad|> with <|image_pad|>*N in the decoded string.
        """
        decoded_str = self.processor.tokenizer.decode(input_ids[0], skip_special_tokens=False)
        pattern = r'(<\|image_pad\|>)+'
        
        def replacer(match):
            count = match.group(0).count('<|image_pad|>')
            return f'<|image_pad|>*{count}'
        
        cleaned_str = re.sub(pattern, replacer, decoded_str)
        return cleaned_str