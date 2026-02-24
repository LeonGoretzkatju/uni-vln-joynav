import argparse
import copy
import itertools
import json
import os
import random
import re
import math
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


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi] range."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def continuous_to_discrete_actions(
    trajectory: torch.Tensor,
    x_scale: float = 1.0,
    y_scale: float = 0.433,
    forward_step: float = 0.125,
    turn_angle_deg: float = 15.0,
    stop_threshold: float = 0.5,
    max_actions: int = 500,
    smooth_trajectory: bool = True,
) -> List[List[int]]:
    """
    Alternative implementation that treats trajectory points as absolute positions
    relative to the starting point, not relative to the previous point.
    
    This version accumulates positions and converts the full path to actions.
    
    Args:
        trajectory: torch.Tensor of shape [batch_size, num_points, 5]
        x_scale: Scale factor for x normalization
        y_scale: Scale factor for y normalization  
        forward_step: Distance moved per FORWARD action
        turn_angle_deg: Degrees turned per LEFT/RIGHT action
        stop_threshold: Threshold for stop flag
        max_actions: Maximum number of actions
        smooth_trajectory: Whether to smooth the trajectory before conversion
    
    Returns:
        List of action lists for each batch item.
    """
    if trajectory.dim() == 2:
        trajectory = trajectory.unsqueeze(0)
    
    batch_size, num_points, feat_dim = trajectory.shape
    assert feat_dim == 5, f"Expected 5 features per point, got {feat_dim}"
    
    turn_angle_rad = math.radians(turn_angle_deg)
    all_actions = []
    
    for b in range(batch_size):
        actions = []
        
        # Current agent state
        agent_x, agent_y = 0.0, 0.0
        agent_yaw = 0.0
        
        for p in range(num_points):
            if len(actions) >= max_actions:
                break
            
            # Target position (denormalized, relative to start)
            target_x = trajectory[b, p, 0].item() * x_scale
            target_y = trajectory[b, p, 1].item() * y_scale
            target_cos = trajectory[b, p, 2].item()
            target_sin = trajectory[b, p, 3].item()
            stop_flag = trajectory[b, p, 4].item()
            
            # Calculate displacement from current agent position
            dx = target_x - agent_x
            dy = target_y - agent_y
            distance = math.sqrt(dx ** 2 + dy ** 2)
            
            if distance > forward_step * 0.1:
                # Angle to target from current position
                angle_to_target = math.atan2(dy, dx)
                angle_diff = normalize_angle(angle_to_target - agent_yaw)
                
                # Turn to face target
                turn_steps = round(angle_diff / turn_angle_rad)
                if turn_steps > 0:
                    actions.extend([2] * min(turn_steps, max_actions - len(actions)))
                elif turn_steps < 0:
                    actions.extend([3] * min(abs(turn_steps), max_actions - len(actions)))
                
                agent_yaw = normalize_angle(agent_yaw + turn_steps * turn_angle_rad)
                
                # Move forward
                forward_steps = round(distance / forward_step)
                actions.extend([1] * min(forward_steps, max_actions - len(actions)))
                
                # Update agent position
                agent_x += forward_steps * forward_step * math.cos(agent_yaw)
                agent_y += forward_steps * forward_step * math.sin(agent_yaw)
            
            # Adjust to target orientation
            target_yaw = math.atan2(target_sin, target_cos)
            final_diff = normalize_angle(target_yaw - agent_yaw)
            final_turn = round(final_diff / turn_angle_rad)
            
            if final_turn > 0:
                actions.extend([2] * min(final_turn, max_actions - len(actions)))
            elif final_turn < 0:
                actions.extend([3] * min(abs(final_turn), max_actions - len(actions)))
            
            agent_yaw = normalize_angle(agent_yaw + final_turn * turn_angle_rad)
            
            if stop_flag > stop_threshold:
                actions.append(0)
                break
        
        all_actions.append(actions)
    
    return all_actions

@dataclass
class Qwen3VLMLPEvaluatorArguments:
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
    

class Qwen3VLMLPEvaluator(BaseEvaluator):
    """VLN evaluator for discrete actions."""
    
    ARGUMENT_CLASS = Qwen3VLMLPEvaluatorArguments
    
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


    def config_env(self) -> Env:
        env = Env(config=self.config)
        # env.episodes = env.episodes[0:1]
        return env

    def process_input_tokens(self, inputs):
        """
        prefix: ...<|vision_end|><|im_end|>\n<|im_start|>assistant\n
        pred:   <|action|>discrete_actions
        --------
        inference: input_ids = input_ids + <|action|>
                                            hidden_states -> action_head -> trajs
        """
        action_special_tokens = self.processor.tokenizer.additional_special_tokens # ['<|action|>']
        action_special_token_ids = self.processor.tokenizer.convert_tokens_to_ids(action_special_tokens)
        batch_size = inputs["input_ids"].shape[0]
        action_token_tensor = torch.tensor([action_special_token_ids] * batch_size, device=inputs["input_ids"].device)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], action_token_tensor], dim=1)
        if "attention_mask" in inputs:
            attention_mask = inputs["attention_mask"]
            ones = torch.ones((batch_size, 1), device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, ones], dim=1)
            inputs["attention_mask"] = attention_mask

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

                        if self.args.model_type == "qwen3_vl_mlp_head":
                            self.process_input_tokens(inputs)

                        input_len = inputs.input_ids.shape[1]
                        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                        cache_position = torch.arange(past_seen_tokens, input_len, device=inputs.input_ids.device) if past_key_values is not None else None

                        with torch.no_grad():
                            
                            if self.args.model_type == "qwen3_vl_mlp_head":
                                outputs = self.model.predict_action(**inputs,past_key_values=past_key_values, cache_position=cache_position)
                            else:
                                outputs = self.model.predict_action(**inputs)

                        action_seq = self.parse_actions(outputs['action_pred'])

                        print(f"episode_id-{episode.episode_id} step_id-{step_id} === llm_outputs: {llm_outputs} === action_seq: {action_seq}")

                        if len(action_seq) > self.action_chunk_num:  
                            action_seq = action_seq[:self.action_chunk_num]  
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

    def parse_actions(self, action_pred: torch.Tensor) -> List[int]:
        """
        Convert continuous trajectory prediction to discrete actions.
        
        Args:
            action_pred: torch.Tensor of shape [batch_size, num_points, 5]
                - [:, :, 0]: x position relative to current (normalized by 1.0)
                - [:, :, 1]: y position relative to current (normalized by 0.433)
                - [:, :, 2]: cos(yaw)
                - [:, :, 3]: sin(yaw)
                - [:, :, 4]: stop flag (0-1)
        
        Returns:
            List of discrete actions: 0=STOP, 1=FORWARD, 2=LEFT, 3=RIGHT
        """
        # Convert continuous trajectory to discrete actions
        actions_list = continuous_to_discrete_actions(
            trajectory=action_pred,
            x_scale=1.0,
            y_scale=0.433,
            forward_step=0.125,
            turn_angle_deg=15.0,
            stop_threshold=0.5,
            max_actions=self.action_chunk_num,
        )
        return actions_list[0]

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