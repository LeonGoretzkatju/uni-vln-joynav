import argparse
import copy
import itertools
import json
import os
import random
import re
from collections import OrderedDict
from pathlib import Path
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
from joynav.dataset.lazy_supervised_dataset import _build_messages as build_training_messages
from joynav.utils.dist import *
from joynav.eval.base_evaluator import BaseEvaluator

DEFAULT_IMAGE_TOKEN = "<image>"

@dataclass
class Qwen3VLLMHeadEvaluatorArguments:
    """Arguments for VLN Evaluator - includes all parameters."""
    # Evaluator selection
    evaluator_type: str = field(default="vln", metadata={"help": "Type of evaluator: vln, etc."})
    
    # Model selection and loading
    model_type: str = field(default="qwen3_vl_discrete", metadata={"help": "Model type: qwen2_5_vl_discrete, qwen3_vl_discrete, qwen3_vl_dit"})
    model_path: str = field(default="", metadata={"help": "Path to pretrained model"})
    
    # Habitat configuration
    habitat_config_path: str = field(default='configs/vln_r2r.yaml', metadata={"help": "Path to Habitat config file"})
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
    num_frames: int = field(default=32, metadata={"help": "Number of environment steps in one interleaved dialogue window"})
    num_history: int = field(default=8, metadata={"help": "Number of sparse history frames at each dialogue window reset"})

    # VLN-specific parameters
    action_chunk_num: int = field(default=8, metadata={"help": "Number of actions to generate per chunk"})
    min_window_size: int = field(default=9, metadata={"help": "Minimum window size for action generation"})
    max_window_size: int = field(default=16, metadata={"help": "Maximum window size for action generation"})
    temporal_interval: int = field(default=4, metadata={"help": "Temporal interval for action generation"})
    
    # Distributed training parameters
    local_rank: int = field(default=0, metadata={"help": "Local rank for distributed training"})
    world_size: int = field(default=1, metadata={"help": "Number of distributed processes"})
    rank: int = field(default=0, metadata={"help": "Global rank"})
    gpu: int = field(default=0, metadata={"help": "GPU id"})
    port: str = field(default='2333', metadata={"help": "Port for distributed training"})
    dist_url: str = field(default='env://', metadata={"help": "URL for distributed training setup"})
    device: str = field(default='cuda', metadata={"help": "Device to use"})


def build_messages(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    base_path = Path(item.get("data_path", ""))
    return build_training_messages(item, base_path)


class QwenVLLMHeadEvaluator(BaseEvaluator):
    """Shared VLN LM-head evaluator for Qwen-VL models."""
    
    ARGUMENT_CLASS = Qwen3VLLMHeadEvaluatorArguments
    
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
        self.conjunctions = [
            'you can see ',
            'in front of you is ',
            'there is ',
            'you can spot ',
            'you are toward the ',
            'ahead of you is ',
            'in your sight is '
        ]

        # VLN-specifc parameters
        self.num_frames = args.num_frames
        self.num_history = args.num_history
        self.action_chunk_num = args.action_chunk_num
        self.min_window_size = args.min_window_size
        self.max_window_size = args.max_window_size
        self.temporal_interval = args.temporal_interval

    def prepare_input_images(self, input_images):
        return input_images

    def prepare_processor_source(self, source):
        return source

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
        sucs, spls, oss, nes, ndtws = [], [], [], [], []
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
                        ndtws.append(res.get('ndtw', 0))

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
                accum_input_ids = None
                llm_outputs = None
                previous_assistant_text = None
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
                            if self.require_refresh_window(step_id) or past_key_values is None:
                                past_key_values = None
                                accum_input_ids = None
                            inputs, text = self.prepare_inputs_use_cache(step_id, episode, rgb_list)
                            inputs["input_ids"] = torch.cat([accum_input_ids, inputs.input_ids], dim=1) if accum_input_ids is not None else inputs.input_ids
                            inputs["attention_mask"] = torch.ones_like(inputs.input_ids)
                        else:
                            inputs, source = self.prepare_inputs_no_cache(
                                source,
                                previous_assistant_text,
                                step_id,
                                episode,
                                rgb_list,
                            )

                        input_len = inputs.input_ids.shape[1]
                        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                        cache_position = torch.arange(past_seen_tokens, input_len, device=inputs.input_ids.device) if past_key_values is not None else None

                        with torch.no_grad():
                            outputs = self.model.generate(**inputs, max_new_tokens=self.args.max_new_tokens, num_beams=1, do_sample=False, use_cache=self.use_cache, return_dict_in_generate=True, past_key_values=past_key_values, cache_position=cache_position)

                        output_ids = outputs.sequences
                        past_key_values = outputs.past_key_values

                        if past_key_values is not None:
                            past_key_values, accum_input_ids = self.update_past_key_values(past_key_values, inputs["input_ids"], accum_input_ids)
                            print(f"KV cache updated: {self.decode_input_ids(accum_input_ids)}")


                        generated_token_ids = output_ids[0][inputs.input_ids.shape[1] :]
                        llm_outputs = self.decode_generated_text(generated_token_ids)
                        if not self.use_cache:
                            previous_assistant_text = llm_outputs

                        action_seq = self.parse_actions(llm_outputs)
                        print(f"episode_id-{episode.episode_id} step_id-{step_id} === llm_outputs: {llm_outputs} === action_seq: {action_seq}")

                        if len(action_seq) > self.action_chunk_num:  
                            action_seq = action_seq[:self.action_chunk_num]  
                        if len(action_seq) == 0: ## if generated llm without Specific values
                            action_seq = [0]

                    action = action_seq.pop(0)
                    
                    observations = env.step(action)
                    step_id += 1
                    if self.should_reset_interleaved_source(step_id):
                        llm_outputs = None
                        previous_assistant_text = None
                        source = {
                            "image": [],
                            "conversations": [],
                        }
                        if not self.use_cache:
                            past_key_values = None
                            accum_input_ids = None

                process_bar.update(1)

                metrics = env.get_metrics()
                if self.should_save_navigation_video(metrics):
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
                ndtws.append(metrics.get("ndtw", 0))
                print(
                    f"scene_episode {scene_id}_{episode_id:04d} success: {metrics['success']}, spl: {metrics['spl']}, os: {metrics['oracle_success']}, ne: {metrics['distance_to_goal']}, ndtw: {metrics.get('ndtw', 0)}"
                )

                result = {
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "success": metrics["success"],
                    "spl": metrics["spl"],
                    "os": metrics['oracle_success'],
                    "ne": metrics["distance_to_goal"],
                    "ndtw": metrics.get("ndtw", 0),
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
            torch.tensor(ndtws).to(self.device),
            torch.tensor(len(sucs)).to(self.device),
        )

    def parse_actions(self, output):
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        # import ipdb; ipdb.set_trace()
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        actions = itertools.chain.from_iterable(actions)
        return list(actions)

    def should_reset_interleaved_source(self, step_id):
        return step_id > 0 and step_id % self.num_frames == 0

    def should_save_navigation_video(self, metrics):
        if not self.save_video:
            return False
        return float(metrics.get("success", 0.0)) == 1.0

    def build_interleaved_source(self, source, previous_assistant_text, step_id, episode, rgb_list):
        history_id = []
        prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
        if previous_assistant_text is None:
            conversation = copy.deepcopy(self.conversation)
            conversation[0]["value"] = conversation[0]["value"].replace(
                '<instruction>.', episode.instruction.instruction_text
            )
            if step_id != 0:
                num_history = max(int(self.num_history), 1)
                history_stride = max(int(step_id) // num_history, 1)
                history_id = np.arange(0, int(step_id), history_stride, dtype=np.int32).tolist()
                if len(history_id) > num_history:
                    history_id = history_id[:num_history]
                history_str = (DEFAULT_IMAGE_TOKEN + '\n') * len(history_id)
                conversation[0]["value"] += f' These are your historical observations: {history_str}.'
            conversation[0]["value"] += f" {prompt}."
        else:
            conversation = [
                {"from": "gpt", "value": previous_assistant_text},
                {"from": "human", "value": f"{prompt}."},
            ]

        input_images = [rgb_list[i] for i in history_id] + rgb_list[-1:]

        source = {
            "image": source["image"] + input_images,
            "conversations": source["conversations"] + conversation,
        }
        return source, history_id

    def prepare_inputs_no_cache(self, source, previous_assistant_text, step_id, episode, rgb_list):
        source, history_id = self.build_interleaved_source(
            source,
            previous_assistant_text,
            step_id,
            episode,
            rgb_list,
        )
        processor_source = self.prepare_processor_source(source)
        messages = build_messages(processor_source)

        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(self.model.device)
        print(f"episode_id-{episode.episode_id} step_id-{step_id} === history_id: {history_id} === decoded input_ids: ```{self.decode_input_ids(inputs['input_ids'])}```")
        return inputs, source

    def require_refresh_window(self, step_id):
        sid = (step_id // self.temporal_interval) % (self.max_window_size - self.min_window_size + 1)  
        return sid == 0

    def prepare_inputs_use_cache(self, step_id, episode, rgb_list):

        if self.require_refresh_window(step_id):
            return self.prepare_inputs_no_cache(
                {"image": [], "conversations": []},
                None,
                step_id,
                episode,
                rgb_list,
            )
        
        prompt = DEFAULT_IMAGE_TOKEN
        conversation = [
            {"from": "human", "value": prompt}
        ]
        input_images = rgb_list[-1:]
        input_images = self.prepare_input_images(input_images)
        source = {
            "image": input_images,
            "conversations": conversation,
        }
        messages = build_messages(source) 

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Remove the extra system prompt
        text = text.replace("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n", "")
        # Remove the extra <|im_start|> token
        text = text.removeprefix("<|im_start|>user\n")

        images, videos = process_vision_info(messages)
        inputs = self.processor(text=text, images=images, videos=videos, return_tensors="pt").to(self.model.device)

        return inputs, text
    
    def update_past_key_values(self, past_key_values, input_ids, accum_input_ids):
        
        # locate the position of <|vision_end|>, get the last occurrence of <|vision_end|> in the input_ids
        vision_end_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        vision_end_positions = (input_ids == vision_end_token_id).nonzero(as_tuple=True)[1][-1].item()  
        retain_kv_cache_length =  vision_end_positions + 1  # +1 to include the <|vision_end|> token itself

        for layer in past_key_values.layers:
            keys, values = layer.keys, layer.values

            keys = keys[:, :, :retain_kv_cache_length, :]
            values = values[:, :, :retain_kv_cache_length, :]
            layer.keys = keys
            layer.values = values

        accum_input_ids = input_ids[:, :retain_kv_cache_length]
        return past_key_values, accum_input_ids


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

    def decode_generated_text(self, generated_token_ids):
        decoded_str = self.processor.tokenizer.decode(generated_token_ids, skip_special_tokens=False)
        terminators = [
            "<|im_end|>",
            getattr(self.processor.tokenizer, "eos_token", None),
            "<|endoftext|>",
        ]
        for terminator in terminators:
            if terminator and terminator in decoded_str:
                decoded_str = decoded_str.split(terminator, 1)[0]
        return decoded_str.strip()


class Qwen3VLLMHeadEvaluator(QwenVLLMHeadEvaluator):
    """Compatibility evaluator name for existing Qwen3-VL scripts."""
