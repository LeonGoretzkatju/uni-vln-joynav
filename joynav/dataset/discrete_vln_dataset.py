"""
This dataset implements Vision-Language Navigation (VLN) with discrete action prediction.
The observation is variable length images, and the model predicts a sequence of discrete actions.
Input:
    [Instruction] + [History images] + [Current image]

Output:
    [Action sequence] (e.g., "↑→↑STOP")
"""

import json
import random
import os
import re
import time
import torch
import copy
import numpy as np
from typing import Dict, Optional, Sequence, List, Tuple, Any
from .lazy_supervised_dataset import (
    LazySupervisedDataset, 
    preprocess_qwen_visual, 
    rank0_print, 
    DEFAULT_IMAGE_TOKEN, 
    DataCollatorForSupervisedDataset
)

from .base_dataset_args import BaseDatasetArguments
from .discrete_vln_dataset_args import DiscreteVLNDatasetArguments


class DiscreteVLNDataset(LazySupervisedDataset):
    """
    Dataset for Vision-Language Navigation with Action Prediction.
    Inherits from LazySupervisedDataset and provides VLN-specific functionality.
    """

    # Specify the corresponding collator class
    ARGUMENT_CLASS = DiscreteVLNDatasetArguments

    def __init__(self, processor, data_args: BaseDatasetArguments):
        """
        Initialize VLN Action Dataset.
        
        Args:
            processor: The processor for handling images/videos
            data_args: VLN-specific dataset arguments
        """
        # Validate args type
        if not isinstance(data_args, DiscreteVLNDatasetArguments):
            raise TypeError(
                f"data_args must be DiscreteVLNDatasetArguments, got {type(data_args)}"
            )
        
        # # VLN-specific attributes (set before calling super().__init__)
        self.min_window_size = data_args.min_window_size
        self.max_window_size = data_args.max_window_size
        self.action_chunk_num = data_args.action_chunk_num
        self.sampling_stride = data_args.sampling_stride

        # Continuous action representation parameters
        self.add_continuous_action = data_args.add_continuous_action
        self.x_norm_factor = data_args.x_norm_factor
        self.y_norm_factor = data_args.y_norm_factor

        super().__init__(processor, data_args)
        
        # VLN-specific setup
        self.idx2actions = {
            '0': 'STOP',
            '1': "↑",
            '2': "←",
            '3': "→",
        }
        
        prompt = (
            "You are an autonomous navigation assistant. Your task is to <instruction>. "
            "Devise an action sequence to follow the instruction using the four actions: "
            "TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE FORWARD (↑) by 25 centimeters, "
            "or STOP."
        )
        answer = ""
        self.conversations = [
            {"from": "human", "value": prompt}, 
            {"from": "gpt", "value": answer}
        ]

        rank0_print(f"================ Sample data ================")
        for i in random.sample(range(len(self.list_data_dict)), 3):
            sample = self.prepare_sources(i)
            rank0_print(f"Sample {i}: {sample}")
            full_result = preprocess_qwen_visual([sample], processor)
            text = self.decode_input_ids(full_result["input_ids"])
            rank0_print(f"Sample {i} - decoded_input_ids: ```{text}```")
        rank0_print(f"=============================================")
    
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
        
    def load_data(self):
        """Load VLN navigation data from video folders."""
        video_folder = self.data_args.video_folder.split(',')
        
        self.nav_data = []
        for vf in video_folder:
            anno_json = json.load(open(os.path.join(vf, 'annotations.json'), 'r'))
            for tdata in anno_json:
                tdata['video'] = os.path.join(vf, tdata['video'])
            self.nav_data += anno_json
        
        list_data_dict = []
        for ep_id, item in enumerate(self.nav_data):
            instructions = item['instructions']
            actions = item['actions']
            actions_len = len(actions)
            if actions_len < 4:
                continue

            if not isinstance(instructions, list):
                instructions = [instructions]
                
            for ins_id in range(len(instructions)):
                valid_idx = 0

                if actions_len - valid_idx < 4:
                    continue
                
                num_rounds = (actions_len - valid_idx) // self.sampling_stride
                for n in range(num_rounds + 1):
                    if n * self.sampling_stride == actions_len - valid_idx:
                        continue
                    list_data_dict.append((ep_id, ins_id, n * self.sampling_stride, valid_idx))
                # for start_idx in range(0, actions_len - valid_idx):
                #     list_data_dict.append((ep_id, ins_id, start_idx, valid_idx))
        rank0_print(f"Loaded {len(list_data_dict)} samples from {len(self.nav_data)} annotations.")
        random.shuffle(list_data_dict)

        self.list_data_dict = list_data_dict

    def actions2text(self, actions):
        converted_sequence = []         
        for action in actions:
            act_text = self.idx2actions[str(action)]
            if type(act_text) == list:
                act_text = random.choice(act_text)
            converted_sequence.append(act_text)
        
        text = ''.join(converted_sequence)
        return text

    
    def prepare_sources(self, i):

        def get_action_chunk(actions, start_idx):
            action_len = len(actions)
            ret = []
            for idx in range(start_idx, len(actions)):
                if actions[idx] == 1:
                    ret.extend([1, 1])  # each move forward is splited into two forward
                else:
                    ret.append(actions[idx])
                if len(ret) >= self.action_chunk_num:
                    break
            ret = ret[:self.action_chunk_num]
            return ret

        ep_id, ins_id, start_idx, valid_idx = self.list_data_dict[i]
        data = self.nav_data[ep_id]
        video_path = data['video']
        video_frames = sorted(os.listdir(os.path.join(video_path, 'rgb')))

        instructions = data.get("instructions", None)
        if not isinstance(instructions, list):
            instructions = [instructions]

        actions = data['actions'][1+valid_idx:] + [0]
        # actions = np.array(actions)[start_idx: start_idx + self.action_chunk_num]
        actions = get_action_chunk(actions, start_idx)
    
        frame_num = random.randint(self.min_window_size, self.max_window_size)
        history_step_ids = []
        if start_idx > 0:
            history_step_ids = np.linspace(valid_idx, valid_idx + start_idx, 
                num=min(frame_num-1, start_idx), endpoint=False, dtype=int).tolist()
        history_step_ids += [valid_idx + start_idx]
        image_files = [os.path.join(video_path, 'rgb', video_frames[idx]) for idx in history_step_ids]

        conversations = copy.deepcopy(self.conversations)
        history_str = (DEFAULT_IMAGE_TOKEN+'\n') * len(image_files)
        conversations[0]["value"] += history_str
        conversations[0]["value"] = conversations[0]["value"].replace('<instruction>.', instructions[ins_id])
        conversations[1]["value"] += self.actions2text(actions)

        sources = {
            "image": image_files,
            "conversations": conversations,
            "actions": actions
        }
        return sources

    def _get_item(self, sources) -> Dict[str, torch.Tensor]:

        def transform_actions(actions):
            forward_distance = 0.125
            rotation_angle = np.radians(15)

            continuous_actions = []
            x_pos, y_pos, theta, is_stop = 0.0, 0.0, 0.0, 0
            for i in range(self.action_chunk_num):
                if i < len(actions):
                    action = actions[i]
                    if action == 0:
                        is_stop = 1
                    elif action == 1:
                        x_pos = x_pos + forward_distance * np.cos(theta)
                        y_pos = y_pos + forward_distance * np.sin(theta)
                    elif action == 2:
                        theta = (theta + rotation_angle) % (2 * np.pi)
                    elif action == 3:
                        theta = (theta + 2 * np.pi - rotation_angle) % (2 * np.pi)
                continuous_actions.append([
                    x_pos / self.x_norm_factor,
                    y_pos / self.y_norm_factor,
                    np.cos(theta),
                    np.sin(theta),
                    is_stop
                ])
            
            continuous_actions = torch.tensor(continuous_actions, dtype=torch.float32)
            return continuous_actions

        actions = sources[0].pop("actions")
        data_dict = super()._get_item(sources)

        if self.add_continuous_action:
            continuous_actions = transform_actions(actions)
            data_dict['continuous_actions'] = continuous_actions

            input_ids = data_dict["input_ids"][0]
            im_end_idx = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

            im_end_pos = (input_ids == im_end_idx).nonzero(as_tuple=True)[0][0]
            select_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            select_mask[:im_end_pos + 1] = 1
            data_dict['select_mask'] = select_mask
        
        return data_dict


class DiscreteVLNCollator(DataCollatorForSupervisedDataset):
    """Collator for VLN Action Dataset with continuous action representation."""
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract continuous actions and select masks
        continuous_actions, select_masks = None, None
        if "continuous_actions" in instances[0]:
            continuous_actions = [instance.pop("continuous_actions") for instance in instances]
            select_masks = [instance.pop("select_mask") for instance in instances]

            # Get max sequence length for padding select_masks
            max_len = max(mask.shape[0] for mask in select_masks)
                    
            # Pad select_masks to max_len
            padded_select_masks = []
            for mask in select_masks:
                if mask.shape[0] < max_len:
                    padding = torch.zeros(max_len - mask.shape[0], dtype=torch.bool)
                    mask = torch.cat([mask, padding], dim=0)
                padded_select_masks.append(mask)
            
            # Stack continuous actions and select masks
            batch_continuous_actions = torch.stack(continuous_actions, dim=0)  # (batch_size, action_chunk_size, action_dim)
            batch_select_masks = torch.stack(padded_select_masks, dim=0)  # (batch_size, max_len)
        
        batch = super().__call__(instances)
        if continuous_actions is not None:
            batch.update(dict(continuous_actions=batch_continuous_actions, select_mask=batch_select_masks))
        return batch


# Register the collator with the dataset
DiscreteVLNDataset.COLLATOR_CLASS = DiscreteVLNCollator
