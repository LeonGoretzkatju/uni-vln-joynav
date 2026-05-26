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
from .vln_action_dataset_args import VLNActionDatasetArguments


class VLNActionDataset(LazySupervisedDataset):
    """
    Dataset for Vision-Language Navigation with Action Prediction.
    Inherits from LazySupervisedDataset and provides VLN-specific functionality.
    """

    # Specify the corresponding collator class
    ARGUMENT_CLASS = VLNActionDatasetArguments

    def __init__(self, processor, data_args: BaseDatasetArguments):
        """
        Initialize VLN Action Dataset.
        
        Args:
            processor: The processor for handling images/videos
            data_args: VLN-specific dataset arguments
        """
        # Validate args type
        if not isinstance(data_args, VLNActionDatasetArguments):
            raise TypeError(
                f"data_args must be VLNActionDatasetArguments, got {type(data_args)}"
            )
        
        # StreamVLN-style sampling attributes.
        self.num_frames = data_args.num_frames
        self.num_history = data_args.num_history
        self.action_chunk_num = data_args.action_chunk_num

        # Legacy attributes kept for compatibility with older scripts/configs.
        self.min_window_size = data_args.min_window_size
        self.max_window_size = data_args.max_window_size
        self.sampling_stride = data_args.sampling_stride
        self.history_sampling_mode = data_args.history_sampling_mode
        self.split_forward = data_args.split_forward
        self.sliding_window_size = data_args.sliding_window_size

        # Continuous action representation parameters
        self.add_continuous_action = data_args.add_continuous_action
        self.x_norm_factor = data_args.x_norm_factor
        self.y_norm_factor = data_args.y_norm_factor

        # Add Special Action Token
        self.action_token = "<|action|>"
        special_tokens_dict = {'additional_special_tokens': [self.action_token]}
        num_new_tokens = processor.tokenizer.add_special_tokens(special_tokens_dict)
        rank0_print(f"Adding {num_new_tokens} new tokens: {special_tokens_dict}")

        super().__init__(processor, data_args)

        # VLN-specific setup
        self.idx2actions = {
            '0': 'STOP',
            '1': "↑",
            '2': "←",
            '3': "→",
        }
        self.conjunctions = [
            'you can see ',
            'in front of you is ',
            'there is ',
            'you can spot ',
            'you are toward the ',
            'ahead of you is ',
            'in your sight is '
        ]
        
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
            splits = vf.split("%")
            vf = splits[0]
            ratio = 100
            if len(splits) > 1:
                ratio = int(splits[1])
                rank0_print(f"Loading {ratio}% of data from {vf}")
    
            with open(os.path.join(vf, 'annotations.json'), 'r') as f:
                anno_json = json.load(f)
            if ratio < 100:
                anno_json = random.sample(anno_json, int(len(anno_json) * ratio / 100))

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
                
                num_rounds = (actions_len - valid_idx) // self.num_frames
                for n in range(num_rounds + 1):
                    if n * self.num_frames == actions_len - valid_idx:
                        continue
                    list_data_dict.append((ep_id, ins_id, n * self.num_frames, valid_idx))
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

    def prepare_conversation(self, conversation, action_chunks):
        sources = []
        for chunk_idx, step_actions in enumerate(action_chunks):
            source = copy.deepcopy(conversation)
            prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
            if chunk_idx == 0:
                source[0]["value"] += f" {prompt}."
            else:
                source[0]["value"] = f"{prompt}."

            source[1]["value"] = self.action_token + self.actions2text(step_actions)
            sources.extend(source)
        return sources

    def _chunk_actions(self, actions):
        chunks = []
        i = 0
        while i < len(actions):
            chunk = actions[i:i + self.action_chunk_num]
            if len(chunk) == 0:
                break
            chunks.append(chunk)
            i += len(chunk)
        return chunks

    def prepare_sources(self, i):

        ep_id, ins_id, start_idx, valid_idx = self.list_data_dict[i]
        data = self.nav_data[ep_id]
        video_path = data['video']
        video_frames = sorted(os.listdir(os.path.join(video_path, 'rgb')))
        video_frames = {
            int(filename.split('.')[0])-1: filename
            for filename in video_frames
        }

        instructions = data.get("instructions", None)
        if not isinstance(instructions, list):
            instructions = [instructions]

        actions = data['actions'][1+valid_idx:] + [0]
        actions_len = len(actions)
        time_ids = np.arange(start_idx, min(start_idx + self.num_frames, actions_len))
        assert len(time_ids) > 0
        window_actions = np.array(actions)[time_ids].tolist()

        sample_start = int(time_ids[0]) + valid_idx
        sample_end = int(time_ids[-1]) + 1 + valid_idx
        sample_step_ids = np.arange(sample_start, sample_end, self.action_chunk_num, dtype=np.int32).tolist()
        sample_frames = [os.path.join(video_path, 'rgb', video_frames[idx]) for idx in sample_step_ids]

        if time_ids[0] != 0:
            num_history = max(int(self.num_history), 1)
            history_stride = max(int(time_ids[0]) // num_history, 1)
            history_step_ids = np.arange(0 + valid_idx, int(time_ids[0]) + valid_idx, history_stride)
            history_frames = [os.path.join(video_path, 'rgb', video_frames[int(idx)]) for idx in history_step_ids]
        else:
            history_frames = []

        image_files = history_frames + sample_frames

        conversations = copy.deepcopy(self.conversations)
        if len(history_frames) > 0:
            history_str = (DEFAULT_IMAGE_TOKEN+'\n') * len(history_frames)
            conversations[0]["value"] += f" These are your historical observations: {history_str}."
        conversations[0]["value"] = conversations[0]["value"].replace('<instruction>.', instructions[ins_id])

        action_chunks = self._chunk_actions(window_actions)
        interleave_conversations = self.prepare_conversation(conversations, action_chunks)

        sources = {
            "image": image_files,
            "conversations": interleave_conversations,
            "actions": action_chunks
        }
        return sources

    def _get_item(self, sources) -> Dict[str, torch.Tensor]:

        def transform_action_chunk(actions):
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

        def transform_actions(actions):
            if len(actions) == 0:
                return torch.empty(0, self.action_chunk_num, 5, dtype=torch.float32)
            first = actions[0]
            if isinstance(first, (list, tuple, np.ndarray)):
                action_chunks = actions
            else:
                action_chunks = [actions]
            return torch.stack([transform_action_chunk(chunk) for chunk in action_chunks], dim=0)

        actions = sources[0].pop("actions")
        data_dict = super()._get_item(sources)

        if self.add_continuous_action:
            continuous_actions = transform_actions(actions)
            data_dict['continuous_actions'] = continuous_actions
            
            input_ids = data_dict["input_ids"][0] # self.processor.tokenizer.decode(input_ids.numpy().tolist())

            action_token_id = self.tokenizer.convert_tokens_to_ids(self.action_token)
            matches = (input_ids == action_token_id).nonzero(as_tuple=True)[0]

            select_mask = torch.zeros_like(input_ids, dtype=torch.bool)

            if len(matches) > 0:
                num_action_chunks = continuous_actions.shape[0]
                selected_matches = matches[-num_action_chunks:] if len(matches) >= num_action_chunks else matches
                select_mask[selected_matches] = True
                if len(selected_matches) != num_action_chunks:
                    rank0_print(
                        f"Warning: found {len(selected_matches)} action tokens for "
                        f"{num_action_chunks} action chunks."
                    )
            else:
                # Fallback: (select_mask: all False)
                rank0_print(f"Warning: Action token {self.action_token} not found in input_ids.")

            data_dict['select_mask'] = select_mask
        
        return data_dict


class VLNActionCollator(DataCollatorForSupervisedDataset):
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
            
            batch_continuous_actions = torch.cat(continuous_actions, dim=0)  # (num_action_chunks, action_chunk_size, action_dim)
            batch_select_masks = torch.stack(padded_select_masks, dim=0)  # (batch_size, max_len)
        
        batch = super().__call__(instances)
        if continuous_actions is not None:
            batch.update(dict(continuous_actions=batch_continuous_actions, select_mask=batch_select_masks))
        return batch


# Register the collator with the dataset
VLNActionDataset.COLLATOR_CLASS = VLNActionCollator
