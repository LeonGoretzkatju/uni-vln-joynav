import json
import random
import os
import re
import time
import torch
import copy
import numpy as np
from typing import Dict, Optional, Sequence, List, Tuple, Any
from .lazy_supervised_dataset import LazySupervisedDataset, preprocess_qwen_visual, rank0_print, DEFAULT_IMAGE_TOKEN

from .base_dataset_args import BaseDatasetArguments
from .streamvln_dataset_args import StreamVLNDatasetArguments


class StreamVLNDataset(LazySupervisedDataset):
    """
    Dataset for Vision-Language Navigation with Action Prediction.
    Inherits from LazySupervisedDataset and provides VLN-specific functionality.
    """

    # Specify the corresponding collator class
    ARGUMENT_CLASS = StreamVLNDatasetArguments

    def __init__(self, processor, data_args: BaseDatasetArguments):
        """
        Initialize VLN Action Dataset.
        
        Args:
            processor: The processor for handling images/videos
            data_args: VLN-specific dataset arguments
        """
        # Validate args type
        if not isinstance(data_args, StreamVLNDatasetArguments):
            raise TypeError(
                f"data_args must be StreamVLNDatasetArguments, got {type(data_args)}"
            )
        
        # VLN-specific attributes (set before calling super().__init__)
        self.num_frames = data_args.num_frames
        self.num_history = data_args.num_history
        self.num_future_steps = data_args.num_future_steps
        self.remove_init_turns = data_args.remove_init_turns

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
                if self.remove_init_turns:
                    valid_idx = self.clean_initial_rotations(instructions[ins_id], actions) # Not implemented
                    if valid_idx != 0:
                        invalid_len += 1

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
    
    def prepare_conversation(self, conversation, actions): 
        i = 0
        sources = []
        t = 0
        while i < len(actions):
            source = copy.deepcopy(conversation)
            prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
            step_actions = actions[i:i+self.num_future_steps]
            answer = self.actions2text(step_actions)
            if i == 0:
                source[0]["value"] += f" {prompt}."
            else:
                source[0]["value"] = f"{prompt}."
            
            source[1]["value"] = answer
            i += len(step_actions)
            t += 1
            sources.extend(source)
        return sources
    
    def prepare_sources(self, i):
        ep_id, ins_id, start_idx, valid_idx = self.list_data_dict[i]
        data = self.nav_data[ep_id]
        video_path = data['video']
        video_frames = sorted(os.listdir(os.path.join(video_path, 'rgb')))

        instructions = data.get("instructions", None)
        if not isinstance(instructions, list):
            instructions = [instructions]

        actions = data['actions'][1+valid_idx:] + [0]
        actions_len = len(actions)
        time_ids = np.arange(start_idx, min(start_idx + self.num_frames, actions_len))
        assert len(time_ids) > 0
        actions = np.array(actions)[time_ids]

        start_idx, end_idx, interval = time_ids[0]+valid_idx, time_ids[-1]+1+valid_idx, self.num_future_steps
        sample_step_ids = np.arange(start_idx, end_idx, interval, dtype=np.int32)
        sample_frames = [os.path.join(video_path, 'rgb', video_frames[i]) for i in sample_step_ids]

        if time_ids[0] != 0:
            history_step_ids = np.arange(0+valid_idx, time_ids[0]+valid_idx, max(time_ids[0] // self.num_history, 1))
            history_frames = [os.path.join(video_path, 'rgb', video_frames[i]) for i in history_step_ids]
        else:
            history_frames = []
            
        image_files = history_frames + sample_frames

        conversations = copy.deepcopy(self.conversations)

        if start_idx != 0:
            history_str = (DEFAULT_IMAGE_TOKEN+'\n') * len(history_frames)
            conversations[0]["value"] += f" These are your historical observations: {history_str}."
        conversations[0]["value"] = conversations[0]["value"].replace('<instruction>.', instructions[ins_id])
        interleave_conversations = self.prepare_conversation(conversations, list(actions))
        sources = {
            "image": image_files,
            "conversations": interleave_conversations
        }
        return sources
