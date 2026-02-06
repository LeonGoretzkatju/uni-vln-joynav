"""
Dataset for Vision-Language Navigation with Continuous Action Prediction.
Predicts continuous actions (x, y, yaw) for future steps using historical observations.
"""
import json
import random
import os
import re
import copy
import numpy as np
import torch
from typing import Dict, Optional, Sequence, List, Tuple, Any

from .lazy_supervised_dataset import LazySupervisedDataset, preprocess_qwen_visual, rank0_print, DEFAULT_IMAGE_TOKEN
from .base_dataset_args import BaseDatasetArguments
from .continuous_action_dataset_args import ContinuousActionDatasetArguments


class ContinuousActionDataset(LazySupervisedDataset):
    """
    Dataset for Vision-Language Navigation with Continuous Action Prediction.
    
    Key features:
    1. Uniformly samples num_history_frames historical frames as observations
    2. Predicts next action_chunk_size continuous actions (each action: x, y, yaw)
    3. Uses special tokens to represent action predictions
    4. Returns select_masks to identify action tokens in input_ids
    5. Reads continuous_actions from JSON as a dict: {step: [[x,y,yaw], ...]}
    """

    ARGUMENT_CLASS = ContinuousActionDatasetArguments

    def __init__(self, processor, data_args: BaseDatasetArguments):
        """
        Initialize Continuous Action Dataset.
        
        Args:
            processor: The processor for handling images/videos
            data_args: Continuous action specific dataset arguments
        """
        # Validate args type
        if not isinstance(data_args, ContinuousActionDatasetArguments):
            raise TypeError(
                f"data_args must be ContinuousActionDatasetArguments, got {type(data_args)}"
            )
        
        # Set attributes before calling super().__init__()
        self.num_history_frames = data_args.num_history_frames
        self.num_special_tokens = data_args.num_special_tokens
        self.action_chunk_size = data_args.action_chunk_size
        self.action_dim = data_args.action_dim

        super().__init__(processor, data_args)
        
        # Define special action tokens
        # Format: <|action_0|>, <|action_1|>, ..., <|action_{num_special_tokens-1}|>
        self.action_token_template = "<|action_{}|>"
        self.special_action_tokens = [
            self.action_token_template.format(i) for i in range(self.num_special_tokens)
        ]

        tokenizer = self.processor.tokenizer
        
        # Add special action tokens to tokenizer if not already present
        num_added = tokenizer.add_tokens(self.special_action_tokens, special_tokens=True)
        rank0_print(f"Added {num_added} special action tokens to tokenizer: {self.special_action_tokens}")
        
        # Get special token IDs (for select_masks computation)
        self.special_token_ids = tokenizer.convert_tokens_to_ids(self.special_action_tokens)
        
        # Conversation template for continuous action prediction
        prompt = (
            "You are an autonomous navigation assistant. Your task is to <instruction>. "
            "Based on the historical observations, predict the next sequence of continuous actions "
            "to follow the instruction."
        )
        answer = ""
        self.conversations = [
            {"from": "human", "value": prompt}, 
            {"from": "gpt", "value": answer}
        ]

        # Sample and print examples
        rank0_print(f"================ Continuous Action Dataset ================")
        rank0_print(f"Special action tokens: {self.special_action_tokens}")
        rank0_print(f"Special token IDs: {self.special_token_ids}")
        rank0_print(f"===========================================================")
        
        if len(self.list_data_dict) > 0:
            rank0_print(f"================ Sample data ================")
            for i in random.sample(range(len(self.list_data_dict)), min(3, len(self.list_data_dict))):
                sample = self.prepare_sources(i)
                rank0_print(f"Sample {i}: instruction={sample.get('conversations', [{}])[0].get('value', '')[:100]}...")
                rank0_print(f"  - Images: {len(sample.get('image', []))} frames")
                rank0_print(f"  - Continuous actions shape: {sample.get('continuous_actions', np.array([])).shape}")
            rank0_print(f"=============================================")
    
    def load_data(self):
        """
        Load VLN navigation data from video folders.
        
        Expected JSON format:
        {
            "video": "episode_folder",
            "instructions": ["instruction1", "instruction2"],
            "continuous_actions": {
                "0": [[x, y, yaw], [x, y, yaw], ...],  # 8x3 actions starting at step 0
                "4": [[x, y, yaw], [x, y, yaw], ...],  # 8x3 actions starting at step 4
                ...
            }
        }
        """
        video_folder = self.data_args.video_folder.split(',')
        
        self.nav_data = []
        for vf in video_folder:
            vf = vf.strip()
            anno_json_path = os.path.join(vf, 'annotations.json')
            if not os.path.exists(anno_json_path):
                rank0_print(f"Warning: {anno_json_path} not found, skipping")
                continue
                
            anno_json = json.load(open(anno_json_path, 'r'))
            for tdata in anno_json:
                tdata['video'] = os.path.join(vf, tdata['video'])
            self.nav_data += anno_json
        
        list_data_dict = []
        skipped_episodes = 0
        
        for ep_id, item in enumerate(self.nav_data):
            instructions = item.get('instructions', [])
            continuous_actions = item.get('continuous_actions', {})
            
            if not continuous_actions:
                rank0_print(f"Warning: Episode {ep_id} missing 'continuous_actions', skipping")
                skipped_episodes += 1
                continue
            
            if not isinstance(instructions, list):
                instructions = [instructions]
            
            # Enumerate all steps in continuous_actions dict
            for step_str, actions in continuous_actions.items():
                try:
                    step = int(step_str)
                except ValueError:
                    rank0_print(f"Warning: Invalid step key '{step_str}' in episode {ep_id}, skipping")
                    continue
                
                # Validate actions shape
                actions_array = np.array(actions)
                expected_shape = (self.action_chunk_size, self.action_dim)
                if actions_array.shape != expected_shape:
                    rank0_print(
                        f"Warning: Episode {ep_id} step {step} has wrong action shape "
                        f"{actions_array.shape}, expected {expected_shape}, skipping"
                    )
                    continue
                
                # Add one entry per instruction
                for ins_id in range(len(instructions)):
                    list_data_dict.append((ep_id, ins_id, step))
        
        rank0_print(f"Loaded {len(list_data_dict)} samples from {len(self.nav_data)} episodes.")
        if skipped_episodes > 0:
            rank0_print(f"Skipped {skipped_episodes} episodes without continuous_actions.")
        
        random.shuffle(list_data_dict)
        self.list_data_dict = list_data_dict

    def prepare_sources(self, i):
        """
        Prepare data sources for index i.
        
        Returns:
            Dictionary containing:
            - image: list of image file paths
            - conversations: conversation template with special tokens as answer
            - continuous_actions: (action_chunk_size, action_dim) array
        """
        ep_id, ins_id, step = self.list_data_dict[i]
        data = self.nav_data[ep_id]
        video_path = data['video']
        
        # Get RGB folder
        rgb_folder = os.path.join(video_path, 'rgb')
        if not os.path.exists(rgb_folder):
            raise ValueError(f"RGB folder not found: {rgb_folder}")
        
        video_frames = sorted(os.listdir(rgb_folder))
        total_frames = len(video_frames)

        instructions = data.get("instructions", [])
        if not isinstance(instructions, list):
            instructions = [instructions]

        continuous_actions = data['continuous_actions']
        
        # Get the continuous actions for this step
        actions = np.array(continuous_actions[str(step)])  # (action_chunk_size, action_dim)
        
        # Uniformly sample num_history_frames from [0, step)
        if step > 0:
            # Uniformly sample historical frames
            history_indices = np.linspace(0, step - 1, self.num_history_frames, dtype=np.int32)
            history_frames = [os.path.join(rgb_folder, video_frames[idx]) for idx in history_indices]
        else:
            # If at step 0, no history
            history_frames = []
        history_frames.append(os.path.join(rgb_folder, video_frames[step]))  # Include current frame
        
        # Create answer with special action tokens
        action_token_str = ''.join(self.special_action_tokens)
        
        # Prepare conversations
        conversations = copy.deepcopy(self.conversations)
        
        # Add historical observations to prompt
        history_str = (DEFAULT_IMAGE_TOKEN + '\n') * len(history_frames)
        conversations[0]["value"] += f" These are your historical observations:\n{history_str}"
        
        # Replace instruction placeholder
        conversations[0]["value"] = conversations[0]["value"].replace('<instruction>', instructions[ins_id])
        
        # Set answer to special action tokens
        conversations[1]["value"] = action_token_str
        
        sources = {
            "image": history_frames,
            "conversations": conversations,
            "continuous_actions": actions,  # (action_chunk_size, action_dim)
        }
        return sources
    
    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        """
        Get a single item with continuous action annotations.
        
        Returns:
            Dictionary containing:
            - input_ids: tokenized input
            - labels: labels for language modeling
            - pixel_values: processed images
            - image_grid_thw: image grid info
            - continuous_actions: (action_chunk_size, action_dim) tensor
            - select_masks: (seq_len,) boolean mask for action token positions
        """
        # Get base data
        sources = self.prepare_sources(i)
        continuous_actions = sources.pop("continuous_actions")
        
        # Preprocess with qwen visual
        data_dict = preprocess_qwen_visual([sources], self.processor)
        
        # Compute select_masks for action tokens
        input_ids = data_dict["input_ids"][0]  # Shape: (seq_len,)
        
        # Create select_masks by comparing with special token IDs range
        select_masks = torch.zeros(len(input_ids), dtype=torch.bool)
        input_ids_list = input_ids.tolist()
        
        # Get select_masks
        select_masks = (input_ids >= self.special_token_ids[0]) & (input_ids <= self.special_token_ids[-1])

        # Package results
        result = {
            "input_ids": data_dict["input_ids"],
            "labels": data_dict["labels"],
            "pixel_values": data_dict.get("pixel_values"),
            "image_grid_thw": data_dict.get("image_grid_thw"),
            "pixel_values_videos": data_dict.get("pixel_values_videos"),
            "video_grid_thw": data_dict.get("video_grid_thw"),
            "continuous_actions": torch.tensor(continuous_actions, dtype=torch.float32),  # (action_chunk_size, action_dim)
            "select_masks": select_masks,  # (seq_len,)
        }
        
        return result
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        """
        Get item with retry logic.
        """
        num_base_retries = 3

        # Try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                return self._get_item(i)
            except Exception as e:
                rank0_print(f"Error loading sample {i} (attempt {attempt_idx + 1}/{num_base_retries}): {e}")
                if attempt_idx == num_base_retries - 1:
                    rank0_print(f"Failed to load sample {i} after {num_base_retries} attempts")
                    raise
                # Try next sample
                i = random.randint(0, len(self.list_data_dict) - 1)
        
        raise RuntimeError("Should not reach here")


class ContinuousActionCollator:
    """
    Collator for continuous action prediction dataset.
    Handles batching of continuous actions and select masks.
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of instances.
        
        Args:
            instances: List of data instances
            
        Returns:
            Batched dictionary
        """
        # Extract continuous actions and select masks
        continuous_actions = [instance.pop("continuous_actions") for instance in instances]
        select_masks = [instance.pop("select_masks") for instance in instances]
        
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
        
        # Collate standard fields
        batch = {}
        
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        
        # Pad input_ids and labels
        max_input_len = max(ids.shape[1] for ids in input_ids)
        padded_input_ids = []
        padded_labels = []
        
        for ids, lbls in zip(input_ids, labels):
            pad_len = max_input_len - ids.shape[1]
            if pad_len > 0:
                ids = torch.cat([ids, torch.full((1, pad_len), self.tokenizer.pad_token_id, dtype=ids.dtype)], dim=1)
                lbls = torch.cat([lbls, torch.full((1, pad_len), -100, dtype=lbls.dtype)], dim=1)
            padded_input_ids.append(ids)
            padded_labels.append(lbls)
        
        batch["input_ids"] = torch.cat(padded_input_ids, dim=0)
        batch["labels"] = torch.cat(padded_labels, dim=0)
        
        # Handle pixel values (if present)
        if instances[0].get("pixel_values") is not None:
            pixel_values = [instance["pixel_values"] for instance in instances]
            batch["pixel_values"] = torch.cat(pixel_values, dim=0) if pixel_values[0] is not None else None
            
            image_grid_thw = [instance["image_grid_thw"] for instance in instances]
            batch["image_grid_thw"] = torch.cat(image_grid_thw, dim=0) if image_grid_thw[0] is not None else None
        
        # Handle video values (if present)
        if instances[0].get("pixel_values_videos") is not None:
            pixel_values_videos = [instance["pixel_values_videos"] for instance in instances]
            batch["pixel_values_videos"] = torch.cat(pixel_values_videos, dim=0) if pixel_values_videos[0] is not None else None
            
            video_grid_thw = [instance["video_grid_thw"] for instance in instances]
            batch["video_grid_thw"] = torch.cat(video_grid_thw, dim=0) if video_grid_thw[0] is not None else None
        
        # Add continuous actions and select masks
        batch["continuous_actions"] = batch_continuous_actions
        batch["select_masks"] = batch_select_masks
        
        return batch


# Register the collator with the dataset
ContinuousActionDataset.COLLATOR_CLASS = ContinuousActionCollator
