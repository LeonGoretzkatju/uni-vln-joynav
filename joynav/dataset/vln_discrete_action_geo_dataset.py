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
from PIL import Image
from torchvision import transforms
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple, Any
from .lazy_supervised_dataset import (
    LazySupervisedDataset, 
    preprocess_qwen_visual, 
    rank0_print, 
    DEFAULT_IMAGE_TOKEN, 
    DataCollatorForSupervisedDataset
)
from .base_dataset_args import BaseDatasetArguments
from .vln_discrete_action_dataset import VLNDiscreteActionDataset
from .vln_discrete_action_dataset_args import VLNDiscreteActionDatasetArguments



class VLNDiscreteActionGeoDataset(VLNDiscreteActionDataset):
    """
    Dataset for Vision-Language Navigation with Action Prediction.
    Inherits from LazySupervisedDataset and provides VLN-specific functionality.
    """

    def __init__(self, processor, data_args: BaseDatasetArguments):
        """
        Initialize VLN Action Dataset.
        
        Args:
            processor: The processor for handling images/videos
            data_args: VLN-specific dataset arguments
        """

        super().__init__(processor, data_args)


    def _get_item(self, sources) -> Dict[str, torch.Tensor]:
        
        image_files = sources[0].get("image")
        # Load images, resize by ratio (14/16 of original size), and convert to tensors
        image_tensors = []
        for img_path in image_files:
            img = Image.open(img_path).convert('RGB')
            width, height = img.size
            new_width, new_height = int(width * (14 / 16)), int(height * (14 / 16))  # Scale to 14/16 of original size
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            img_tensor = transforms.ToTensor()(img)  # Convert to [0, 1] float and permute to (C, H, W)
            image_tensors.append(img_tensor)
        
        image_tensors = torch.stack(image_tensors)

        item = super()._get_item(sources)
        item["image_tensors"] = image_tensors
        return item


class VLNDiscreteActionGeoCollator(DataCollatorForSupervisedDataset):
    """Collator for VLN Action Dataset with continuous action representation."""
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        batch_image_tensors = None
        if "image_tensors" in instances[0]:
            image_tensors = [instance.pop("image_tensors") for instance in instances]
            batch_image_tensors = torch.stack(image_tensors, dim=0)  # (batch_size, num_frames, C, H, W)

        batch = super().__call__(instances)
        batch["image_tensors"] = batch_image_tensors
        return batch


# Register the collator with the dataset
VLNDiscreteActionGeoDataset.COLLATOR_CLASS = VLNDiscreteActionGeoCollator



