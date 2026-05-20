import torch
from PIL import Image
from dataclasses import dataclass, field
from typing import Dict, Sequence
from torchvision import transforms

from .base_dataset_args import BaseDatasetArguments
from .vln_action_dataset import VLNActionCollator, VLNActionDataset
from .vln_action_dataset_args import VLNActionDatasetArguments
from .vln_action_spatial_forcing_dataset import resize_image_to_qwen_grid


@dataclass
class VLNActionOmegaSpatialForcingDatasetArguments(VLNActionDatasetArguments):
    spatial_forcing_teacher_patch_size: int = field(
        default=16,
        metadata={"help": "VGGT-Omega patch size used for Spatial Forcing target images."},
    )


class VLNActionOmegaSpatialForcingDataset(VLNActionDataset):
    ARGUMENT_CLASS = VLNActionOmegaSpatialForcingDatasetArguments

    def __init__(self, processor, data_args: BaseDatasetArguments):
        self.qwen_patch_size = getattr(processor.image_processor, "patch_size", 16)
        self.teacher_patch_size = data_args.spatial_forcing_teacher_patch_size
        self.spatial_forcing_transform = transforms.ToTensor()
        super().__init__(processor, data_args)

    def _get_item(self, sources) -> Dict[str, torch.Tensor]:
        item = super()._get_item(sources)
        image_files = sources[0].get("image") or []
        if isinstance(image_files, str):
            image_files = [image_files]

        sf_image_tensors = []
        for image_path, grid_thw in zip(image_files, item["image_grid_thw"]):
            image = Image.open(image_path).convert("RGB")
            image = resize_image_to_qwen_grid(
                image,
                grid_thw,
                patch_size=self.qwen_patch_size,
                teacher_patch_size=self.teacher_patch_size,
            )
            sf_image_tensors.append(self.spatial_forcing_transform(image))

        if sf_image_tensors:
            item["sf_image_tensors"] = sf_image_tensors
        return item


class VLNActionOmegaSpatialForcingCollator(VLNActionCollator):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sf_image_tensors = []
        if "sf_image_tensors" in instances[0]:
            for instance in instances:
                sf_image_tensors.append(torch.stack(instance.pop("sf_image_tensors"), dim=0))

        batch = super().__call__(instances)
        if sf_image_tensors:
            batch["sf_image_tensors"] = sf_image_tensors
        return batch


VLNActionOmegaSpatialForcingDataset.COLLATOR_CLASS = VLNActionOmegaSpatialForcingCollator
