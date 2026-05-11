import torch
from PIL import Image
from dataclasses import dataclass, field
from typing import Dict, Sequence
from torchvision import transforms

from .base_dataset_args import BaseDatasetArguments
from .vln_action_dataset import VLNActionCollator, VLNActionDataset
from .vln_action_dataset_args import VLNActionDatasetArguments


def _round_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def resize_image_to_qwen_grid(
    image: Image.Image,
    grid_thw: torch.Tensor,
    patch_size: int,
    teacher_patch_size: int = 14,
) -> Image.Image:
    _, grid_h, grid_w = [int(v) for v in grid_thw.tolist()]
    resample = getattr(Image, "Resampling", Image).BICUBIC
    width = _round_to_multiple(grid_w * patch_size, teacher_patch_size)
    height = _round_to_multiple(grid_h * patch_size, teacher_patch_size)
    return image.resize((width, height), resample)


@dataclass
class VLNActionSpatialForcingDatasetArguments(VLNActionDatasetArguments):
    spatial_forcing_image_size: int = field(
        default=518,
        metadata={"help": "Square image size for Depth Anything V2 Spatial Forcing targets."},
    )


class VLNActionSpatialForcingDataset(VLNActionDataset):
    ARGUMENT_CLASS = VLNActionSpatialForcingDatasetArguments

    def __init__(self, processor, data_args: BaseDatasetArguments):
        self.spatial_forcing_image_size = data_args.spatial_forcing_image_size
        self.spatial_forcing_patch_size = getattr(processor.image_processor, "patch_size", 16)
        self.spatial_forcing_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        super().__init__(processor, data_args)

    def _get_item(self, sources) -> Dict[str, torch.Tensor]:
        item = super()._get_item(sources)
        image_files = sources[0].get("image") or []
        if isinstance(image_files, str):
            image_files = [image_files]

        image_grid_thw = item.get("image_grid_thw")
        sf_image_tensors = []
        for image_path, grid_thw in zip(image_files, image_grid_thw):
            image = Image.open(image_path).convert("RGB")
            image = resize_image_to_qwen_grid(image, grid_thw, self.spatial_forcing_patch_size)
            sf_image_tensors.append(self.spatial_forcing_transform(image))

        if sf_image_tensors:
            item["sf_image_tensors"] = sf_image_tensors
        return item


class VLNActionSpatialForcingCollator(VLNActionCollator):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sf_image_tensors = []
        if "sf_image_tensors" in instances[0]:
            for instance in instances:
                sf_image_tensors.extend(instance.pop("sf_image_tensors"))

        batch = super().__call__(instances)
        if sf_image_tensors:
            batch["sf_image_tensors"] = sf_image_tensors
        return batch


VLNActionSpatialForcingDataset.COLLATOR_CLASS = VLNActionSpatialForcingCollator
