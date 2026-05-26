import torch
import copy
from dataclasses import dataclass, field
from typing import Dict, Sequence

from PIL import Image

from .base_dataset_args import BaseDatasetArguments
from .vln_action_dataset import VLNActionCollator, VLNActionDataset
from .vln_action_dataset_args import VLNActionDatasetArguments
from vggt_omega.utils.load_fn import (
    _balanced_target_shape,
    _crop_to_supported_aspect_ratio,
    _load_rgb_image,
    load_and_preprocess_images,
)


def _load_or_copy_rgb_image(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return _load_rgb_image(image)


def prepare_qwen_images_for_omega_direct(
    images,
    omega_hw=None,
    spatial_merge_size=2,
    image_resolution=256,
    patch_size=16,
):
    cropped_images = [_crop_to_supported_aspect_ratio(_load_or_copy_rgb_image(image)) for image in images]
    if omega_hw is None:
        target_shapes = [
            _balanced_target_shape(image.size[1] / max(image.size[0], 1), image_resolution, patch_size)
            for image in cropped_images
        ]
        max_h = max(target_h for target_h, _ in target_shapes)
        max_w = max(target_w for _, target_w in target_shapes)
        target_shapes = [(max_h, max_w)] * len(cropped_images)
    else:
        target_shapes = [(int(omega_hw[0]), int(omega_hw[1]))] * len(cropped_images)

    qwen_images = []
    for image, (target_h, target_w) in zip(cropped_images, target_shapes):
        qwen_images.append(image.resize((target_w * spatial_merge_size, target_h * spatial_merge_size)))
    return qwen_images


def load_qwen_images_for_omega_direct(image_path_list, omega_hw, spatial_merge_size):
    return prepare_qwen_images_for_omega_direct(
        image_path_list,
        omega_hw=omega_hw,
        spatial_merge_size=spatial_merge_size,
    )



@dataclass
class VLNActionOmegaSpatialForcingDatasetArguments(VLNActionDatasetArguments):
    spatial_forcing_teacher_patch_size: int = field(
        default=16,
        metadata={"help": "VGGT-Omega patch size used for Spatial Forcing target images."},
    )
    spatial_forcing_image_resolution: int = field(
        default=512,
        metadata={"help": "VGGT-Omega balanced image-resolution budget."},
    )


class VLNActionOmegaSpatialForcingDataset(VLNActionDataset):
    ARGUMENT_CLASS = VLNActionOmegaSpatialForcingDatasetArguments

    def __init__(self, processor, data_args: BaseDatasetArguments):
        self.teacher_patch_size = data_args.spatial_forcing_teacher_patch_size
        self.image_resolution = data_args.spatial_forcing_image_resolution
        self.omega_mode = getattr(data_args, "omega_mode", "")
        self.spatial_merge_size = getattr(processor.image_processor, "merge_size", 2)
        super().__init__(processor, data_args)

    def _get_item(self, sources) -> Dict[str, torch.Tensor]:
        image_files = sources[0].get("image") or []
        if isinstance(image_files, str):
            image_files = [image_files]

        sf_image_tensors = None
        if image_files:
            sf_image_tensors = load_and_preprocess_images(
                image_files,
                image_resolution=self.image_resolution,
                patch_size=self.teacher_patch_size,
            )

        if self.omega_mode == "text_align_force_qwen":
            qwen_sources = copy.deepcopy(sources)
            qwen_sources[0]["image"] = load_qwen_images_for_omega_direct(
                image_files,
                sf_image_tensors.shape[-2:],
                self.spatial_merge_size,
            )
            item = super()._get_item(qwen_sources)
        else:
            item = super()._get_item(sources)

        if sf_image_tensors is not None:
            item["sf_image_tensors"] = sf_image_tensors
        return item


class VLNActionOmegaSpatialForcingCollator(VLNActionCollator):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sf_image_tensors = []
        if "sf_image_tensors" in instances[0]:
            for instance in instances:
                sf_image_tensors.append(instance.pop("sf_image_tensors"))

        batch = super().__call__(instances)
        if sf_image_tensors:
            batch["sf_image_tensors"] = sf_image_tensors
        return batch


VLNActionOmegaSpatialForcingDataset.COLLATOR_CLASS = VLNActionOmegaSpatialForcingCollator
