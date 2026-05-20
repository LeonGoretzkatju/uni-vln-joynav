import torch
from dataclasses import dataclass, field
from typing import Dict, Sequence

from .base_dataset_args import BaseDatasetArguments
from .vln_action_dataset import VLNActionCollator, VLNActionDataset
from .vln_action_dataset_args import VLNActionDatasetArguments
from vggt_omega.utils.load_fn import load_and_preprocess_images


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
        super().__init__(processor, data_args)

    def _get_item(self, sources) -> Dict[str, torch.Tensor]:
        item = super()._get_item(sources)
        image_files = sources[0].get("image") or []
        if isinstance(image_files, str):
            image_files = [image_files]

        if image_files:
            item["sf_image_tensors"] = load_and_preprocess_images(
                image_files,
                image_resolution=self.image_resolution,
                patch_size=self.teacher_patch_size,
            )
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
