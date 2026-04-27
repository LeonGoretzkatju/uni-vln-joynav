import torch
from PIL import Image
from dataclasses import dataclass, field
from typing import Dict, Sequence
from torchvision import transforms

from .base_dataset_args import BaseDatasetArguments
from .vln_action_dataset import VLNActionCollator, VLNActionDataset
from .vln_action_dataset_args import VLNActionDatasetArguments


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
        self.spatial_forcing_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        super().__init__(processor, data_args)

    def _resize_image(self, image: Image.Image) -> Image.Image:
        resample = getattr(Image, "Resampling", Image).BICUBIC
        width, height = image.size
        scale = self.spatial_forcing_image_size / max(width, height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        image = image.resize((resized_width, resized_height), resample)
        canvas = Image.new("RGB", (self.spatial_forcing_image_size, self.spatial_forcing_image_size), (255, 255, 255))
        left = (self.spatial_forcing_image_size - resized_width) // 2
        top = (self.spatial_forcing_image_size - resized_height) // 2
        canvas.paste(image, (left, top))
        return canvas

    def _get_item(self, sources) -> Dict[str, torch.Tensor]:
        image_files = sources[0].get("image") or []
        if isinstance(image_files, str):
            image_files = [image_files]

        sf_image_tensors = []
        for image_path in image_files:
            image = Image.open(image_path).convert("RGB")
            image = self._resize_image(image)
            sf_image_tensors.append(self.spatial_forcing_transform(image))

        item = super()._get_item(sources)
        if sf_image_tensors:
            item["sf_image_tensors"] = torch.stack(sf_image_tensors, dim=0)
        return item


class VLNActionSpatialForcingCollator(VLNActionCollator):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sf_image_tensors = None
        if "sf_image_tensors" in instances[0]:
            sf_image_tensors = [instance.pop("sf_image_tensors") for instance in instances]

        batch = super().__call__(instances)
        if sf_image_tensors is not None:
            batch["sf_image_tensors"] = torch.cat(sf_image_tensors, dim=0)
        return batch


VLNActionSpatialForcingDataset.COLLATOR_CLASS = VLNActionSpatialForcingCollator
