"""DepthAnything geometry encoder implementation."""
from contextlib import nullcontext

import torch
import logging
from .base import BaseGeometryEncoder, GeometryEncoderConfig
from .merger import GeometryPatchMerger

logger = logging.getLogger(__name__)

class DepthAnythingEncoder(BaseGeometryEncoder):
    """DepthAnything geometry encoder wrapper."""
    
    def __init__(self, config: GeometryEncoderConfig):
        super().__init__(config)
        
        # from qwen_vl.model.depth_anything_v2.dpt import DepthAnythingV2
        from joynav.model.geometry_encoder.depth_anything_v2.dpt import DepthAnythingV2
        # Define model configurations
        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        self.encoder_type = "vitl"

        # Initialize DepthAnythingV2 model
        self.model = DepthAnythingV2(**self.model_configs[self.encoder_type])
        if config.model_path:
            self.load_model(config.model_path)
        
        # Freeze parameters if required
        if self.freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
                
        self.patch_size = 14 
        self.merger = GeometryPatchMerger(
            in_hidden_size=self.get_feature_dim(),
            hidden_size=4096,
            out_hidden_size=config.out_hidden_size,
            spatial_merge_size=config.spatial_merge_size,
            merger_type="mlp",
            use_postshuffle_norm=False
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images using DepthAnything's ViT encoder."""
        self.model.eval()
        if images.dim() == 4:
            images = images.unsqueeze(0)
        B, N, C, H, W = images.shape
        images = images.reshape(B * N, C, H, W)
        
        use_cuda_amp = images.is_cuda and torch.cuda.is_available()
        dtype = torch.bfloat16 if use_cuda_amp and torch.cuda.get_device_capability(images.device)[0] >= 8 else torch.float16
        autocast_context = torch.cuda.amp.autocast(dtype=dtype) if use_cuda_amp else nullcontext()
        with torch.no_grad():
            with autocast_context:
                features = self.model.pretrained.get_intermediate_layers(
                    images, self.model.intermediate_layer_idx[self.model.encoder], 
                    return_class_token=True
                )[-1][0]
        
        features = features.reshape(B, N, -1, features.shape[-1])
        return features
    
    def get_feature_dim(self) -> int:
        """Get DepthAnything feature dimension based on encoder type."""
        # These are the standard hidden dimensions for ViT backbones
        dim_map = {
            'vits': 384,
            'vitb': 768,
            'vitl': 1024,
            'vitg': 1536
        }
        return dim_map.get(self.encoder_type, 1024) # Default to vitl dim
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass for compatibility."""
        B, N, _, H, W = images.shape
        features = self.encode(images)
        features = features.reshape(B*N, H // self.patch_size, W // self.patch_size, -1)
        
        use_cuda_amp = features.is_cuda and torch.cuda.is_available()
        dtype = torch.bfloat16 if use_cuda_amp and torch.cuda.get_device_capability(features.device)[0] >= 8 else torch.float16
        autocast_context = torch.cuda.amp.autocast(dtype=dtype) if use_cuda_amp else nullcontext()
        with autocast_context:
            outputs = self.merger(features)
        outputs = outputs.reshape(B, N, *outputs.shape[1:])     # [B, N, H, W, C]
        return outputs
    
        
    def load_model(self, model_path: str) -> None:
        """Load pretrained DepthAnything model."""
        # Load checkpoint
        ckpt = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(ckpt)
        self.model.eval()
                
        # Freeze parameters if required
        if self.freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
