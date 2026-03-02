"""Factory for creating geometry encoders."""

from typing import Optional
from .base import BaseGeometryEncoder, GeometryEncoderConfig
from .pi3_encoder import Pi3Encoder
from .depth_anything_encoder import DepthAnythingEncoder


def create_geometry_encoder(config) -> BaseGeometryEncoder:
    """
    Factory function to create geometry encoders.
    
    Args:
        encoder_type: Type of encoder ("vggt", "pi3", etc.)
        model_path: Path to pretrained model
        reference_frame: Reference frame setting
        freeze_encoder: Whether to freeze encoder parameters
        **encoder_kwargs: Additional encoder-specific arguments
        
    Returns:
        Geometry encoder instance
    """
    encoder_type = config.encoder_type.lower()
    if encoder_type == "vggt":
        return VGGTEncoder(config)
    elif encoder_type == "pi3":
        return Pi3Encoder(config)
    elif encoder_type == "da3":
        return DA3Encoder(config)
    elif encoder_type == "da2":
        return DepthAnythingEncoder(config)
    else:
        raise ValueError(f"Unknown geometry encoder type: {encoder_type}")
