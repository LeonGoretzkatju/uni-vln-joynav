"""Factory for creating geometry encoders."""

from .base import BaseGeometryEncoder, GeometryEncoderConfig
from .depth_anything_encoder import DepthAnythingEncoder
from .vggt_omega_encoder import VGGTOmegaEncoder

try:
    from .vggt_encoder import VGGTEncoder
except ImportError:
    VGGTEncoder = None

try:
    from .pi3_encoder import Pi3Encoder
except ImportError:
    Pi3Encoder = None

try:
    from .da3_encoder import DA3Encoder
except ImportError:
    DA3Encoder = None


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
        if VGGTEncoder is None:
            raise ImportError("VGGTEncoder is not available in this checkout.")
        return VGGTEncoder(config)
    elif encoder_type == "pi3":
        if Pi3Encoder is None:
            raise ImportError("Pi3Encoder is not available in this checkout.")
        return Pi3Encoder(config)
    elif encoder_type == "da3":
        if DA3Encoder is None:
            raise ImportError("DA3Encoder is not available in this checkout.")
        return DA3Encoder(config)
    elif encoder_type == "da2":
        return DepthAnythingEncoder(config)
    elif encoder_type == "vggt_omega":
        return VGGTOmegaEncoder(config)
    else:
        raise ValueError(f"Unknown geometry encoder type: {encoder_type}")
