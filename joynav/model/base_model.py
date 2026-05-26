from abc import ABC, abstractmethod
from typing import Optional
import torch
import torch.nn as nn
from .base_argument import BaseArguments

class BaseModel(ABC):
    """
    Abstract base class for all JoyNav models.
    Provides common interface and utility methods.
    """

    ARGUMENT_CLASS = BaseArguments
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward pass of the model.
        
        Returns:
            Model outputs
        """
        pass

    @classmethod
    def get_argument_class(self) -> BaseArguments:
        """
        Returns the argument class associated with the model.
        
        Returns:
            BaseArguments: The argument class for the model.
        """
        return self.ARGUMENT_CLASS
    
    @classmethod
    def post_update_model(self):
        """
        Hook for additional model updates after initialization.
        Can be overridden by subclasses.
        """
        pass

    @classmethod
    def from_pretrained(self, pretrained_model_name_or_path: Optional[str] = None, **kwargs):
        model_args = kwargs.pop("model_args", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        
        if model_args is not None:
            if "config" not in kwargs:
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            else:
                config = kwargs.pop("config")
            
            for key, value in model_args.__dict__.items():
                setattr(config, key, value)
            kwargs["config"] = config
        
        model, loading_info = super().from_pretrained(
            pretrained_model_name_or_path,
            output_loading_info=True,
            **kwargs,
        )
        model._hf_loading_info = loading_info
        if output_loading_info:
            return model, loading_info
        return model
