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