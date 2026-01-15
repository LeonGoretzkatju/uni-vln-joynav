"""
Base dataset class that all specific datasets should inherit from.
Provides the most abstract interface without specific implementation.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Type
from torch.utils.data import Dataset

from .base_dataset_args import BaseDatasetArguments


class BaseDataCollator(ABC):
    """
    Abstract base class for data collators.
    Each dataset should have its own corresponding collator.
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    @abstractmethod
    def __call__(self, instances):
        """
        Collate a batch of instances.
        
        Args:
            instances: List of data instances from the dataset
            
        Returns:
            Dict containing batched tensors
        """
        pass


class BaseDataset(Dataset, ABC):
    """
    Most abstract base dataset class.
    Defines the core interface that all datasets must implement.
    
    Subclasses should:
    1. Set COLLATOR_CLASS class variable
    2. Implement __getitem__() method
    """
    
    # Each subclass should specify its corresponding collator class
    COLLATOR_CLASS = BaseDataCollator
    ARGUMENT_CLASS = BaseDatasetArguments
    
    def __init__(self, processor, data_args: BaseDatasetArguments):
        """
        Initialize the base dataset.
        
        Args:
            processor: The processor for handling images/videos
            data_args: Dataset-specific arguments (must inherit from BaseDatasetArguments)
        """
        super().__init__()
        
        # Validate that data_args is of the correct type
        if not isinstance(data_args, self.ARGUMENT_CLASS):
            raise TypeError(
                f"data_args must be an instance of BaseDatasetArguments or its subclass, "
                f"got {type(data_args)}"
            )
    
    @abstractmethod
    def __len__(self):
        """Return the number of samples in the dataset."""
        pass
    
    @abstractmethod
    def __getitem__(self, i) -> Dict[str, Any]:
        """
        Get a single item from the dataset.
        
        Args:
            i: Index of the sample
            
        Returns:
            Processed sample as a dictionary
        """
        pass
    
    @classmethod
    def get_collator_class(cls) -> Type[BaseDataCollator]:
        """
        Get the corresponding collator class for this dataset.
        
        Returns:
            The collator class associated with this dataset
            
        Raises:
            NotImplementedError: If COLLATOR_CLASS is not set
        """
        if cls.COLLATOR_CLASS is None:
            raise NotImplementedError(
                f"{cls.__name__} must define COLLATOR_CLASS attribute"
            )
        return cls.COLLATOR_CLASS
    
    @classmethod
    def create_collator(cls, tokenizer):
        """
        Create an instance of the corresponding collator for this dataset.
        
        Args:
            tokenizer: The tokenizer to use in the collator
            
        Returns:
            An instance of the dataset's collator
        """
        collator_class = cls.get_collator_class()
        return collator_class(tokenizer)


    @classmethod
    def get_argument_class(cls) -> Type[BaseDatasetArguments]:
        """
        Get the corresponding argument class for this dataset.
        
        Returns:
            The argument class associated with this dataset
            
        Raises:
            NotImplementedError: If ARGUMENT_CLASS is not set
        """
        if cls.ARGUMENT_CLASS is None:
            raise NotImplementedError(
                f"{cls.__name__} must define ARGUMENT_CLASS attribute"
            )
        return cls.ARGUMENT_CLASS