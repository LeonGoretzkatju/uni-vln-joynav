"""
Base arguments for datasets.
Each dataset can inherit from BaseDatasetArguments and add its own specific arguments.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BaseDatasetArguments:
    """Base arguments for all datasets."""

    dataset_type: str = field(
        default="base",
        metadata={"help": "Type of the dataset"}
    )
    
    def __post_init__(self):
        """Validate arguments after initialization."""
        self.validate()
    
    def validate(self):
        """Validate the arguments. Can be overridden by subclasses."""
        pass