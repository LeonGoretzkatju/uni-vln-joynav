# Dataset Module

This module provides a comprehensive dataset framework for the JoyNav project, supporting Vision-Language Navigation (VLN) with action prediction capabilities. It implements a hierarchical architecture with abstract base classes and concrete implementations for handling different data types and tasks.

## Directory Structure

```
dataset/
├── __init__.py                          # Module initialization and component registration
├── base_dataset_args.py                 # Abstract base class for dataset arguments
├── base_dataset.py                      # Abstract base dataset class
├── lazy_supervised_dataset_args.py      # Arguments for supervised fine-tuning datasets
├── lazy_supervised_dataset.py           # Generic supervised dataset implementation
├── vln_action_dataset_args.py           # VLN-specific dataset arguments
├── vln_action_dataset.py                # VLN action prediction dataset implementation
└── utils/
    ├── qwen3_data_list.py               # Data registry and dataset configuration
    └── rope2d.py                        # Rotary positional embeddings for 2D/3D tensors
```

## Core Components

### 1. Base Classes (`base_dataset.py`)

Provides the abstract foundation for all datasets in the framework.

#### `BaseDataCollator`
- Abstract base class for data collators
- Each dataset has a corresponding collator for batch processing
- Implements `__call__()` to collate instances into batched tensors

#### `BaseDataset`
- Abstract base class inheriting from `torch.utils.data.Dataset`
- Defines the core interface all datasets must implement:
  - `__len__()` - Return dataset size
  - `__getitem__(i)` - Retrieve a single sample
  - `get_collator_class()` - Get the corresponding collator

### 2. Argument Classes

#### `BaseDatasetArguments` (`base_dataset_args.py`)
- Base dataclass for all dataset arguments
- Supports argument validation via `validate()` method

#### `LazySupervisedDatasetArguments` (`lazy_supervised_dataset_args.py`)
- Extends `BaseDatasetArguments` for supervised learning tasks
- Key parameters:
  - `dataset_use`: Comma-separated list of datasets to mix (e.g., 'cambrian_737k,demo')
  - `model_type`: Model type (qwen2.5vl, qwen2vl, qwen3vl)
  - Image processing: `max_pixels`, `min_pixels`
  - Video processing: `video_max_frames`, `video_min_frames`, `video_max_pixels`, `video_min_pixels`

#### `VLNActionDatasetArguments` (`vln_action_dataset_args.py`)
- Extends `LazySupervisedDatasetArguments` with VLN-specific parameters:
  - `video_folder`: Path(s) to video data (comma-separated)
  - `num_frames`: Number of frames to sample from trajectory
  - `num_history`: Number of historical frames to include
  - `num_future_steps`: Number of future action steps to predict
  - `remove_init_turns`: Whether to remove initial rotation actions

### 3. Dataset Implementations

#### `LazySupervisedDataset` (`lazy_supervised_dataset.py`)
- Generic supervised learning dataset supporting multiple data sources
- Features:
  - Dynamic dataset mixing and sampling
  - Multi-modal support (images and videos)
  - Processor-based image/video handling with configurable pixel ranges
  - ROPE (Rotary Position Embedding) integration for 2D/3D spatial encoding
  - Distributed training support via PyTorch distributed
  - Debugging utilities for sample inspection

#### `VLNActionDataset` (`vln_action_dataset.py`)
- Specialized dataset for Vision-Language Navigation with action prediction
- Inherits from `LazySupervisedDataset`
- VLN-specific features:
  - Action space: STOP (0), FORWARD (1), LEFT (2), RIGHT (3)
  - Multi-modal action descriptions with conjunctions
  - Navigation prompt generation for instruction following
  - Sample validation and debugging output

## Utility Modules

### `qwen3_data_list.py`
- Data registry defining available datasets and their paths
- Supports dynamic dataset configurations
- Implements sampling rate parsing for dataset mixing
- Current supported datasets:
  - `cambrian_737k` / `cambrian_737k_pack`: Large-scale vision-language dataset
  - `mp_doc`: Document understanding dataset
  - `clevr_mc`: Visual reasoning dataset
  - `videochatgpt`: Video understanding dataset
  - `demo`: Demo dataset for testing

### `rope2d.py`
- Implements 2D/3D Rotary Position Embeddings (ROPE)
- Key functions:
  - `get_rope_index_2()`: 2D spatial position indices
  - `get_rope_index_3()`: 3D spatial + temporal position indices (for Qwen3VL)
  - Supports vision-language token integration
  - Handles variable-length sequences and attention masks

## Usage Examples

### Creating a VLN Dataset

```python
from joynav.dataset import VLNActionDataset
from joynav.dataset.vln_action_dataset_args import VLNActionDatasetArguments

# Define dataset arguments
data_args = VLNActionDatasetArguments(
    video_folder="path/to/videos",
    num_frames=32,
    num_history=8,
    num_future_steps=4,
    model_type="qwen3vl",
    dataset_use="r2r,rxr"
)

# Create dataset
processor = ...  # Your vision-language processor
dataset = VLNActionDataset(processor, data_args)

# Access samples
sample = dataset[0]
```

### Using the Registry

```python
from joynav.utils.registry import register_component

# Datasets are automatically registered
dataset_cls = get_component('dataset', 'vln_action')
```
