"""
Universal component registry and argument parsing system.
Supports dataset, model, and any other configurable components.

This unified system replaces:
- dataset_registry.py
- argument_parser.py  
- dataset_factory.py

Key Features:
1. Register any component type (dataset, model, optimizer, etc.)
2. Two-stage argument parsing for any component
3. Automatic factory creation based on args type
4. Simple, unified API
"""
import sys
from typing import Dict, Type, Tuple, Any, Optional, List
from dataclasses import dataclass, field
import transformers


# ============================================================================
# Component Registry
# ============================================================================

class ComponentRegistry:
    """Registry for managing component types and their argument classes."""
    
    def __init__(self):
        # component_type -> {component_id -> (ComponentClass, ArgsClass)}
        self._registry: Dict[str, Dict[str, Tuple[Type, Type]]] = {}
    
    def register(self, component_type: str, component_id: str, 
                 component_class: Type):
        """
        Register a component with its arguments class.
        
        Args:
            component_type: Type of component ('dataset', 'model', etc.)
            component_id: Unique ID for this component ('vln_action', 'qwen3vl', etc.)
            component_class: The component implementation class
        
        Example:
            registry.register('dataset', 'vln_action', VLNActionDataset)
            registry.register('model', 'qwen3vl', Qwen3VLModel)
        """
        if component_type not in self._registry:
            self._registry[component_type] = {}
        self._registry[component_type][component_id] = component_class
    
    def get(self, component_type: str, component_id: str) -> Tuple[Type, Type]:
        """Get component class and args class."""
        if component_type not in self._registry:
            raise KeyError(f"Unknown component type: {component_type}")
        if component_id not in self._registry[component_type]:
            available = ', '.join(self._registry[component_type].keys())
            raise KeyError(
                f"Unknown {component_type} ID: {component_id}. "
                f"Available: {available}"
            )
        return self._registry[component_type][component_id]
    
    def list_types(self) -> List[str]:
        """List all registered component types."""
        return list(self._registry.keys())
    
    def list_ids(self, component_type: str) -> List[str]:
        """List all registered component IDs for a given type."""
        return list(self._registry[component_type].keys()) if component_type in self._registry else []
    
    def find_by_args_type(self, component_type: str, args_instance: Any) -> Tuple[Type, Type]:
        """Find component by args instance type."""
        for component_id, comp_class in self._registry[component_type].items():
            args_class = comp_class.get_argument_class()
            if isinstance(args_instance, args_class):
                return comp_class, args_class
        raise TypeError(
            f"No {component_type} registered for args type: {type(args_instance).__name__}"
        )


# Global registry instance
_REGISTRY = ComponentRegistry()


def register_component(component_type: str, component_id: str, 
                       component_class: Type):
    """Register a component. Wrapper for global registry."""
    _REGISTRY.register(component_type, component_id, component_class)


def get_component(component_type: str, component_id: str) -> Dict[str, Any]:
    """Get component class and args class info."""
    comp_class = _REGISTRY.get(component_type, component_id)
    return comp_class

# ============================================================================
# Component Selector (for two-stage parsing)
# ============================================================================

@dataclass
class ComponentSelector:
    """
    Minimal selector to determine which components to use.
    Parsed in stage 1, then full args are parsed in stage 2.
    """
    # Dataset selection
    dataset_type: str = field(
        default="supervised",
        metadata={"help": "Dataset type: 'supervised', 'vln_action', etc."}
    )
    
    # Model selection (for future extension)
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "Model type: 'qwen2.5vl', 'qwen3vl', etc. If None, uses model_name_or_path"}
    )
    
    def get_dataset_classes(self) -> Tuple[Type, Type]:
        """Get dataset class and args class."""
        return _REGISTRY.get('dataset', self.dataset_type)
    
    def get_model_classes(self) -> Optional[Tuple[Type, Type]]:
        """Get model class and args class if model_type is specified."""
        if self.model_type:
            return _REGISTRY.get('model', self.model_type)
        return None


# ============================================================================
# Two-Stage Argument Parser
# ============================================================================

def parse_component_args(
    additional_args_classes: Optional[tuple] = None,
    component_types: Optional[List[str]] = None,
    return_remaining_strings: bool = False
) -> Tuple:
    """
    Universal two-stage argument parsing for any components.
    
    Stage 1: Parse ComponentSelector to determine which components to use
    Stage 2: Parse full arguments with correct component-specific args classes
    
    Args:
        additional_args_classes: Additional args classes (e.g., TrainingArguments)
        component_types: List of component types to parse (e.g., ['dataset', 'model'])
                        If None, only parses 'dataset'
        return_remaining_strings: Whether to return unparsed strings
    
    Returns:
        Tuple of parsed arguments
    
    Example:
        # Parse only dataset args
        data_args = parse_component_args()
        
        # Parse dataset + training args
        training_args, data_args = parse_component_args(
            additional_args_classes=(TrainingArguments,)
        )
        
        # Parse dataset + model + training args
        training_args, data_args, model_args = parse_component_args(
            additional_args_classes=(TrainingArguments,),
            component_types=['dataset', 'model']
        )
    """
    if component_types is None:
        component_types = ['dataset']
    
    # Stage 1: Parse selector
    selector_parser = transformers.HfArgumentParser(ComponentSelector)
    selector, _ = selector_parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )
    # Stage 2: Build full args classes list
    args_classes = []
    
    # Add additional args first (e.g., TrainingArguments)
    if additional_args_classes:
        args_classes.extend(additional_args_classes)
    
    # Add component-specific args
    for comp_type in component_types:
        if comp_type == 'dataset':
            args_class = selector.get_dataset_classes().get_argument_class()
            args_classes.append(args_class)
        elif comp_type == 'model':
            args_class = selector.get_model_classes().get_argument_class()
            args_classes.append(args_class)
    
    # Parse full arguments
    full_parser = transformers.HfArgumentParser(tuple(args_classes))
    full_args = sys.argv[1:]
    
    if return_remaining_strings:
        return full_parser.parse_args_into_dataclasses(
            args=full_args,
            return_remaining_strings=True
        )
    else:
        return full_parser.parse_args_into_dataclasses(args=full_args)
