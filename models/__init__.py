"""
Unified model module
Provides model implementations, core functionality and neural network components
"""

from .base import (
    BaseSequenceModel,
    BaseTransformerModel,
    BaseRNNModel,
    BaseDiffusionModel,
)

from .transformer import SyriacTransformerModel
from .bert import SyriacBERTModel
from .lstm import SyriacLSTMModel
from .mdlm import MDLMModel
from .cutting_edge import (
    SyriacMambaModel,
    SyriacBiMambaModel,
    SyriacBiMambaLargeModel,
    SyriacBiMambaXLModel,
    SyriacRetNetModel,
    SyriacSwitchModel,
    SyriacMambaLargeModel,
    SyriacRWKV7Model,
    SyriacRWKV7LargeModel,
    SyriacRWKV7EfficientModel
)

# Model registry
MODEL_REGISTRY = {
    # Core architectures
    'transformer': SyriacTransformerModel,
    'bert': SyriacBERTModel,
    'lstm': SyriacLSTMModel,
    'mdlm': MDLMModel,

    # Other model architectures
    'mamba': SyriacMambaModel,
    'bimamba': SyriacBiMambaModel,
    'bimamba_large': SyriacBiMambaLargeModel,
    'bimamba_xl': SyriacBiMambaXLModel,
    'retnet': SyriacRetNetModel,
    'switch': SyriacSwitchModel,
    'mamba_large': SyriacMambaLargeModel,

    # RWKV-7 series
    'rwkv7': SyriacRWKV7Model,
    'rwkv7_large': SyriacRWKV7LargeModel,
    'rwkv7_efficient': SyriacRWKV7EfficientModel,
}

def get_model_class(model_type: str):
    """Get model class by type"""
    model_type = model_type.lower()
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_type]

def list_available_models():
    """List all available models"""
    return list(MODEL_REGISTRY.keys())

# Re-export core and components modules for backward compatibility
from . import core
from . import components

__all__ = [
    # Base classes
    'BaseSequenceModel',
    'BaseTransformerModel',
    'BaseRNNModel',
    'BaseDiffusionModel',

    # Core models
    'SyriacTransformerModel',
    'SyriacBERTModel',
    'SyriacLSTMModel',
    'MDLMModel',

    # Other model architectures
    'SyriacMambaModel',
    'SyriacBiMambaModel',
    'SyriacRetNetModel',
    'SyriacSwitchModel',
    'SyriacMambaLargeModel',

    # RWKV-7 series models
    'SyriacRWKV7Model',
    'SyriacRWKV7LargeModel',
    'SyriacRWKV7EfficientModel',

    # Utility functions
    'get_model_class',
    'list_available_models',
    'MODEL_REGISTRY',

    # Submodules
    'core',
    'components',
]