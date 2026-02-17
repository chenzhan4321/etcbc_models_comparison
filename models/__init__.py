"""
统一模型模块
提供模型实现、核心功能和神经网络组件
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

# 模型注册表
MODEL_REGISTRY = {
    # 经典架构
    'transformer': SyriacTransformerModel,
    'bert': SyriacBERTModel,
    'lstm': SyriacLSTMModel,
    'mdlm': MDLMModel,

    # 2023-2024前沿架构
    'mamba': SyriacMambaModel,
    'bimamba': SyriacBiMambaModel,
    'bimamba_large': SyriacBiMambaLargeModel,
    'bimamba_xl': SyriacBiMambaXLModel,
    'retnet': SyriacRetNetModel,
    'switch': SyriacSwitchModel,
    'mamba_large': SyriacMambaLargeModel,

    # 2025前沿架构
    'rwkv7': SyriacRWKV7Model,
    'rwkv7_large': SyriacRWKV7LargeModel,
    'rwkv7_efficient': SyriacRWKV7EfficientModel,
}

def get_model_class(model_type: str):
    """获取模型类"""
    model_type = model_type.lower()
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"不支持的模型类型: {model_type}. 支持的类型: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_type]

def list_available_models():
    """列出所有可用的模型"""
    return list(MODEL_REGISTRY.keys())

# 重新导出core和components模块以保持向后兼容
from . import core
from . import components

__all__ = [
    # 基类
    'BaseSequenceModel',
    'BaseTransformerModel',
    'BaseRNNModel',
    'BaseDiffusionModel',

    # 经典模型
    'SyriacTransformerModel',
    'SyriacBERTModel',
    'SyriacLSTMModel',
    'MDLMModel',
    'FullMDLMModel',

    # 前沿架构模型 (2023-2024)
    'SyriacMambaModel',
    'SyriacBiMambaModel',
    'SyriacRetNetModel',
    'SyriacSwitchModel',
    'SyriacMambaLargeModel',

    # 2025最新架构模型
    'SyriacRWKV7Model',
    'SyriacRWKV7LargeModel',
    'SyriacRWKV7EfficientModel',

    # 工具函数
    'get_model_class',
    'list_available_models',
    'MODEL_REGISTRY',

    # 子模块
    'core',
    'components',
]