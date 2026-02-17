"""
模型工厂模块
统一管理所有模型的创建和配置

重构版本：使用新的模型架构
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from pathlib import Path

from models.core import (
    ConfigError,
    log_info,
    log_warning,
    get_vocab_size,
    get_device_manager,
)
from models import (
    MODEL_REGISTRY,
    get_model_class,
    list_available_models,
)


class ModelFactory:
    """
    模型工厂类
    
    负责根据配置创建不同类型的模型实例。
    提供统一的接口来处理所有支持的模型类型。
    """
    
    def __init__(self):
        """初始化模型工厂"""
        # 默认配置模板
        self._default_configs = {
            'transformer': {
                'd_model': 512,
                'num_layers': 6,
                'num_heads': 8,
                'dim_feedforward': 2048,
                'dropout': 0.1,
                'max_length': 512,
                'activation': 'gelu',
                'layer_norm_eps': 1e-5,
            },
            'bert': {
                'd_model': 768,
                'num_layers': 12,
                'num_heads': 12,
                'dim_feedforward': 3072,
                'dropout': 0.1,
                'attention_dropout': 0.1,
                'max_length': 512,
                'layer_norm_eps': 1e-12,
                'activation': 'gelu',
            },
            'bilstm': {
                'embedding_dim': 128,
                'hidden_size': 256,
                'num_layers': 2,
                'dropout': 0.3,
                'bidirectional': True,
                'use_attention': True,
                'attention_heads': 8,
            },
            'lstm': {
                'embedding_dim': 128,
                'hidden_size': 256,
                'num_layers': 2,
                'dropout': 0.3,
                'bidirectional': True,
                'use_attention': True,
                'attention_heads': 8,
                'use_crf': False,
            },
            'diffusion': {
                'd_model': 256,
                'num_layers': 6,
                'num_heads': 8,
                'dim_feedforward': 1024,
                'dropout': 0.1,
                'max_length': 256,
                'num_timesteps': 10,
                'beta_start': 0.0001,
                'beta_end': 0.02,
                'beta_schedule': 'linear',
            }
        }
    
    def create_model(
        self,
        model_type: str,
        vocab_size: Optional[int] = None,
        num_classes: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> nn.Module:
        """
        创建模型实例
        
        Args:
            model_type: 模型类型
            vocab_size: 词汇表大小（如果None，根据模型类型自动选择）
            num_classes: 分类类别数（如果None，默认309）
            config: 模型配置参数
            device: 设备（如果None，自动选择）
            
        Returns:
            创建的模型实例
            
        Raises:
            ConfigError: 不支持的模型类型或配置错误
        """
        # 验证模型类型
        model_type = model_type.lower()
        if model_type not in list_available_models():
            raise ConfigError(
                f"不支持的模型类型: {model_type}. "
                f"支持的类型: {list_available_models()}"
            )
        
        # 自动设置词汇表大小
        if vocab_size is None:
            vocab_size = get_vocab_size(model_type)
            log_info(f"自动设置 {model_type} 模型词汇表大小: {vocab_size}")
        
        # 设置默认类别数
        if num_classes is None:
            if model_type == 'diffusion':
                num_classes = vocab_size  # 扩散模型输出与词汇表相同
            else:
                num_classes = 309  # 默认的形态标签数
        
        # 合并配置
        final_config = self._merge_configs(model_type, config)
        
        # 获取模型类
        model_class = get_model_class(model_type)
        
        # 创建模型
        log_info(f"创建 {model_type} 模型...")
        model = model_class(
            vocab_size=vocab_size,
            num_classes=num_classes,
            config=final_config
        )
        
        # 移动到设备
        if device is None:
            device = get_device_manager().device
        model = model.to(device)
        
        # 打印模型信息
        self._print_model_info(model, model_type)
        
        return model
    
    def _merge_configs(
        self,
        model_type: str,
        user_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        合并默认配置和用户配置
        
        Args:
            model_type: 模型类型
            user_config: 用户提供的配置
            
        Returns:
            合并后的配置
        """
        # 获取默认配置
        default_config = self._default_configs.get(model_type, {}).copy()
        
        # 合并用户配置
        if user_config:
            default_config.update(user_config)
        
        return default_config
    
    def _print_model_info(self, model: nn.Module, model_type: str):
        """打印模型信息"""
        param_count = model.get_num_parameters()
        trainable_count = model.get_num_parameters(only_trainable=True)
        
        log_info(f"模型创建成功:")
        log_info(f"  - 类型: {model_type}")
        log_info(f"  - 总参数: {param_count:,}")
        log_info(f"  - 可训练参数: {trainable_count:,}")
        log_info(f"  - 设备: {next(model.parameters()).device}")
    
    def get_default_config(self, model_type: str) -> Dict[str, Any]:
        """
        获取模型的默认配置
        
        Args:
            model_type: 模型类型
            
        Returns:
            默认配置字典
        """
        model_type = model_type.lower()
        if model_type not in self._default_configs:
            raise ConfigError(f"未知的模型类型: {model_type}")
        
        return self._default_configs[model_type].copy()
    
    def list_models(self) -> List[str]:
        """列出所有支持的模型类型"""
        return list_available_models()
    
    @staticmethod
    def from_pretrained(
        model_path: str,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> nn.Module:
        """
        从预训练模型加载
        
        Args:
            model_path: 模型路径
            device: 设备
            **kwargs: 其他参数
            
        Returns:
            加载的模型
        """
        model_path = Path(model_path)
        
        # 加载配置
        config_path = model_path / "config.json"
        if not config_path.exists():
            raise ConfigError(f"配置文件不存在: {config_path}")
        
        import json
        with open(config_path, 'r', encoding='utf-8') as f:
            saved_config = json.load(f)
        
        # 获取模型类型和参数
        model_type = saved_config.get('model_type')
        if not model_type:
            # 尝试从类名推断
            model_type = saved_config.get('model_class', '').replace('Syriac', '').replace('Model', '').lower()
        
        vocab_size = saved_config['vocab_size']
        num_classes = saved_config['num_classes']
        config = saved_config.get('config', {})
        
        # 获取模型类
        model_class = get_model_class(model_type)
        
        # 创建模型
        model = model_class(vocab_size, num_classes, config)
        
        # 加载权重
        weights_path = model_path / "model.pth"
        if weights_path.exists():
            if device is None:
                device = get_device_manager().device
            
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            model = model.to(device)
            
            log_info(f"成功加载预训练模型: {model_path}")
        else:
            log_warning(f"权重文件不存在: {weights_path}")
        
        return model


# 全局工厂实例
_model_factory = None

def get_model_factory() -> ModelFactory:
    """获取全局模型工厂实例"""
    global _model_factory
    if _model_factory is None:
        _model_factory = ModelFactory()
    return _model_factory


def create_model(
    model_type: str,
    vocab_size: Optional[int] = None,
    num_classes: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    便捷函数：创建模型
    
    Args:
        model_type: 模型类型
        vocab_size: 词汇表大小
        num_classes: 类别数
        config: 配置
        device: 设备
        
    Returns:
        模型实例
    """
    factory = get_model_factory()
    return factory.create_model(model_type, vocab_size, num_classes, config, device)


def get_supported_models() -> List[str]:
    """获取支持的模型列表"""
    return list_available_models()


def get_model_config(model_type: str) -> Dict[str, Any]:
    """获取模型默认配置"""
    factory = get_model_factory()
    return factory.get_default_config(model_type)


__all__ = [
    'ModelFactory',
    'get_model_factory',
    'create_model',
    'get_supported_models',
    'get_model_config',
]