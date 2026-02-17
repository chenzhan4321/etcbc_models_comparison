"""
模型基础类
定义所有模型的基础类和接口
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union
import json
from pathlib import Path

from .core import log_info, log_warning, get_device
from .components import init_weights

class BaseSequenceModel(nn.Module, ABC):
    """所有序列模型的基类"""
    
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化基础模型
        
        Args:
            vocab_size: 词汇表大小
            num_classes: 分类类别数
            config: 模型配置
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.config = config or {}
        
        # 保存模型类型
        self.model_type = self.__class__.__name__
        
        # 构建模型
        self.build_model()
        
        # 初始化权重
        self.apply(self._init_weights)
        
        log_info(f"初始化 {self.model_type} - vocab_size: {vocab_size}, num_classes: {num_classes}")
    
    @abstractmethod
    def build_model(self):
        """构建模型架构（子类实现）"""
        pass
    
    @abstractmethod
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        编码输入序列
        
        Args:
            input_ids: 输入ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            **kwargs: 其他参数
            
        Returns:
            编码表示 [batch_size, seq_len, hidden_size]
        """
        pass
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播
        
        Args:
            input_ids: 输入ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            labels: 标签 [batch_size, seq_len]（可选）
            **kwargs: 其他参数
            
        Returns:
            如果提供labels，返回(loss, logits)
            否则返回logits
        """
        # 编码
        hidden_states = self.encode(input_ids, attention_mask, **kwargs)
        
        # 分类
        if hasattr(self, 'classifier'):
            logits = self.classifier(hidden_states)
        else:
            raise NotImplementedError("模型必须定义classifier")
        
        # 计算损失（如果提供标签）
        if labels is not None:
            loss = self._compute_loss(logits, labels, attention_mask)
            return loss, logits
        
        return logits
    
    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算损失
        
        Args:
            logits: 模型输出 [batch_size, seq_len, num_classes]
            labels: 真实标签 [batch_size, seq_len]
            mask: 掩码 [batch_size, seq_len]
        """
        # 重塑为二维进行损失计算
        batch_size, seq_len = labels.shape
        logits_2d = logits.view(-1, self.num_classes)
        labels_1d = labels.view(-1)
        
        # 计算交叉熵损失
        loss = nn.functional.cross_entropy(
            logits_2d, labels_1d, reduction='none'
        )
        
        # 应用掩码
        if mask is not None:
            mask_1d = mask.view(-1).float()
            loss = loss * mask_1d
            loss = loss.sum() / mask_1d.sum().clamp(min=1e-5)
        else:
            loss = loss.mean()
        
        return loss
    
    def _init_weights(self, module):
        """初始化权重（可被子类覆盖）"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def save_config(self, save_path: str):
        """保存模型配置"""
        config_path = Path(save_path) / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_to_save = {
            "model_type": self.model_type,
            "vocab_size": self.vocab_size,
            "num_classes": self.num_classes,
            "config": self.config
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_to_save, f, indent=2, ensure_ascii=False)
        
        log_info(f"配置已保存到: {config_path}")
    
    @classmethod
    def from_config(cls, config_path: str, **kwargs):
        """从配置文件加载模型"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        return cls(
            vocab_size=config["vocab_size"],
            num_classes=config["num_classes"],
            config=config.get("config", {}),
            **kwargs
        )
    
    def get_num_parameters(self, only_trainable: bool = False) -> int:
        """获取参数数量"""
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def to_device(self, device: Optional[torch.device] = None):
        """将模型移动到指定设备"""
        if device is None:
            device = get_device()
        return self.to(device)


class BaseTransformerModel(BaseSequenceModel):
    """Transformer系列模型的基类"""
    
    def __init__(self, vocab_size: int, num_classes: int, config: Optional[Dict[str, Any]] = None):
        # 设置默认的Transformer配置
        default_config = {
            'd_model': 512,
            'num_layers': 6,
            'num_heads': 8,
            'dim_feedforward': 2048,
            'dropout': 0.1,
            'max_length': 512,
            'layer_norm_eps': 1e-5,
            'activation': 'gelu',
        }
        
        # 合并用户配置
        if config:
            default_config.update(config)
        
        super().__init__(vocab_size, num_classes, default_config)
    
    def create_padding_mask(
        self,
        input_ids: torch.Tensor,
        pad_token_id: int = 0
    ) -> torch.Tensor:
        """创建填充掩码"""
        return (input_ids != pad_token_id).float()
    
    def create_attention_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False
    ) -> Optional[torch.Tensor]:
        """创建注意力掩码"""
        seq_len = input_ids.size(1)
        device = input_ids.device
        
        if is_causal:
            # 创建因果掩码
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device),
                diagonal=1
            ).bool()
            
            if attention_mask is not None:
                # 结合填充掩码和因果掩码
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attention_mask = attention_mask & ~causal_mask
            else:
                attention_mask = ~causal_mask
        
        return attention_mask


class BaseRNNModel(BaseSequenceModel):
    """RNN系列模型的基类"""
    
    def __init__(self, vocab_size: int, num_classes: int, config: Optional[Dict[str, Any]] = None):
        # 设置默认的RNN配置
        default_config = {
            'hidden_size': 256,
            'num_layers': 2,
            'dropout': 0.1,
            'bidirectional': True,
            'rnn_type': 'lstm',  # 'lstm', 'gru', 'rnn'
            'use_attention': True,
        }
        
        # 合并用户配置
        if config:
            default_config.update(config)
        
        super().__init__(vocab_size, num_classes, default_config)
    
    def get_rnn_class(self, rnn_type: str):
        """获取RNN类"""
        rnn_classes = {
            'lstm': nn.LSTM,
            'gru': nn.GRU,
            'rnn': nn.RNN,
        }
        return rnn_classes.get(rnn_type.lower(), nn.LSTM)


class BaseDiffusionModel(BaseSequenceModel):
    """扩散模型的基类"""
    
    def __init__(self, vocab_size: int, num_classes: int, config: Optional[Dict[str, Any]] = None):
        # 设置默认的扩散模型配置
        default_config = {
            'd_model': 256,
            'num_layers': 4,
            'num_heads': 8,
            'dim_feedforward': 1024,
            'dropout': 0.1,
            'max_length': 128,
            'num_timesteps': 10,
            'beta_start': 0.0001,
            'beta_end': 0.02,
            'beta_schedule': 'linear',
        }
        
        # 合并用户配置
        if config:
            default_config.update(config)
        
        super().__init__(vocab_size, num_classes, default_config)
    
    @abstractmethod
    def add_noise(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """添加噪声（子类实现）"""
        pass
    
    @abstractmethod
    def denoise(self, x: torch.Tensor, t: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """去噪（子类实现）"""
        pass