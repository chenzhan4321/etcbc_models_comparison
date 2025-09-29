"""
Transformer模型实现
使用统一组件和基类的重构版本
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from .base import BaseTransformerModel
from .components import (
    UnifiedEmbedding,
    MultiHeadAttention,
    FeedForward,
    ResidualConnection,
    LayerNorm,
    ClassifierHead,
    create_padding_mask,
)
from .core import log_info, get_vocab_size


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = False,
    ):
        super().__init__()
        
        # 自注意力
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        # 前馈网络
        self.feed_forward = FeedForward(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation
        )
        
        # 残差连接
        self.residual1 = ResidualConnection(d_model, dropout, norm_first)
        self.residual2 = ResidualConnection(d_model, dropout, norm_first)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向传播"""
        # 自注意力 + 残差
        def attention_fn(x):
            # 处理掩码参数
            attn_mask = None
            if mask is not None:
                attn_mask = mask
            elif src_key_padding_mask is not None:
                # 将padding mask转换为attention mask
                attn_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(1)
                attn_mask = (1.0 - attn_mask.float()) * -1e4

            output, _ = self.self_attn(x, x, x, mask=attn_mask, need_weights=False)
            return output

        x = self.residual1(x, attention_fn)
        
        # 前馈网络 + 残差
        x = self.residual2(x, self.feed_forward)
        
        return x


class TransformerEncoder(nn.Module):
    """Transformer编码器"""
    
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = False,
    ):
        super().__init__()
        
        # 编码器层
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model, num_heads, dim_feedforward,
                dropout, activation, norm_first
            )
            for _ in range(num_layers)
        ])
        
        # 最终层归一化
        self.final_norm = LayerNorm(d_model) if norm_first else None
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向传播"""
        for layer in self.layers:
            x = layer(x, mask, src_key_padding_mask)
        
        if self.final_norm is not None:
            x = self.final_norm(x)
        
        return x


class SyriacTransformerModel(BaseTransformerModel):
    """叙利亚文Transformer模型"""
    
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化Transformer模型
        
        Args:
            vocab_size: 词汇表大小
            num_classes: 分类类别数
            config: 模型配置
        """
        super().__init__(vocab_size, num_classes, config)
        log_info(f"Transformer模型初始化完成: {self.get_num_parameters():,} 参数")
    
    def build_model(self):
        """构建模型架构"""
        # 从配置中获取参数
        d_model = self.config['d_model']
        num_layers = self.config['num_layers']
        num_heads = self.config['num_heads']
        dim_feedforward = self.config['dim_feedforward']
        dropout = self.config['dropout']
        max_length = self.config['max_length']
        activation = self.config.get('activation', 'gelu')
        norm_first = self.config.get('norm_first', False)
        
        # 嵌入层（使用统一组件）
        self.embedding = UnifiedEmbedding(
            vocab_size=self.vocab_size,
            d_model=d_model,
            max_length=max_length,
            dropout=dropout,
            padding_idx=0,  # 假设0是padding token
            learnable_pos=False,  # 使用正弦位置编码
        )
        
        # Transformer编码器
        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
        )
        
        # 分类头（使用统一组件）
        self.classifier = ClassifierHead(
            d_model=d_model,
            num_classes=self.num_classes,
            dropout=dropout,
            activation=activation,
            use_pooling=False,  # 序列标注任务不需要池化
            num_layers=2,
        )
    
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
            
        Returns:
            编码表示 [batch_size, seq_len, d_model]
        """
        # 嵌入
        x = self.embedding(input_ids)
        
        # 准备掩码
        if attention_mask is not None:
            # 将填充掩码转换为键填充掩码
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        
        # 编码
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        return x
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        前向传播
        
        继承自BaseTransformerModel，提供完整的forward实现
        """
        return super().forward(input_ids, attention_mask, labels, **kwargs)