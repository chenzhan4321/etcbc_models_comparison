"""
BERT模型实现
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
)
from .core import log_info


class BERTLayer(nn.Module):
    """BERT编码器层"""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        
        # 多头自注意力
        self.attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # 前馈网络
        self.feed_forward = FeedForward(
            d_model=d_model,
            d_ff=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        
        # 残差连接和层归一化
        self.attention_residual = ResidualConnection(
            d_model=d_model,
            dropout=dropout,
            norm_first=False,  # BERT使用Post-LN
            norm_type="layer",
        )
        
        self.ff_residual = ResidualConnection(
            d_model=d_model,
            dropout=dropout,
            norm_first=False,
            norm_type="layer",
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            hidden_states: [batch_size, seq_len, d_model]
            attention_mask: [batch_size, seq_len] 或 [batch_size, 1, seq_len, seq_len]
        """
        # 自注意力
        def attention_fn(x):
            return self.attention(
                query=x,
                key=x,
                value=x,
                mask=attention_mask,
                need_weights=False
            )[0]
        
        hidden_states = self.attention_residual(hidden_states, attention_fn)
        
        # 前馈网络
        hidden_states = self.ff_residual(hidden_states, self.feed_forward)
        
        return hidden_states


class BERTEncoder(nn.Module):
    """BERT编码器"""
    
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            BERTLayer(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in range(num_layers)
        ])
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向传播通过所有层"""
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        return hidden_states


class SyriacBERTModel(BaseTransformerModel):
    """叙利亚文BERT模型"""
    
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化BERT模型
        
        Args:
            vocab_size: 词汇表大小
            num_classes: 分类类别数
            config: 模型配置
        """
        # BERT特定的默认配置
        bert_config = {
            'd_model': 768,
            'num_layers': 12,
            'num_heads': 12,
            'dim_feedforward': 3072,
            'dropout': 0.1,
            'attention_dropout': 0.1,
            'max_length': 512,
            'layer_norm_eps': 1e-12,
            'activation': 'gelu',
            'initializer_range': 0.02,
        }
        
        # 合并用户配置
        if config:
            bert_config.update(config)
        
        super().__init__(vocab_size, num_classes, bert_config)
        log_info(f"BERT模型初始化完成: {self.get_num_parameters():,} 参数")
    
    def build_model(self):
        """构建BERT模型架构"""
        # 从配置中获取参数
        d_model = self.config['d_model']
        num_layers = self.config['num_layers']
        num_heads = self.config['num_heads']
        dim_feedforward = self.config['dim_feedforward']
        dropout = self.config['dropout']
        max_length = self.config['max_length']
        activation = self.config['activation']
        layer_norm_eps = self.config['layer_norm_eps']
        
        # BERT嵌入层（包含token、position和segment嵌入）
        self.embeddings = UnifiedEmbedding(
            vocab_size=self.vocab_size,
            d_model=d_model,
            max_length=max_length,
            dropout=dropout,
            padding_idx=0,
            learnable_pos=True,  # BERT使用可学习的位置编码
            use_segment=True,    # BERT使用段嵌入
            num_segments=2,
        )
        
        # BERT编码器
        self.encoder = BERTEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
        )
        
        # 分类头
        self.classifier = ClassifierHead(
            d_model=d_model,
            num_classes=self.num_classes,
            dropout=dropout,
            activation=activation,
            use_pooling=False,  # 序列标注不需要池化
            num_layers=2,
        )
        
        # 初始化权重
        self.apply(self._init_bert_weights)

        # 可选 CRF 层（BERT-token-classifier + CRF；回应 R2-M4/R3-2）。
        # 放在 apply 之后，CRF 保持自身初始化（其参数不被 _init_bert_weights 触及）。
        if self.config.get('use_crf', False):
            try:
                from torchcrf import CRF
                self.crf = CRF(self.num_classes, batch_first=True)
                log_info("BERT 启用 CRF 层")
            except ImportError:
                log_info("未安装 torchcrf，跳过 CRF 层")
                self.crf = None
        else:
            self.crf = None

    def _init_bert_weights(self, module):
        """BERT特定的权重初始化（数值稳定版本）"""
        initializer_range = self.config.get('initializer_range', 0.01)  # 降低初始化标准差
        
        if isinstance(module, nn.Linear):
            # 使用更保守的正态分布初始化
            module.weight.data.normal_(mean=0.0, std=initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 嵌入层使用更小的标准差
            module.weight.data.normal_(mean=0.0, std=initializer_range * 0.5)
            if hasattr(module, 'padding_idx') and module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LayerNorm):
            # LayerNorm的标准初始化
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        编码输入序列
        
        Args:
            input_ids: 输入token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            token_type_ids: 段ID [batch_size, seq_len]
            
        Returns:
            编码表示 [batch_size, seq_len, d_model]
        """
        # 如果没有提供token_type_ids，创建全0的
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # 嵌入
        embeddings = self.embeddings(
            input_ids=input_ids,
            segment_ids=token_type_ids,
        )
        
        # 准备注意力掩码（数值稳定版本）
        if attention_mask is not None:
            # FIX(2026-06-07): MultiHeadAttention 用 masked_fill(mask==0,-1e9) 约定;
            # 原加性掩码符号与之相反 → 会 mask 掉真实 token、只 attend padding → 塌缩到多数类(0.71)。
            # 直接传原始 0/1 掩码(1=真实,0=padding),MHA 内部自会扩展维度。
            extended_attention_mask = attention_mask
        else:
            extended_attention_mask = None
        
        # 通过编码器
        encoder_outputs = self.encoder(
            hidden_states=embeddings,
            attention_mask=extended_attention_mask,
        )
        
        return encoder_outputs
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            token_type_ids: 段ID（用于NSP任务）
            labels: 标签（可选）
        """
        # 编码
        sequence_output = self.encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs
        )
        
        # 分类
        logits = self.classifier(sequence_output)
        
        # 计算损失
        if labels is not None:
            loss = self._compute_loss(logits, labels, attention_mask)
            return loss, logits
        
        return logits