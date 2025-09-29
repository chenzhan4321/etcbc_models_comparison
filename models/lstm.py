"""
LSTM模型实现
使用统一组件和基类的重构版本
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple

from .base import BaseRNNModel
from .components import (
    UnifiedEmbedding,
    SelfAttention,
    ClassifierHead,
    LayerNorm,
    create_padding_mask,
)
from .components.embeddings import SimpleEmbedding
from .components.layers import ClassifierHead
from .components.attention import MultiHeadAttention
from .core import log_info


class LSTMEncoder(nn.Module):
    """LSTM编码器"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
        use_attention: bool = True,
        attention_heads: int = 8,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 计算输出维度
        self.output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # 可选的自注意力层
        if use_attention:
            self.attention = SelfAttention(
                d_model=self.output_size,
                dropout=dropout,
            )
            self.attention_norm = LayerNorm(self.output_size)
            self.attention_dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, input_size]
            mask: 掩码 [batch_size, seq_len]
            lengths: 序列实际长度 [batch_size]
            
        Returns:
            output: 编码输出 [batch_size, seq_len, output_size]
            (h_n, c_n): 最终隐藏状态和细胞状态
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        # 如果提供了长度，使用pack_padded_sequence优化
        if lengths is not None:
            # 按长度排序
            sorted_lengths, sorted_idx = lengths.sort(0, descending=True)
            sorted_x = x[sorted_idx]
            
            # Pack序列
            packed_x = nn.utils.rnn.pack_padded_sequence(
                sorted_x, sorted_lengths.cpu(), batch_first=True, enforce_sorted=True
            )
            
            # 通过LSTM
            packed_output, (h_n, c_n) = self.lstm(packed_x)
            
            # Unpack序列
            output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True, total_length=seq_len
            )
            
            # 恢复原始顺序
            _, unsorted_idx = sorted_idx.sort(0)
            output = output[unsorted_idx]
            h_n = h_n[:, unsorted_idx]
            c_n = c_n[:, unsorted_idx]
        else:
            # 直接通过LSTM
            output, (h_n, c_n) = self.lstm(x)
        
        # 应用自注意力（如果启用）
        if self.use_attention:
            # 残差连接 + 自注意力
            attn_output, _ = self.attention(output, mask=mask)
            output = self.attention_norm(output + self.attention_dropout(attn_output))
        
        return output, (h_n, c_n)


class SyriacLSTMModel(BaseRNNModel):
    """叙利亚文LSTM模型"""
    
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化LSTM模型
        
        Args:
            vocab_size: 词汇表大小
            num_classes: 分类类别数
            config: 模型配置
        """
        # LSTM特定的默认配置
        lstm_config = {
            'embedding_dim': 128,
            'hidden_size': 256,
            'num_layers': 2,
            'dropout': 0.3,
            'bidirectional': True,
            'use_attention': True,
            'attention_heads': 8,
            'use_crf': False,  # 可选的CRF层
        }
        
        # 合并用户配置
        if config:
            lstm_config.update(config)
        
        super().__init__(vocab_size, num_classes, lstm_config)
        log_info(f"LSTM模型初始化完成: {self.get_num_parameters():,} 参数")
    
    def build_model(self):
        """构建LSTM模型架构"""
        # 从配置中获取参数
        embedding_dim = self.config['embedding_dim']
        hidden_size = self.config['hidden_size']
        num_layers = self.config['num_layers']
        dropout = self.config['dropout']
        bidirectional = self.config['bidirectional']
        use_attention = self.config['use_attention']
        attention_heads = self.config.get('attention_heads', 8)
        
        # 嵌入层
        self.embedding = SimpleEmbedding(
            vocab_size=self.vocab_size,
            embed_dim=embedding_dim,
        )
        
        # LSTM编码器
        self.encoder = LSTMEncoder(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            use_attention=use_attention,
            attention_heads=attention_heads,
        )
        
        # 分类头
        encoder_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.classifier = ClassifierHead(
            input_size=encoder_output_size,
            num_classes=self.num_classes,
            dropout=dropout,
        )
        
        # 可选的CRF层（用于序列标注）
        if self.config.get('use_crf', False):
            try:
                from torchcrf import CRF
                self.crf = CRF(self.num_classes, batch_first=True)
                log_info("使用CRF层进行序列标注")
            except ImportError:
                log_info("未安装torchcrf，跳过CRF层")
                self.crf = None
        else:
            self.crf = None
    
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
            编码表示 [batch_size, seq_len, hidden_size]
        """
        # 嵌入
        embedded = self.embedding(input_ids)
        
        # 计算序列长度（用于优化）
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).long()
        else:
            lengths = None
        
        # LSTM编码
        encoded, _ = self.encoder(embedded, mask=attention_mask, lengths=lengths)
        
        return encoded
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: 输入ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            labels: 标签 [batch_size, seq_len]
        """
        # 编码
        sequence_output = self.encode(input_ids, attention_mask, **kwargs)
        
        # 分类
        logits = self.classifier(sequence_output)
        
        # 计算损失
        if labels is not None:
            if hasattr(self, 'crf') and self.crf is not None:
                # 使用CRF计算损失
                mask = attention_mask.bool() if attention_mask is not None else None
                loss = -self.crf(logits, labels, mask=mask, reduction='mean')
            else:
                # 使用标准交叉熵损失
                loss = self._compute_loss(logits, labels, attention_mask)
            
            return loss, logits
        
        return logits
    
    def decode(
        self,
        logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        解码预测结果（用于CRF）
        
        Args:
            logits: 模型输出 [batch_size, seq_len, num_classes]
            attention_mask: 掩码 [batch_size, seq_len]
            
        Returns:
            预测标签 [batch_size, seq_len]
        """
        if hasattr(self, 'crf') and self.crf is not None:
            mask = attention_mask.bool() if attention_mask is not None else None
            return self.crf.decode(logits, mask=mask)
        else:
            return logits.argmax(dim=-1)