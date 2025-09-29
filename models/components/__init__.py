"""
神经网络组件模块
包含各种可复用的神经网络层和组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


def init_weights(module):
    """初始化网络权重"""
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def create_padding_mask(input_ids: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """创建填充掩码"""
    return (input_ids != pad_token_id).float()


class UnifiedEmbedding(nn.Module):
    """统一的嵌入层，包含位置编码和可选的段嵌入"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_length: int = 512,
        dropout: float = 0.1,
        padding_idx: Optional[int] = None,
        learnable_pos: bool = True,
        use_segment: bool = False,
        num_segments: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length
        self.learnable_pos = learnable_pos
        self.use_segment = use_segment

        # Token嵌入
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

        # 位置编码
        if learnable_pos:
            # 可学习的位置编码（如BERT）
            self.position_embedding = nn.Embedding(max_length, d_model)
        else:
            # 固定的正弦位置编码（如原始Transformer）
            self.register_buffer('pos_encoding', self._create_sinusoidal_encoding(max_length, d_model))

        # 段嵌入（用于BERT等需要句子对的模型）
        if use_segment:
            self.segment_embedding = nn.Embedding(num_segments, d_model)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('position_ids', torch.arange(max_length).unsqueeze(0))

    def _create_sinusoidal_encoding(self, max_length: int, d_model: int) -> torch.Tensor:
        """创建正弦位置编码"""
        encoding = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        return encoding.unsqueeze(0)

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.size()

        # Token嵌入
        token_emb = self.token_embedding(input_ids)

        # 位置编码
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_len].expand(batch_size, -1)

        if self.learnable_pos:
            pos_emb = self.position_embedding(position_ids)
        else:
            pos_emb = self.pos_encoding[:, :seq_len, :].expand(batch_size, -1, -1)

        embeddings = token_emb + pos_emb

        # 段嵌入
        if self.use_segment:
            if segment_ids is None:
                segment_ids = torch.zeros_like(input_ids)
            segment_emb = self.segment_embedding(segment_ids)
            embeddings = embeddings + segment_emb

        return self.dropout(embeddings)


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = query.size(0)

        # 线性变换和重塑
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            # 扩展掩码维度以匹配attention scores
            # mask: (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            # 对于自注意力，创建序列到序列的掩码
            # mask: (batch_size, 1, 1, seq_len) -> (batch_size, 1, seq_len, seq_len)
            if mask.size(-2) == 1:
                mask = mask.expand(-1, -1, scores.size(-2), -1)
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(context)

        if need_weights:
            return output, attention
        else:
            return output, None


class SelfAttention(MultiHeadAttention):
    """自注意力层（简化版）"""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__(d_model, 1, dropout)  # 单头注意力

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        output, _ = super().forward(x, x, x, mask, need_weights=False)
        return output


class FeedForward(nn.Module):
    """前馈网络"""

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        # 兼容两种参数名称
        if d_ff is not None:
            feedforward_dim = d_ff
        elif dim_feedforward is not None:
            feedforward_dim = dim_feedforward
        else:
            feedforward_dim = d_model * 4  # 默认值

        self.linear1 = nn.Linear(d_model, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, d_model)
        self.dropout = nn.Dropout(dropout)

        # 支持不同的激活函数
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "swish" or activation == "silu":
            self.activation = F.silu
        else:
            self.activation = F.gelu  # 默认使用GELU

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class ResidualConnection(nn.Module):
    """残差连接"""

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        norm_first: bool = True,
        norm_type: str = "layer"
    ):
        super().__init__()
        self.norm_first = norm_first
        self.dropout = nn.Dropout(dropout)

        if norm_type == "layer":
            self.norm = nn.LayerNorm(d_model)
        else:
            self.norm = nn.LayerNorm(d_model)  # 默认使用LayerNorm

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        if self.norm_first:
            # Pre-LayerNorm (现代架构)
            return x + self.dropout(sublayer(self.norm(x)))
        else:
            # Post-LayerNorm (原始Transformer/BERT)
            return self.norm(x + self.dropout(sublayer(x)))


class LayerNorm(nn.LayerNorm):
    """层归一化（继承PyTorch的LayerNorm）"""
    pass


class ClassifierHead(nn.Module):
    """分类头"""

    def __init__(
        self,
        d_model: Optional[int] = None,
        input_size: Optional[int] = None,
        num_classes: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_pooling: bool = False,
        num_layers: int = 1
    ):
        super().__init__()

        # 兼容两种参数名称
        if input_size is not None:
            input_dim = input_size
        elif d_model is not None:
            input_dim = d_model
        else:
            raise ValueError("必须提供 input_size 或 d_model 参数")

        self.use_pooling = use_pooling
        self.dropout = nn.Dropout(dropout)

        # 支持多层分类头
        if num_layers == 1:
            self.classifier = nn.Linear(input_dim, num_classes)
        else:
            layers = []
            current_dim = input_dim

            for i in range(num_layers - 1):
                layers.append(nn.Linear(current_dim, current_dim // 2))
                layers.append(nn.Dropout(dropout))
                if activation == "gelu":
                    layers.append(nn.GELU())
                elif activation == "relu":
                    layers.append(nn.ReLU())
                else:
                    layers.append(nn.GELU())
                current_dim = current_dim // 2

            layers.append(nn.Linear(current_dim, num_classes))
            self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_pooling:
            # 全局平均池化（用于句子级别分类）
            x = x.mean(dim=1)

        return self.classifier(self.dropout(x))


# 导出所有组件
__all__ = [
    'init_weights',
    'create_padding_mask',
    'UnifiedEmbedding',
    'MultiHeadAttention',
    'SelfAttention',
    'FeedForward',
    'ResidualConnection',
    'LayerNorm',
    'ClassifierHead',
]