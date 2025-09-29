"""
MDLM (Masked Discrete Language Model) 模型实现
基于插值任务的掩码离散语言模型 - 重新设计的生成器版本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import warnings
import math
import random

from .base import BaseSequenceModel


class MDLMModel(BaseSequenceModel):
    """
    MDLM模型 - 重新设计为生成器
    
    MDLM的本质：
    1. 是生成器，不是分类器
    2. 基于插值扩散过程
    3. 输入字符和输出标签都在同一个词汇空间
    4. 使用掩码和去噪训练
    """
    
    def __init__(self, vocab_size: int, num_classes: int = None, config: Optional[Dict[str, Any]] = None):
        # MDLM需要扩展词汇表：输入字符 + 输出标签 + 特殊标记
        self.input_vocab_size = vocab_size  # 输入字符词汇表
        self.label_vocab_size = num_classes or 309  # 标签词汇表
        
        # 统一词汇表：字符 + 标签 + 特殊标记
        unified_vocab_size = vocab_size + self.label_vocab_size + 4  # +4 for <MASK>, <PAD>, <START>, <END>
        
        super().__init__(unified_vocab_size, self.label_vocab_size, config)
        
        # 特殊标记的索引
        self.mask_token_id = vocab_size + self.label_vocab_size  # <MASK>
        self.pad_token_id = vocab_size + self.label_vocab_size + 1  # <PAD>
        self.start_token_id = vocab_size + self.label_vocab_size + 2  # <START>
        self.end_token_id = vocab_size + self.label_vocab_size + 3  # <END>
        
        # 标签ID偏移
        self.label_offset = vocab_size
        
    def build_model(self):
        """构建MDLM生成器模型"""
        
        # 默认配置
        default_config = {
            'd_model': 384,
            'num_layers': 8,
            'num_heads': 8,
            'dropout': 0.1,
            'max_length': 256,
            'mask_ratio': 0.3,  # 掩码比例
            'diffusion_steps': 10,  # 扩散步数
        }
        
        for k, v in default_config.items():
            if k not in self.config:
                self.config[k] = v
        
        # 词嵌入 - 统一的字符和标签嵌入 - 修复初始化
        self.embeddings = nn.Embedding(self.vocab_size, self.config['d_model'])
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.02)
        
        # 位置编码 - 修复初始化
        self.pos_embeddings = nn.Embedding(self.config['max_length'], self.config['d_model'])
        nn.init.normal_(self.pos_embeddings.weight, mean=0.0, std=0.02)
        
        # 时间步嵌入（用于扩散过程） - 修复初始化
        self.time_embeddings = nn.Embedding(self.config['diffusion_steps'], self.config['d_model'])
        nn.init.normal_(self.time_embeddings.weight, mean=0.0, std=0.02)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config['d_model'],
            nhead=self.config['num_heads'],
            dropout=self.config['dropout'],
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.config['num_layers']
        )
        
        # 输出投影：生成下一个token - 修复初始化
        self.output_projection = nn.Linear(self.config['d_model'], self.vocab_size)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        
        # 层归一化 - 修复初始化
        self.layer_norm = nn.LayerNorm(self.config['d_model'])
        nn.init.ones_(self.layer_norm.weight)
        nn.init.zeros_(self.layer_norm.bias)
        
        # 初始化所有transformer参数
        self._init_transformer_weights()
        
        log_info = print  # 临时处理
        log_info(f"MDLM模型构建完成:")
        log_info(f"  统一词汇表大小: {self.vocab_size}")
        log_info(f"  输入字符: {self.input_vocab_size}, 标签: {self.label_vocab_size}")
        log_info(f"  模型维度: {self.config['d_model']}")
        log_info(f"  层数: {self.config['num_layers']}")
    
    def _init_transformer_weights(self):
        """初始化transformer权重 - 强制覆盖默认初始化"""
        def init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                # 特别处理MultiheadAttention的权重
                if hasattr(module, 'in_proj_weight') and module.in_proj_weight is not None:
                    nn.init.xavier_uniform_(module.in_proj_weight)
                if hasattr(module, 'in_proj_bias') and module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
                if hasattr(module, 'out_proj') and module.out_proj.weight is not None:
                    nn.init.xavier_uniform_(module.out_proj.weight)
                    if module.out_proj.bias is not None:
                        nn.init.zeros_(module.out_proj.bias)
        
        # 应用到整个transformer
        self.transformer.apply(init_weights)
    
    def add_noise(self, sequence: torch.Tensor, timestep: int) -> torch.Tensor:
        """添加扩散噪声（掩码）"""
        batch_size, seq_len = sequence.shape
        
        # 计算掩码比例（随时间步递减）
        mask_ratio = self.config['mask_ratio'] * (timestep / self.config['diffusion_steps'])
        
        # 创建掩码
        mask = torch.rand(batch_size, seq_len, device=sequence.device) < mask_ratio
        
        # 应用掩码
        noisy_sequence = sequence.clone()
        noisy_sequence[mask] = self.mask_token_id
        
        return noisy_sequence
    
    def forward(self, input_ids: torch.Tensor, target_ids: Optional[torch.Tensor] = None, 
                timestep: Optional[int] = None) -> torch.Tensor:
        """
        前向传播 - 简化版本，暂时去除复杂的时间步逻辑
        
        Args:
            input_ids: 输入序列 (B, T)
            target_ids: 目标序列 (B, T) - 训练时使用
            timestep: 扩散时间步 - 暂时忽略
        """
        batch_size, seq_len = input_ids.shape
        
        # 简化：直接使用输入序列，不进行复杂的插值
        x = self.embeddings(input_ids)
        
        # 位置编码
        pos_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embeddings(pos_ids)
        x = x + pos_emb
        
        # 暂时跳过时间步编码，避免索引问题
        # time_emb = self.time_embeddings(torch.tensor(0, device=x.device, dtype=torch.long))
        # x = x + time_emb.unsqueeze(0).unsqueeze(0).expand(batch_size, x.size(1), -1)
        
        # Transformer处理
        x = self.layer_norm(x)
        x = self.transformer(x)
        
        # 输出投影
        logits = self.output_projection(x)
        
        return logits
    
    def create_interleaved_sequence(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """创建字符+标签交替的插值序列"""
        batch_size, seq_len = input_ids.shape
        
        # 创建插值序列：[char1, label1, char2, label2, ...]
        interleaved = torch.zeros(batch_size, seq_len * 2, dtype=torch.long, device=input_ids.device)
        
        # 填充字符（偶数位置）
        interleaved[:, 0::2] = input_ids
        
        # 填充标签（奇数位置），添加偏移量
        interleaved[:, 1::2] = target_ids + self.label_offset
        
        return interleaved
    
    def generate(self, input_ids: torch.Tensor, max_length: Optional[int] = None, temperature: float = 1.0) -> torch.Tensor:
        """生成序列 - 简化版本，直接预测标签"""
        self.eval()
        
        batch_size, seq_len = input_ids.shape
        
        with torch.no_grad():
            # 前向传播获取logits
            logits = self.forward(input_ids)  # [batch, seq_len, vocab_size]
            
            # 提取标签相关的logits
            label_logits = logits[:, :, self.label_offset:self.label_offset + self.label_vocab_size]  # [batch, seq_len, label_vocab_size]
            
            # 使用温度缩放和贪婪解码
            if temperature > 0:
                scaled_logits = label_logits / temperature
                probs = F.softmax(scaled_logits, dim=-1)
                predicted_labels = torch.argmax(probs, dim=-1)
            else:
                predicted_labels = torch.argmax(label_logits, dim=-1)
        
        return predicted_labels  # [batch, seq_len] - 直接返回预测的标签
    
    def encode(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """编码序列为特征表示"""
        with torch.no_grad():
            x = self.embeddings(input_ids)
            
            # 位置编码
            batch_size, seq_len = input_ids.shape
            pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            pos_emb = self.pos_embeddings(pos_ids)
            x = x + pos_emb
            
            # Transformer编码
            x = self.layer_norm(x)
            features = self.transformer(x)
            
            return features
    
    def compute_loss(self, input_ids: torch.Tensor, target_ids: torch.Tensor, 
                     class_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算MDLM损失 - 超简化版本，就像标准分类器"""
        batch_size, seq_len = input_ids.shape
        
        # 前向传播，直接获取logits
        logits = self.forward(input_ids, target_ids)  # [batch, seq_len, vocab_size]
        
        # 只使用标签相关的logits
        # 提取标签词汇表对应的logits
        label_logits = logits[:, :, self.label_offset:self.label_offset + self.label_vocab_size]  # [batch, seq_len, label_vocab_size]
        
        # 确保target_ids维度匹配
        min_len = min(label_logits.size(1), target_ids.size(1))
        if min_len == 0:
            return torch.tensor(0.0, device=input_ids.device, requires_grad=True)
            
        label_logits = label_logits[:, :min_len, :]  # [batch, min_len, label_vocab_size]
        target_ids = target_ids[:, :min_len]         # [batch, min_len]
        
        # 重塑为2D进行损失计算
        label_logits_2d = label_logits.contiguous().view(-1, self.label_vocab_size)  # [batch*min_len, label_vocab_size]
        target_ids_1d = target_ids.contiguous().view(-1)  # [batch*min_len]
        
        # 创建掩码，排除无效标签
        valid_mask = (target_ids_1d >= 0) & (target_ids_1d < self.label_vocab_size)
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=input_ids.device, requires_grad=True)
        
        # 只对有效位置计算损失
        valid_logits = label_logits_2d[valid_mask]  # [num_valid, label_vocab_size]
        valid_targets = target_ids_1d[valid_mask]   # [num_valid]
        
        # 交叉熵损失 - 支持类别权重
        if class_weights is not None and class_weights.size(0) == self.label_vocab_size:
            # 使用类别权重
            loss = F.cross_entropy(valid_logits, valid_targets, weight=class_weights, reduction='mean')
        else:
            # 标准交叉熵损失
            loss = F.cross_entropy(valid_logits, valid_targets, reduction='mean')
        
        return loss
    
    def save_config(self) -> Dict[str, Any]:
        """保存模型配置"""
        return {
            'model_type': 'mdlm',
            'input_vocab_size': self.input_vocab_size,
            'label_vocab_size': self.label_vocab_size,
            'vocab_size': self.vocab_size,
            'num_classes': self.num_classes,
            **self.config
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'MDLMModel':
        """从配置创建模型"""
        return cls(
            vocab_size=config['input_vocab_size'],
            num_classes=config['label_vocab_size'],
            config=config
        )