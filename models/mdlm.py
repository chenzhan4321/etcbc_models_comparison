"""
MDLM (Masked Discrete Language Model) - 插值Diffusion模型实现

核心设计：
1. 插值序列结构：[c₁, l₁, c₂, l₂, c₃, l₃, ...] 字符和标签交替
2. 统一embedding空间：字符(0-39) + 标签(40-348) + MASK(349)
3. 训练时随机mask标签位置，模拟扩散中间状态
4. 推理时迭代去噪，逐步unmask最confident的位置

序列长度说明：
- 原始序列长度为 L
- 插值后序列长度为 2*L（字符和标签交替）
- 偶数位置(0,2,4,...)：字符（已知，不mask）
- 奇数位置(1,3,5,...)：标签（训练时部分mask，推理时全mask开始）
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
    插值Diffusion模型

    关键特性：
    1. 字符和标签在同一序列中交替出现
    2. 使用统一的embedding空间
    3. 训练时随机mask标签位置
    4. 推理时迭代refinement
    """

    def __init__(self, vocab_size: int, num_classes: int = None, config: Optional[Dict[str, Any]] = None):
        """
        初始化插值Diffusion模型

        Args:
            vocab_size: 输入字符词汇表大小（通常为40）
            num_classes: 标签类别数（通常为309或329）
            config: 模型配置字典
        """
        # 保存原始参数
        self.input_vocab_size = vocab_size  # 字符词汇表大小 (0-39)
        self.label_vocab_size = num_classes or 309  # 标签词汇表大小

        # 统一词汇表设计：
        # - 字符: 0 ~ (input_vocab_size - 1)
        # - 标签: input_vocab_size ~ (input_vocab_size + label_vocab_size - 1)
        # - MASK: input_vocab_size + label_vocab_size
        self.char_offset = 0
        self.label_offset = self.input_vocab_size  # 标签从40开始
        self.MASK_TOKEN_ID = self.input_vocab_size + self.label_vocab_size  # MASK = 349

        # 统一词汇表大小
        self.unified_vocab_size = self.input_vocab_size + self.label_vocab_size + 1  # +1 for MASK

        # 传递给BaseSequenceModel
        super().__init__(vocab_size, self.label_vocab_size, config)

    def build_model(self):
        """构建插值Diffusion模型"""

        # 默认配置
        default_config = {
            'd_model': 384,
            'num_layers': 8,
            'num_heads': 8,
            'dropout': 0.1,
            'max_length': 256,  # 原始序列最大长度
            'diffusion_steps': 10,  # 扩散步数
        }

        for k, v in default_config.items():
            if k not in self.config:
                self.config[k] = v

        # 计算插值后的最大序列长度（翻倍）
        self.interleaved_max_length = self.config['max_length'] * 2

        # === 统一Embedding层 ===
        # 包含：字符(0-39) + 标签(40-348) + MASK(349)
        self.unified_embedding = nn.Embedding(self.unified_vocab_size, self.config['d_model'])
        nn.init.normal_(self.unified_embedding.weight, mean=0.0, std=0.02)

        # 位置编码（针对插值后的长度）
        self.pos_embeddings = nn.Embedding(self.interleaved_max_length, self.config['d_model'])
        nn.init.normal_(self.pos_embeddings.weight, mean=0.0, std=0.02)

        # 时间步嵌入（扩散过程）
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

        # 输出投影：预测统一词汇表中的token
        self.output_projection = nn.Linear(self.config['d_model'], self.unified_vocab_size)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

        # 层归一化
        self.layer_norm = nn.LayerNorm(self.config['d_model'])
        nn.init.ones_(self.layer_norm.weight)
        nn.init.zeros_(self.layer_norm.bias)

        # 初始化transformer权重
        self._init_transformer_weights()

        log_info = print
        log_info(f"MDLM插值Diffusion模型构建完成:")
        log_info(f"  字符vocab: 0 ~ {self.input_vocab_size - 1}")
        log_info(f"  标签vocab: {self.label_offset} ~ {self.label_offset + self.label_vocab_size - 1}")
        log_info(f"  MASK token: {self.MASK_TOKEN_ID}")
        log_info(f"  统一vocab大小: {self.unified_vocab_size}")
        log_info(f"  原始max_length: {self.config['max_length']}")
        log_info(f"  插值max_length: {self.interleaved_max_length}")
        log_info(f"  模型维度: {self.config['d_model']}")
        log_info(f"  层数: {self.config['num_layers']}")
        log_info(f"  扩散步数: {self.config['diffusion_steps']}")

    def _init_transformer_weights(self):
        """初始化transformer权重"""
        def init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                if hasattr(module, 'in_proj_weight') and module.in_proj_weight is not None:
                    nn.init.xavier_uniform_(module.in_proj_weight)
                if hasattr(module, 'in_proj_bias') and module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
                if hasattr(module, 'out_proj') and module.out_proj.weight is not None:
                    nn.init.xavier_uniform_(module.out_proj.weight)
                    if module.out_proj.bias is not None:
                        nn.init.zeros_(module.out_proj.bias)

        self.transformer.apply(init_weights)

    def _create_interleaved_sequence(self, char_ids: torch.Tensor,
                                      label_ids: Optional[torch.Tensor] = None,
                                      mask_ratio: float = 1.0) -> torch.Tensor:
        """
        创建插值序列：[c₁, l₁, c₂, l₂, ...]

        Args:
            char_ids: [B, L] 字符ID（0-39范围）
            label_ids: [B, L] 标签ID（0-308范围），如果为None则使用MASK
            mask_ratio: 标签位置的mask比例（0.0=不mask，1.0=全mask）

        Returns:
            interleaved: [B, 2*L] 插值序列，使用统一词汇表ID
        """
        batch_size, seq_len = char_ids.shape
        device = char_ids.device

        # 创建插值序列
        interleaved = torch.zeros(batch_size, seq_len * 2, dtype=torch.long, device=device)

        # 偶数位置放字符（字符ID保持不变，因为offset=0）
        interleaved[:, 0::2] = char_ids  # 字符ID已经在0-39范围

        # 奇数位置放标签（需要加上label_offset）
        if label_ids is not None:
            # 处理padding值（-100或其他负值）
            # 将padding位置先替换成0，然后在后面的逻辑中会被MASK覆盖
            valid_labels = torch.clamp(label_ids, min=0)  # 将负值变成0
            padding_mask = (label_ids < 0)  # 记录padding位置

            # 将标签ID转换为统一词汇表ID
            unified_label_ids = valid_labels + self.label_offset  # 变成40-368范围

            if mask_ratio > 0 or padding_mask.any():
                # 创建mask：随机选择mask_ratio比例的位置 + padding位置
                random_mask = torch.rand(batch_size, seq_len, device=device) < mask_ratio
                combined_mask = random_mask | padding_mask  # 合并随机mask和padding mask
                masked_labels = torch.where(combined_mask,
                                           torch.full_like(unified_label_ids, self.MASK_TOKEN_ID),
                                           unified_label_ids)
                interleaved[:, 1::2] = masked_labels
            else:
                interleaved[:, 1::2] = unified_label_ids
        else:
            # 全部使用MASK
            interleaved[:, 1::2] = self.MASK_TOKEN_ID

        return interleaved

    def _extract_label_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        从输出logits中提取标签位置的预测

        Args:
            logits: [B, 2*L, unified_vocab_size] 完整输出

        Returns:
            label_logits: [B, L, label_vocab_size] 标签预测logits
        """
        # 提取奇数位置（标签位置）
        label_position_logits = logits[:, 1::2, :]  # [B, L, unified_vocab_size]

        # 只保留标签部分的logits（从label_offset到label_offset+label_vocab_size）
        label_logits = label_position_logits[:, :, self.label_offset:self.label_offset + self.label_vocab_size]

        return label_logits

    def forward(self, input_ids: torch.Tensor, prev_labels: Optional[torch.Tensor] = None,
                timestep: Optional[int] = None, return_full_logits: bool = False) -> torch.Tensor:
        """
        插值Diffusion前向传播

        Args:
            input_ids: [B, L] 输入字符序列（0-39范围）
            prev_labels: [B, L] 上一步预测的标签（0-308范围），如果为None则全部mask
            timestep: 扩散时间步 (0 到 diffusion_steps-1)
            return_full_logits: 是否返回完整logits（包含字符位置）

        Returns:
            如果return_full_logits=False:
                label_logits: [B, L, label_vocab_size] 标签预测logits
            如果return_full_logits=True:
                full_logits: [B, 2*L, unified_vocab_size] 完整logits
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # 确保序列长度不超过最大长度
        if seq_len > self.config['max_length']:
            input_ids = input_ids[:, :self.config['max_length']]
            if prev_labels is not None:
                prev_labels = prev_labels[:, :self.config['max_length']]
            seq_len = self.config['max_length']

        # 创建插值序列
        # prev_labels如果为None，mask_ratio=1.0（全mask）
        # prev_labels如果提供，mask_ratio=0.0（使用提供的标签）
        if prev_labels is None:
            interleaved = self._create_interleaved_sequence(input_ids, None, mask_ratio=1.0)
        else:
            # 检查prev_labels中的MASK位置（用self.label_vocab_size表示）
            # 需要将这些位置在插值序列中也设为MASK
            interleaved = self._create_interleaved_sequence(input_ids, prev_labels, mask_ratio=0.0)
            # 处理prev_labels中已经是MASK的位置
            mask_positions = (prev_labels >= self.label_vocab_size)
            if mask_positions.any():
                # 将这些位置设为MASK_TOKEN_ID
                label_positions = torch.arange(1, seq_len * 2, 2, device=device)
                for b in range(batch_size):
                    masked_indices = mask_positions[b].nonzero(as_tuple=True)[0]
                    if len(masked_indices) > 0:
                        interleaved[b, label_positions[masked_indices]] = self.MASK_TOKEN_ID

        # Embedding
        x = self.unified_embedding(interleaved)  # [B, 2*L, d_model]

        # 位置编码
        interleaved_len = seq_len * 2
        pos_ids = torch.arange(interleaved_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embeddings(pos_ids)
        x = x + pos_emb

        # 时间步编码
        if timestep is None:
            timestep = 0
        timestep = max(0, min(timestep, self.config['diffusion_steps'] - 1))
        time_emb = self.time_embeddings(torch.tensor(timestep, device=device, dtype=torch.long))
        x = x + time_emb.unsqueeze(0).unsqueeze(0).expand(batch_size, interleaved_len, -1)

        # Transformer处理
        x = self.layer_norm(x)
        x = self.transformer(x)

        # 输出投影
        logits = self.output_projection(x)  # [B, 2*L, unified_vocab_size]

        if return_full_logits:
            return logits
        else:
            # 提取标签位置的logits
            return self._extract_label_logits(logits)

    def encode(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """编码序列为特征表示"""
        with torch.no_grad():
            batch_size, seq_len = input_ids.shape

            # 创建全mask的插值序列
            interleaved = self._create_interleaved_sequence(input_ids, None, mask_ratio=1.0)

            # Embedding
            x = self.unified_embedding(interleaved)

            # 位置编码
            interleaved_len = seq_len * 2
            pos_ids = torch.arange(interleaved_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            pos_emb = self.pos_embeddings(pos_ids)
            x = x + pos_emb

            # Transformer编码
            x = self.layer_norm(x)
            features = self.transformer(x)

            return features

    def generate(self, input_ids: torch.Tensor, num_iterations: int = None,
                 temperature: float = 1.0) -> torch.Tensor:
        """
        迭代生成标签（真正的diffusion推理）

        Args:
            input_ids: [B, L] 输入字符序列
            num_iterations: 迭代次数（默认使用diffusion_steps）
            temperature: 采样温度

        Returns:
            predicted_labels: [B, L] 预测的标签（0-308范围）
        """
        self.eval()
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if num_iterations is None:
            num_iterations = self.config['diffusion_steps']

        # 确保序列长度不超过最大长度
        if seq_len > self.config['max_length']:
            input_ids = input_ids[:, :self.config['max_length']]
            seq_len = self.config['max_length']

        # 初始化：所有标签位置都是MASK
        # 使用一个特殊值表示MASK状态（在标签空间中用label_vocab_size表示）
        prev_labels = torch.full((batch_size, seq_len), self.label_vocab_size,
                                dtype=torch.long, device=device)

        with torch.no_grad():
            # 从高噪声到低噪声迭代
            for t in range(num_iterations - 1, -1, -1):
                # 前向传播
                label_logits = self.forward(input_ids, prev_labels, timestep=t)  # [B, L, label_vocab_size]

                if temperature > 0:
                    probs = F.softmax(label_logits / temperature, dim=-1)
                else:
                    probs = F.softmax(label_logits, dim=-1)

                # 获取预测和置信度
                confidence, predictions = probs.max(dim=-1)  # [B, L]

                # 计算本轮应该unmask的比例
                if num_iterations > 1:
                    unmask_ratio = 1.0 / num_iterations
                else:
                    unmask_ratio = 1.0

                # 只更新还是MASK的位置中最confident的那些
                still_masked = (prev_labels == self.label_vocab_size)

                if still_masked.any():
                    for b in range(batch_size):
                        mask_indices = still_masked[b].nonzero(as_tuple=True)[0]
                        if len(mask_indices) > 0:
                            num_to_unmask = max(1, int(len(mask_indices) * unmask_ratio))
                            _, top_indices = confidence[b, mask_indices].topk(
                                min(num_to_unmask, len(mask_indices)))
                            unmask_positions = mask_indices[top_indices]
                            prev_labels[b, unmask_positions] = predictions[b, unmask_positions]

        # 最后一步：确保所有位置都有预测
        still_masked = (prev_labels == self.label_vocab_size)
        if still_masked.any():
            # 对剩余的MASK位置做最终预测
            label_logits = self.forward(input_ids, prev_labels, timestep=0)
            _, final_predictions = label_logits.max(dim=-1)
            prev_labels = torch.where(still_masked, final_predictions, prev_labels)

        return prev_labels  # [B, L]，范围0-308

    def compute_loss(self, input_ids: torch.Tensor, target_ids: torch.Tensor,
                     class_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算插值Diffusion损失

        训练策略：
        1. 随机采样时间步t
        2. 根据t计算mask_ratio = (t + 1) / diffusion_steps
        3. 创建部分masked的插值序列
        4. 只对MASK位置计算损失

        Args:
            input_ids: [B, L] 输入字符序列
            target_ids: [B, L] 目标标签序列
            class_weights: 类别权重（用于处理不平衡数据）

        Returns:
            loss: 标量损失值
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # 确保序列长度不超过最大长度
        if seq_len > self.config['max_length']:
            input_ids = input_ids[:, :self.config['max_length']]
            target_ids = target_ids[:, :self.config['max_length']]
            seq_len = self.config['max_length']

        # 步骤1: 随机采样时间步
        if self.config['diffusion_steps'] > 1:
            timestep = torch.randint(0, self.config['diffusion_steps'], (1,)).item()
        else:
            timestep = 0

        # 步骤2: 计算mask_ratio
        # t=0时mask_ratio较小，t=T-1时mask_ratio接近1
        mask_ratio = (timestep + 1) / self.config['diffusion_steps']

        # 步骤3: 创建部分masked的插值序列
        interleaved = self._create_interleaved_sequence(input_ids, target_ids, mask_ratio=mask_ratio)

        # 记录哪些标签位置被mask了
        target_in_unified = target_ids + self.label_offset  # 转换为统一词汇表ID
        actual_labels = interleaved[:, 1::2]  # 插值序列中的标签位置
        masked_positions = (actual_labels == self.MASK_TOKEN_ID)  # [B, L]

        # 步骤4: 前向传播
        # Embedding
        x = self.unified_embedding(interleaved)  # [B, 2*L, d_model]

        # 位置编码
        interleaved_len = seq_len * 2
        pos_ids = torch.arange(interleaved_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embeddings(pos_ids)
        x = x + pos_emb

        # 时间步编码
        time_emb = self.time_embeddings(torch.tensor(timestep, device=device, dtype=torch.long))
        x = x + time_emb.unsqueeze(0).unsqueeze(0).expand(batch_size, interleaved_len, -1)

        # Transformer处理
        x = self.layer_norm(x)
        x = self.transformer(x)

        # 输出投影
        logits = self.output_projection(x)  # [B, 2*L, unified_vocab_size]

        # 步骤5: 只对MASK位置计算损失
        # 提取标签位置的logits
        label_position_logits = logits[:, 1::2, :]  # [B, L, unified_vocab_size]

        # 只保留标签部分的logits
        label_logits = label_position_logits[:, :, self.label_offset:self.label_offset + self.label_vocab_size]

        # 展平用于损失计算
        label_logits_flat = label_logits.reshape(-1, self.label_vocab_size)  # [B*L, label_vocab_size]
        target_flat = target_ids.reshape(-1)  # [B*L]
        masked_positions_flat = masked_positions.reshape(-1)  # [B*L]

        # 有效性检查：目标必须在有效范围内
        # 注意：虽然输入时部分标签被mask，但我们对所有位置计算loss
        # 这样训练目标与验证目标一致，最大化Levenshtein准确率
        valid_mask = (target_flat >= 0) & (target_flat < self.label_vocab_size)

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 提取有效位置
        valid_logits = label_logits_flat[valid_mask]
        valid_targets = target_flat[valid_mask]

        # 计算交叉熵损失
        if class_weights is not None and class_weights.size(0) == self.label_vocab_size:
            loss = F.cross_entropy(valid_logits, valid_targets, weight=class_weights, reduction='mean')
        else:
            loss = F.cross_entropy(valid_logits, valid_targets, reduction='mean')

        return loss

    def save_config(self) -> Dict[str, Any]:
        """保存模型配置"""
        return {
            'model_type': 'mdlm',
            'input_vocab_size': self.input_vocab_size,
            'label_vocab_size': self.label_vocab_size,
            'unified_vocab_size': self.unified_vocab_size,
            'MASK_TOKEN_ID': self.MASK_TOKEN_ID,
            'label_offset': self.label_offset,
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
