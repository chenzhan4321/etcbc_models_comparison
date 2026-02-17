"""
MDLM (Masked Discrete Language Model) - 完整扩散实现
基于离散扩散模型的序列标注，真正实现Output Dependency

关键特性：
1. 迭代去噪训练和生成
2. 字符-标签交替序列 [x1, y1, x2, y2, ...]
3. 时间步条件建模
4. Output-to-Output信息流
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import math
import random

from .base import BaseSequenceModel


class FullMDLMModel(BaseSequenceModel):
    """
    完整MDLM模型 - 真正的离散扩散实现

    核心机制：
    1. 训练时：从干净序列添加噪声，学习去噪
    2. 推理时：从全MASK序列迭代去噪到干净序列
    3. Output Dependency：每步去噪时能看到其他位置的当前状态
    """

    def __init__(self, vocab_size: int, num_classes: int = None, config: Optional[Dict[str, Any]] = None):
        # 词汇表设计
        self.input_vocab_size = vocab_size  # 输入字符词汇表 (e.g., 40)
        self.label_vocab_size = num_classes or 309  # 标签词汇表 (e.g., 309)

        # 统一词汇表：字符 + 标签 + 特殊标记
        # 结构: [char_0...char_39, label_0...label_308, MASK, PAD, START, END]
        unified_vocab_size = vocab_size + self.label_vocab_size + 4

        super().__init__(unified_vocab_size, self.label_vocab_size, config)

        # 特殊token索引
        self.mask_token_id = vocab_size + self.label_vocab_size  # MASK
        self.pad_token_id = vocab_size + self.label_vocab_size + 1  # PAD
        self.start_token_id = vocab_size + self.label_vocab_size + 2  # START
        self.end_token_id = vocab_size + self.label_vocab_size + 3  # END

        # 标签在统一词汇表中的偏移
        self.label_offset = vocab_size

    def build_model(self):
        """构建完整MDLM架构"""

        # 默认配置
        default_config = {
            'd_model': 384,
            'num_layers': 8,
            'num_heads': 8,
            'dropout': 0.1,
            'max_length': 512,  # 支持更长序列（因为交替后长度翻倍）
            'diffusion_steps': 10,  # T步扩散
            'noise_schedule': 'linear',  # 线性噪声调度
        }

        for k, v in default_config.items():
            if k not in self.config:
                self.config[k] = v

        # 词嵌入（统一字符和标签）
        self.embeddings = nn.Embedding(self.vocab_size, self.config['d_model'])
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.02)

        # 位置编码（需要支持2倍长度的交替序列）
        # 交替序列长度 = 原始序列长度 * 2，所以位置编码需要 max_length * 2
        self.pos_embeddings = nn.Embedding(self.config['max_length'] * 2, self.config['d_model'])
        nn.init.normal_(self.pos_embeddings.weight, mean=0.0, std=0.02)

        # 时间步嵌入（扩散步数）
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

        # 输出投影：预测整个统一词汇表
        self.output_projection = nn.Linear(self.config['d_model'], self.vocab_size)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

        # 层归一化
        self.layer_norm = nn.LayerNorm(self.config['d_model'])
        nn.init.ones_(self.layer_norm.weight)
        nn.init.zeros_(self.layer_norm.bias)

        # 初始化transformer权重
        self._init_transformer_weights()

        # 预计算噪声调度
        self._prepare_noise_schedule()

        log_info = print
        log_info(f"完整MDLM模型构建完成:")
        log_info(f"  统一词汇表: {self.vocab_size} (字符:{self.input_vocab_size}, 标签:{self.label_vocab_size})")
        log_info(f"  扩散步数: {self.config['diffusion_steps']}")
        log_info(f"  噪声调度: {self.config['noise_schedule']}")
        log_info(f"  序列策略: 完整交替 [x1,y1,x2,y2,...]")

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

        self.transformer.apply(init_weights)

    def _prepare_noise_schedule(self):
        """
        准备噪声调度表
        Linear schedule: α_t = t / T
        """
        T = self.config['diffusion_steps']

        if self.config['noise_schedule'] == 'linear':
            # 线性调度：t=1时mask 10%，t=T时mask 100%
            self.noise_schedule = torch.linspace(0.1, 1.0, T)
        elif self.config['noise_schedule'] == 'cosine':
            # 余弦调度（备用）
            steps = torch.arange(T, dtype=torch.float32)
            self.noise_schedule = torch.cos((steps / T + 0.008) / 1.008 * math.pi / 2) ** 2
        else:
            raise ValueError(f"Unknown noise schedule: {self.config['noise_schedule']}")

        # 转为可学习参数（但不更新梯度）
        self.register_buffer('alpha_schedule', self.noise_schedule)

    def create_interleaved_sequence(self, input_ids: torch.Tensor, label_ids: torch.Tensor) -> torch.Tensor:
        """
        创建字符-标签交替序列

        输入:
            input_ids: [batch, seq_len] 字符序列
            label_ids: [batch, seq_len] 标签序列
        输出:
            interleaved: [batch, seq_len*2] 交替序列 [x1, y1, x2, y2, ...]
        """
        batch_size, seq_len = input_ids.shape

        # 创建2倍长度的序列
        interleaved = torch.zeros(batch_size, seq_len * 2, dtype=torch.long, device=input_ids.device)

        # 偶数位置：字符 (索引 0, 2, 4, ...)
        interleaved[:, 0::2] = input_ids

        # 奇数位置：标签+偏移 (索引 1, 3, 5, ...)
        interleaved[:, 1::2] = label_ids + self.label_offset

        return interleaved

    def extract_labels_from_interleaved(self, interleaved: torch.Tensor) -> torch.Tensor:
        """
        从交替序列中提取标签

        输入:
            interleaved: [batch, seq_len*2] 交替序列
        输出:
            labels: [batch, seq_len] 标签序列
        """
        # 提取奇数位置（标签位置）
        label_ids_with_offset = interleaved[:, 1::2]

        # 减去偏移，恢复原始标签ID
        labels = label_ids_with_offset - self.label_offset

        return labels

    def add_noise(self, sequence: torch.Tensor, timestep: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        添加扩散噪声（掩码）

        关键：只对标签位置添加噪声，字符位置保持不变

        输入:
            sequence: [batch, seq_len*2] 交替序列
            timestep: 时间步 t ∈ [0, T-1]
        输出:
            noisy_seq: [batch, seq_len*2] 加噪后的序列
            mask: [batch, seq_len*2] 被mask的位置（用于计算损失）
        """
        batch_size, total_len = sequence.shape
        device = sequence.device

        # 获取当前时间步的噪声比例
        alpha_t = self.alpha_schedule[timestep].item()

        # 创建mask：只对标签位置（奇数索引）进行mask
        mask = torch.zeros(batch_size, total_len, dtype=torch.bool, device=device)

        # 标签位置的索引
        label_positions = torch.arange(1, total_len, 2, device=device)

        # 对每个样本独立采样要mask的标签位置
        for b in range(batch_size):
            num_labels = len(label_positions)
            num_to_mask = int(num_labels * alpha_t)

            # 随机选择要mask的标签位置
            if num_to_mask > 0:
                mask_indices = torch.randperm(num_labels, device=device)[:num_to_mask]
                positions_to_mask = label_positions[mask_indices]
                mask[b, positions_to_mask] = True

        # 应用mask
        noisy_seq = sequence.clone()
        noisy_seq[mask] = self.mask_token_id

        return noisy_seq, mask

    def forward(self, sequence: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                timestep: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播 - 条件于时间步

        输入:
            sequence: [batch, seq_len] 输入序列（可能是交替序列）
            attention_mask: [batch, seq_len] 注意力掩码（兼容训练器，暂不使用）
            timestep: [batch] 或 标量，时间步索引
        输出:
            logits: [batch, seq_len, vocab_size] 对统一词汇表的预测
        """
        batch_size, seq_len = sequence.shape
        device = sequence.device

        # Token嵌入
        x = self.embeddings(sequence)  # [batch, seq_len, d_model]

        # 位置编码
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embeddings(pos_ids)
        x = x + pos_emb

        # 时间步编码（如果提供且是有效的时间步）
        if timestep is not None and timestep.dim() <= 1:
            # 处理timestep的形状
            if timestep.dim() == 0:  # 标量
                timestep = timestep.unsqueeze(0).expand(batch_size)
            elif timestep.dim() == 1 and timestep.size(0) == 1:  # [1]
                timestep = timestep.expand(batch_size)

            # 确保在有效范围内
            timestep = torch.clamp(timestep, 0, self.config['diffusion_steps'] - 1)

            time_emb = self.time_embeddings(timestep)  # [batch, d_model]
            # 广播到所有位置
            x = x + time_emb.unsqueeze(1)  # [batch, seq_len, d_model]

        # Transformer编码
        x = self.layer_norm(x)
        x = self.transformer(x)

        # 输出投影
        logits = self.output_projection(x)  # [batch, seq_len, vocab_size]

        return logits

    def compute_loss(self, input_ids: torch.Tensor, target_ids: torch.Tensor,
                     class_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算MDLM扩散损失

        训练流程：
        1. 创建交替序列 [x1, y1, x2, y2, ...]
        2. 随机采样时间步 t
        3. 添加噪声（mask部分标签）
        4. 模型预测原始序列
        5. 只对被mask的位置计算损失
        """
        batch_size = input_ids.size(0)
        device = input_ids.device

        # 重要：处理padding值-100，避免创建交替序列时出现负数索引
        # -100是PyTorch用于标记padding的特殊值，应该替换为有效的标签ID
        # 我们用0替换（任意有效标签都可以，因为这些位置不会参与损失计算）
        target_ids_clean = target_ids.clone()
        target_ids_clean[target_ids == -100] = 0

        # 1. 创建交替序列（使用清理后的标签）
        clean_seq = self.create_interleaved_sequence(input_ids, target_ids_clean)  # [batch, seq_len*2]

        # 记录原始的padding mask，用于后续排除这些位置的损失
        padding_mask = (target_ids == -100)  # [batch, seq_len]

        # 2. 随机采样时间步 t ∈ [0, T-1]
        t = torch.randint(0, self.config['diffusion_steps'], (batch_size,), device=device)

        # 3. 对每个样本添加对应时间步的噪声
        total_loss = 0.0
        num_masked_total = 0

        for b in range(batch_size):
            # 单个样本处理
            clean_single = clean_seq[b:b+1]  # [1, seq_len*2]
            t_single = t[b].item()

            # 添加噪声
            noisy_single, mask_single = self.add_noise(clean_single, t_single)

            # 创建padding mask的交替序列版本（只关注标签位置）
            # padding_mask[b]形状: [seq_len]，我们需要映射到交替序列的标签位置（奇数索引）
            seq_len = input_ids.size(1)
            padding_mask_interleaved = torch.zeros(1, seq_len * 2, dtype=torch.bool, device=device)
            padding_mask_interleaved[0, 1::2] = padding_mask[b]  # 奇数位置（标签位置）

            # 合并mask：只对被噪声mask且不是padding的位置计算损失
            valid_mask = mask_single & ~padding_mask_interleaved

            # 4. 前向传播预测
            t_tensor = torch.tensor([t_single], device=device, dtype=torch.long)
            logits = self.forward(noisy_single, timestep=t_tensor)  # [1, seq_len*2, vocab_size]

            # 5. 只对有效的被mask位置计算损失（排除padding）
            if valid_mask.sum() > 0:
                masked_logits = logits[valid_mask]  # [num_valid_masked, vocab_size]
                masked_targets = clean_single[valid_mask]  # [num_valid_masked]

                # 交叉熵损失
                if class_weights is not None:
                    # 需要调整class_weights到统一词汇表
                    # 创建完整权重向量
                    full_weights = torch.ones(self.vocab_size, device=device)
                    full_weights[self.label_offset:self.label_offset + self.label_vocab_size] = class_weights
                    loss_single = F.cross_entropy(masked_logits, masked_targets, weight=full_weights, reduction='sum')
                else:
                    loss_single = F.cross_entropy(masked_logits, masked_targets, reduction='sum')

                total_loss += loss_single
                num_masked_total += valid_mask.sum().item()

        # 平均损失
        if num_masked_total > 0:
            avg_loss = total_loss / num_masked_total
        else:
            avg_loss = torch.tensor(0.0, device=device, requires_grad=True)

        return avg_loss

    def encode(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        编码序列为特征表示

        输入:
            input_ids: [batch, seq_len] 输入序列
            attention_mask: 可选的注意力掩码
        输出:
            features: [batch, seq_len, d_model] 特征表示
        """
        with torch.no_grad():
            batch_size, seq_len = input_ids.shape
            device = input_ids.device

            # Token嵌入
            x = self.embeddings(input_ids)

            # 位置编码
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            pos_emb = self.pos_embeddings(pos_ids)
            x = x + pos_emb

            # Transformer编码
            x = self.layer_norm(x)
            features = self.transformer(x)

            return features

    def generate(self, input_ids: torch.Tensor, num_steps: Optional[int] = None,
                 temperature: float = 1.0, return_intermediate: bool = False) -> torch.Tensor:
        """
        迭代去噪生成 - 真正的MDLM推理

        流程：
        1. 初始化：创建全MASK的标签序列
        2. 迭代 t = T-1 → 0:
           - 创建交替序列 [x1, y_t, x2, y_t, ...]
           - 前向传播获取预测
           - 更新标签位置
        3. 返回最终去噪的标签

        关键：每步都能看到其他位置的当前标签状态 → Output Dependency

        参数：
            return_intermediate: 如果为True，返回 (final_labels, intermediate_steps)
                               intermediate_steps是列表，包含每个时间步的预测
        """
        self.eval()

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if num_steps is None:
            num_steps = self.config['diffusion_steps']

        # 用于保存中间步骤的预测
        intermediate_predictions = [] if return_intermediate else None

        with torch.no_grad():
            # 初始化：全MASK的标签序列
            y_t = torch.full((batch_size, seq_len), self.mask_token_id,
                           dtype=torch.long, device=device)

            # 迭代去噪：t = T-1 → 0
            for t in reversed(range(num_steps)):
                # 创建交替序列
                interleaved = self.create_interleaved_sequence(input_ids, y_t - self.label_offset)

                # 前向传播（条件于时间步t）
                t_tensor = torch.tensor([t], device=device, dtype=torch.long)
                logits = self.forward(interleaved, timestep=t_tensor)  # [batch, seq_len*2, vocab_size]

                # 提取标签位置的logits（奇数索引）
                label_logits = logits[:, 1::2, :]  # [batch, seq_len, vocab_size]

                # 只关注标签部分的预测
                label_logits_subset = label_logits[:, :, self.label_offset:self.label_offset + self.label_vocab_size]

                # 采样或贪婪解码
                if temperature > 0:
                    probs = F.softmax(label_logits_subset / temperature, dim=-1)
                    # 贪婪解码（也可以用采样）
                    y_t_new = torch.argmax(probs, dim=-1)  # [batch, seq_len]
                else:
                    y_t_new = torch.argmax(label_logits_subset, dim=-1)

                # 更新标签（加上偏移以保持在统一词汇表空间）
                y_t = y_t_new + self.label_offset

                # 保存中间步骤（如果需要）
                if return_intermediate:
                    intermediate_predictions.append((y_t - self.label_offset).clone())

            # 最终标签（去掉偏移）
            final_labels = y_t - self.label_offset

        if return_intermediate:
            return final_labels, intermediate_predictions
        else:
            return final_labels

    def save_config(self) -> Dict[str, Any]:
        """保存模型配置"""
        return {
            'model_type': 'mdlm_full',
            'input_vocab_size': self.input_vocab_size,
            'label_vocab_size': self.label_vocab_size,
            'vocab_size': self.vocab_size,
            'num_classes': self.num_classes,
            **self.config
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'FullMDLMModel':
        """从配置创建模型"""
        return cls(
            vocab_size=config['input_vocab_size'],
            num_classes=config['label_vocab_size'],
            config=config
        )
