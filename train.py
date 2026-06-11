import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import seaborn as sns
from datetime import datetime
import time
import sys
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import subprocess
from typing import Dict, Any, List

# Import unified modules - 重构后的导入
from models.model_factory import create_model, get_supported_models
from models.config_manager import load_config, Config
from models.data_utils import create_data_loaders, get_num_classes, get_class_weights, DataAugmentation, compute_enhanced_class_weights, get_vocab_for_model
from models.core import get_device_manager, create_optimized_config, log_info, log_warning, get_vocab_size

# Device configuration availability
DEVICE_CONFIG_AVAILABLE = False

class FocalLoss(nn.Module):
    """
    改进的焦点损失，用于处理类别不平衡问题
    特别适用于极度不平衡的多分类任务
    添加了数值稳定性改进
    """
    def __init__(self, alpha=1.0, gamma=2.0, ignore_index=-100, weight=None, label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.weight = weight
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs, targets):
        # 添加数值稳定性检查
        if torch.isnan(inputs).any() or torch.isinf(inputs).any():
            print(f"⚠️ FocalLoss输入包含nan或inf值")
            print(f"   输入形状: {inputs.shape}")
            print(f"   输入范围: [{inputs.min().item():.4f}, {inputs.max().item():.4f}]")
            print(f"   nan数量: {torch.isnan(inputs).sum().item()}")
            print(f"   inf数量: {torch.isinf(inputs).sum().item()}")
            # 使用更安全的数值替换策略
            inputs = torch.nan_to_num(inputs, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # 计算标准交叉熵损失（带标签平滑）
        ce_loss = F.cross_entropy(inputs, targets, 
                                 ignore_index=self.ignore_index, 
                                 weight=self.weight, 
                                 reduction='none',
                                 label_smoothing=self.label_smoothing)
        
        # 数值稳定的概率计算
        # 避免使用torch.exp(-ce_loss)，而是直接从softmax概率计算
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = F.softmax(inputs, dim=-1)
        
        # 获取目标类别的概率
        valid_mask = targets != self.ignore_index
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
            
        valid_targets = targets[valid_mask]
        valid_probs = probs[valid_mask]
        valid_ce_loss = ce_loss[valid_mask]
        
        # 获取目标类别的概率
        pt = valid_probs.gather(1, valid_targets.unsqueeze(1)).squeeze(1)
        
        # 限制pt的范围，避免数值问题
        pt = torch.clamp(pt, min=1e-8, max=1.0-1e-8)
        
        # 应用焦点损失公式：FL = -α(1-pt)^γ * log(pt)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        focal_loss = focal_weight * valid_ce_loss
        
        # 添加数值稳定性检查
        if torch.isnan(focal_loss).any() or torch.isinf(focal_loss).any():
            print(f"⚠️ FocalLoss输出包含nan或inf，使用备用损失")
            return valid_ce_loss.mean()
        
        return focal_loss.mean()

def calculate_class_weights(train_loader, num_classes, device):
    """
    计算平衡的类别权重来处理不平衡问题
    使用更稳定的权重计算方案，避免极端权重值
    
    参数：
        train_loader: 训练数据加载器
        num_classes: 类别数量
        device: 计算设备
        
    返回：
        torch.Tensor: 类别权重张量
    """
    print("🔍 计算平衡类别权重...")
    
    # 统计每个类别的频率
    class_counts = torch.zeros(num_classes, dtype=torch.long)
    
    for _, (_, labels, _) in enumerate(train_loader):
        # 展平标签并过滤填充位置
        labels_flat = labels.view(-1)
        mask = labels_flat != -100
        valid_labels = labels_flat[mask]
        
        # 统计类别频率
        for label in valid_labels:
            if 0 <= label < num_classes:
                class_counts[label] += 1
    
    # 使用更稳定的权重计算方案
    # 1. 为频率添加平滑项，避免除以零和极端值
    smoothed_counts = class_counts.float() + 1.0
    
    # 2. 计算权重与频率成反比
    weights = 1.0 / smoothed_counts
    
    # 3. 将权重归一化，使最小的权重为1
    weights = weights / weights.min()
    
    # 4. 限制权重的最大值，避免过度倾斜（进一步降低上限）
    max_weight = 2.0  # 进一步降低权重上限，避免梯度爆炸
    weights = torch.clamp(weights, min=1.0, max=max_weight)
    
    print(f"   类别计数: {class_counts[:10].tolist()}...")
    print(f"   平衡权重: {weights[:10].tolist()}...")
    print(f"   权重范围: {weights.min():.3f} - {weights.max():.3f}")
    
    return weights.to(device)

class ModelTrainer:
    def __init__(self, model, device, num_classes, learning_rate=0.001, weight_decay=0.01, log_dir=None, class_weights=None, 
                 use_warmup=False, warmup_steps=500, use_scheduler=True, total_steps=None):
        self.log_dir = log_dir
        self.model = model.to(device)
        self.device = device
        import os as _os
        self.ddp_world = int(_os.environ.get("WORLD_SIZE", 1))
        self.ddp_rank = int(_os.environ.get("RANK", 0))
        self.num_classes = num_classes
        self.class_weights = class_weights  # 添加这一行
        
        # Loss function: 使用FocalLoss来更好地处理极度不平衡问题
        # 临时方案：对于初始训练，可以先使用标准CrossEntropyLoss
        use_focal_loss = True  # 可以设置为False来使用标准CE
        
        if use_focal_loss:
            if class_weights is not None:
                print(f"✅ 使用FocalLoss + 类别权重处理数据不平衡问题")
                print(f"   权重张量形状: {class_weights.shape}")
                print(f"   权重范围: {class_weights.min():.3f} - {class_weights.max():.3f}")
                # 使用更小的gamma值和标签平滑来提高稳定性
                self.criterion = FocalLoss(alpha=1.0, gamma=1.0, ignore_index=-100, weight=class_weights, label_smoothing=0.1)
            else:
                print(f"⚠️  使用FocalLoss但未使用类别权重")
                self.criterion = FocalLoss(alpha=1.0, gamma=1.0, ignore_index=-100, label_smoothing=0.1)
        else:
            # 使用标准的CrossEntropyLoss作为基准
            print(f"📊 使用标准CrossEntropyLoss（更稳定的基准）")
            if class_weights is not None:
                # 对权重进行进一步缩放，避免过大
                scaled_weights = torch.sqrt(class_weights)  # 取平方根缓和权重
                self.criterion = nn.CrossEntropyLoss(weight=scaled_weights, ignore_index=-100, label_smoothing=0.1)
            else:
                self.criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
        
        # Optimizer - 使用更保守的设置
        # 对于warmup，初始学习率设置为很小的值
        initial_lr = learning_rate * 0.05 if use_warmup else learning_rate * 0.5  # 进一步降低学习率
        self.optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay, 
                                     betas=(0.9, 0.95), eps=1e-8)  # 更保守的beta2和更大的eps
        
        # 学习率调度设置
        self.use_warmup = use_warmup
        self.warmup_steps = warmup_steps
        self.use_scheduler = use_scheduler
        self.total_steps = total_steps or 10000
        self.current_step = 0
        self.base_lr = learning_rate
        
        # Learning rate scheduler: 根据模型类型选择合适的调度器
        if use_scheduler:
            if use_warmup and total_steps:
                # 使用warmup + cosine衰减调度器
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=total_steps, eta_min=learning_rate * 0.01
                )
            else:
                # 使用ReduceLROnPlateau调度器
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='max', factor=0.5, patience=8, min_lr=learning_rate * 0.01
                )
        else:
            self.scheduler = None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
        # 检查模型参数的数值稳定性
        self._check_model_stability()
    
    def _check_model_stability(self):
        """检查模型参数的数值稳定性"""
        print("🔍 检查模型参数稳定性...")
        total_params = 0
        unstable_params = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                total_params += 1
                param_norm = param.data.norm().item()
                
                # 检查参数范数是否过大
                if param_norm > 10.0:
                    print(f"   ⚠️ {name}: {param_norm:.4f}")
                    unstable_params += 1
                
                # 检查是否包含nan或inf
                if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                    print(f"   ❌ {name}: 包含nan或inf")
                    unstable_params += 1
        
        if unstable_params > 0:
            print(f"⚠️ 发现 {unstable_params}/{total_params} 个不稳定参数")
        else:
            print("✅ 模型参数稳定性检查通过")
        
        # TensorBoard
        if self.log_dir is not None:
            self.writer = SummaryWriter(log_dir=self.log_dir)
        else:
            self.writer = None
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        num_batches = len(dataloader)
        is_tty = sys.stdout.isatty()
        N = 500
        if is_tty:
            progress_bar = tqdm(dataloader, desc="Training")
        else:
            progress_bar = dataloader
        for batch_idx, (input_ids, labels, attention_mask) in enumerate(progress_bar):
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            # 检查输入数据的有效性
            if torch.isnan(input_ids).any() or torch.isinf(input_ids).any():
                print(f"⚠️ 输入数据包含nan或inf，跳过该批次")
                continue
            
            # 检查索引范围，防止CUDA设备断言错误
            vocab_size = self.model.vocab_size
            if input_ids.max() >= vocab_size or input_ids.min() < 0:
                print(f"⚠️ 输入数据索引超出范围 [{input_ids.min()}, {input_ids.max()}]，词汇表大小 {vocab_size}")
                print(f"   将索引裁剪到有效范围 [0, {vocab_size-1}]")
                input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
                
            self.optimizer.zero_grad()
            
            # 使用混合精度训练提高数值稳定性（如果设备支持）
            try:
                outputs = self.model(input_ids, attention_mask)
                
                # 检查模型输出
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(f"⚠️ 模型输出包含nan或inf")
                    print(f"   输出范围: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
                    # 打印更多调试信息
                    print(f"   输出中nan的数量: {torch.isnan(outputs).sum().item()}")
                    print(f"   输出中inf的数量: {torch.isinf(outputs).sum().item()}")
                    
                    # 尝试修复输出
                    outputs = torch.nan_to_num(outputs, nan=0.0, posinf=10.0, neginf=-10.0)
                    print(f"   已修复输出中的nan/inf值")
                    
            except RuntimeError as e:
                if "nan" in str(e).lower() or "inf" in str(e).lower():
                    print(f"⚠️ 模型前向传播出现数值问题: {e}")
                    continue
                else:
                    raise e
            
            # MDLM模型特殊处理：使用自己的compute_loss方法
            if hasattr(self.model, 'compute_loss') and hasattr(self.model, 'input_vocab_size'):
                # MDLM模型使用自己的损失计算，传递类别权重
                if self.class_weights is not None:
                    loss = self.model.compute_loss(input_ids, labels, self.class_weights)
                else:
                    loss = self.model.compute_loss(input_ids, labels)
            elif getattr(self.model, 'crf', None) is not None:
                # CRF 模型：用 CRF 负对数似然作为损失（outputs 为 3D logits [B,T,C]）
                safe_labels, crf_mask = _crf_prepare(labels, attention_mask)
                loss = -self.model.crf(outputs, safe_labels, mask=crf_mask, reduction='mean')
            else:
                # 标准模型的损失计算
                outputs = outputs.view(-1, self.num_classes)
                labels = labels.view(-1)
                
                # 添加数值稳定性检查
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    outputs = torch.nan_to_num(outputs, nan=0.0, posinf=10.0, neginf=-10.0)
                
                loss = self.criterion(outputs, labels)
            
            # 检查损失和梯度的数值稳定性
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️ 检测到不正常损失: {loss.item()}, 跳过该批次")
                continue
                
            loss.backward()
            # === DDP: 跨卡同步(求和后平均)梯度 ===
            if getattr(self, 'ddp_world', 1) > 1:
                import torch.distributed as _dist
                for _p in self.model.parameters():
                    if _p.grad is not None:
                        _dist.all_reduce(_p.grad, op=_dist.ReduceOp.SUM)
                        _p.grad.div_(self.ddp_world)

            # 检查梯度状态
            total_norm = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        print(f"⚠️ 检测到不正常梯度")
                        p.grad.data.zero_()  # 清零不正常梯度
            total_norm = total_norm ** (1. / 2)
            
            # 梯度裁剪（使用更保守的阈值）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            
            # 应用warmup和学习率调度
            if self.use_warmup and self.current_step < self.warmup_steps:
                # Warmup阶段：线性增加学习率
                warmup_lr = self.base_lr * (self.current_step + 1) / self.warmup_steps
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            
            self.optimizer.step()
            
            # 更新学习率调度器（仅在warmup后或不使用warmup时）
            if self.scheduler and (not self.use_warmup or self.current_step >= self.warmup_steps):
                if isinstance(self.scheduler, optim.lr_scheduler.CosineAnnealingLR):
                    self.scheduler.step()
            
            self.current_step += 1
            total_loss += loss.item()
            if is_tty:
                progress_bar.set_postfix({'损失': f'{loss.item():.4f}'})
            else:
                if (batch_idx+1) % N == 0 or (batch_idx+1) == num_batches:
                    print(f"训练进度: {batch_idx+1}/{num_batches} 损失={loss.item():.4f}")
        return total_loss / num_batches
    
    def validate_epoch(self, dataloader):
        """
        验证一个epoch
        
        参数：
            dataloader: 验证数据加载器
            
        返回：
            Tuple[float, float]: (平均损失, 准确率)
        """
        self.model.eval()  # 设置模型为评估模式
        total_loss = 0     # 累计损失
        all_predictions = []  # 所有预测结果
        all_labels = []       # 所有真实标签
        is_tty = sys.stdout.isatty()
        N = 500
        # 验证阶段不需要计算梯度，节省内存和计算资源
        with torch.no_grad():
            if is_tty:
                val_iter = tqdm(dataloader, desc="验证中")
            else:
                val_iter = dataloader
            for batch_idx, (input_ids, labels, attention_mask) in enumerate(val_iter):
                # 将数据移动到指定设备
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                attention_mask = attention_mask.to(self.device)
                
                # 前向传播
                outputs = self.model(input_ids, attention_mask)
                
                # MDLM模型特殊处理：使用自己的compute_loss方法
                if hasattr(self.model, 'compute_loss') and hasattr(self.model, 'input_vocab_size'):
                    # MDLM模型使用自己的损失计算，传递类别权重
                    if self.class_weights is not None:
                        loss = self.model.compute_loss(input_ids, labels, self.class_weights)
                    else:
                        loss = self.model.compute_loss(input_ids, labels)
                    
                    # 为了获取预测，我们需要生成序列
                    predictions = self.model.generate(input_ids)  # 现在直接返回预测标签 [batch, seq_len]
                    
                    # 截断到与labels相同的长度
                    min_len = min(predictions.size(1), labels.size(1))
                    predictions = predictions[:, :min_len]
                    labels_for_eval = labels[:, :min_len]
                    
                    # 展平用于评估
                    predictions_flat = predictions.view(-1)
                    labels_flat = labels_for_eval.view(-1)
                elif getattr(self.model, 'crf', None) is not None:
                    # CRF 模型：CRF NLL 作损失，Viterbi 解码作预测
                    safe_labels, crf_mask = _crf_prepare(labels, attention_mask)
                    loss = -self.model.crf(outputs, safe_labels, mask=crf_mask, reduction='mean')
                    pred_2d = _crf_decode_to_tensor(self.model, outputs, attention_mask, labels)
                    predictions_flat = pred_2d.view(-1)
                    labels_flat = labels.view(-1)
                else:
                    # 标准模型的处理
                    # 为损失计算重塑形状
                    outputs_flat = outputs.view(-1, self.num_classes)
                    labels_flat = labels.view(-1)

                    # 计算损失
                    loss = self.criterion(outputs_flat, labels_flat)

                    # 获取预测结果
                    predictions_flat = torch.argmax(outputs_flat, dim=1)
                
                # 检查验证损失的数值稳定性
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️ 验证损失不正常: {loss.item()}")
                    loss = torch.tensor(float('inf'), device=loss.device)
                total_loss += loss.item()
                
                # 过滤掉填充位置（标签为-100）
                # 只评估非填充位置的性能
                mask = labels_flat != -100
                predictions = predictions_flat[mask]
                labels_filtered = labels_flat[mask]
                
                # 收集所有预测和标签用于最终评估
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels_filtered.cpu().numpy())
                if not is_tty and ((batch_idx+1) % N == 0 or (batch_idx+1) == len(dataloader)):
                    print(f"验证进度: {batch_idx+1}/{len(dataloader)} 损失={loss.item():.4f}")
        
        # 计算平均损失和多个评估指标
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # 使用weighted平均作为主要指标（更适合不平衡数据）
        f1_weighted = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        precision_weighted = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall_weighted = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        
        # 同时计算macro平均作为参考
        f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
        precision_macro = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
        recall_macro = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
        
        # 分析预测分布
        from collections import Counter
        pred_dist = Counter(all_predictions)
        label_dist = Counter(all_labels)
        
        print(f"\n📊 预测分布分析:")
        print(f"  - 预测的前5个类别: {dict(pred_dist.most_common(5))}")
        print(f"  - 标签的前5个类别: {dict(label_dist.most_common(5))}")
        
        # 检查是否所有预测都是同一类别
        if len(pred_dist) == 1:
            print(f"  ⚠️  警告：所有预测都是类别 {list(pred_dist.keys())[0]}")
        
        # 返回weighted指标作为主要指标
        return avg_loss, accuracy, f1_weighted, f1_macro, precision_weighted, recall_weighted, all_predictions, all_labels
    
    def generate_sample_outputs(self, dataloader, num_samples=3, sample_type="训练"):
        """
        生成样本输出展示
        
        参数:
            dataloader: 数据加载器
            num_samples: 要显示的样本数量
            sample_type: 样本类型描述
        """
        # 导入解码函数
        try:
            from models.data_utils import idx_to_char
        except ImportError:
            # 如果无法导入，提供简单的解码函数
            def idx_to_char(idx):
                return str(idx)
        
        self.model.eval()
        samples_shown = 0
        
        print(f"\n生成的{sample_type}样本:")
        print("-" * 80)
        
        with torch.no_grad():
            for input_ids, labels, attention_mask in dataloader:
                if samples_shown >= num_samples:
                    break
                    
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                attention_mask = attention_mask.to(self.device)
                
                # MDLM模型特殊处理
                if hasattr(self.model, 'compute_loss') and hasattr(self.model, 'input_vocab_size'):
                    # MDLM模型使用生成方式获取预测
                    predictions = self.model.generate(input_ids)  # 现在直接返回预测标签 [batch, seq_len]
                    
                    # 截断到与labels相同的长度
                    min_len = min(predictions.size(1), labels.size(1))
                    predictions = predictions[:, :min_len]
                else:
                    # 标准模型处理
                    outputs = self.model(input_ids, attention_mask)
                    predictions = torch.argmax(outputs, dim=-1)
                
                # 处理每个样本
                batch_size = min(num_samples - samples_shown, input_ids.size(0))
                for i in range(batch_size):
                    # 获取有效长度（非填充部分）
                    valid_mask = attention_mask[i] == 1
                    valid_length = valid_mask.sum().item()
                    
                    # 解码输入、目标和预测
                    input_seq = input_ids[i][:valid_length].cpu().numpy()
                    target_seq = labels[i][:valid_length].cpu().numpy()
                    pred_seq = predictions[i][:valid_length].cpu().numpy()
                    
                    # 过滤掉填充标记(-100)
                    target_filtered = target_seq[target_seq != -100]
                    pred_filtered = pred_seq[:len(target_filtered)]
                    input_filtered = input_seq[:len(target_filtered)]
                    
                    # 计算准确率
                    if len(target_filtered) > 0 and len(pred_filtered) > 0:
                        matches = (pred_filtered == target_filtered).sum()
                        total = len(target_filtered)
                        accuracy = matches / total
                    else:
                        accuracy = 0.0
                        matches = 0
                        total = 0
                    
                    # 解码为文本
                    try:
                        input_text = ''.join([idx_to_char(idx) for idx in input_filtered])
                        target_text = ' '.join([str(idx) for idx in target_filtered])
                        pred_text = ' '.join([str(idx) for idx in pred_filtered])
                    except:
                        input_text = str(input_filtered.tolist())
                        target_text = str(target_filtered.tolist())
                        pred_text = str(pred_filtered.tolist())
                    
                    print(f"输入:      {input_text}")
                    print(f"目标:     {target_text}")
                    print(f"预测:     {pred_text}")
                    print(f"准确率:    {accuracy:.4f} ({matches}/{total})")
                    print("-" * 40)
                    
                    samples_shown += 1
                    if samples_shown >= num_samples:
                        break
                
                if samples_shown >= num_samples:
                    break
    
    def train(self, train_loader, val_loader, num_epochs, save_path):
        """
        模型训练主函数
        
        参数：
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs (int): 训练轮数
            save_path (str): 模型保存路径
            
        返回：
            float: 最佳验证准确率
        """
        print(f"开始训练，共{num_epochs}个epoch...")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 检查模型初始状态
        with torch.no_grad():
            # 检查参数范围
            param_norms = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param_norm = param.data.norm(2).item()
                    param_norms.append(param_norm)
                    if param_norm > 10.0:
                        print(f"⚠️ 参数 {name} 初始范数过大: {param_norm:.4f}")
            
            avg_param_norm = sum(param_norms) / len(param_norms) if param_norms else 0
            print(f"📊 模型参数平均范数: {avg_param_norm:.4f}")
        
        # 早停机制参数（针对极度不平衡数据调整）
        best_val_accuracy = 0   # 最佳验证准确率
        best_epoch = -1         # 记录最佳模型的epoch
        patience_counter = 0    # 耐心计数器
        patience = getattr(self, 'patience', 50)  # 耐心阈值：可由 trainer.patience 覆盖
        
        # 训练循环
        for epoch in range(num_epochs):
            if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)  # DDP: 每 epoch 重排分片
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # 训练阶段
            train_loss = self.train_epoch(train_loader)
            
            # 验证阶段
            val_loss, val_accuracy, val_f1_weighted, val_f1_macro, val_precision_weighted, val_recall_weighted, val_predictions, val_labels = self.validate_epoch(val_loader)
            
            # 更新训练历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            
            # 学习率调度：根据调度器类型处理
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    # ReduceLROnPlateau需要传入指标
                    self.scheduler.step(val_f1_weighted)
                elif isinstance(self.scheduler, optim.lr_scheduler.CosineAnnealingLR):
                    # CosineAnnealingLR在train_epoch中已经更新，这里不需要再次调用
                    pass
            
            # 输出训练统计信息
            print(f"训练损失: {train_loss:.4f}")
            print(f"验证损失: {val_loss:.4f}")
            # 计算验证集的正确样本数
            val_correct = int(val_accuracy * len(val_loader.dataset))
            val_total = len(val_loader.dataset)
            print(f"验证准确率: {val_accuracy:.4f} ({val_correct}/{val_total})")
            print(f"学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 生成样本输出：训练集3个样本，验证集2个样本（仅 rank0,避免 DDP 多进程重复）
            if self.ddp_rank == 0:
                self.generate_sample_outputs(train_loader, num_samples=3, sample_type="训练")
                self.generate_sample_outputs(val_loader, num_samples=2, sample_type="验证")
            
            # 保存最佳模型（基于weighted F1分数）
            if val_f1_weighted > best_val_accuracy:
                best_val_accuracy = val_f1_weighted
                best_epoch = epoch + 1  # 记录最佳epoch（从1开始计数）
                patience_counter = 0  # 重置耐心计数器
                
                # 保存模型检查点（包含所有必要信息）
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_accuracy': val_accuracy,
                    'train_history': {
                        'train_losses': self.train_losses,
                        'val_losses': self.val_losses,
                        'val_accuracies': self.val_accuracies
                    }
                }
                
                # 只有在调度器存在时才保存其状态
                if self.scheduler is not None:
                    checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
                
                if self.ddp_rank == 0:
                    torch.save(checkpoint, save_path)

                print(f"💾 保存新的最佳模型！验证F1(weighted): {val_f1_weighted:.4f} (Epoch {epoch + 1})")
            else:
                patience_counter += 1  # 增加耐心计数器
                
            # 内存清理：防止内存泄漏
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                if hasattr(torch.backends.mps, 'empty_cache'):
                    torch.backends.mps.empty_cache()
                
            # 早停机制：连续多个epoch没有改善则停止训练
            if patience_counter >= patience:
                print(f"⏹️  连续{patience}个epoch没有改善，触发早停机制")
                break
        
        print(f"\n训练完成！")
        print(f"📌 最佳模型保存于第 {best_epoch} 轮")
        return best_val_accuracy
    
    def plot_training_history(self, save_path):
        """
        绘制训练历史图表
        
        参数：
            save_path (str): 图表保存路径
        """
        # 创建包含两个子图的图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线图
        ax1.plot(self.train_losses, label='训练损失')
        ax1.plot(self.val_losses, label='验证损失')
        ax1.set_title('训练和验证损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('损失')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线图
        ax2.plot(self.val_accuracies, label='验证准确率')
        ax2.set_title('验证准确率')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('准确率')
        ax2.legend()
        ax2.grid(True)
        
        # 调整布局并保存图表
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def _crf_prepare(labels, attention_mask):
    """把以 -100 作 padding 的 labels 转成 torchcrf 需要的 (safe_labels, bool_mask)。
    torchcrf 要求 labels∈[0,C) 且 mask 为 bool 且每条序列首位为 True。"""
    if attention_mask is not None:
        mask = attention_mask.bool().clone()
    else:
        mask = (labels != -100)
    mask = mask & (labels != -100)
    mask[:, 0] = True  # torchcrf 要求每条序列首位有效
    safe_labels = labels.clone()
    safe_labels[labels == -100] = 0
    return safe_labels, mask


def _crf_decode_to_tensor(model, logits, attention_mask, labels):
    """用 CRF Viterbi 解码并把变长路径 pad 回 [batch, seq_len] tensor（与 labels 对齐）。"""
    _, mask = _crf_prepare(labels, attention_mask)
    paths = model.crf.decode(logits, mask=mask)  # List[List[int]]
    out = torch.zeros_like(labels)
    for i, p in enumerate(paths):
        if len(p) > 0:
            out[i, :len(p)] = torch.as_tensor(p, device=labels.device, dtype=out.dtype)
    return out


def evaluate_model(model, dataloader, device, num_classes, return_per_sample=False):
    """
    评估模型并返回详细指标
    
    参数：
        model: 要评估的模型
        dataloader: 测试数据加载器
        device: 计算设备
        num_classes (int): 分类类别数
        return_per_sample (bool): 是否返回每个样本的预测（按原始顺序）
        
    返回：
        如果return_per_sample=False:
            Tuple: (准确率, 分类报告, 预测结果, 真实标签, 正确预测数, 总预测数)
        如果return_per_sample=True:
            额外返回: 每个样本的预测列表（按原始顺序）
    """
    model.eval()  # 设置为评估模式
    all_predictions = []  # 所有预测结果（展平）
    all_labels = []       # 所有真实标签（展平）
    
    # 用于保存每个样本的完整预测（不展平）
    per_sample_predictions = [] if return_per_sample else None
    # 用于保存行号（如果提供）
    sample_line_numbers = [] if return_per_sample else None
    
    # 不计算梯度以节省内存
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="评估中")):
            # 解包批次数据（可能包含行号）
            if len(batch_data) == 4:  # 包含行号
                input_ids, labels, attention_mask, line_numbers = batch_data
                line_numbers = line_numbers.cpu().numpy().tolist()
            else:  # 不包含行号
                input_ids, labels, attention_mask = batch_data
                line_numbers = None
            # 将数据移动到指定设备
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)
            
            # 模型预测
            if hasattr(model, 'compute_loss') and hasattr(model, 'input_vocab_size'):
                # MDLM模型特殊处理
                # 生成序列并直接获取预测标签
                predictions = model.generate(input_ids)  # 现在直接返回预测标签 [batch, seq_len]
                
                # 截断到与labels相同的长度
                min_len = min(predictions.size(1), labels.size(1))
                predictions = predictions[:, :min_len]
                labels = labels[:, :min_len]
            elif getattr(model, 'crf', None) is not None:
                # CRF 模型：Viterbi 解码
                logits = model(input_ids, attention_mask)
                predictions = _crf_decode_to_tensor(model, logits, attention_mask, labels)
                min_len = min(predictions.size(1), labels.size(1))
                predictions = predictions[:, :min_len]
                labels = labels[:, :min_len]
            else:
                # 标准模型预测
                outputs = model(input_ids, attention_mask)
                predictions = torch.argmax(outputs, dim=-1)

            # 如果需要保存每个样本的预测
            if return_per_sample:
                # 对批次中的每个样本
                for i in range(predictions.size(0)):
                    # 获取单个样本的预测和标签
                    sample_pred = predictions[i]
                    sample_label = labels[i]
                    
                    # 只保留非填充位置的预测
                    valid_mask = sample_label != -100
                    valid_pred = sample_pred[valid_mask].cpu().numpy().tolist()
                    
                    # 保存该样本的预测
                    per_sample_predictions.append(valid_pred)
                    
                    # 如果有行号，也保存行号
                    if line_numbers is not None and i < len(line_numbers):
                        sample_line_numbers.append(line_numbers[i])
            
            # 展平并过滤填充位置（用于计算指标）
            predictions_flat = predictions.view(-1)
            labels_flat = labels.view(-1)
            
            # 只评估非填充位置
            mask = labels_flat != -100
            predictions_filtered = predictions_flat[mask]
            labels_filtered = labels_flat[mask]
            
            # 收集结果
            all_predictions.extend(predictions_filtered.cpu().numpy())
            all_labels.extend(labels_filtered.cpu().numpy())
    
    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # 计算正确预测数和总预测数
    correct_predictions = sum(1 for pred, label in zip(all_predictions, all_labels) if pred == label)
    total_predictions = len(all_labels)

    # 获取实际出现的类别
    from sklearn.utils.multiclass import unique_labels
    labels_unique = list(unique_labels(all_labels, all_predictions))

    # 生成分类报告
    class_report = classification_report(
        all_labels, all_predictions,
        labels=labels_unique,
        target_names=[str(i) for i in labels_unique],
        output_dict=True
    )
    
    # 生成测试样本输出展示（5个样本）
    generate_test_sample_outputs(model, dataloader, device, num_samples=5)
    
    if return_per_sample:
        # 如果有行号，需要按行号排序
        if sample_line_numbers:
            # 将预测和行号配对，然后按行号排序
            paired_results = list(zip(sample_line_numbers, per_sample_predictions))
            paired_results.sort(key=lambda x: x[0])  # 按行号排序
            # 分离排序后的结果
            sorted_predictions = [pred for _, pred in paired_results]
            return accuracy, class_report, all_predictions, all_labels, correct_predictions, total_predictions, sorted_predictions
        else:
            return accuracy, class_report, all_predictions, all_labels, correct_predictions, total_predictions, per_sample_predictions
    else:
        return accuracy, class_report, all_predictions, all_labels, correct_predictions, total_predictions

def generate_test_sample_outputs(model, dataloader, device, num_samples=5, save_path=None):
    """
    生成测试样本输出展示
    
    参数:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        num_samples: 要显示的样本数量
        save_path: 可选，保存样本输出的文件路径
    """
    # 导入解码函数
    try:
        from models.data_utils import idx_to_char
    except ImportError:
        # 如果无法导入，提供简单的解码函数
        def idx_to_char(idx):
            return str(idx)
    
    model.eval()
    samples_shown = 0
    
    print(f"\n生成的测试样本:")
    print("-" * 80)
    
    # 如果指定了保存路径，准备保存文件
    sample_outputs = []
    
    with torch.no_grad():
        for batch_data in dataloader:
            if samples_shown >= num_samples:
                break
            
            # 处理可能包含行号的批次数据
            if len(batch_data) == 4:  # 包含行号
                input_ids, labels, attention_mask, _ = batch_data
            else:  # 不包含行号
                input_ids, labels, attention_mask = batch_data
                
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)
            
            # 获取模型预测 - 与evaluate_model函数保持一致
            if hasattr(model, 'compute_loss') and hasattr(model, 'input_vocab_size'):
                # MDLM模型特殊处理
                predictions = model.generate(input_ids)  # 直接返回预测标签
                
                # 截断到与labels相同的长度
                min_len = min(predictions.size(1), labels.size(1))
                predictions = predictions[:, :min_len]
                labels = labels[:, :min_len]
            else:
                # 标准模型预测
                outputs = model(input_ids, attention_mask)
                predictions = torch.argmax(outputs, dim=-1)
            
            # 处理每个样本
            batch_size = min(num_samples - samples_shown, input_ids.size(0))
            for i in range(batch_size):
                # 获取有效长度（非填充部分）
                valid_mask = attention_mask[i] == 1
                valid_length = valid_mask.sum().item()
                
                # 解码输入、目标和预测
                input_seq = input_ids[i][:valid_length].cpu().numpy()
                target_seq = labels[i][:valid_length].cpu().numpy()
                pred_seq = predictions[i][:valid_length].cpu().numpy()
                
                # 过滤掉填充标记(-100)
                target_filtered = target_seq[target_seq != -100]
                pred_filtered = pred_seq[:len(target_filtered)]
                input_filtered = input_seq[:len(target_filtered)]
                
                # 解码为文本
                try:
                    input_text = ''.join([idx_to_char(idx) for idx in input_filtered])
                    target_text = ' '.join([str(idx) for idx in target_filtered])
                    pred_text = ' '.join([str(idx) for idx in pred_filtered])
                except:
                    input_text = str(input_filtered.tolist())
                    target_text = str(target_filtered.tolist())
                    pred_text = str(pred_filtered.tolist())
                
                print(f"输入:     {input_text}")
                print(f"目标:     {target_text}")
                print(f"预测:     {pred_text}")
                print("-" * 40)
                
                # 保存到列表
                if save_path:
                    sample_outputs.append({
                        'sample_id': samples_shown,
                        'input': input_text,
                        'target': target_text,
                        'prediction': pred_text,
                        'input_ids': input_seq.tolist(),
                        'target_ids': target_seq.tolist(),
                        'prediction_ids': pred_seq.tolist()
                    })
                
                samples_shown += 1
                if samples_shown >= num_samples:
                    break
            
            if samples_shown >= num_samples:
                break
    
    # 保存样本输出到文件
    if save_path and sample_outputs:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(sample_outputs, f, indent=2, ensure_ascii=False)
        # print(f"✅ 测试样本输出已保存到: {save_path}")  # 不输出此行

def get_device_info():
    """
    获取设备信息和推荐配置
    """
    if DEVICE_CONFIG_AVAILABLE:
        device_manager = get_device_manager()
        device_manager.print_device_info()
        return device_manager
    else:
        # 基本设备检测
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"🚀 使用CUDA GPU: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print(f"🍎 使用Apple MPS")
        else:
            device = torch.device('cpu')
            print(f"💻 使用CPU")
        return None

def apply_smart_config(args, model_type):
    """
    应用智能配置优化
    """
    if DEVICE_CONFIG_AVAILABLE:
        config = create_optimized_config(model_type, args.batch_size)
        
        # 应用优化配置
        args.batch_size = config['batch_size']
        
        # 根据模型类型应用不同的参数
        if model_type == 'bilstm':
            args.hidden_size = config['model_config']['hidden_size']
            args.num_layers = config['model_config']['num_layers']
            args.max_length = config['model_config']['max_length']
        else:
            # transformer, bert, diffusion模型使用d_model
            args.d_model = config['model_config']['d_model']
            args.num_layers = config['model_config']['num_layers']
            if 'num_heads' in config['model_config']:
                args.num_heads = config['model_config']['num_heads']
            args.max_length = config['model_config']['max_length']
        
        # 根据模型类型和设备优化学习率
        device_type = config['device'].type
        
        # 为Transformer和BERT模型设置合适的学习率和架构
        if model_type in ['transformer', 'bert']:
            if model_type == 'transformer':
                # Transformer 优化学习率以达到90%+性能
                optimal_lr = 4e-5  # 稍微降低学习率以提高稳定性
                # Transformer特殊优化：增加模型容量
                if args.num_layers == 3:  # 默认层数
                    args.num_layers = 4  # 增加到4层
                if args.d_model == 256:  # 默认维度
                    args.d_model = 384  # 增加维度比BERT稍大
                if args.num_heads == 4:  # 默认头数
                    args.num_heads = 8   # 增加注意力头数
                # 调整权重衰减
                if args.weight_decay == 0.02:  # 默认权重衰减
                    args.weight_decay = 0.01  # 减少权重衰减
            else:  # bert
                # BERT 恢复到之前93.12%的高性能配置
                optimal_lr = 5e-6  # 使用更低的学习率避免梯度爆炸
                # BERT保持默认架构配置，专注于学习率优化
                # 之前93.12%是在较简单架构下实现的
                # 恢复原始架构参数
                if args.num_layers == 4:  # 如果被智能配置改了
                    args.num_layers = 3   # 恢复到原始层数
                if args.num_heads == 8:   # 如果被智能配置改了
                    args.num_heads = 4    # 恢复到原始头数
            
            # 只有当当前学习率是默认值时才设置，否则保持用户指定的值
            if args.learning_rate <= 0.0001:  # 默认或较低值
                args.learning_rate = optimal_lr
                print(f"🔧 {model_type.upper()}学习率优化: {args.learning_rate}")
        elif model_type == 'mdlm':
            # MDLM模型特殊优化
            optimal_lr = 1e-4  # MDLM使用中等学习率
            # MDLM架构优化
            if args.d_model == 256:  # 默认维度
                args.d_model = 512  # 增加维度
            if args.num_layers == 3:  # 默认层数
                args.num_layers = 6  # 增加层数
            if args.num_heads == 4:  # 默认头数
                args.num_heads = 8   # 增加注意力头数
            # 调整权重衰减
            if args.weight_decay == 0.02:  # 默认权重衰减
                args.weight_decay = 0.01  # 减少权重衰减
            
            # 只有当当前学习率是默认值时才设置，否则保持用户指定的值
            if args.learning_rate <= 0.0001:  # 默认或较低值
                args.learning_rate = optimal_lr
                print(f"🔧 MDLM学习率优化: {args.learning_rate}")
        elif model_type in ['rwkv7', 'rwkv7_large', 'rwkv7_efficient']:
            # RWKV-7模型深度优化 - 提升到96%+性能
            optimal_lr = 1e-4  # 更低的学习率，确保稳定收敛
            print(f"🔧 RWKV-7模型深度优化:")
            
            # 更强的正则化防止过拟合
            if args.weight_decay == 0.02:  # 默认权重衰减
                args.weight_decay = 0.01  # 增强权重衰减
                print(f"   权重衰减: {args.weight_decay}")
            
            # 更保守的学习率策略
            if args.learning_rate >= 0.0001:  # 如果学习率过高
                args.learning_rate = optimal_lr
                print(f"   学习率优化: {args.learning_rate}")
                
            # 增加模型容量
            if args.d_model < 512:
                args.d_model = 512  # 增大模型维度
                print(f"   模型维度: {args.d_model}")
                
            if args.num_layers < 10:
                args.num_layers = 12  # 增加层数
                print(f"   层数: {args.num_layers}")
                
            # RWKV-7特定的高性能优化
            print(f"   启用RWKV-7高性能优化模式")
        else:
            # 对于BiLSTM等其他模型，使用较高的学习率
            if device_type == 'cuda':
                args.learning_rate = max(args.learning_rate, 0.001)  # 提高CUDA学习率
            elif device_type == 'mps':
                args.learning_rate = max(args.learning_rate, 0.0005)  # 提高MPS学习率
            else:  # CPU
                args.learning_rate = max(args.learning_rate, 0.001)  # CPU使用较高学习率
        
        # 对于Diffusion模型，使用更保守的配置以避免内存不足
        if model_type == 'diffusion':
            args.d_model = min(args.d_model, 512)  # 限制模型维度
            args.num_layers = min(args.num_layers, 8)  # 限制层数
            args.batch_size = min(args.batch_size, 16)  # 限制批次大小
            args.max_length = 256  # Diffusion模型固定为256（需要START/END token）
            print(f"🔧 Diffusion模型优化配置:")
            print(f"   模型维度: {args.d_model}")
            print(f"   层数: {args.num_layers}")
            print(f"   批次大小: {args.batch_size}")
            print(f"   序列长度: {args.max_length} (固定为256)")
        
        print(f"✅ 智能配置已应用:")
        print(f"   批次大小: {args.batch_size}")
        if model_type == 'bilstm':
            print(f"   隐藏层大小: {args.hidden_size}")
        else:
            print(f"   模型维度: {args.d_model}")
        print(f"   层数: {args.num_layers}")
        print(f"   学习率: {args.learning_rate}")
        
        return config
    else:
        print("⚠️  使用默认配置")
        return None

def run_model_comparison(data_dir: str, models_config: str, verbose: bool = False):
    """
    运行模型比较
    """
    if not os.path.exists(models_config):
        print(f"❌ 模型配置文件不存在: {models_config}")
        return False
    
    try:
        cmd = [
            sys.executable, 'compare_models.py',
            f'--data_dir={data_dir}',
            f'--models_config={models_config}',
            '--save_dir=./comparisons'
        ]
        
        if verbose:
            print(f"🔧 执行命令: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=not verbose, text=True)
        
        if result.returncode != 0:
            if not verbose and result.stderr:
                print(f"❌ 模型比较失败: {result.stderr}")
            return False
        
        print("✅ 模型比较完成")
        return True
        
    except Exception as e:
        print(f"❌ 模型比较过程中发生错误: {e}")
        return False

def main():
    """
    主函数：处理命令行参数并启动训练流程
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='训练叙利亚文形态分析模型 - 智能版本')
    
    # 运行模式选择
    parser.add_argument('--mode', type=str, choices=['train', 'compare'], 
                      default='train', help='运行模式：训练或比较')
    
    # 模型架构选择
    supported_models = get_supported_models()
    parser.add_argument('--model_type', type=str, choices=supported_models, 
                      default='transformer', help='要使用的模型架构')
    
    # 数据相关参数
    parser.add_argument('--data_dir', type=str, 
                      default='./data',
                      help='包含训练/验证/测试数据的目录')
    parser.add_argument('--data_subdir', type=str, 
                      default=None,
                      help='数据子目录名称（可选，如s4-in_s4-out_vs_s2-in_comparison_split）')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--max_length', type=int, default=64, help='最大序列长度')
    
    # 训练参数（针对极度不平衡数据优化）
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='学习率（针对BERT/Transformer优化）')
    parser.add_argument('--num_epochs', type=int, default=100, help='训练轮数（增加以充分学习罕见类别）')
    parser.add_argument('--epochs', type=int, help='训练轮数（别名，兼容run_training.py）')
    parser.add_argument('--save_dir', type=str, default='./outputs', help='模型保存目录')
    parser.add_argument('--weight_decay', type=float, default=0.02, help='权重衰减（稍微提高正则化）')
    
    # 模型特定超参数
    parser.add_argument('--d_model', type=int, default=256, help='模型维度（降低以减少复杂度）')
    parser.add_argument('--num_layers', type=int, default=3, help='层数（降低以减少复杂度）')
    parser.add_argument('--num_heads', type=int, default=4, help='注意力头数（降低以减少复杂度）')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout概率（提高以避免过拟合）')
    parser.add_argument('--num_timesteps', type=int, default=1000, help='Diffusion时间步数')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--use_crf', action='store_true', help='在分类头之上加 CRF 层（用于 BiLSTM-CRF / Encoder+CRF baseline，回应 R2-M4/R3-2）')

    # 智能训练功能
    parser.add_argument('--smart_config', action='store_true', help='启用智能配置优化')
    parser.add_argument('--force_device', type=str, choices=['cpu', 'cuda', 'mps'], 
                       help='强制使用指定设备')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    parser.add_argument('--test_only', action='store_true', help='仅测试模式，不训练')
    parser.add_argument('--model_path', type=str, help='指定要加载的模型文件路径')
    
    # 模型比较功能
    parser.add_argument('--models_config', type=str, 
                       help='模型比较配置文件（用于比较模式）')
    
    # 解析命令行参数
    args = parser.parse_args()

    # 设置随机种子（可复现性）
    import random as _random
    _random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    # 处理兼容性参数
    if args.epochs is not None:
        args.num_epochs = args.epochs
    
    # 构造实际的数据目录路径
    # 测试模式时，优先使用模型目录下的data文件
    if args.test_only and args.model_path:
        # 检查模型目录下是否有data目录
        model_dir = os.path.dirname(args.model_path)
        model_data_dir = os.path.join(model_dir, 'data')
        
        if os.path.exists(model_data_dir):
            # 检查必要的文件是否存在
            required_files = ['test.in', 'test.out']
            all_files_exist = all(
                os.path.exists(os.path.join(model_data_dir, f)) 
                for f in required_files
            )
            
            if all_files_exist:
                actual_data_dir = model_data_dir
                print(f"🔄 使用模型目录下的数据: {actual_data_dir}")
            else:
                # 使用默认数据目录
                actual_data_dir = args.data_dir
                if args.data_subdir:
                    actual_data_dir = os.path.join(args.data_dir, args.data_subdir)
                print(f"⚠️  模型目录下缺少数据文件，使用默认数据: {actual_data_dir}")
        else:
            # 没有模型目录下的data，使用默认
            actual_data_dir = args.data_dir
            if args.data_subdir:
                actual_data_dir = os.path.join(args.data_dir, args.data_subdir)
            print(f"📁 使用默认数据目录: {actual_data_dir}")
    else:
        # 训练模式，使用指定的数据目录
        actual_data_dir = args.data_dir
        if args.data_subdir:
            actual_data_dir = os.path.join(args.data_dir, args.data_subdir)
            print(f"📁 使用数据子目录: {actual_data_dir}")
    
    # 处理比较模式
    if args.mode == 'compare':
        if not args.models_config:
            print("❌ 比较模式需要提供模型配置文件 (--models_config)")
            return
        
        print(f"📊 开始模型比较...")
        success = run_model_comparison(actual_data_dir, args.models_config, args.verbose)
        
        if success:
            print(f"📈 模型比较完成！")
        else:
            print("❌ 模型比较失败")
        return
    
    # 测试模式处理
    if args.test_only:
        if not args.model_path:
            print("❌ 测试模式需要提供模型文件路径 (--model_path)")
            return
        
        if not os.path.exists(args.model_path):
            print(f"❌ 模型文件不存在: {args.model_path}")
            return
        
        print(f"🧪 测试模式：加载模型 {args.model_path}")
        
        # 从模型路径推断模型类型（如果没有指定）
        if not args.model_type:
            model_filename = os.path.basename(args.model_path)
            if 'mdlm' in model_filename.lower():
                args.model_type = 'mdlm'
            elif 'lstm' in model_filename.lower():
                args.model_type = 'lstm'
            elif 'transformer' in model_filename.lower():
                args.model_type = 'transformer'
            else:
                print("❌ 无法从文件名推断模型类型，请使用 --model_type 参数指定")
                return
            print(f"📝 推断模型类型: {args.model_type}")
    
    # 训练模式
    if not args.test_only:
        print(f"🚀 开始训练 {args.model_type} 模型...")
    
    # 生成统一的输出目录 - 使用outputs目录而不是models目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.test_only:
        # 测试模式: {model}_test_{timestamp}
        output_dir = os.path.join("outputs", f"{args.model_type}_test_{timestamp}")
    else:
        # 训练模式: {model}_train_{timestamp}
        output_dir = os.path.join("outputs", f"{args.model_type}_train_{timestamp}")
    
    # 创建子目录结构
    tensorboard_dir = os.path.join(output_dir, "tensorboard")
    plots_dir = os.path.join(output_dir, "plots")
    results_dir = os.path.join(output_dir, "results")
    data_dir = os.path.join(output_dir, "data")  # 添加data目录用于存储输入文件
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # 更新args.save_dir为新的输出目录
    args.save_dir = output_dir
    
    print(f"📁 输出目录: {output_dir}")
    
    # 复制重要的数据文件到输出目录
    import shutil
    
    # 复制patterns.csv
    patterns_path = os.path.join(actual_data_dir, 'patterns.csv')
    if os.path.exists(patterns_path):
        shutil.copy2(patterns_path, os.path.join(data_dir, 'patterns.csv'))
        print(f"  ✓ 复制patterns.csv")
    
    # 复制测试集文件
    test_in_path = os.path.join(actual_data_dir, 'test.in')
    test_out_path = os.path.join(actual_data_dir, 'test.out')
    if os.path.exists(test_in_path):
        shutil.copy2(test_in_path, os.path.join(data_dir, 'test.in'))
        print(f"  ✓ 复制test.in")
    if os.path.exists(test_out_path):
        shutil.copy2(test_out_path, os.path.join(data_dir, 'test.out'))
        print(f"  ✓ 复制test.out")
    
    # 训练模式还需要复制训练和验证数据
    if not args.test_only:
        train_in_path = os.path.join(actual_data_dir, 'train.in')
        train_out_path = os.path.join(actual_data_dir, 'train.out')
        val_in_path = os.path.join(actual_data_dir, 'val.in')
        val_out_path = os.path.join(actual_data_dir, 'val.out')
        
        if os.path.exists(train_in_path):
            shutil.copy2(train_in_path, os.path.join(data_dir, 'train.in'))
            print(f"  ✓ 复制train.in")
        if os.path.exists(train_out_path):
            shutil.copy2(train_out_path, os.path.join(data_dir, 'train.out'))
            print(f"  ✓ 复制train.out")
        if os.path.exists(val_in_path):
            shutil.copy2(val_in_path, os.path.join(data_dir, 'val.in'))
            print(f"  ✓ 复制val.in")
        if os.path.exists(val_out_path):
            shutil.copy2(val_out_path, os.path.join(data_dir, 'val.out'))
            print(f"  ✓ 复制val.out")
    
    # 保存数据集信息
    dataset_info = {
        'original_data_dir': actual_data_dir,
        'timestamp': timestamp,
        'model_type': args.model_type,
        'mode': 'test' if args.test_only else 'train',
        'max_length': args.max_length,
        'batch_size': args.batch_size if hasattr(args, 'batch_size') else 16
    }
    
    # 如果是测试模式且使用了模型目录下的数据，记录这个信息
    if args.test_only and 'model_data_dir' in locals() and actual_data_dir == model_data_dir:
        dataset_info['using_model_data'] = True
        dataset_info['model_path'] = args.model_path
    
    with open(os.path.join(data_dir, 'dataset_info.json'), 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        print(f"  ✓ 创建数据集信息文件")
    
    # 获取设备信息
    device_manager = get_device_info()
    
    # 应用智能配置
    if args.smart_config or DEVICE_CONFIG_AVAILABLE:
        print("🧠 启用智能配置优化...")
        config = apply_smart_config(args, args.model_type)
        
        # 设置设备优化
        if device_manager and hasattr(device_manager, 'setup_device_optimization'):
            device_manager.setup_device_optimization()
    
    # 智能设备选择：优先使用GPU，考虑MPS（苹果硅芯片）和CUDA兼容性
    if args.force_device:
        if args.force_device == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
        elif args.force_device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        print(f"🔧 强制使用设备: {device}")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 使用CUDA GPU: {torch.cuda.get_device_name()}")
        print(f"📊 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        # 设置GPU内存分配策略
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True  # 优化CUDA性能
        # === DDP (torchrun) 初始化:在 torchrun 下绑定本卡并建进程组 ===
        if "LOCAL_RANK" in os.environ:
            import torch.distributed as _dist
            _lrk = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(_lrk)
            if not _dist.is_initialized():
                _dist.init_process_group(backend="nccl")
            device = torch.device(f"cuda:{_lrk}")
            print(f"🔗 DDP rank {os.environ.get('RANK')}/{os.environ.get('WORLD_SIZE')} local={_lrk} device={device}")

        # 根据GPU内存调整批次大小建议
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory_gb < 6:
            print("⚠️  GPU内存较小，建议使用较小的批次大小和模型尺寸")
            
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"🍎 使用Apple MPS (Metal Performance Shaders)")
        print("💡 MPS优化提示：适合M1/M2芯片，批次大小建议8-16")
        # MPS特定优化
        if hasattr(torch.backends.mps, 'empty_cache'):
            torch.backends.mps.empty_cache()
        
    else:
        device = torch.device('cpu')
        print(f"💻 使用CPU")
        # CPU优化：设置线程数
        num_threads = min(8, torch.get_num_threads())
        torch.set_num_threads(num_threads)
        print(f"🧵 CPU线程数: {num_threads}")
        print("💡 CPU优化提示：建议使用较小的模型和批次大小")
    
    # 根据设备类型和算力调整批次大小
    adjusted_batch_size = args.batch_size
    if device.type == 'cpu':
        # CPU性能有限，使用较小的批次大小
        adjusted_batch_size = min(args.batch_size, 8)
        print(f"🔧 CPU优化：批次大小调整为 {adjusted_batch_size}")
    elif device.type == 'mps':
        # MPS对内存管理较为敏感，使用中等批次大小
        adjusted_batch_size = min(args.batch_size, 16)
        print(f"🔧 MPS优化：批次大小调整为 {adjusted_batch_size}")
    elif device.type == 'cuda':
        # CUDA可以处理更大的批次，但需要考虑GPU内存
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory_gb < 6:
            adjusted_batch_size = min(args.batch_size, 8)
            print(f"🔧 CUDA优化（低内存）：批次大小调整为 {adjusted_batch_size}")
    
    # 创建数据加载器
    print("📂 正在加载数据...")
    
    # 根据模型类型获取正确的词汇表
    char_to_idx, idx_to_char, vocab_size = get_vocab_for_model(args.model_type)
    
    # 如果是MDLM模型的测试模式，先加载模型以获取正确的max_length
    if args.test_only and args.model_type.lower() in ['mdlm', 'diffusion']:
        # 先加载模型检查配置
        checkpoint = torch.load(args.model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            # 检查是否有位置编码以推断max_length
            if 'pos_embeddings.weight' in state_dict:
                inferred_max_length = state_dict['pos_embeddings.weight'].shape[0]
                if inferred_max_length != args.max_length:
                    print(f"⚠️  从模型推断max_length = {inferred_max_length} (原始值: {args.max_length})")
                    args.max_length = inferred_max_length
        del checkpoint  # 释放内存
    
    # 创建数据加载器（新的data_utils返回四个值）
    # 所有模型的测试集都返回行号以保持顺序
    test_return_line_numbers = True  # 对所有模型都启用，确保预测顺序与test.in一致
    
    train_loader, val_loader, test_loader, _ = create_data_loaders(
        train_input=os.path.join(actual_data_dir, 'train.in'),
        train_output=os.path.join(actual_data_dir, 'train.out'),
        val_input=os.path.join(actual_data_dir, 'val.in'),
        val_output=os.path.join(actual_data_dir, 'val.out'),
        test_input=os.path.join(actual_data_dir, 'test.in'),
        test_output=os.path.join(actual_data_dir, 'test.out'),
        batch_size=adjusted_batch_size,
        max_length=args.max_length,  # 使用可能已更新的max_length
        model_type=args.model_type,  # 传递模型类型
        test_return_line_numbers=test_return_line_numbers  # 传递是否返回行号
    )
    # === DDP: 用 DistributedSampler 把训练集分片(每卡只载 1/world,真正加速) ===
    if "LOCAL_RANK" in os.environ and int(os.environ.get("WORLD_SIZE", 1)) > 1:
        from torch.utils.data.distributed import DistributedSampler as _DSamp
        from torch.utils.data import DataLoader as _DL
        _ws = int(os.environ["WORLD_SIZE"]); _rk = int(os.environ["RANK"])
        _tds = train_loader.dataset
        _samp = _DSamp(_tds, num_replicas=_ws, rank=_rk, shuffle=True, seed=getattr(args, 'seed', 42))
        train_loader = _DL(_tds, batch_size=adjusted_batch_size, sampler=_samp,
                           collate_fn=train_loader.collate_fn,
                           num_workers=getattr(train_loader, 'num_workers', 0), pin_memory=True)
        if _rk == 0:
            print(f"🔗 DDP 训练分片: world={_ws}, batch/卡={adjusted_batch_size}, 等效batch={adjusted_batch_size*_ws}")
    
    # 对于非diffusion模型，计算分类类别数和权重
    if args.model_type != 'diffusion':
        # 获取分类类别数（从实际数据）
        print("📊 分析数据集...")
        try:
            output_files = [
                os.path.join(actual_data_dir, 'train.out'),
                os.path.join(actual_data_dir, 'val.out'),
                os.path.join(actual_data_dir, 'test.out')
            ]
            num_classes = get_num_classes(output_files)  # 从实际数据统计
            print(f"[DEBUG] 实际num_classes = {num_classes}")
        except Exception as e:
            print(f"❌ 获取类别数失败: {e}")
            return
        
        # 计算类别权重以处理极度数据不平衡
        print("⚖️  计算类别权重...")
        try:
            class_weights = calculate_class_weights(train_loader, num_classes, device)
            print(f"✅ 类别权重计算完成，num_classes={num_classes}")
        except Exception as e:
            print(f"❌ 计算类别权重失败: {e}, 将不使用权重。")
            class_weights = None
    else:
        # Diffusion模型不需要类别数和权重
        print("🎯 Diffusion模型：跳过类别分析和权重计算")
        num_classes = None
        class_weights = None
    
    # 使用新的词汇表管理器获取vocab_size
    vocab_size = get_vocab_size(args.model_type)
    
    # 打印数据统计信息
    print(f"📊 数据统计:")
    print(f"   词汇表大小: {vocab_size}")
    if args.model_type != 'diffusion':
        print(f"   分类类别数: {num_classes}")
    print(f"   训练批次: {len(train_loader)}")
    print(f"   验证批次: {len(val_loader)}")
    print(f"   测试批次: {len(test_loader)}")
    print(f"   批次大小: {adjusted_batch_size}")
    print(f"   最大序列长度: {args.max_length}")
    
    # 创建模型
    print(f"🏗️  创建{args.model_type}模型...")
    try:
        if args.model_type == 'diffusion':
            # 对于diffusion模型，我们需要调用专门的训练脚本
            print("🔄 Diffusion模型需要使用专门的训练脚本")
            print("🚀 正在调用train_diffusion.py...")
            
            # 构建train_diffusion.py的命令（跨平台实时输出）
            import platform
            if platform.system() == 'Linux':
                # Linux系统使用stdbuf确保实时输出
                cmd = ['stdbuf', '-oL', '-eL', sys.executable, '-u', 'train_diffusion.py']
            else:
                # macOS/Windows系统直接使用Python -u参数
                cmd = [sys.executable, '-u', 'train_diffusion.py']
            
            # 添加参数
            cmd.extend([
                f'--data_dir={actual_data_dir}',
                f'--batch_size={args.batch_size}',
                f'--num_epochs={args.num_epochs}',
                f'--d_model={args.d_model}',
                f'--num_layers={args.num_layers}',
                f'--num_heads={args.num_heads}',
                f'--learning_rate={args.learning_rate}',
                f'--save_dir={args.save_dir}',
                f'--max_length={args.max_length}',
                f'--dropout={args.dropout}',
                f'--weight_decay={args.weight_decay}',
                f'--num_timesteps={args.num_timesteps}'
            ])
            
            # train_diffusion.py 不支持 --verbose 参数，跳过
            
            if args.verbose:
                print(f"🔧 执行命令: {' '.join(cmd)}")
            
            # 调用train_diffusion.py（配置实时输出）
            try:
                # 在SLURM环境中确保实时输出
                env = os.environ.copy()
                env['PYTHONUNBUFFERED'] = '1'  # 强制Python不缓冲输出
                
                result = subprocess.run(
                    cmd, 
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                    env=env,
                    stdout=None,  # 继承父进程的stdout
                    stderr=None,  # 继承父进程的stderr
                    bufsize=0     # 无缓冲
                )
                if result.returncode == 0:
                    print("✅ Diffusion模型训练完成")
                else:
                    print("❌ Diffusion模型训练失败")
                return
            except Exception as e:
                print(f"❌ 调用train_diffusion.py失败: {e}")
                return
        else:
            # 使用统一的模型工厂创建模型
            if args.model_type in ['lstm', 'bilstm']:
                model_config = {
                    'embedding_dim': getattr(args, 'embedding_dim', 128),
                    'hidden_size': getattr(args, 'hidden_size', 256),
                    'num_layers': args.num_layers,
                    'dropout': args.dropout,
                    'bidirectional': True,
                    'use_attention': True,
                    'attention_heads': 8,
                }
            else:
                model_config = {
                    'd_model': args.d_model,
                    'num_layers': args.num_layers,
                    'num_heads': args.num_heads,
                    'dropout': args.dropout,
                    'max_length': args.max_length,
                }
            
            model_config['use_crf'] = getattr(args, 'use_crf', False)
            # FIX(2026-06-07): 训练时让 mdlm/diffusion 的 diffusion_steps 真正取自 --num_timesteps。
            # 否则 create_model 落到 mdlm.py 内置默认 10,time_emb 永远 10 槽,
            # 导致"按不同步数训练"的 T-sweep 全部失效(实测 --num_timesteps 3 仍训成 steps=10)。
            if args.model_type.lower() in ('mdlm', 'diffusion'):
                model_config['diffusion_steps'] = args.num_timesteps
            model = create_model(
                model_type=args.model_type,
                vocab_size=vocab_size,  # 使用根据模型类型获取的词汇表大小
                num_classes=num_classes,
                config=model_config
            )
            
        # 对BERT模型应用特殊的初始化修复
        if args.model_type == 'bert':
            print("🔧 应用BERT初始化修复...")
            import math
            
            # 先打印参数名称以调试
            fixed_count = 0
            for name, param in model.named_parameters():
                # 修复LayerNorm权重 - 关键修复
                if 'norm' in name and 'weight' in name:  # 匹配所有归一化层的权重
                    # 对所有LayerNorm使用小的初始化值
                    param.data.fill_(0.1)  # 使用0.1而不是1.0，大幅降低初始范数
                    fixed_count += 1
                    print(f"   修复LayerNorm: {name}, shape: {param.shape}")
                
                # 修复LayerNorm偏置
                elif 'norm' in name and 'bias' in name:
                    param.data.zero_()  # 偏置初始化为0
                    fixed_count += 1
                
                # 修复前馈网络权重
                elif 'feed_forward' in name and 'weight' in name:
                    # 使用更小的初始化
                    fan_in, fan_out = param.size(1), param.size(0)
                    std = 0.01  # 使用固定的小std值
                    param.data.normal_(mean=0.0, std=std)
                    fixed_count += 1
                    print(f"   修复前馈网络: {name}, std: {std}")
                
                # 修复注意力投影权重
                elif ('q_proj' in name or 'k_proj' in name or 'v_proj' in name or 'out_proj' in name) and 'weight' in name:
                    # 注意力层也使用更小的初始化
                    nn.init.normal_(param, mean=0.0, std=0.01)
                    fixed_count += 1
                    print(f"   修复注意力投影: {name}")
                
                # 修复嵌入层
                elif 'embedding' in name and 'weight' in name:
                    nn.init.normal_(param, mean=0.0, std=0.01)
                    fixed_count += 1
                    print(f"   修复嵌入层: {name}")
            
            print(f"   ✅ 修复了 {fixed_count} 个参数的初始化")
            
            # 再次检查参数范数
            print("🔍 重新检查参数范数...")
            param_norms = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param_norm = param.data.norm(2).item()
                    param_norms.append(param_norm)
                    if param_norm > 5.0:
                        print(f"   ⚠️ {name}: {param_norm:.4f}")
            avg_norm = sum(param_norms) / len(param_norms) if param_norms else 0
            print(f"   修复后平均参数范数: {avg_norm:.4f}")
        
        # 计算模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"🔢 模型参数:")
        print(f"   总参数量: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
        print(f"   模型大小: {total_params * 4 / 1024**2:.1f} MB")
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return
    
    # 创建训练器
    print("🎯 创建训练器...")
    # 使用统一的TensorBoard目录
    log_dir = tensorboard_dir
    # 计算总训练步数用于warmup
    total_training_steps = len(train_loader) * args.num_epochs
    warmup_steps = min(1000, int(0.1 * total_training_steps))  # warmup占总步数的10%或1000步
    
    trainer = ModelTrainer(
        model, device, num_classes,
        args.learning_rate,
        weight_decay=args.weight_decay,
        log_dir=log_dir,
        class_weights=class_weights,
        use_warmup=True,  # 启用warmup
        warmup_steps=warmup_steps,
        use_scheduler=True,
        total_steps=total_training_steps
    )
    trainer.patience = args.patience  # apply --patience CLI override
    
    # timestamp已经在创建输出目录时生成了，不需要重新生成
    
    if args.test_only:
        # 测试模式：直接加载模型
        print("🧪 跳过训练，直接加载模型进行测试...")
        save_path = args.model_path
        best_accuracy = 0.0  # 测试模式下没有验证准确率
        training_time = 0
        
        # 先加载checkpoint以获取模型配置
        print(f"📥 加载checkpoint: {save_path}")
        checkpoint = torch.load(save_path, map_location=device)
        
        # 从checkpoint中获取模型配置（如果存在）
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
            print(f"📝 从checkpoint读取模型配置: {model_config}")
            
            # 重新创建模型（使用正确的配置）
            model = create_model(
                model_type=args.model_type,
                vocab_size=vocab_size,
                num_classes=num_classes,
                config=model_config
            )
            model = model.to(device)
            
            # 重新计算模型参数
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"🔢 重新创建的模型参数:")
            print(f"   总参数量: {total_params:,}")
            print(f"   可训练参数: {trainable_params:,}")
            print(f"   模型大小: {total_params * 4 / 1024 / 1024:.1f} MB")
        else:
            # 如果checkpoint中没有配置，尝试从模型state_dict推断
            print("⚠️ Checkpoint中没有模型配置，尝试从state_dict推断...")
            state_dict = checkpoint['model_state_dict']
            
            # 根据模型类型进行不同的推断
            if args.model_type.lower() in ['mdlm', 'diffusion']:
                # MDLM模型的推断逻辑
                d_model = None
                max_length = args.max_length  # 默认值
                
                # 推断d_model和num_layers
                for key in state_dict.keys():
                    if 'embeddings.weight' in key:
                        d_model = state_dict[key].shape[1]
                        print(f"   推断d_model = {d_model}")
                        break
                
                # 推断max_length (从位置编码推断)
                if 'pos_embeddings.weight' in state_dict:
                    inferred_max_length = state_dict['pos_embeddings.weight'].shape[0]
                    if inferred_max_length != max_length:
                        print(f"   推断max_length = {inferred_max_length} (从位置编码推断)")
                        max_length = inferred_max_length
                        args.max_length = max_length  # 更新args中的值
                
                # 如果没有找到d_model，使用默认值
                if d_model is None:
                    d_model = 256  # 默认值
                    print(f"   使用默认d_model = {d_model}")
                
                # 推断num_classes（从输出层推断）
                # MDLM模型的统一词汇表大小 = vocab_size + num_classes + 4
                inferred_num_classes = num_classes  # 默认使用当前数据集的值
                if 'output_projection.weight' in state_dict:
                    # MDLM模型：output_projection的大小就是统一词汇表大小
                    unified_vocab_size = state_dict['output_projection.weight'].shape[0]
                    # 统一词汇表 = vocab_size(26) + num_classes + 4个特殊标记
                    inferred_num_classes = unified_vocab_size - vocab_size - 4
                    print(f"   推断num_classes = {inferred_num_classes} (从统一词汇表大小{unified_vocab_size}推断)")
                    # 使用推断出的num_classes而不是当前数据集的
                    num_classes = inferred_num_classes
                
                # 计算transformer层数
                layer_keys = [k for k in state_dict.keys() if 'transformer.layers.' in k]
                layer_indices = set()
                for key in layer_keys:
                    if 'transformer.layers.' in key:
                        try:
                            layer_idx = int(key.split('transformer.layers.')[1].split('.')[0])
                            layer_indices.add(layer_idx)
                        except:
                            pass
                num_layers = len(layer_indices)
                print(f"   推断num_layers = {num_layers}")
                
                # 重新创建模型配置
                model_config = {
                    'd_model': d_model,
                    'num_layers': num_layers,
                    'num_heads': 8,  # 默认值，因为无法从state_dict推断
                    'dropout': args.dropout,
                    'max_length': max_length,  # 使用推断出的max_length
                }
            else:
                # 其他模型（LSTM, Transformer等）使用默认配置或不需要特殊配置
                print(f"   {args.model_type}模型使用默认配置")
                model_config = {}
                
                # 对于LSTM模型，推断层数
                if args.model_type.lower() in ['lstm', 'bilstm']:
                    # 查找LSTM层数
                    lstm_layers = set()
                    for key in state_dict.keys():
                        if 'lstm.weight_ih_l' in key:
                            # 提取层号
                            try:
                                layer_num = int(key.split('lstm.weight_ih_l')[1].split('_')[0])
                                lstm_layers.add(layer_num)
                            except:
                                pass
                    num_lstm_layers = len(lstm_layers)
                    if num_lstm_layers > 0:
                        print(f"   推断LSTM层数 = {num_lstm_layers}")
                        model_config['num_layers'] = num_lstm_layers
                
                # 对于所有模型，推断num_classes
                # 查找最后的分类器层（通常是最大索引的层）
                classifier_layers = {}
                for key in state_dict.keys():
                    if 'classifier' in key and 'weight' in key and len(state_dict[key].shape) == 2:
                        # 提取层索引（如果有）
                        try:
                            # 尝试提取数字，如 classifier.classifier.5.weight
                            import re
                            match = re.search(r'classifier\.(\d+)\.weight', key)
                            if match:
                                layer_idx = int(match.group(1))
                                classifier_layers[layer_idx] = state_dict[key].shape[0]
                            elif 'classifier' in key:
                                # 没有索引的情况，使用-1作为默认索引
                                classifier_layers[-1] = state_dict[key].shape[0]
                        except:
                            pass
                
                # 使用最大索引的分类器层的输出维度作为num_classes
                if classifier_layers:
                    max_idx = max(classifier_layers.keys())
                    num_classes = classifier_layers[max_idx]
                    print(f"   从分类器层推断num_classes = {num_classes}")
            
            # 重新创建模型（使用推断出的配置）
            model = create_model(
                model_type=args.model_type,
                vocab_size=vocab_size,
                num_classes=num_classes,
                config=model_config
            )
            model = model.to(device)
            
            # 重新计算模型参数
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"🔢 重新创建的模型参数:")
            print(f"   总参数量: {total_params:,}")
            print(f"   可训练参数: {trainable_params:,}")
            print(f"   模型大小: {total_params * 4 / 1024 / 1024:.1f} MB")
        
        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ 模型权重加载成功")
        
        # 从模型文件名中提取时间戳（如果有）
        model_filename = os.path.basename(save_path)
        import re
        timestamp_match = re.search(r'(\d{8}_\d{6})', model_filename)
        if timestamp_match:
            timestamp = timestamp_match.group(1)
            print(f"📝 使用模型文件的时间戳: {timestamp}")
    else:
        # 训练模式
        print("🚀 开始训练...")
        start_time = time.time()
        # 模型文件直接保存在输出目录根目录
        save_path = os.path.join(output_dir, f'{args.model_type}_model.pt')
        
        # 训练模型
        best_accuracy = trainer.train(train_loader, val_loader, args.num_epochs, save_path)
        
        # 记录训练时间
        training_time = time.time() - start_time
        print(f"⏱️  训练耗时: {training_time/60:.1f} 分钟")
    
    try:
        if not args.test_only:
            # 绘制训练历史
            # 训练历史图表保存到plots子目录
            plot_path = os.path.join(plots_dir, f'{args.model_type}_training_history.png')
            trainer.plot_training_history(plot_path)
            
            # 加载最佳模型进行最终评估
            print("\n📥 加载最佳模型进行最终评估...")
            checkpoint = torch.load(save_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # 测试集评估
        # 所有模型都使用return_per_sample=True以保持test.in的顺序
        test_accuracy, test_report, test_predictions, test_labels, test_correct, test_total, per_sample_preds = evaluate_model(
            model, test_loader, device, num_classes, return_per_sample=True
        )
        
        # 生成并保存测试样本输出（已注释，不生成该文件）
        # test_dir = os.path.join(os.path.dirname(args.save_dir), 'test')
        # if not os.path.exists(test_dir):
        #     os.makedirs(test_dir)
        # sample_outputs_path = os.path.join(test_dir, f'{args.model_type}_sample_outputs_{timestamp}.json')
        # generate_test_sample_outputs(model, test_loader, device, num_samples=10, save_path=sample_outputs_path)
        
        # 打印最终测试结果
        print(f"   - 🎯 最终测试结果:")
        print(f"   - ✅ 测试准确率: {test_accuracy:.4f} ({test_correct:,}/{test_total:,})")
        
        # 显示分类报告的主要指标
        if 'weighted avg' in test_report:
            weighted_avg = test_report['weighted avg']
        
        # 保存详细结果
        results = {
            'model_type': args.model_type,
            'training_time_minutes': training_time / 60,
            'best_val_accuracy': best_accuracy,
            'test_accuracy': test_accuracy,
            'test_report': test_report,
            'num_classes': num_classes,
            'vocab_size': vocab_size,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'hyperparameters': vars(args),
            'timestamp': timestamp,
            'device': str(device)
        }
        
        # 保存结果文件和词汇表（仅在训练模式下）
        if not args.test_only:
            # 训练模式：保存results和vocab文件
            # 训练结果保存到results子目录（带时间戳）
            results_path = os.path.join(results_dir, f'{args.model_type}_results_{timestamp}.json')
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # 保存词汇表
            # 词汇表保存到results子目录（带时间戳）
            vocab_path = os.path.join(results_dir, f'vocab_{timestamp}.json')
            with open(vocab_path, 'w', encoding='utf-8') as f:
                json.dump(char_to_idx, f, indent=2, ensure_ascii=False)
        else:
            # test_only模式：保存测试结果到新文件，避免覆盖原有文件
            # 测试结果保存到results子目录（带时间戳）
            test_results_path = os.path.join(results_dir, f'{args.model_type}_test_results_{timestamp}.json')
            with open(test_results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"📊 测试结果已保存到: {test_results_path}")
        
        # 所有测试相关文件都保存到results子目录
        test_dir = results_dir
        test_timestamp = timestamp
        print(f"📁 测试结果将保存到: {test_dir}")
        
        # 保存测试预测结果
        # 预测结果保存到results子目录（带时间戳）
        test_predictions_path = os.path.join(results_dir, f'{args.model_type}_predictions_{timestamp}.out')
        
        # 所有模型都使用per_sample_preds以保持test.in的顺序
        # 读取测试数据以获取行数（用于验证）
        test_input_path = os.path.join(actual_data_dir, 'test.in')
        with open(test_input_path, 'r', encoding='utf-8') as f:
            test_lines = f.readlines()
        
        # 检查最长序列长度（仅用于日志输出）
        max_seq_length = max(len(line.rstrip('\n\r')) for line in test_lines) if test_lines else 0
        
        # 使用evaluate_model返回的per_sample_preds（已经按原始顺序排列）
        print(f"\n📝 保存{args.model_type}模型预测结果（最长序列: {max_seq_length}字符，使用批量评估结果）...")
        
        # 将per_sample_preds转换为字符串格式
        predictions_lines = []
        for pred_list in per_sample_preds:
            # 将预测列表转换为空格分隔的字符串
            predictions_lines.append(' '.join(map(str, pred_list)))
        
        # 保存预测结果
        with open(test_predictions_path, 'w', encoding='utf-8') as f:
            for line in predictions_lines:
                f.write(line + '\n')
        
        print(f"✅ 预测结果已保存到: {test_predictions_path}")
        print(f"   预测行数: {len(predictions_lines)}")
        print(f"   原始行数: {len(test_lines)}")
        
        # 验证行数匹配
        if len(predictions_lines) != len(test_lines):
            print(f"⚠️ 警告：预测行数({len(predictions_lines)})与原始行数({len(test_lines)})不匹配！")
        
        # 删除旧的else分支，因为现在所有模型都使用相同的方法
        if False:  # 永远不会执行，保留代码供参考
            # 直接保存原始预测结果（与evaluate_model返回的test_predictions对应）
            # 注意：test_predictions是展平的一维列表，需要重新组织成每行对应一个样本
            with open(test_predictions_path, 'w', encoding='utf-8') as f:
                # 获取测试数据集以了解每个样本的长度
                test_dataset = test_loader.dataset
                current_idx = 0
                
                for sample_idx in range(len(test_dataset)):
                    # 获取当前样本的长度
                    # SyriacDataset返回三个值：(input, labels, attention_mask)
                    sample_data = test_dataset[sample_idx]
                    if len(sample_data) == 3:
                        _, labels, attention_mask = sample_data
                    else:
                        _, labels = sample_data
                        attention_mask = torch.ones_like(labels)
                    
                    # 重要修复：只计算非填充位置（labels != -100）的数量
                    # 因为evaluate_model中过滤了填充位置
                    valid_mask = labels != -100
                    sample_length = valid_mask.sum().item()
                    
                    # 提取对应的预测结果
                    sample_predictions = []
                    for _ in range(sample_length):
                        if current_idx < len(test_predictions):
                            sample_predictions.append(str(test_predictions[current_idx]))
                            current_idx += 1
                        else:
                            break
                    
                    # 写入一行预测结果
                    if sample_predictions:
                        f.write(' '.join(sample_predictions) + '\n')
                    
                    if current_idx >= len(test_predictions):
                        break
        
        # 保存测试详细报告
        # 测试报告保存到results子目录（带时间戳）
        test_report_path = os.path.join(results_dir, f'{args.model_type}_test_report_{timestamp}.json')
        test_report_data = {
            'model_type': args.model_type,
            'timestamp': test_timestamp,
            'test_accuracy': test_accuracy,
            'test_report': test_report,
            'total_test_samples': len(test_labels),
            'total_predictions': len(test_predictions),
            'num_classes': num_classes,
            'model_path': save_path,
            'data_dir': actual_data_dir
        }
        with open(test_report_path, 'w', encoding='utf-8') as f:
            json.dump(test_report_data, f, indent=2, ensure_ascii=False)
        
        # 打印保存路径
        print(f"\n💾 文件保存路径:")
        print(f"   输出目录: {output_dir}")
        if not args.test_only:
            print(f"   ├── {args.model_type}_model.pt")
            print(f"   ├── data/")
            print(f"   │   ├── patterns.csv")
            print(f"   │   ├── test.in")
            print(f"   │   ├── test.out")
            print(f"   │   └── dataset_info.json")
            print(f"   ├── tensorboard/")
            print(f"   ├── plots/")
            print(f"   │   └── {args.model_type}_training_history.png")
            print(f"   └── results/")
            print(f"       ├── {args.model_type}_results_{timestamp}.json")
            print(f"       ├── vocab_{timestamp}.json")
            print(f"       ├── {args.model_type}_predictions_{timestamp}.out")
            print(f"       └── {args.model_type}_test_report_{timestamp}.json")
        else:
            print(f"   ├── data/")
            print(f"   │   ├── patterns.csv")
            print(f"   │   ├── test.in")
            print(f"   │   ├── test.out")
            print(f"   │   └── dataset_info.json")
            print(f"   ├── tensorboard/")
            print(f"   └── results/")
            print(f"       ├── {args.model_type}_test_results_{timestamp}.json")
            print(f"       ├── {args.model_type}_predictions_{timestamp}.out")
            print(f"       └── {args.model_type}_test_report_{timestamp}.json")
        
        if args.test_only:
            print(f"\n🎉 {args.model_type}模型测试完成！")
        else:
            print(f"\n🎉 {args.model_type}模型训练完成！")
        
    except KeyboardInterrupt:
        print("\n⏹️  训练被用户中断")
        return
    except Exception as e:
        print(f"\n❌ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return
    finally:
        # 清理内存
        if device_manager and hasattr(device_manager, 'clean_memory'):
            device_manager.clean_memory()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            if hasattr(torch.backends.mps, 'empty_cache'):
                torch.backends.mps.empty_cache()
        print("🧹 内存清理完成")

if __name__ == "__main__":
    main()