"""
超参数优化(HPO)组件
专门针对Transformer和MDLM模型的超参数调优

支持多种优化算法：
- Optuna (贝叶斯优化)
- Hyperopt (TPE, Random Search)
- Grid Search
- Random Search

特性：
- 智能搜索空间定义
- 早停机制
- 资源管理
- 多目标优化
- 实验跟踪
"""

import os
import json
import time
import math
import random
import logging
import warnings
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# HPO库导入 (可选依赖)
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
except Exception as e:
    # 记录非ImportError异常但仍设置为不可用
    print(f"警告: optuna导入时遇到异常: {type(e).__name__}: {e}")
    OPTUNA_AVAILABLE = False

try:
    import hyperopt
    from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from ..core import log_info, log_warning, ModelError


@dataclass
class HPOConfig:
    """HPO配置类"""
    # 基础设置
    study_name: str = "syriac_hpo"
    n_trials: int = 50
    timeout: Optional[int] = None  # 秒
    n_jobs: int = 1

    # 优化算法
    optimizer: str = "optuna"  # optuna, hyperopt, grid, random
    sampler: str = "tpe"  # tpe, random, cmaes, grid

    # 目标优化
    direction: str = "maximize"  # maximize, minimize
    metric: str = "val_accuracy"  # val_accuracy, val_f1, val_loss

    # 早停设置
    enable_pruning: bool = True
    pruning_patience: int = 5
    min_trials_for_pruning: int = 10

    # 资源管理
    max_epochs_per_trial: int = 20
    min_epochs_per_trial: int = 5
    memory_limit_gb: Optional[float] = None
    gpu_memory_fraction: float = 0.8

    # 实验跟踪
    save_best_model: bool = True
    save_all_trials: bool = False
    log_to_wandb: bool = False

    # 搜索空间限制
    max_model_size_mb: float = 500  # 最大模型大小
    max_training_time_minutes: float = 60  # 单次训练最大时间

    def __post_init__(self):
        """验证配置"""
        if self.optimizer not in ['optuna', 'hyperopt', 'grid', 'random']:
            raise ValueError(f"不支持的优化器: {self.optimizer}")

        if self.optimizer == 'optuna' and not OPTUNA_AVAILABLE:
            raise ImportError("需要安装optuna: pip install optuna")

        if self.optimizer == 'hyperopt' and not HYPEROPT_AVAILABLE:
            raise ImportError("需要安装hyperopt: pip install hyperopt")


@dataclass
class HPOSearchSpace:
    """HPO搜索空间定义 - 支持所有13种模型"""
    model_type: str  # 支持所有模型类型

    # 通用参数空间
    d_model: List[int] = None
    num_layers: List[int] = None
    num_heads: List[int] = None  # 对于支持的模型
    dropout: Tuple[float, float] = (0.0, 0.5)
    learning_rate: Tuple[float, float] = (1e-5, 1e-3)
    batch_size: List[int] = None
    weight_decay: Tuple[float, float] = (0.0, 0.1)

    # Transformer专用
    dim_feedforward: List[int] = None
    activation: List[str] = None
    layer_norm_eps: Tuple[float, float] = (1e-6, 1e-4)

    # MDLM专用
    num_timesteps: List[int] = None
    mask_ratio: Tuple[float, float] = (0.1, 0.5)
    diffusion_steps: List[int] = None
    max_length: List[int] = None

    # LSTM专用
    hidden_size: List[int] = None
    bidirectional: List[bool] = None

    # Mamba/BiMamba专用
    d_state: List[int] = None
    d_conv: List[int] = None
    expand: List[int] = None

    # RWKV专用
    use_dynamic_state: List[bool] = None
    max_sequence_length: List[int] = None

    # RetNet专用
    retention_heads: List[int] = None
    value_factor: List[float] = None

    # Switch Transformer专用
    num_experts: List[int] = None
    expert_capacity_factor: Tuple[float, float] = (1.0, 2.0)

    # BERT专用
    intermediate_size: List[int] = None
    hidden_act: List[str] = None

    def __post_init__(self):
        """设置所有模型类型的默认搜索空间"""
        # 通用默认值
        if self.batch_size is None:
            self.batch_size = [16, 32, 64]

        # 根据模型类型设置特定默认值
        if self.model_type in ['transformer', 'bert']:
            self._set_transformer_defaults()
        elif self.model_type == 'mdlm':
            self._set_mdlm_defaults()
        elif self.model_type == 'lstm':
            self._set_lstm_defaults()
        elif self.model_type in ['mamba', 'bimamba', 'mamba_large', 'bimamba_large', 'bimamba_xl']:
            self._set_mamba_defaults()
        elif self.model_type in ['rwkv7', 'rwkv7_large', 'rwkv7_efficient']:
            self._set_rwkv_defaults()
        elif self.model_type == 'retnet':
            self._set_retnet_defaults()
        elif self.model_type == 'switch':
            self._set_switch_defaults()
        else:
            self._set_generic_defaults()

    def _set_transformer_defaults(self):
        """Transformer和BERT模型默认值"""
        if self.d_model is None:
            self.d_model = [256, 384, 512, 640]
        if self.num_layers is None:
            self.num_layers = [3, 4, 6, 8]
        if self.num_heads is None:
            self.num_heads = [4, 6, 8, 12]
        if self.dim_feedforward is None:
            self.dim_feedforward = [1024, 1536, 2048]
        if self.activation is None:
            self.activation = ['relu', 'gelu', 'swish']
        if self.intermediate_size is None:
            self.intermediate_size = [1024, 2048, 3072]
        if self.hidden_act is None:
            self.hidden_act = ['relu', 'gelu', 'swish']

    def _set_mdlm_defaults(self):
        """MDLM模型默认值"""
        if self.d_model is None:
            self.d_model = [256, 384, 512]
        if self.num_layers is None:
            self.num_layers = [3, 4, 6, 8]
        if self.num_timesteps is None:
            self.num_timesteps = [10]
        if self.diffusion_steps is None:
            self.diffusion_steps = [5, 10, 20]
        if self.max_length is None:
            self.max_length = [64, 128, 256]

    def _set_lstm_defaults(self):
        """LSTM模型默认值"""
        if self.d_model is None:
            self.d_model = [256, 384, 512]
        if self.hidden_size is None:
            self.hidden_size = [256, 384, 512]
        if self.num_layers is None:
            self.num_layers = [2, 3, 4]
        if self.bidirectional is None:
            self.bidirectional = [True, False]

    def _set_mamba_defaults(self):
        """Mamba/BiMamba模型默认值"""
        if self.d_model is None:
            self.d_model = [256, 384, 512]
        if self.num_layers is None:
            self.num_layers = [4, 6, 8]
        if self.d_state is None:
            self.d_state = [16, 32, 64]
        if self.d_conv is None:
            self.d_conv = [4, 6, 8]
        if self.expand is None:
            self.expand = [2, 4]

    def _set_rwkv_defaults(self):
        """RWKV模型默认值"""
        if self.d_model is None:
            self.d_model = [256, 384, 512]
        if self.num_layers is None:
            self.num_layers = [4, 6, 8]
        if self.use_dynamic_state is None:
            self.use_dynamic_state = [True, False]
        if self.max_sequence_length is None:
            self.max_sequence_length = [128, 256, 512]

    def _set_retnet_defaults(self):
        """RetNet模型默认值"""
        if self.d_model is None:
            self.d_model = [256, 384, 512]
        if self.num_layers is None:
            self.num_layers = [4, 6, 8]
        if self.retention_heads is None:
            self.retention_heads = [4, 8, 12]
        if self.value_factor is None:
            self.value_factor = [1.0, 2.0]

    def _set_switch_defaults(self):
        """Switch Transformer模型默认值"""
        if self.d_model is None:
            self.d_model = [256, 384, 512]
        if self.num_layers is None:
            self.num_layers = [4, 6, 8]
        if self.num_experts is None:
            self.num_experts = [4, 8, 16]

    def _set_generic_defaults(self):
        """通用默认值"""
        if self.d_model is None:
            self.d_model = [256, 384, 512]
        if self.num_layers is None:
            self.num_layers = [3, 4, 6, 8]


class HPOObjective:
    """HPO目标函数"""

    def __init__(self,
                 config: HPOConfig,
                 search_space: HPOSearchSpace,
                 train_func: Callable,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 output_dir: str):
        self.config = config
        self.search_space = search_space
        self.train_func = train_func
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 跟踪最佳结果
        self.best_score = float('-inf') if config.direction == 'maximize' else float('inf')
        self.best_params = None
        self.trial_results = []

        # 设置日志
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger(f'hpo_{self.config.study_name}')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            # 文件处理器
            fh = logging.FileHandler(self.output_dir / 'hpo.log')
            fh.setLevel(logging.INFO)

            # 控制台处理器
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)

            # 格式化器
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)

            logger.addHandler(fh)
            logger.addHandler(ch)

        return logger

    def suggest_hyperparameters(self, trial) -> Dict[str, Any]:
        """建议超参数"""
        params = {}

        # 通用参数
        params['d_model'] = trial.suggest_categorical('d_model', self.search_space.d_model)
        params['num_layers'] = trial.suggest_categorical('num_layers', self.search_space.num_layers)
        params['dropout'] = trial.suggest_float('dropout', *self.search_space.dropout)
        params['learning_rate'] = trial.suggest_loguniform('learning_rate', *self.search_space.learning_rate)
        params['batch_size'] = trial.suggest_categorical('batch_size', self.search_space.batch_size)
        params['weight_decay'] = trial.suggest_float('weight_decay', *self.search_space.weight_decay)

        # 模型特定参数
        if self.search_space.model_type in ['transformer', 'bert']:
            if self.search_space.num_heads:
                params['num_heads'] = trial.suggest_categorical('num_heads', self.search_space.num_heads)
            if self.search_space.dim_feedforward:
                params['dim_feedforward'] = trial.suggest_categorical('dim_feedforward', self.search_space.dim_feedforward)
            if self.search_space.activation:
                params['activation'] = trial.suggest_categorical('activation', self.search_space.activation)
            if self.search_space.layer_norm_eps:
                params['layer_norm_eps'] = trial.suggest_loguniform('layer_norm_eps', *self.search_space.layer_norm_eps)

            # 确保d_model可被num_heads整除
            if 'num_heads' in params:
                while params['d_model'] % params['num_heads'] != 0:
                    params['num_heads'] = trial.suggest_categorical('num_heads_retry', self.search_space.num_heads)

            # BERT特定参数
            if self.search_space.model_type == 'bert':
                if self.search_space.intermediate_size:
                    params['intermediate_size'] = trial.suggest_categorical('intermediate_size', self.search_space.intermediate_size)
                if self.search_space.hidden_act:
                    params['hidden_act'] = trial.suggest_categorical('hidden_act', self.search_space.hidden_act)

        elif self.search_space.model_type == 'mdlm':
            if self.search_space.num_timesteps:
                params['num_timesteps'] = trial.suggest_categorical('num_timesteps', self.search_space.num_timesteps)
            if self.search_space.mask_ratio:
                params['mask_ratio'] = trial.suggest_float('mask_ratio', *self.search_space.mask_ratio)
            if self.search_space.diffusion_steps:
                params['diffusion_steps'] = trial.suggest_categorical('diffusion_steps', self.search_space.diffusion_steps)
            if self.search_space.max_length:
                params['max_length'] = trial.suggest_categorical('max_length', self.search_space.max_length)

        elif self.search_space.model_type == 'lstm':
            if self.search_space.hidden_size:
                params['hidden_size'] = trial.suggest_categorical('hidden_size', self.search_space.hidden_size)
            if self.search_space.bidirectional:
                params['bidirectional'] = trial.suggest_categorical('bidirectional', self.search_space.bidirectional)

        elif self.search_space.model_type in ['mamba', 'bimamba', 'mamba_large', 'bimamba_large', 'bimamba_xl']:
            if self.search_space.d_state:
                params['d_state'] = trial.suggest_categorical('d_state', self.search_space.d_state)
            if self.search_space.d_conv:
                params['d_conv'] = trial.suggest_categorical('d_conv', self.search_space.d_conv)
            if self.search_space.expand:
                params['expand'] = trial.suggest_categorical('expand', self.search_space.expand)

        elif self.search_space.model_type in ['rwkv7', 'rwkv7_large', 'rwkv7_efficient']:
            if self.search_space.use_dynamic_state:
                params['use_dynamic_state'] = trial.suggest_categorical('use_dynamic_state', self.search_space.use_dynamic_state)
            if self.search_space.max_sequence_length:
                params['max_sequence_length'] = trial.suggest_categorical('max_sequence_length', self.search_space.max_sequence_length)

        elif self.search_space.model_type == 'retnet':
            if self.search_space.retention_heads:
                params['retention_heads'] = trial.suggest_categorical('retention_heads', self.search_space.retention_heads)
            if self.search_space.value_factor:
                params['value_factor'] = trial.suggest_categorical('value_factor', self.search_space.value_factor)

        elif self.search_space.model_type == 'switch':
            if self.search_space.num_experts:
                params['num_experts'] = trial.suggest_categorical('num_experts', self.search_space.num_experts)
            if self.search_space.expert_capacity_factor:
                params['expert_capacity_factor'] = trial.suggest_float('expert_capacity_factor', *self.search_space.expert_capacity_factor)

        # 参数映射：将通用的d_model映射到模型特定参数
        params = self._map_model_params(params)

        return params

    def _map_model_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """将通用参数映射到模型特定参数"""
        mapped_params = params.copy()

        # 对于LSTM、BiLSTM和基于mamba的模型，将d_model映射到合适的参数
        if self.search_space.model_type in ['lstm', 'bilstm']:
            # LSTM使用embedding_dim和hidden_size而不是d_model
            if 'd_model' in mapped_params:
                d_model = mapped_params.pop('d_model')  # 移除d_model
                mapped_params['embedding_dim'] = min(128, d_model // 2)  # 嵌入维度通常较小
                mapped_params['hidden_size'] = d_model  # hidden_size对应d_model
                mapped_params['bidirectional'] = True  # 启用双向LSTM
                mapped_params['use_attention'] = False  # 暂时禁用注意力机制避免维度问题
                mapped_params['attention_heads'] = 8

        elif self.search_space.model_type in ['mamba', 'bimamba', 'mamba_large', 'bimamba_large', 'bimamba_xl']:
            # Mamba模型使用d_model，但需要确保与其他参数兼容
            pass  # d_model保持不变

        elif self.search_space.model_type in ['rwkv7', 'rwkv7_large', 'rwkv7_efficient']:
            # RWKV模型使用d_model，保持不变
            pass  # d_model保持不变

        return mapped_params

    def __call__(self, trial) -> float:
        """执行一次HPO试验"""
        trial_id = trial.number if hasattr(trial, 'number') else len(self.trial_results)

        try:
            # 建议超参数
            params = self.suggest_hyperparameters(trial)

            self.logger.info(f"Trial {trial_id}: 开始训练，参数: {params}")

            # 估算模型大小
            estimated_size = self._estimate_model_size(params)
            if estimated_size > self.config.max_model_size_mb:
                self.logger.warning(f"Trial {trial_id}: 模型太大 ({estimated_size:.1f}MB)，跳过")
                return float('-inf') if self.config.direction == 'maximize' else float('inf')

            # 训练开始时间
            start_time = time.time()

            # 执行训练
            result = self.train_func(
                model_type=self.search_space.model_type,
                params=params,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                max_epochs=self.config.max_epochs_per_trial,
                trial_id=trial_id,
                trial=trial if self.config.enable_pruning else None
            )

            training_time = time.time() - start_time

            # 检查训练时间限制
            if training_time > self.config.max_training_time_minutes * 60:
                self.logger.warning(f"Trial {trial_id}: 训练时间过长 ({training_time/60:.1f}分钟)")

            # 获取目标指标
            score = result.get(self.config.metric, 0.0)

            # 记录试验结果
            trial_result = {
                'trial_id': trial_id,
                'params': params,
                'score': score,
                'training_time': training_time,
                'model_size_mb': estimated_size,
                'all_metrics': result
            }
            self.trial_results.append(trial_result)

            # 更新最佳结果
            is_better = (self.config.direction == 'maximize' and score > self.best_score) or \
                       (self.config.direction == 'minimize' and score < self.best_score)

            if is_better:
                self.best_score = score
                self.best_params = params.copy()
                self.logger.info(f"Trial {trial_id}: 新的最佳结果! {self.config.metric}={score:.4f}")

                # 保存最佳模型
                if self.config.save_best_model:
                    self._save_best_result(trial_result)

            # 保存试验历史
            self._save_trial_history()

            # Wandb日志
            if self.config.log_to_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'trial_id': trial_id,
                    'score': score,
                    'training_time': training_time,
                    **params,
                    **result
                })

            self.logger.info(f"Trial {trial_id}: 完成，{self.config.metric}={score:.4f}, 用时={training_time/60:.1f}分钟")

            return score

        except Exception as e:
            self.logger.error(f"Trial {trial_id}: 训练失败 - {e}")
            return float('-inf') if self.config.direction == 'maximize' else float('inf')

    def _estimate_model_size(self, params: Dict[str, Any]) -> float:
        """估算模型大小(MB)"""
        vocab_size = 26  # S2数据集默认
        num_classes = 329

        if self.search_space.model_type == 'transformer':
            # Transformer参数估算
            d_model = params['d_model']
            num_layers = params['num_layers']
            dim_feedforward = params['dim_feedforward']

            # 嵌入层
            embed_params = vocab_size * d_model

            # Transformer层
            # 自注意力: Q, K, V, O
            attention_params = 4 * d_model * d_model * num_layers
            # FFN: 两个线性层
            ffn_params = 2 * d_model * dim_feedforward * num_layers
            # Layer Norm
            ln_params = 2 * d_model * num_layers

            # 分类头
            classifier_params = d_model * num_classes

            total_params = embed_params + attention_params + ffn_params + ln_params + classifier_params

        elif self.search_space.model_type == 'mdlm':
            # MDLM参数估算（简化）
            d_model = params['d_model']
            num_layers = params['num_layers']
            vocab_size = 40  # MDLM使用扩展词汇表

            # 统一词汇表大小计算
            unified_vocab_size = vocab_size + num_classes + 4  # 特殊标记

            # 嵌入层
            embed_params = unified_vocab_size * d_model

            # Transformer层（简化估算）
            layer_params = 4 * d_model * d_model * num_layers  # 自注意力
            layer_params += 2 * d_model * (4 * d_model) * num_layers  # FFN
            layer_params += 2 * d_model * num_layers  # Layer Norm

            # 分类头
            classifier_params = d_model * num_classes

            total_params = embed_params + layer_params + classifier_params

        elif self.search_space.model_type in ['mamba', 'mamba_large']:
            # Mamba参数估算
            d_model = params['d_model']
            num_layers = params['num_layers']
            d_state = params.get('d_state', 16)
            d_conv = params.get('d_conv', 4)
            expand = params.get('expand', 2)

            # 嵌入层
            embed_params = vocab_size * d_model

            # Mamba块参数
            # 投影层: in_proj (d_model -> expand*d_model*2)
            proj_params = d_model * (expand * d_model * 2) * num_layers
            # 卷积层: conv1d (expand*d_model, d_conv)
            conv_params = (expand * d_model) * d_conv * num_layers
            # SSM参数: A, B, C, dt_proj等
            ssm_params = (d_state * expand * d_model) * 3 * num_layers  # A, B, C矩阵
            ssm_params += (d_model * expand * d_model) * num_layers  # dt_proj

            # 分类头
            classifier_params = d_model * num_classes

            total_params = embed_params + proj_params + conv_params + ssm_params + classifier_params

        elif self.search_space.model_type in ['bimamba', 'bimamba_large', 'bimamba_xl']:
            # BiMamba参数估算（双向Mamba）
            d_model = params['d_model']
            num_layers = params['num_layers']
            d_state = params.get('d_state', 16)
            d_conv = params.get('d_conv', 4)
            expand = params.get('expand', 2)

            # 嵌入层
            embed_params = vocab_size * d_model

            # BiMamba块参数（双向，所以参数量大约是单向的2倍）
            single_mamba_params = d_model * (expand * d_model * 2) * num_layers  # 投影
            single_mamba_params += (expand * d_model) * d_conv * num_layers  # 卷积
            single_mamba_params += (d_state * expand * d_model) * 3 * num_layers  # SSM
            single_mamba_params += (d_model * expand * d_model) * num_layers  # dt_proj

            # 双向参数
            bimamba_params = single_mamba_params * 2

            # 分类头
            classifier_params = d_model * num_classes

            total_params = embed_params + bimamba_params + classifier_params

        elif self.search_space.model_type in ['rwkv7', 'rwkv7_large', 'rwkv7_efficient']:
            # RWKV7参数估算
            d_model = params['d_model']
            num_layers = params['num_layers']
            ffn_size = params.get('ffn_size', d_model * 4)

            # 嵌入层
            embed_params = vocab_size * d_model

            # RWKV7块参数
            # Time mixing: key, value, receptance, output投影
            time_mixing_params = 4 * d_model * d_model * num_layers
            # Channel mixing: key, receptance, value投影
            channel_mixing_params = (d_model * ffn_size + d_model * d_model + ffn_size * d_model) * num_layers
            # Layer norm
            ln_params = 2 * d_model * num_layers
            # 时间衰减和其他参数
            time_params = d_model * 2 * num_layers

            # 分类头
            classifier_params = d_model * num_classes

            total_params = embed_params + time_mixing_params + channel_mixing_params + ln_params + time_params + classifier_params

        elif self.search_space.model_type == 'switch':
            # Switch Transformer参数估算
            d_model = params['d_model']
            num_layers = params['num_layers']
            dim_feedforward = params['dim_feedforward']
            num_experts = params.get('num_experts', 8)

            # 嵌入层
            embed_params = vocab_size * d_model

            # 自注意力（与标准Transformer相同）
            attention_params = 4 * d_model * d_model * num_layers

            # Switch FFN（每个expert都有独立的FFN）
            expert_params = num_experts * 2 * d_model * dim_feedforward * num_layers
            # 门控网络
            gate_params = d_model * num_experts * num_layers

            # Layer Norm
            ln_params = 2 * d_model * num_layers

            # 分类头
            classifier_params = d_model * num_classes

            total_params = embed_params + attention_params + expert_params + gate_params + ln_params + classifier_params

        else:
            # 对未知模型类型使用通用估算
            d_model = params.get('d_model', 384)
            num_layers = params.get('num_layers', 6)

            # 简化估算：嵌入 + 层参数 + 分类头
            embed_params = vocab_size * d_model
            layer_params = 8 * d_model * d_model * num_layers  # 保守估算
            classifier_params = d_model * num_classes

            total_params = embed_params + layer_params + classifier_params

        # 转换为MB（假设每个参数4字节）
        size_mb = total_params * 4 / (1024 * 1024)
        return size_mb

    def _save_best_result(self, trial_result: Dict[str, Any]):
        """保存最佳结果"""
        best_result_path = self.output_dir / 'best_result.json'
        with open(best_result_path, 'w', encoding='utf-8') as f:
            json.dump(trial_result, f, indent=2, ensure_ascii=False)

    def _save_trial_history(self):
        """保存试验历史"""
        history_path = self.output_dir / 'trial_history.json'
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.trial_results, f, indent=2, ensure_ascii=False)


class HPOOptimizer:
    """HPO优化器主类"""

    def __init__(self,
                 config: HPOConfig,
                 search_space: HPOSearchSpace,
                 train_func: Callable,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 output_dir: str):
        self.config = config
        self.search_space = search_space
        self.objective = HPOObjective(
            config, search_space, train_func,
            train_loader, val_loader, output_dir
        )

        log_info(f"HPO优化器初始化完成:")
        log_info(f"  - 模型类型: {search_space.model_type}")
        log_info(f"  - 优化算法: {config.optimizer}")
        log_info(f"  - 试验次数: {config.n_trials}")
        log_info(f"  - 目标指标: {config.metric} ({config.direction})")

    def optimize(self) -> Dict[str, Any]:
        """开始优化"""
        if self.config.optimizer == 'optuna':
            return self._optimize_with_optuna()
        elif self.config.optimizer == 'hyperopt':
            return self._optimize_with_hyperopt()
        elif self.config.optimizer == 'random':
            return self._optimize_with_random_search()
        elif self.config.optimizer == 'grid':
            return self._optimize_with_grid_search()
        else:
            raise ValueError(f"不支持的优化器: {self.config.optimizer}")

    def _optimize_with_optuna(self) -> Dict[str, Any]:
        """使用Optuna优化"""
        if not OPTUNA_AVAILABLE:
            raise ImportError("需要安装optuna: pip install optuna")

        # 创建study
        direction = 'maximize' if self.config.direction == 'maximize' else 'minimize'

        # 选择sampler
        if self.config.sampler == 'tpe':
            sampler = optuna.samplers.TPESampler()
        elif self.config.sampler == 'random':
            sampler = optuna.samplers.RandomSampler()
        elif self.config.sampler == 'cmaes':
            sampler = optuna.samplers.CmaEsSampler()
        else:
            sampler = optuna.samplers.TPESampler()

        # 创建pruner（如果启用）
        pruner = None
        if self.config.enable_pruning:
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=self.config.min_trials_for_pruning,
                n_warmup_steps=self.config.pruning_patience
            )

        study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            study_name=self.config.study_name
        )

        # 优化
        log_info(f"开始Optuna优化，目标: {self.config.metric} ({direction})")

        study.optimize(
            self.objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs
        )

        # 返回结果
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'best_trial': study.best_trial,
            'n_trials': len(study.trials),
            'study': study
        }

    def _optimize_with_hyperopt(self) -> Dict[str, Any]:
        """使用Hyperopt优化"""
        if not HYPEROPT_AVAILABLE:
            raise ImportError("需要安装hyperopt: pip install hyperopt")

        # 定义搜索空间
        space = self._build_hyperopt_space()

        # 创建trials对象
        trials = Trials()

        # 优化
        log_info(f"开始Hyperopt优化，目标: {self.config.metric}")

        # Hyperopt目标函数包装
        def hyperopt_objective(params):
            score = self.objective.suggest_hyperparameters_and_train(params)
            # Hyperopt总是最小化，所以需要转换
            if self.config.direction == 'maximize':
                score = -score
            return {'loss': score, 'status': STATUS_OK}

        best = fmin(
            fn=hyperopt_objective,
            space=space,
            algo=tpe.suggest,
            max_evals=self.config.n_trials,
            trials=trials
        )

        return {
            'best_params': best,
            'best_value': trials.best_trial['result']['loss'],
            'trials': trials,
            'n_trials': len(trials)
        }

    def _build_hyperopt_space(self) -> Dict:
        """构建Hyperopt搜索空间"""
        space = {
            'd_model': hp.choice('d_model', self.search_space.d_model),
            'num_layers': hp.choice('num_layers', self.search_space.num_layers),
            'dropout': hp.uniform('dropout', *self.search_space.dropout),
            'learning_rate': hp.loguniform('learning_rate',
                                         math.log(self.search_space.learning_rate[0]),
                                         math.log(self.search_space.learning_rate[1])),
            'batch_size': hp.choice('batch_size', self.search_space.batch_size),
            'weight_decay': hp.uniform('weight_decay', *self.search_space.weight_decay),
        }

        if self.search_space.model_type == 'transformer':
            space.update({
                'num_heads': hp.choice('num_heads', self.search_space.num_heads),
                'dim_feedforward': hp.choice('dim_feedforward', self.search_space.dim_feedforward),
                'activation': hp.choice('activation', self.search_space.activation),
                'layer_norm_eps': hp.loguniform('layer_norm_eps',
                                               math.log(self.search_space.layer_norm_eps[0]),
                                               math.log(self.search_space.layer_norm_eps[1]))
            })
        elif self.search_space.model_type == 'mdlm':
            space.update({
                'num_timesteps': hp.choice('num_timesteps', self.search_space.num_timesteps),
                'mask_ratio': hp.uniform('mask_ratio', *self.search_space.mask_ratio),
                'diffusion_steps': hp.choice('diffusion_steps', self.search_space.diffusion_steps),
                'max_length': hp.choice('max_length', self.search_space.max_length)
            })

        return space

    def _optimize_with_random_search(self) -> Dict[str, Any]:
        """随机搜索"""
        log_info(f"开始随机搜索优化，目标: {self.config.metric}")

        best_score = float('-inf') if self.config.direction == 'maximize' else float('inf')
        best_params = None

        for trial_id in range(self.config.n_trials):
            # 随机采样参数
            params = self._random_sample_params()

            # 模拟trial对象
            class RandomTrial:
                def __init__(self, trial_id, params):
                    self.number = trial_id
                    self._params = params

                def suggest_categorical(self, name, choices):
                    return self._params[name]

                def suggest_float(self, name, low, high):
                    return self._params[name]

                def suggest_loguniform(self, name, low, high):
                    return self._params[name]

            trial = RandomTrial(trial_id, params)
            score = self.objective(trial)

            # 更新最佳结果
            is_better = (self.config.direction == 'maximize' and score > best_score) or \
                       (self.config.direction == 'minimize' and score < best_score)

            if is_better:
                best_score = score
                best_params = params.copy()

        return {
            'best_params': best_params,
            'best_value': best_score,
            'n_trials': self.config.n_trials
        }

    def _random_sample_params(self) -> Dict[str, Any]:
        """随机采样参数"""
        params = {}

        # 通用参数
        params['d_model'] = random.choice(self.search_space.d_model)
        params['num_layers'] = random.choice(self.search_space.num_layers)
        params['dropout'] = random.uniform(*self.search_space.dropout)
        params['learning_rate'] = random.uniform(*self.search_space.learning_rate)
        params['batch_size'] = random.choice(self.search_space.batch_size)
        params['weight_decay'] = random.uniform(*self.search_space.weight_decay)

        # 模型特定参数
        if self.search_space.model_type in ['transformer', 'bert']:
            if self.search_space.num_heads:
                params['num_heads'] = random.choice(self.search_space.num_heads)
            if self.search_space.dim_feedforward:
                params['dim_feedforward'] = random.choice(self.search_space.dim_feedforward)
            if self.search_space.activation:
                params['activation'] = random.choice(self.search_space.activation)
            if self.search_space.layer_norm_eps:
                params['layer_norm_eps'] = random.uniform(*self.search_space.layer_norm_eps)

            # 确保d_model可被num_heads整除
            if 'num_heads' in params:
                while params['d_model'] % params['num_heads'] != 0:
                    params['num_heads'] = random.choice(self.search_space.num_heads)

            # BERT特定参数
            if self.search_space.model_type == 'bert':
                if self.search_space.intermediate_size:
                    params['intermediate_size'] = random.choice(self.search_space.intermediate_size)
                if self.search_space.hidden_act:
                    params['hidden_act'] = random.choice(self.search_space.hidden_act)

        elif self.search_space.model_type == 'mdlm':
            if self.search_space.num_timesteps:
                params['num_timesteps'] = random.choice(self.search_space.num_timesteps)
            if self.search_space.mask_ratio:
                params['mask_ratio'] = random.uniform(*self.search_space.mask_ratio)
            if self.search_space.diffusion_steps:
                params['diffusion_steps'] = random.choice(self.search_space.diffusion_steps)
            if self.search_space.max_length:
                params['max_length'] = random.choice(self.search_space.max_length)

        elif self.search_space.model_type == 'lstm':
            if self.search_space.hidden_size:
                params['hidden_size'] = random.choice(self.search_space.hidden_size)
            if self.search_space.bidirectional:
                params['bidirectional'] = random.choice(self.search_space.bidirectional)

        elif self.search_space.model_type in ['mamba', 'bimamba', 'mamba_large', 'bimamba_large', 'bimamba_xl']:
            if self.search_space.d_state:
                params['d_state'] = random.choice(self.search_space.d_state)
            if self.search_space.d_conv:
                params['d_conv'] = random.choice(self.search_space.d_conv)
            if self.search_space.expand:
                params['expand'] = random.choice(self.search_space.expand)

        elif self.search_space.model_type in ['rwkv7', 'rwkv7_large', 'rwkv7_efficient']:
            if self.search_space.use_dynamic_state:
                params['use_dynamic_state'] = random.choice(self.search_space.use_dynamic_state)
            if self.search_space.max_sequence_length:
                params['max_sequence_length'] = random.choice(self.search_space.max_sequence_length)

        elif self.search_space.model_type == 'retnet':
            if self.search_space.retention_heads:
                params['retention_heads'] = random.choice(self.search_space.retention_heads)
            if self.search_space.value_factor:
                params['value_factor'] = random.choice(self.search_space.value_factor)

        elif self.search_space.model_type == 'switch':
            if self.search_space.num_experts:
                params['num_experts'] = random.choice(self.search_space.num_experts)
            if self.search_space.expert_capacity_factor:
                params['expert_capacity_factor'] = random.uniform(*self.search_space.expert_capacity_factor)

        # 参数映射：将通用的d_model映射到模型特定参数
        params = self._map_model_params(params)

        return params

    def _optimize_with_grid_search(self) -> Dict[str, Any]:
        """网格搜索"""
        log_info(f"开始网格搜索优化，目标: {self.config.metric}")
        log_warning("网格搜索可能需要大量试验，建议降低搜索空间大小")

        # 生成所有参数组合（简化版本）
        param_combinations = self._generate_grid_combinations()

        # 限制组合数量
        if len(param_combinations) > self.config.n_trials:
            log_warning(f"网格组合数({len(param_combinations)})超过试验限制({self.config.n_trials})，随机采样")
            param_combinations = random.sample(param_combinations, self.config.n_trials)

        best_score = float('-inf') if self.config.direction == 'maximize' else float('inf')
        best_params = None

        for trial_id, params in enumerate(param_combinations):
            class GridTrial:
                def __init__(self, trial_id, params):
                    self.number = trial_id
                    self._params = params

                def suggest_categorical(self, name, choices):
                    return self._params[name]

                def suggest_float(self, name, low, high):
                    return self._params[name]

                def suggest_loguniform(self, name, low, high):
                    return self._params[name]

            trial = GridTrial(trial_id, params)
            score = self.objective(trial)

            # 更新最佳结果
            is_better = (self.config.direction == 'maximize' and score > best_score) or \
                       (self.config.direction == 'minimize' and score < best_score)

            if is_better:
                best_score = score
                best_params = params.copy()

        return {
            'best_params': best_params,
            'best_value': best_score,
            'n_trials': len(param_combinations)
        }

    def _generate_grid_combinations(self) -> List[Dict[str, Any]]:
        """生成网格搜索的参数组合"""
        import itertools

        # 离散化连续参数
        dropout_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        lr_values = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
        weight_decay_values = [0.0, 0.01, 0.05, 0.1]

        base_params = [
            self.search_space.d_model,
            self.search_space.num_layers,
            dropout_values,
            lr_values,
            self.search_space.batch_size,
            weight_decay_values
        ]

        param_names = ['d_model', 'num_layers', 'dropout', 'learning_rate', 'batch_size', 'weight_decay']

        if self.search_space.model_type == 'transformer':
            base_params.extend([
                self.search_space.num_heads,
                self.search_space.dim_feedforward,
                self.search_space.activation,
                [1e-6, 1e-5, 1e-4]  # layer_norm_eps
            ])
            param_names.extend(['num_heads', 'dim_feedforward', 'activation', 'layer_norm_eps'])

        elif self.search_space.model_type == 'mdlm':
            base_params.extend([
                self.search_space.num_timesteps,
                [0.15, 0.25, 0.35],  # mask_ratio
                self.search_space.diffusion_steps,
                self.search_space.max_length
            ])
            param_names.extend(['num_timesteps', 'mask_ratio', 'diffusion_steps', 'max_length'])

        # 生成所有组合
        combinations = []
        for combo in itertools.product(*base_params):
            params = dict(zip(param_names, combo))

            # 验证Transformer的约束条件
            if self.search_space.model_type == 'transformer':
                if params['d_model'] % params['num_heads'] != 0:
                    continue

            combinations.append(params)

        return combinations


# ============================================================================
# 工具函数
# ============================================================================

def create_hpo_optimizer(
    model_type: str,
    train_func: Callable,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: str,
    config: Optional[HPOConfig] = None,
    search_space: Optional[HPOSearchSpace] = None
) -> HPOOptimizer:
    """创建HPO优化器的便捷函数"""

    if config is None:
        config = HPOConfig()

    if search_space is None:
        search_space = HPOSearchSpace(model_type=model_type)

    return HPOOptimizer(
        config=config,
        search_space=search_space,
        train_func=train_func,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir
    )


def analyze_hpo_results(results_dir: str) -> Dict[str, Any]:
    """分析HPO结果"""
    results_path = Path(results_dir)

    # 读取试验历史
    history_path = results_path / 'trial_history.json'
    if not history_path.exists():
        raise FileNotFoundError(f"未找到试验历史文件: {history_path}")

    with open(history_path, 'r', encoding='utf-8') as f:
        trials = json.load(f)

    if not trials:
        return {'error': '没有试验数据'}

    # 基本统计
    scores = [trial['score'] for trial in trials]
    best_trial = max(trials, key=lambda x: x['score'])

    # 参数重要性分析（简单版本）
    param_importance = {}
    for param_name in best_trial['params'].keys():
        param_values = [trial['params'].get(param_name) for trial in trials]
        param_scores = [trial['score'] for trial in trials]

        # 计算相关性（简化）
        if all(isinstance(v, (int, float)) for v in param_values if v is not None):
            try:
                correlation = np.corrcoef(param_values, param_scores)[0, 1]
                param_importance[param_name] = abs(correlation) if not np.isnan(correlation) else 0.0
            except:
                param_importance[param_name] = 0.0
        else:
            param_importance[param_name] = 0.0

    analysis = {
        'total_trials': len(trials),
        'best_score': max(scores),
        'worst_score': min(scores),
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'best_trial': best_trial,
        'parameter_importance': dict(sorted(param_importance.items(), key=lambda x: x[1], reverse=True)),
        'score_distribution': {
            'percentile_95': np.percentile(scores, 95),
            'percentile_75': np.percentile(scores, 75),
            'median': np.median(scores),
            'percentile_25': np.percentile(scores, 25),
            'percentile_5': np.percentile(scores, 5)
        }
    }

    # 保存分析结果
    analysis_path = results_path / 'hpo_analysis.json'
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    return analysis


__all__ = [
    'HPOConfig',
    'HPOSearchSpace',
    'HPOObjective',
    'HPOOptimizer',
    'create_hpo_optimizer',
    'analyze_hpo_results'
]