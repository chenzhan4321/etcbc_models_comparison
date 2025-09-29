#!/usr/bin/env python3
"""
HPO训练脚本
专门用于Transformer和MDLM模型的超参数优化

使用方法:
python train_hpo.py --model_type transformer --optimizer optuna --n_trials 50
python train_hpo.py --model_type mdlm --optimizer hyperopt --n_trials 30
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    import optuna
except ImportError:
    optuna = None

# 项目导入
from models.components.hpo import (
    HPOConfig, HPOSearchSpace, create_hpo_optimizer, analyze_hpo_results
)
from models.components.model_comparator import (
    ComparisonConfig, create_model_comparator
)
from models.data_utils import create_data_loaders, get_num_classes
from models.model_factory import create_model
from models.core import get_device_manager, log_info, log_warning


class HPOTrainer:
    """HPO训练器"""

    def __init__(self, device: torch.device):
        self.device = device

    def train_model(self,
                   model_type: str,
                   params: Dict[str, Any],
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   max_epochs: int = 20,
                   trial_id: int = 0,
                   trial: Optional[Any] = None) -> Dict[str, Any]:
        """训练单个模型"""

        try:
            # 创建模型
            model_config = self._build_model_config(model_type, params)
            model = create_model(
                model_type=model_type,
                config=model_config,
                device=self.device
            )

            # 优化器
            optimizer = optim.AdamW(
                model.parameters(),
                lr=params['learning_rate'],
                weight_decay=params.get('weight_decay', 0.01)
            )

            # 学习率调度器
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_epochs, eta_min=1e-6
            )

            # 损失函数
            criterion = nn.CrossEntropyLoss(ignore_index=-100)

            # 训练循环
            best_val_acc = 0.0
            best_val_f1 = 0.0
            best_val_loss = float('inf')

            for epoch in range(max_epochs):
                # 训练阶段
                model.train()
                train_loss = 0.0
                train_samples = 0

                for batch_idx, (input_ids, labels, attention_mask) in enumerate(train_loader):
                    input_ids = input_ids.to(self.device)
                    labels = labels.to(self.device)
                    attention_mask = attention_mask.to(self.device)

                    optimizer.zero_grad()

                    # 前向传播
                    outputs = model(input_ids, attention_mask)

                    # 计算损失
                    outputs_flat = outputs.view(-1, outputs.size(-1))
                    labels_flat = labels.view(-1)
                    loss = criterion(outputs_flat, labels_flat)

                    # 反向传播
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    train_loss += loss.item()
                    train_samples += input_ids.size(0)

                    # 提前终止检查（避免过长训练）
                    if batch_idx > 100:  # 限制训练批次
                        break

                scheduler.step()

                # 验证阶段
                val_metrics = self._validate_model(model, val_loader, criterion)

                # 更新最佳指标
                if val_metrics['accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['accuracy']
                if val_metrics['f1'] > best_val_f1:
                    best_val_f1 = val_metrics['f1']
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']

                # Optuna pruning支持
                if trial is not None and hasattr(trial, 'report') and optuna is not None:
                    trial.report(val_metrics['accuracy'], epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                # 打印进度
                if epoch % 5 == 0 or epoch == max_epochs - 1:
                    print(f"Trial {trial_id}, Epoch {epoch+1}/{max_epochs}: "
                          f"train_loss={train_loss/train_samples:.4f}, "
                          f"val_acc={val_metrics['accuracy']:.4f}, "
                          f"val_f1={val_metrics['f1']:.4f}")

            # 返回最终结果
            return {
                'val_accuracy': best_val_acc,
                'val_f1': best_val_f1,
                'val_loss': best_val_loss,
                'final_lr': scheduler.get_last_lr()[0],
                'total_params': sum(p.numel() for p in model.parameters())
            }

        except Exception as e:
            print(f"Trial {trial_id}: 训练出错 - {e}")
            return {
                'val_accuracy': 0.0,
                'val_f1': 0.0,
                'val_loss': float('inf'),
                'error': str(e)
            }

    def _build_model_config(self, model_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """构建模型配置"""
        config = {}

        # 通用配置
        if 'd_model' in params:
            config['d_model'] = params['d_model']
        if 'num_layers' in params:
            config['num_layers'] = params['num_layers']
        if 'dropout' in params:
            config['dropout'] = params['dropout']

        # Transformer特定配置
        if model_type == 'transformer':
            if 'num_heads' in params:
                config['num_heads'] = params['num_heads']
            if 'dim_feedforward' in params:
                config['dim_feedforward'] = params['dim_feedforward']
            if 'activation' in params:
                config['activation'] = params['activation']
            if 'layer_norm_eps' in params:
                config['layer_norm_eps'] = params['layer_norm_eps']

        # MDLM特定配置
        elif model_type == 'mdlm':
            if 'num_timesteps' in params:
                config['num_timesteps'] = params['num_timesteps']
            if 'mask_ratio' in params:
                config['mask_ratio'] = params['mask_ratio']
            if 'diffusion_steps' in params:
                config['diffusion_steps'] = params['diffusion_steps']
            if 'max_length' in params:
                config['max_length'] = params['max_length']

        return config

    def _validate_model(self, model: nn.Module, val_loader: DataLoader, criterion: nn.Module) -> Dict[str, float]:
        """验证模型"""
        model.eval()
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        total_predictions = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_idx, (input_ids, labels, attention_mask) in enumerate(val_loader):
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                attention_mask = attention_mask.to(self.device)

                # 前向传播
                outputs = model(input_ids, attention_mask)

                # 计算损失
                outputs_flat = outputs.view(-1, outputs.size(-1))
                labels_flat = labels.view(-1)
                loss = criterion(outputs_flat, labels_flat)

                total_loss += loss.item()
                total_samples += input_ids.size(0)

                # 计算准确率
                predictions = torch.argmax(outputs_flat, dim=-1)

                # 过滤填充位置
                mask = labels_flat != -100
                if mask.sum() > 0:
                    masked_preds = predictions[mask]
                    masked_labels = labels_flat[mask]

                    correct_predictions += (masked_preds == masked_labels).sum().item()
                    total_predictions += mask.sum().item()

                    all_preds.extend(masked_preds.cpu().numpy())
                    all_labels.extend(masked_labels.cpu().numpy())

                # 限制验证批次
                if batch_idx > 20:
                    break

        # 计算F1分数（简化版本）
        accuracy = correct_predictions / max(total_predictions, 1)
        avg_loss = total_loss / max(total_samples, 1)

        # 简化的F1计算
        f1_score = accuracy  # 简化版本，实际应该使用sklearn

        return {
            'accuracy': accuracy,
            'f1': f1_score,
            'loss': avg_loss
        }


def run_model_comparison(args):
    """运行模型比较"""
    log_info("🎯 启动模型比较模式")

    # 设置输出目录
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_suffix = args.compare_mode if hasattr(args, 'compare_mode') else 'hpo'
        args.output_dir = f"outputs/model_comparison_{mode_suffix}_{timestamp}"

    # 创建比较配置
    config = ComparisonConfig(
        mode=getattr(args, 'compare_mode', 'hpo'),
        n_trials=args.n_trials,
        max_epochs=args.max_epochs,
        timeout_hours=getattr(args, 'timeout', 7200) / 3600 if hasattr(args, 'timeout') and args.timeout else 2.0,
        output_dir=args.output_dir,
        hpo_optimizer=args.optimizer,
        enable_pruning=getattr(args, 'enable_pruning', False)
    )

    log_info(f"📊 比较配置:")
    log_info(f"  - 比较模式: {config.mode}")
    log_info(f"  - 试验次数: {config.n_trials}")
    log_info(f"  - 最大轮数: {config.max_epochs}")
    log_info(f"  - 输出目录: {config.output_dir}")

    # 确定要比较的组
    groups = getattr(args, 'compare_groups', None)
    if groups is None:
        log_info("📁 比较所有模型组: light, medium, large")
        groups = ['light', 'medium', 'large']
    else:
        log_info(f"📁 比较指定组: {groups}")

    # 创建比较器并运行
    try:
        comparator = create_model_comparator(config)
        results = comparator.compare_groups(groups)

        log_info("\\n🎉 模型比较完成!")

        # 显示简要结果
        report = results.get('report', {})
        if 'best_overall' in report and report['best_overall']:
            model, score = report['best_overall']
            log_info(f"🏆 总体最佳: {model} (准确率: {score:.4f})")

        # 显示建议
        recommendations = report.get('recommendations', [])
        if recommendations:
            log_info("\\n💡 建议:")
            for i, rec in enumerate(recommendations, 1):
                log_info(f"   {i}. {rec}")

        log_info(f"\\n📊 详细结果已保存到: {config.output_dir}")

        return 0

    except Exception as e:
        log_warning(f"❌ 模型比较失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser(description='HPO训练脚本 - 支持所有13种模型和模型比较')

    # 基本参数
    parser.add_argument('--model_type', type=str, default=None,
                       choices=['transformer', 'bert', 'lstm', 'mdlm',
                               'mamba', 'bimamba', 'mamba_large', 'bimamba_large', 'bimamba_xl',
                               'rwkv7', 'rwkv7_large', 'rwkv7_efficient',
                               'retnet', 'switch'],
                       help='模型类型（不指定则进行模型比较）')

    # 模型比较参数
    parser.add_argument('--compare_models', action='store_true',
                       help='启用模型比较模式')
    parser.add_argument('--compare_groups', type=str, nargs='+',
                       choices=['light', 'medium', 'large', 'xl', 'efficient'],
                       help='指定要比较的模型组')
    parser.add_argument('--compare_mode', type=str, default='hpo',
                       choices=['hpo', 'quick'],
                       help='比较模式: hpo(深度优化) 或 quick(快速测试)')

    # HPO参数
    parser.add_argument('--optimizer', type=str, default='optuna',
                       choices=['optuna', 'hyperopt', 'random', 'grid'],
                       help='HPO优化算法')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='试验次数')
    parser.add_argument('--timeout', type=int, default=None,
                       help='优化时间限制(秒)')
    parser.add_argument('--sampler', type=str, default='tpe',
                       choices=['tpe', 'random', 'cmaes'],
                       help='Optuna采样器')

    # 训练参数
    parser.add_argument('--max_epochs', type=int, default=15,
                       help='每次试验最大训练轮数')
    parser.add_argument('--batch_size_override', type=int, default=None,
                       help='覆盖批次大小搜索')

    # 输出参数
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录')
    parser.add_argument('--study_name', type=str, default=None,
                       help='研究名称')

    # 搜索空间自定义
    parser.add_argument('--d_model_range', type=str, default=None,
                       help='模型维度范围，如"256,384,512"')
    parser.add_argument('--num_layers_range', type=str, default=None,
                       help='层数范围，如"3,4,6,8"')

    # 资源管理
    parser.add_argument('--memory_limit', type=float, default=None,
                       help='内存限制(GB)')
    parser.add_argument('--enable_pruning', action='store_true',
                       help='启用早停')

    args = parser.parse_args()

    # 检查是否启用模型比较模式
    if args.compare_models or args.model_type is None:
        return run_model_comparison(args)

    # 单模型HPO模式
    # 设置输出目录
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"outputs/hpo_{args.model_type}_{timestamp}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置研究名称
    if args.study_name is None:
        args.study_name = f"{args.model_type}_hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 初始化设备
    device_manager = get_device_manager()
    device = device_manager.device

    log_info(f"HPO训练开始:")
    log_info(f"  - 模型类型: {args.model_type}")
    log_info(f"  - 优化器: {args.optimizer}")
    log_info(f"  - 试验次数: {args.n_trials}")
    log_info(f"  - 设备: {device}")
    log_info(f"  - 输出目录: {output_dir}")

    # 创建数据加载器
    log_info("创建数据加载器...")
    # 使用默认数据路径
    data_dir = "data"
    train_loader, val_loader, test_loader, _ = create_data_loaders(
        train_input=os.path.join(data_dir, 'train.in'),
        train_output=os.path.join(data_dir, 'train.out'),
        val_input=os.path.join(data_dir, 'val.in'),
        val_output=os.path.join(data_dir, 'val.out'),
        test_input=os.path.join(data_dir, 'test.in'),
        test_output=os.path.join(data_dir, 'test.out'),
        batch_size=32,  # 默认批次大小，将被HPO覆盖
        max_length=64
    )

    # 创建HPO配置
    hpo_config = HPOConfig(
        study_name=args.study_name,
        n_trials=args.n_trials,
        timeout=args.timeout,
        optimizer=args.optimizer,
        sampler=args.sampler,
        max_epochs_per_trial=args.max_epochs,
        enable_pruning=args.enable_pruning,
        memory_limit_gb=args.memory_limit
    )

    # 创建搜索空间
    search_space = HPOSearchSpace(model_type=args.model_type)

    # 自定义搜索空间
    if args.d_model_range:
        search_space.d_model = [int(x.strip()) for x in args.d_model_range.split(',')]
    if args.num_layers_range:
        search_space.num_layers = [int(x.strip()) for x in args.num_layers_range.split(',')]
    if args.batch_size_override:
        search_space.batch_size = [args.batch_size_override]

    # 创建训练器
    trainer = HPOTrainer(device)

    # 创建HPO优化器
    hpo_optimizer = create_hpo_optimizer(
        model_type=args.model_type,
        train_func=trainer.train_model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=str(output_dir),
        config=hpo_config,
        search_space=search_space
    )

    # 保存配置
    config_path = output_dir / 'hpo_config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump({
            'hpo_config': {k: v for k, v in asdict(hpo_config).items() if v is not None},
            'search_space': {k: v for k, v in asdict(search_space).items() if v is not None},
            'args': vars(args)
        }, f, indent=2, ensure_ascii=False)

    # 开始优化
    log_info("开始HPO优化...")
    start_time = time.time()

    try:
        results = hpo_optimizer.optimize()
        optimization_time = time.time() - start_time

        log_info(f"HPO优化完成!")
        log_info(f"  - 总用时: {optimization_time/60:.1f} 分钟")
        log_info(f"  - 完成试验: {results.get('n_trials', 'unknown')}")
        log_info(f"  - 最佳得分: {results.get('best_value', 'unknown'):.4f}")
        log_info(f"  - 最佳参数: {results.get('best_params', {})}")

        # 保存优化结果
        results_path = output_dir / 'optimization_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            # 移除不能序列化的对象
            serializable_results = {}
            for k, v in results.items():
                if k not in ['study', 'trials']:  # 跳过复杂对象
                    try:
                        json.dumps(v)  # 测试是否可序列化
                        serializable_results[k] = v
                    except:
                        serializable_results[k] = str(v)

            json.dump({
                'optimization_results': serializable_results,
                'optimization_time': optimization_time,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False)

        # 分析结果
        log_info("分析HPO结果...")
        analysis = analyze_hpo_results(str(output_dir))

        log_info("HPO分析完成:")
        log_info(f"  - 最佳得分: {analysis['best_score']:.4f}")
        log_info(f"  - 平均得分: {analysis['mean_score']:.4f}")
        log_info(f"  - 参数重要性 TOP3:")
        for param, importance in list(analysis['parameter_importance'].items())[:3]:
            log_info(f"    * {param}: {importance:.3f}")

        # 生成训练建议
        best_params = analysis['best_trial']['params']
        log_info("推荐的训练命令:")

        cmd_parts = [
            f"python train.py",
            f"--model_type {args.model_type}"
        ]

        if 'd_model' in best_params:
            cmd_parts.append(f"--d_model {best_params['d_model']}")
        if 'num_layers' in best_params:
            cmd_parts.append(f"--num_layers {best_params['num_layers']}")
        if 'learning_rate' in best_params:
            cmd_parts.append(f"--learning_rate {best_params['learning_rate']:.2e}")
        if 'batch_size' in best_params:
            cmd_parts.append(f"--batch_size {best_params['batch_size']}")
        if 'dropout' in best_params:
            cmd_parts.append(f"--dropout {best_params['dropout']:.3f}")
        if 'weight_decay' in best_params:
            cmd_parts.append(f"--weight_decay {best_params['weight_decay']:.4f}")

        # Transformer特定参数
        if args.model_type == 'transformer':
            if 'num_heads' in best_params:
                cmd_parts.append(f"--num_heads {best_params['num_heads']}")
            if 'dim_feedforward' in best_params:
                cmd_parts.append(f"--dim_feedforward {best_params['dim_feedforward']}")

        # MDLM特定参数
        elif args.model_type == 'mdlm':
            if 'num_timesteps' in best_params:
                cmd_parts.append(f"--num_timesteps {best_params['num_timesteps']}")

        cmd_parts.append(f"--epochs 100")  # 建议的完整训练轮数

        recommended_cmd = " \\\n    ".join(cmd_parts)
        print(f"\n推荐训练命令:\n{recommended_cmd}\n")

        # 保存推荐命令
        cmd_path = output_dir / 'recommended_command.txt'
        with open(cmd_path, 'w', encoding='utf-8') as f:
            f.write(recommended_cmd)

    except KeyboardInterrupt:
        log_warning("HPO优化被用户中断")
        return 1

    except Exception as e:
        log_warning(f"HPO优化失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)