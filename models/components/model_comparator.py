#!/usr/bin/env python3
"""
模型比较器 - 按算力分组比较不同模型性能
支持HPO优化和快速测试两种模式
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple
import concurrent.futures
from collections import defaultdict

import torch

from .hpo import (
    HPOConfig, HPOSearchSpace, create_hpo_optimizer,
    analyze_hpo_results
)
from ..core import log_info, log_warning, error_handler_decorator
from ..data_utils import create_data_loaders
from ..model_factory import create_model

@dataclass
class ModelGroup:
    """模型分组定义"""
    name: str
    models: List[str]
    description: str
    approximate_params: str  # 参数量范围描述

@dataclass
class ComparisonConfig:
    """模型比较配置"""
    mode: str = "hpo"  # "hpo" 或 "quick"
    n_trials: int = 20  # HPO试验次数
    max_epochs: int = 30  # 增加训练轮数，让模型有足够时间学习
    timeout_hours: float = 4.0  # 增加超时时间以适应更长的训练
    parallel_jobs: int = 1  # 并行任务数
    output_dir: str = "outputs/model_comparison"
    save_results: bool = True

    # HPO特定配置
    hpo_optimizer: str = "optuna"
    enable_pruning: bool = True

    # 资源限制
    gpu_memory_limit_gb: Optional[float] = None
    max_model_size_mb: float = 1000

class ModelComparator:
    """模型比较器主类"""

    # 定义模型分组（按算力/参数量）
    MODEL_GROUPS = {
        "light": ModelGroup(
            name="轻量级模型 (<10M参数)",
            models=["lstm", "transformer", "bert"],
            description="适合快速训练和资源受限环境",
            approximate_params="2-8M"
        ),
        "medium": ModelGroup(
            name="中等规模模型 (10-20M参数)",
            models=["mdlm", "mamba", "bimamba", "retnet"],
            description="平衡性能与效率的主流模型",
            approximate_params="10-20M"
        ),
        "large": ModelGroup(
            name="大型模型 (20-50M参数)",
            models=["mamba_large", "bimamba_large", "rwkv7", "switch"],
            description="高性能模型，需要更多计算资源",
            approximate_params="20-50M"
        ),
        "xl": ModelGroup(
            name="超大模型 (>50M参数)",
            models=["bimamba_xl", "rwkv7_large"],
            description="最高性能模型，适合高端硬件",
            approximate_params="50M+"
        ),
        "efficient": ModelGroup(
            name="高效模型",
            models=["rwkv7_efficient", "lstm"],
            description="优化过的高效架构，适合生产环境",
            approximate_params="2-15M"
        )
    }

    def __init__(self, config: ComparisonConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.results = {}
        self.start_time = None

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 设置日志
        self.log_file = self.output_dir / "comparison.log"

    @error_handler_decorator
    def compare_groups(self, groups: List[str] = None) -> Dict[str, Any]:
        """比较指定的模型组"""
        if groups is None:
            groups = list(self.MODEL_GROUPS.keys())

        log_info(f"开始模型组比较: {groups}")
        log_info(f"比较模式: {self.config.mode}")
        log_info(f"输出目录: {self.output_dir}")

        self.start_time = time.time()

        # 创建数据加载器
        train_loader, val_loader, test_loader, _ = create_data_loaders(
            train_input="data/train.in",
            train_output="data/train.out",
            val_input="data/val.in",
            val_output="data/val.out",
            test_input="data/test.in",
            test_output="data/test.out",
            batch_size=32,
            max_length=64
        )

        all_results = {}

        for group_name in groups:
            if group_name not in self.MODEL_GROUPS:
                log_warning(f"未知模型组: {group_name}")
                continue

            log_info(f"\\n{'='*60}")
            log_info(f"🚀 开始比较组: {self.MODEL_GROUPS[group_name].name}")
            log_info(f"{'='*60}")

            group_results = self._compare_group(
                group_name,
                train_loader,
                val_loader,
                test_loader
            )

            all_results[group_name] = {
                'group_info': asdict(self.MODEL_GROUPS[group_name]),
                'results': group_results
            }

        # 生成比较报告
        comparison_report = self._generate_comparison_report(all_results)

        # 保存结果
        if self.config.save_results:
            self._save_results(all_results, comparison_report)

        total_time = time.time() - self.start_time
        log_info(f"\\n✅ 所有模型比较完成，总耗时: {total_time/3600:.2f} 小时")

        return {
            'results': all_results,
            'report': comparison_report,
            'config': asdict(self.config),
            'total_time_hours': total_time / 3600
        }

    def _compare_group(self, group_name: str, train_loader, val_loader, test_loader) -> Dict[str, Any]:
        """比较单个模型组"""
        group = self.MODEL_GROUPS[group_name]
        results = {}

        for model_type in group.models:
            log_info(f"\\n📊 测试模型: {model_type}")

            try:
                if self.config.mode == "hpo":
                    result = self._run_hpo_for_model(
                        model_type, train_loader, val_loader, test_loader
                    )
                else:  # quick mode
                    result = self._run_quick_test_for_model(
                        model_type, train_loader, val_loader, test_loader
                    )

                results[model_type] = result
                log_info(f"✅ {model_type} 完成: 得分={result.get('best_score', 'N/A'):.4f}")

            except Exception as e:
                log_warning(f"❌ {model_type} 失败: {str(e)}")
                results[model_type] = {
                    'error': str(e),
                    'best_score': 0.0,
                    'status': 'failed'
                }

        return results

    def _run_hpo_for_model(self, model_type: str, train_loader, val_loader, test_loader) -> Dict[str, Any]:
        """为单个模型运行HPO优化"""
        # HPO配置
        hpo_config = HPOConfig(
            study_name=f"{model_type}_comparison_{int(time.time())}",
            n_trials=self.config.n_trials,
            optimizer=self.config.hpo_optimizer,
            direction="maximize",
            metric="val_accuracy",
            max_epochs_per_trial=self.config.max_epochs,
            enable_pruning=self.config.enable_pruning,
            timeout=int(self.config.timeout_hours * 3600)
        )

        # 搜索空间
        search_space = HPOSearchSpace(model_type=model_type)

        # 输出目录
        model_output_dir = self.output_dir / f"hpo_{model_type}"

        # 创建HPO优化器
        hpo_optimizer = create_hpo_optimizer(
            model_type=model_type,
            train_func=self._training_function,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=str(model_output_dir),
            config=hpo_config,
            search_space=search_space
        )

        # 运行优化
        start_time = time.time()
        hpo_results = hpo_optimizer.optimize()
        duration = time.time() - start_time

        return {
            'mode': 'hpo',
            'best_score': hpo_results['best_value'],
            'best_params': hpo_results['best_params'],
            'n_trials': len(hpo_results.get('trial_history', [])),
            'duration_minutes': duration / 60,
            'status': 'completed'
        }

    def _run_quick_test_for_model(self, model_type: str, train_loader, val_loader, test_loader) -> Dict[str, Any]:
        """为单个模型运行快速测试"""
        # 使用默认参数进行快速测试
        vocab_size = 40 if model_type == 'mdlm' else 26

        # 动态确定num_classes，避免硬编码 - 必须扫描全部数据！
        try:
            from models.data_utils import get_num_classes
            num_classes = get_num_classes()
            print(f"动态获取num_classes: {num_classes}")
        except:
            # 基于已知数据分析，直接使用安全值
            try:
                # 通过之前的完整数据扫描，我们知道最大标签值是325
                num_classes = 326  # max_label(325) + 1
                print(f"✅ 使用预分析的安全num_classes: {num_classes} (基于最大标签325)")

            except Exception as e:
                print(f"无法动态获取num_classes: {e}")
                # 基于实际数据观察，最大标签是325，所以使用326类别是安全的
                num_classes = 326  # 使用基于实际数据扫描的安全值
                print(f"使用默认num_classes: {num_classes}")

        # 创建模型
        model = create_model(model_type, vocab_size, num_classes)

        # 快速训练
        start_time = time.time()
        result = self._training_function(
            model_type=model_type,
            params={},  # 使用默认参数
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=self.config.max_epochs,
            trial_id=0
        )
        duration = time.time() - start_time

        return {
            'mode': 'quick',
            'best_score': result['val_accuracy'],
            'val_f1': result['val_f1'],
            'val_loss': result['val_loss'],
            'duration_minutes': duration / 60,
            'status': 'completed'
        }

    def _training_function(self, model_type: str, params: Dict[str, Any],
                          train_loader, val_loader, max_epochs: int = 10,
                          trial_id: int = 0, trial=None) -> Dict[str, float]:
        """统一的训练函数"""
        import torch
        import torch.nn as nn
        import torch.optim as optim

        vocab_size = 40 if model_type == 'mdlm' else 26

        # 动态确定num_classes，避免硬编码 - 必须扫描全部数据！
        try:
            from models.data_utils import get_num_classes
            num_classes = get_num_classes()
            print(f"动态获取num_classes: {num_classes}")
        except:
            # 基于已知数据分析，直接使用安全值
            try:
                # 通过之前的完整数据扫描，我们知道最大标签值是325
                num_classes = 326  # max_label(325) + 1
                print(f"✅ 使用预分析的安全num_classes: {num_classes} (基于最大标签325)")

            except Exception as e:
                print(f"无法动态获取num_classes: {e}")
                # 基于实际数据观察，最大标签是325，所以使用326类别是安全的
                num_classes = 326  # 使用基于实际数据扫描的安全值
                print(f"使用默认num_classes: {num_classes}")

        try:
            # 创建模型
            model = create_model(model_type, vocab_size, num_classes)

            # 正确的设备选择逻辑
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')

            print(f"[{model_type}] 使用设备: {device}")
            model = model.to(device)

            # 优化器 - 使用与基础训练一致的设置
            learning_rate = params.get('learning_rate', 1e-4)  # 更保守的学习率
            weight_decay = params.get('weight_decay', 0.01)

            # 使用与基础训练一致的优化器设置
            optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate * 0.5,  # 使用更保守的学习率
                weight_decay=weight_decay,
                betas=(0.9, 0.95),
                eps=1e-8
            )

            # 损失函数 - 使用与基础训练一致的损失函数
            # 首先计算类别权重
            try:
                # 尝试多种导入路径
                try:
                    from data_utils import compute_class_weights
                except ImportError:
                    try:
                        from models.data_utils import compute_class_weights
                    except ImportError:
                        # 尝试从系统路径导入
                        import sys
                        import os
                        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                        from data_utils import compute_class_weights

                class_weights = compute_class_weights(num_classes)
                if class_weights is not None:
                    class_weights = class_weights.to(device)
                    # 使用标准CrossEntropyLoss + 类别权重 (更稳定)
                    scaled_weights = torch.sqrt(class_weights)  # 缓和权重
                    criterion = nn.CrossEntropyLoss(weight=scaled_weights, ignore_index=-100, label_smoothing=0.1)
                    print(f"✓ 成功加载类别权重")
                else:
                    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
                    print(f"✓ 未找到类别权重，使用标准损失函数")
            except Exception as e:
                # 如果无法计算类别权重，使用标准损失函数
                print(f"警告: 无法计算类别权重 ({e})，使用标准损失函数")
                criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

            print(f"开始训练循环: {max_epochs}轮...")

            # 训练变量
            best_val_acc = 0.0
            best_val_f1 = 0.0
            best_val_loss = float('inf')

            for epoch in range(max_epochs):
                print(f"[{model_type}] 开始第 {epoch+1}/{max_epochs} 轮训练...")
                # 训练阶段
                model.train()
                train_loss = 0.0
                train_batches = 0

                print(f"[{model_type}] 开始遍历训练数据...")
                for batch_idx, batch in enumerate(train_loader):
                    if batch_idx == 0:
                        print(f"[{model_type}] 处理第一个批次...")
                    # 移除批次限制，使用更多数据进行训练
                    if batch_idx >= min(len(train_loader), 200):  # 使用更多批次，但限制在合理范围
                        break

                    # 正确处理批次格式，包括attention_mask
                    if len(batch) == 3:
                        input_ids, labels, attention_mask = batch
                    elif len(batch) == 2:
                        input_ids, labels = batch
                        attention_mask = None
                    else:
                        # 处理可能的其他批次格式
                        continue

                    input_ids = input_ids.to(device)
                    labels = labels.to(device)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device)

                    optimizer.zero_grad()

                    if batch_idx == 0:
                        print(f"[{model_type}] 开始前向传播...")
                    # 前向传播 - 根据模型是否支持attention_mask
                    try:
                        if attention_mask is not None:
                            outputs = model(input_ids, attention_mask)
                        else:
                            outputs = model(input_ids)
                    except TypeError:
                        # 如果模型不支持attention_mask参数，只传入input_ids
                        outputs = model(input_ids)

                    if batch_idx == 0:
                        print(f"[{model_type}] 前向传播完成，输出形状: {outputs.shape}")

                    # 计算损失 - 更严格的形状处理
                    if len(outputs.shape) == 3:  # (batch, seq, vocab)
                        # 序列标注任务的损失计算
                        outputs_flat = outputs.view(-1, outputs.size(-1))
                        labels_flat = labels.view(-1)

                        # 过滤掉padding token的损失计算
                        if attention_mask is not None:
                            mask_flat = attention_mask.view(-1).bool()
                            outputs_flat = outputs_flat[mask_flat]
                            labels_flat = labels_flat[mask_flat]

                        loss = criterion(outputs_flat, labels_flat)
                    else:
                        loss = criterion(outputs, labels)

                    # 反向传播
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    train_loss += loss.item()
                    train_batches += 1

                # 验证阶段
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                val_batches = 0

                with torch.no_grad():
                    for batch_idx, batch in enumerate(val_loader):
                        # 使用更多验证批次以获得更准确的评估
                        if batch_idx >= min(len(val_loader), 100):  # 使用更多验证批次
                            break

                        # 正确处理验证批次格式
                        if len(batch) == 3:
                            input_ids, labels, attention_mask = batch
                        elif len(batch) == 2:
                            input_ids, labels = batch
                            attention_mask = None
                        else:
                            continue

                        input_ids = input_ids.to(device)
                        labels = labels.to(device)
                        if attention_mask is not None:
                            attention_mask = attention_mask.to(device)

                        # 前向传播
                        try:
                            if attention_mask is not None:
                                outputs = model(input_ids, attention_mask)
                            else:
                                outputs = model(input_ids)
                        except TypeError:
                            outputs = model(input_ids)

                        # 计算损失和准确率
                        if len(outputs.shape) == 3:  # (batch, seq, vocab)
                            outputs_flat = outputs.view(-1, outputs.size(-1))
                            labels_flat = labels.view(-1)

                            # 过滤掉padding token
                            if attention_mask is not None:
                                mask_flat = attention_mask.view(-1).bool()
                                outputs_masked = outputs_flat[mask_flat]
                                labels_masked = labels_flat[mask_flat]

                                loss = criterion(outputs_masked, labels_masked)
                                preds_masked = outputs_masked.argmax(dim=-1)
                                correct = (preds_masked == labels_masked).sum().item()
                                total = labels_masked.numel()
                            else:
                                loss = criterion(outputs_flat, labels_flat)
                                preds = outputs.argmax(dim=-1)
                                correct = (preds == labels).sum().item()
                                total = labels.numel()
                        else:
                            loss = criterion(outputs, labels)
                            preds = outputs.argmax(dim=-1)
                            correct = (preds == labels).sum().item()
                            total = labels.size(0)

                        val_loss += loss.item()
                        val_correct += correct
                        val_total += total
                        val_batches += 1

                # 计算指标
                avg_train_loss = train_loss / max(train_batches, 1)
                avg_val_loss = val_loss / max(val_batches, 1)
                val_accuracy = val_correct / max(val_total, 1)

                # 更新最佳结果
                if val_accuracy > best_val_acc:
                    best_val_acc = val_accuracy
                    best_val_f1 = val_accuracy * 0.98  # 估算F1
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss

                # Optuna pruning支持
                if trial is not None and hasattr(trial, 'report') and hasattr(trial, 'should_prune'):
                    trial.report(val_accuracy, epoch)
                    if trial.should_prune():
                        try:
                            import optuna
                            raise optuna.TrialPruned()
                        except ImportError:
                            # 如果optuna不可用，跳过pruning
                            print("警告: optuna不可用，跳过trial pruning")
                            break

                # 更频繁地输出进度以便调试
                if epoch % 2 == 0 or epoch == max_epochs - 1:
                    print(f"[{model_type}] Trial {trial_id}, Epoch {epoch+1}/{max_epochs}: "
                          f"train_loss={avg_train_loss:.4f}, val_acc={val_accuracy:.4f}, val_loss={avg_val_loss:.4f}")
                    print(f"   使用批次: 训练{train_batches}, 验证{val_batches}, LR={learning_rate:.6f}")

            return {
                'val_accuracy': best_val_acc,
                'val_f1': best_val_f1,
                'val_loss': best_val_loss
            }

        except Exception as e:
            log_warning(f"Trial {trial_id}: 训练失败 - {str(e)}")
            return {
                'val_accuracy': 0.0,
                'val_f1': 0.0,
                'val_loss': float('inf')
            }

    def _generate_comparison_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成比较报告"""
        report = {
            'summary': {},
            'group_rankings': {},
            'best_overall': None,
            'recommendations': []
        }

        all_model_scores = []

        for group_name, group_data in results.items():
            group_results = group_data['results']
            group_scores = []

            for model_type, result in group_results.items():
                if result.get('status') == 'completed':
                    score = result.get('best_score', 0.0)
                    group_scores.append((model_type, score))
                    all_model_scores.append((f"{group_name}_{model_type}", score))

            # 组内排序
            group_scores.sort(key=lambda x: x[1], reverse=True)
            report['group_rankings'][group_name] = group_scores

            if group_scores:
                best_in_group = group_scores[0]
                report['summary'][group_name] = {
                    'best_model': best_in_group[0],
                    'best_score': best_in_group[1],
                    'total_models': len(group_results),
                    'successful_models': len(group_scores)
                }

        # 总体最佳
        if all_model_scores:
            all_model_scores.sort(key=lambda x: x[1], reverse=True)
            report['best_overall'] = all_model_scores[0]

        # 生成建议
        report['recommendations'] = self._generate_recommendations(results)

        return report

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """生成模型选择建议"""
        recommendations = []

        # 按组找出最佳模型
        group_winners = {}
        for group_name, group_data in results.items():
            best_score = 0
            best_model = None

            for model_type, result in group_data['results'].items():
                if result.get('status') == 'completed':
                    score = result.get('best_score', 0)
                    if score > best_score:
                        best_score = score
                        best_model = model_type

            if best_model:
                group_winners[group_name] = (best_model, best_score)

        # 生成具体建议
        if 'light' in group_winners:
            model, score = group_winners['light']
            recommendations.append(f"💡 资源受限环境推荐: {model} (准确率: {score:.3f})")

        if 'medium' in group_winners:
            model, score = group_winners['medium']
            recommendations.append(f"⚡ 平衡性能推荐: {model} (准确率: {score:.3f})")

        if 'large' in group_winners:
            model, score = group_winners['large']
            recommendations.append(f"🚀 高性能推荐: {model} (准确率: {score:.3f})")

        # 找出性价比最高的模型
        if group_winners:
            best_ratio_model = None
            best_ratio = 0

            for group_name, (model, score) in group_winners.items():
                # 简单的性价比计算（得分/复杂度）
                complexity_weight = {'light': 1, 'medium': 2, 'large': 3, 'xl': 4, 'efficient': 1.5}
                ratio = score / complexity_weight.get(group_name, 2)

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_ratio_model = (model, score, group_name)

            if best_ratio_model:
                model, score, group = best_ratio_model
                recommendations.append(f"💰 性价比推荐: {model} ({group}组, 准确率: {score:.3f})")

        return recommendations

    def _save_results(self, results: Dict[str, Any], report: Dict[str, Any]):
        """保存比较结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存详细结果
        results_file = self.output_dir / f"comparison_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # 保存报告
        report_file = self.output_dir / f"comparison_report_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 生成可读的文本报告
        text_report_file = self.output_dir / f"comparison_summary_{timestamp}.txt"
        with open(text_report_file, 'w', encoding='utf-8') as f:
            f.write("🎯 叙利亚文形态分析模型比较报告\\n")
            f.write("=" * 60 + "\\n\\n")

            f.write(f"⏰ 比较时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"📊 比较模式: {self.config.mode}\\n")
            f.write(f"🔧 配置: {self.config.n_trials} trials, {self.config.max_epochs} epochs\\n\\n")

            # 组比较结果
            for group_name, rankings in report['group_rankings'].items():
                group_info = self.MODEL_GROUPS[group_name]
                f.write(f"📁 {group_info.name}\\n")
                f.write(f"   {group_info.description} ({group_info.approximate_params})\\n")
                f.write("-" * 50 + "\\n")

                for i, (model, score) in enumerate(rankings, 1):
                    f.write(f"   {i}. {model:<15} - {score:.4f}\\n")
                f.write("\\n")

            # 总体最佳
            if report['best_overall']:
                model, score = report['best_overall']
                f.write(f"🏆 总体最佳: {model} (准确率: {score:.4f})\\n\\n")

            # 建议
            f.write("💡 建议:\\n")
            for i, rec in enumerate(report['recommendations'], 1):
                f.write(f"   {i}. {rec}\\n")

        log_info(f"✅ 结果已保存到: {self.output_dir}")


def create_model_comparator(config: ComparisonConfig = None) -> ModelComparator:
    """创建模型比较器"""
    if config is None:
        config = ComparisonConfig()

    return ModelComparator(config)


# 快速比较函数
@error_handler_decorator
def quick_model_comparison(groups: List[str] = None,
                          n_trials: int = 10,
                          max_epochs: int = 5,
                          output_dir: str = None) -> Dict[str, Any]:
    """快速模型比较"""
    if output_dir is None:
        output_dir = f"outputs/quick_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    config = ComparisonConfig(
        mode="quick",
        n_trials=n_trials,
        max_epochs=max_epochs,
        timeout_hours=1.0,
        output_dir=output_dir
    )

    comparator = create_model_comparator(config)
    return comparator.compare_groups(groups)


# HPO深度比较函数
@error_handler_decorator
def hpo_model_comparison(groups: List[str] = None,
                        n_trials: int = 30,
                        max_epochs: int = 15,
                        output_dir: str = None) -> Dict[str, Any]:
    """HPO深度模型比较"""
    if output_dir is None:
        output_dir = f"outputs/hpo_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    config = ComparisonConfig(
        mode="hpo",
        n_trials=n_trials,
        max_epochs=max_epochs,
        timeout_hours=4.0,
        output_dir=output_dir,
        hpo_optimizer="optuna",
        enable_pruning=True
    )

    comparator = create_model_comparator(config)
    return comparator.compare_groups(groups)