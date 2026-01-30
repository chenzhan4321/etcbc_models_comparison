#!/usr/bin/env python3
"""
对比 .original 文件的 Levenshtein 正确率
处理特殊的 Predicted/Truevalue 交替格式
"""

import os
from pathlib import Path

def levenshtein_distance(s1, s2):
    """计算编辑距离"""
    m, n = len(s1), len(s2)

    # 创建DP表
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 初始化
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # 填充DP表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,    # 删除
                    dp[i][j-1] + 1,    # 插入
                    dp[i-1][j-1] + 1   # 替换
                )

    return dp[m][n]

def extract_predictions_and_truth_from_file(file_path):
    """从包含 Predicted/Truevalue 标记的文件中提取预测和真实值"""
    predictions = []
    truevalues = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')

            if line.startswith("Predicted "):
                predictions.append(line[len("Predicted "):])
            elif line.startswith("Truevalue "):
                truevalues.append(line[len("Truevalue "):])

    return predictions, truevalues

def read_simple_file(file_path):
    """读取简单格式的文件（每行一个样本）"""
    lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            lines.append(line.rstrip('\n'))
    return lines

def calculate_accuracy_from_lists(pred_lines, truth_lines, model_name=""):
    """从预测和真实值列表计算正确率"""

    total_chars = 0
    total_errors = 0
    line_count = 0
    error_lines = 0
    perfect_lines = 0

    min_lines = min(len(pred_lines), len(truth_lines))

    if len(pred_lines) != len(truth_lines):
        print(f"  ⚠️  警告: 行数不匹配 - 预测: {len(pred_lines)} vs 真实: {len(truth_lines)}")
        print(f"  ⚠️  将仅对比前 {min_lines} 行")

    for i in range(min_lines):
        pred_line = pred_lines[i].strip()
        truth_line = truth_lines[i].strip()

        if not truth_line:
            continue

        line_count += 1

        # 计算这一行的编辑距离
        distance = levenshtein_distance(pred_line, truth_line)

        total_chars += len(truth_line)
        total_errors += distance

        if distance > 0:
            error_lines += 1
        else:
            perfect_lines += 1

    accuracy = (total_chars - total_errors) / total_chars * 100 if total_chars > 0 else 0

    return {
        'total_chars': total_chars,
        'total_errors': total_errors,
        'correct_chars': total_chars - total_errors,
        'accuracy': accuracy,
        'line_count': line_count,
        'error_lines': error_lines,
        'perfect_lines': perfect_lines
    }

def main():
    """主函数"""

    # 定义真实标签文件
    s2_truth_file = Path("/Users/zhanchen/Library/CloudStorage/Dropbox/Projects/etcbc_update/data/raw_s2_on_s2/test.out.original")

    # 定义要测试的三个模型
    test_configs = [
        {
            'name': 'transformer_s2_on_s2',
            'pred': "/Users/zhanchen/Library/CloudStorage/Dropbox/Projects/etcbc_update/outputs/transformer_s2_on_s2/test.out.original",
            'has_labels': False,  # 简单格式，没有 Predicted/Truevalue 标记
            'truth': s2_truth_file
        },
        {
            'name': 'transformer_s4_on_s2',
            'pred': "/Users/zhanchen/Library/CloudStorage/Dropbox/Projects/etcbc_update/outputs/transformer_s4_on_s2/test.out.original",
            'has_labels': True,  # 包含 Predicted/Truevalue 标记
            'truth': None  # 使用文件内的 Truevalue
        },
        {
            'name': 'transformer_s4_on_s4',
            'pred': "/Users/zhanchen/Library/CloudStorage/Dropbox/Projects/etcbc_update/outputs/transformer_s4_on_s4/test.out.original",
            'has_labels': True,  # 包含 Predicted/Truevalue 标记
            'truth': None  # 使用文件内的 Truevalue
        }
    ]

    print("=" * 80)
    print("Transformer 模型 - Levenshtein 字符正确率分析 (.original 格式)")
    print("=" * 80)
    print()

    results = []

    for config in test_configs:
        pred_file = Path(config['pred'])
        model_name = config['name']

        if not pred_file.exists():
            print(f"警告: 预测文件不存在 - {pred_file}")
            continue

        print(f"正在分析: {model_name}")
        print("-" * 80)

        if config['has_labels']:
            # 文件包含 Predicted/Truevalue 标记，直接提取
            pred_lines, truth_lines = extract_predictions_and_truth_from_file(pred_file)
        else:
            # 简单格式，需要外部真实标签
            pred_lines = read_simple_file(pred_file)

            if config['truth'] and config['truth'].exists():
                truth_lines = read_simple_file(config['truth'])
            else:
                print(f"  错误: 需要外部真实标签文件")
                continue

        result = calculate_accuracy_from_lists(pred_lines, truth_lines, model_name)

        print(f"  总字符数:        {result['total_chars']:,}")
        print(f"  正确字符数:      {result['correct_chars']:,}")
        print(f"  错误字符数:      {result['total_errors']:,}")
        print(f"  字符正确率:      {result['accuracy']:.2f}%")
        print(f"  总行数:          {result['line_count']:,}")

        if result['line_count'] > 0:
            print(f"  完美预测行数:    {result['perfect_lines']:,} ({result['perfect_lines']/result['line_count']*100:.1f}%)")
            print(f"  有错误行数:      {result['error_lines']:,} ({result['error_lines']/result['line_count']*100:.1f}%)")
        else:
            print(f"  ⚠️  无法计算行级统计（缺少对应的真实标签文件）")
        print()

        # 只添加有效结果（至少有一些数据）
        if result['line_count'] > 0:
            results.append({
                'model': model_name,
                **result
            })

    # 排序并显示排名
    results.sort(key=lambda x: x['accuracy'], reverse=True)

    print("=" * 80)
    print("排名汇总 (按字符正确率)")
    print("=" * 80)
    print(f"{'排名':<6} {'模型':<30} {'字符正确率':<14} {'完美行率':<12}")
    print("-" * 80)

    for idx, result in enumerate(results, 1):
        if result['line_count'] > 0:
            perfect_rate = result['perfect_lines'] / result['line_count'] * 100
            perfect_rate_str = f"{perfect_rate:>10.1f}%"
        else:
            perfect_rate_str = "N/A".rjust(12)

        emoji = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else "  "
        print(f"{emoji} {idx:<4} {result['model']:<30} {result['accuracy']:>11.2f}%  {perfect_rate_str}")

    print()

    # 显示最佳和最差模型对比
    if len(results) >= 2:
        print("=" * 80)
        print("性能对比")
        print("=" * 80)

        best = results[0]
        worst = results[-1]

        print(f"\n🏆 最佳模型: {best['model']}")
        print(f"   字符正确率: {best['accuracy']:.2f}%")
        print(f"   完美预测行: {best['perfect_lines']:,} / {best['line_count']:,} ({best['perfect_lines']/best['line_count']*100:.1f}%)")
        print(f"   错误字符数: {best['total_errors']:,}")

        print(f"\n📊 最差模型: {worst['model']}")
        print(f"   字符正确率: {worst['accuracy']:.2f}%")
        print(f"   完美预测行: {worst['perfect_lines']:,} / {worst['line_count']:,} ({worst['perfect_lines']/worst['line_count']*100:.1f}%)")
        print(f"   错误字符数: {worst['total_errors']:,}")

        print(f"\n📈 性能差距: {best['accuracy'] - worst['accuracy']:.2f} 个百分点")
        if worst['total_errors'] > best['total_errors'] and worst['total_errors'] > 0:
            error_reduction = (worst['total_errors'] - best['total_errors']) / worst['total_errors'] * 100
            print(f"   错误减少: {worst['total_errors'] - best['total_errors']:,} 个字符 ({error_reduction:.1f}%)")
        print()

if __name__ == "__main__":
    main()
