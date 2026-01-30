#!/usr/bin/env python3
"""
对比 .original 文件的 Levenshtein 正确率
按字符级别计算编辑距离
处理特殊格式和前缀
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

def clean_line(line):
    """清理行内容，去除可能的前缀"""
    line = line.rstrip('\n')

    # 去除 "Predicted " 前缀
    if line.startswith("Predicted "):
        line = line[len("Predicted "):]

    # 去除 "Truth " 前缀
    if line.startswith("Truth "):
        line = line[len("Truth "):]

    return line

def calculate_accuracy(pred_file, truth_file, model_name=""):
    """计算字符级别的正确率"""

    total_chars = 0
    total_errors = 0
    line_count = 0
    error_lines = 0
    perfect_lines = 0

    # 读取预测文件
    with open(pred_file, 'r', encoding='utf-8') as f:
        pred_lines = f.readlines()

    # 读取真实标签文件
    with open(truth_file, 'r', encoding='utf-8') as f:
        truth_lines = f.readlines()

    # 检查行数
    if len(pred_lines) != len(truth_lines):
        print(f"  ⚠️  警告: 行数不匹配 - 预测: {len(pred_lines)} vs 真实: {len(truth_lines)}")
        min_lines = min(len(pred_lines), len(truth_lines))
        print(f"  ⚠️  将仅对比前 {min_lines} 行")
    else:
        min_lines = len(pred_lines)

    for i in range(min_lines):
        pred_line = clean_line(pred_lines[i])
        truth_line = clean_line(truth_lines[i])

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

    # 定义要测试的三个模型及其对应的真实标签
    test_configs = [
        {
            'name': 'transformer_s2_on_s2',
            'pred': "/Users/zhanchen/Library/CloudStorage/Dropbox/Projects/etcbc_update/outputs/transformer_s2_on_s2/test.out.original",
            'truth': "/Users/zhanchen/Library/CloudStorage/Dropbox/Projects/etcbc_update/data/raw_s2_on_s2/test.out.original"
        },
        {
            'name': 'transformer_s4_on_s2',
            'pred': "/Users/zhanchen/Library/CloudStorage/Dropbox/Projects/etcbc_update/outputs/transformer_s4_on_s2/test.out.original",
            'truth': "/Users/zhanchen/Library/CloudStorage/Dropbox/Projects/etcbc_update/data/raw_s2_on_s2/test.out.original"
        },
        {
            'name': 'transformer_s4_on_s4',
            'pred': "/Users/zhanchen/Library/CloudStorage/Dropbox/Projects/etcbc_update/outputs/transformer_s4_on_s4/test.out.original",
            'truth': "/Users/zhanchen/Library/CloudStorage/Dropbox/Projects/etcbc_update/data/raw_s2_on_s2/test.out.original"  # 暂时用 S2，后续如果有 S4 完整测试集再更新
        }
    ]

    print("=" * 80)
    print("Transformer 模型 - Levenshtein 字符正确率分析 (.original 格式)")
    print("=" * 80)
    print()

    results = []

    for config in test_configs:
        pred_file = Path(config['pred'])
        truth_file = Path(config['truth'])
        model_name = config['name']

        if not pred_file.exists():
            print(f"警告: 预测文件不存在 - {pred_file}")
            continue

        if not truth_file.exists():
            print(f"警告: 真实标签文件不存在 - {truth_file}")
            continue

        print(f"正在分析: {model_name}")
        print("-" * 80)

        result = calculate_accuracy(pred_file, truth_file, model_name)

        print(f"  总字符数:        {result['total_chars']:,}")
        print(f"  正确字符数:      {result['correct_chars']:,}")
        print(f"  错误字符数:      {result['total_errors']:,}")
        print(f"  字符正确率:      {result['accuracy']:.2f}%")
        print(f"  总行数:          {result['line_count']:,}")
        print(f"  完美预测行数:    {result['perfect_lines']:,} ({result['perfect_lines']/result['line_count']*100:.1f}%)")
        print(f"  有错误行数:      {result['error_lines']:,} ({result['error_lines']/result['line_count']*100:.1f}%)")
        print()

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
        perfect_rate = result['perfect_lines'] / result['line_count'] * 100
        emoji = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else "  "
        print(f"{emoji} {idx:<4} {result['model']:<30} {result['accuracy']:>11.2f}%  {perfect_rate:>10.1f}%")

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
        if worst['total_errors'] > 0:
            error_reduction = (worst['total_errors'] - best['total_errors']) / worst['total_errors'] * 100
            print(f"   错误减少: {worst['total_errors'] - best['total_errors']:,} 个字符 ({error_reduction:.1f}%)")
        print()

if __name__ == "__main__":
    main()
