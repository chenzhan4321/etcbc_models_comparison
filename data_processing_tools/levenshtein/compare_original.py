#!/usr/bin/env python3
"""
对比 .original 文件的 Levenshtein 正确率
按字符级别计算编辑距离
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

def calculate_accuracy(pred_file, truth_file):
    """计算字符级别的正确率"""

    total_chars = 0
    total_errors = 0
    line_count = 0
    error_lines = 0
    perfect_lines = 0

    with open(pred_file, 'r', encoding='utf-8') as pf, \
         open(truth_file, 'r', encoding='utf-8') as tf:

        for pred_line, truth_line in zip(pf, tf):
            # 去除行尾换行符，但保留其他空格
            pred_line = pred_line.rstrip('\n')
            truth_line = truth_line.rstrip('\n')

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
    truth_file = Path("/Users/zhanchen/Library/CloudStorage/Dropbox/Projects/etcbc_update/data/raw_s2_on_s2/test.out.original")

    # 定义要测试的三个模型
    test_models = [
        "/Users/zhanchen/Library/CloudStorage/Dropbox/Projects/etcbc_update/outputs/transformer_s2_on_s2/test.out.original",
        "/Users/zhanchen/Library/CloudStorage/Dropbox/Projects/etcbc_update/outputs/transformer_s4_on_s2/test.out.original",
        "/Users/zhanchen/Library/CloudStorage/Dropbox/Projects/etcbc_update/outputs/transformer_s4_on_s4/test.out.original"
    ]

    print("=" * 80)
    print("Transformer 模型 - Levenshtein 字符正确率分析 (.original 格式)")
    print("=" * 80)
    print()

    if not truth_file.exists():
        print(f"错误: 找不到真实标签文件 {truth_file}")
        return

    results = []

    for pred_file_path in test_models:
        pred_file = Path(pred_file_path)

        if not pred_file.exists():
            print(f"警告: 文件不存在 - {pred_file}")
            continue

        model_name = pred_file.parent.name

        print(f"正在分析: {model_name}")
        print("-" * 80)

        result = calculate_accuracy(pred_file, truth_file)

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

        print(f"\n📊 最差模型: {worst['model']}")
        print(f"   字符正确率: {worst['accuracy']:.2f}%")
        print(f"   完美预测行: {worst['perfect_lines']:,} / {worst['line_count']:,} ({worst['perfect_lines']/worst['line_count']*100:.1f}%)")

        print(f"\n📈 性能差距: {best['accuracy'] - worst['accuracy']:.2f} 个百分点")
        print(f"   错误减少: {worst['total_errors'] - best['total_errors']:,} 个字符 ({(worst['total_errors'] - best['total_errors'])/worst['total_errors']*100:.1f}%)")
        print()

if __name__ == "__main__":
    main()
