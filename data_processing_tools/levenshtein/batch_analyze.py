#!/usr/bin/env python3
"""
批量分析所有模型的 Levenshtein 正确率
"""

import os
import json
from pathlib import Path

def parse_out_final_line(line):
    """解析.out.final格式的行，提取所有单位（标签和字母）"""
    units = []
    line = line.strip()
    if not line:
        return units

    i = 0
    while i < len(line):
        # 读取数字（标签）
        if line[i].isdigit():
            label = ""
            while i < len(line) and line[i].isdigit():
                label += line[i]
                i += 1
            units.append(int(label))
        # 读取字母
        elif line[i].isalpha() or line[i] in '<>':
            units.append(line[i])
            i += 1
        # 跳过空格
        elif line[i] == ' ':
            i += 1
        else:
            i += 1

    return units

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
    """计算总体正确率"""

    total_units = 0
    total_errors = 0
    line_count = 0
    error_lines = 0

    with open(pred_file, 'r', encoding='utf-8') as pf, \
         open(truth_file, 'r', encoding='utf-8') as tf:

        for pred_line, truth_line in zip(pf, tf):
            pred_units = parse_out_final_line(pred_line)
            truth_units = parse_out_final_line(truth_line)

            if not truth_units:
                continue

            line_count += 1

            # 计算这一行的编辑距离
            distance = levenshtein_distance(pred_units, truth_units)

            total_units += len(truth_units)
            total_errors += distance

            if distance > 0:
                error_lines += 1

    accuracy = (total_units - total_errors) / total_units * 100 if total_units > 0 else 0

    return {
        'total_units': total_units,
        'total_errors': total_errors,
        'correct_units': total_units - total_errors,
        'accuracy': accuracy,
        'line_count': line_count,
        'error_lines': error_lines,
        'perfect_lines': line_count - error_lines
    }

def main():
    """批量分析所有模型"""

    tested_dir = Path("tested")
    truth_file = "test.out.final.truth"

    if not truth_file:
        print(f"错误: 找不到真实标签文件 {truth_file}")
        return

    # 获取所有测试文件
    test_files = sorted(tested_dir.glob("test.out.final.*"))

    if not test_files:
        print(f"错误: 在 {tested_dir} 目录下找不到测试文件")
        return

    print("=" * 80)
    print("叙利亚文形态分析 - Levenshtein 正确率批量分析")
    print("=" * 80)
    print()

    results_summary = []

    for test_file in test_files:
        model_name = test_file.name.replace("test.out.final.", "")

        print(f"正在分析: {model_name}")
        print("-" * 80)

        result = calculate_accuracy(test_file, truth_file)

        print(f"  总单位数:        {result['total_units']:,}")
        print(f"  正确单位数:      {result['correct_units']:,}")
        print(f"  错误单位数:      {result['total_errors']:,}")
        print(f"  单位正确率:      {result['accuracy']:.2f}%")
        print(f"  总行数:          {result['line_count']:,}")
        print(f"  完美预测行数:    {result['perfect_lines']:,}")
        print(f"  有错误行数:      {result['error_lines']:,}")
        print()

        results_summary.append({
            'model': model_name,
            **result
        })

    # 按准确率排序
    results_summary.sort(key=lambda x: x['accuracy'], reverse=True)

    print("=" * 80)
    print("排名汇总 (按单位正确率)")
    print("=" * 80)
    print(f"{'排名':<6} {'模型':<30} {'单位正确率':<12} {'完美行数':<12}")
    print("-" * 80)

    for idx, result in enumerate(results_summary, 1):
        print(f"{idx:<6} {result['model']:<30} {result['accuracy']:>10.2f}%  {result['perfect_lines']:>10,}")

    print()

    # 保存详细结果
    output_file = "batch_accuracy_report.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print(f"详细结果已保存到: {output_file}")
    print()

    # 显示最佳和最差模型
    print("=" * 80)
    print("性能对比")
    print("=" * 80)

    best = results_summary[0]
    worst = results_summary[-1]

    print(f"\n🏆 最佳模型: {best['model']}")
    print(f"   单位正确率: {best['accuracy']:.2f}%")
    print(f"   完美预测行: {best['perfect_lines']:,} / {best['line_count']:,}")

    print(f"\n📊 最差模型: {worst['model']}")
    print(f"   单位正确率: {worst['accuracy']:.2f}%")
    print(f"   完美预测行: {worst['perfect_lines']:,} / {worst['line_count']:,}")

    print(f"\n📈 性能差距: {best['accuracy'] - worst['accuracy']:.2f} 个百分点")
    print()

if __name__ == "__main__":
    main()
