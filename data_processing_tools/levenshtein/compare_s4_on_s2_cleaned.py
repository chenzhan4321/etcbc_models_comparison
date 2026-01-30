#!/usr/bin/env python3
"""
对比清理后的 transformer_s4_on_s2 的 .original 文件与标准文件
精确字符对比
"""

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
    """计算字符级别的正确率 - 精确对比"""

    total_chars = 0
    total_errors = 0
    line_count = 0
    error_lines = 0
    perfect_lines = 0

    with open(pred_file, 'r', encoding='utf-8') as pf, \
         open(truth_file, 'r', encoding='utf-8') as tf:

        for pred_line, truth_line in zip(pf, tf):
            # 只去除换行符，保留所有其他字符
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

    pred_file = Path("/Users/zhanchen/Library/CloudStorage/Dropbox/Projects/etcbc_update/outputs/transformer_s4_on_s2/test.out.original")
    truth_file = Path("/Users/zhanchen/Library/CloudStorage/Dropbox/Projects/etcbc_update/data/raw_s2_on_s2/test.out.original")

    print("=" * 80)
    print("transformer_s4_on_s2 - Levenshtein 字符正确率分析")
    print("（清理 Truevalue 行和 Predicted 前缀后的精确对比）")
    print("=" * 80)
    print()
    print(f"预测文件: {pred_file}")
    print(f"标准文件: {truth_file}")
    print()

    if not pred_file.exists():
        print(f"❌ 错误: 预测文件不存在")
        return

    if not truth_file.exists():
        print(f"❌ 错误: 标准文件不存在")
        return

    print("正在计算...")
    result = calculate_accuracy(pred_file, truth_file)

    print()
    print("=" * 80)
    print("分析结果")
    print("=" * 80)
    print(f"  总字符数:        {result['total_chars']:,}")
    print(f"  正确字符数:      {result['correct_chars']:,}")
    print(f"  错误字符数:      {result['total_errors']:,}")
    print(f"  字符正确率:      {result['accuracy']:.4f}%")
    print()
    print(f"  总行数:          {result['line_count']:,}")
    print(f"  完美预测行数:    {result['perfect_lines']:,} ({result['perfect_lines']/result['line_count']*100:.2f}%)")
    print(f"  有错误行数:      {result['error_lines']:,} ({result['error_lines']/result['line_count']*100:.2f}%)")
    print()

    # 显示一些错误示例
    if result['error_lines'] > 0:
        print("=" * 80)
        print("错误示例（前5个）")
        print("=" * 80)

        with open(pred_file, 'r', encoding='utf-8') as pf, \
             open(truth_file, 'r', encoding='utf-8') as tf:

            error_count = 0
            for line_num, (pred_line, truth_line) in enumerate(zip(pf, tf), 1):
                pred_line = pred_line.rstrip('\n')
                truth_line = truth_line.rstrip('\n')

                if not truth_line:
                    continue

                distance = levenshtein_distance(pred_line, truth_line)

                if distance > 0 and error_count < 5:
                    print(f"\n第 {line_num} 行 (编辑距离: {distance}):")
                    print(f"  预测: {pred_line[:100]}{'...' if len(pred_line) > 100 else ''}")
                    print(f"  真实: {truth_line[:100]}{'...' if len(truth_line) > 100 else ''}")
                    error_count += 1

                if error_count >= 5:
                    break

    print()

if __name__ == "__main__":
    main()
