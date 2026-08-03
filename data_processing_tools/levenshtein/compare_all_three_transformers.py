#!/usr/bin/env python3
"""
综合对比三个 Transformer 模型的 Levenshtein 正确率
- transformer_s2_on_s2: S2 训练，S2 测试
- transformer_s4_on_s2: S4 训练，S2 测试
- transformer_s4_on_s4: S4 训练，S4 测试
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
    """计算字符级别的正确率"""

    total_chars = 0
    total_errors = 0
    line_count = 0
    error_lines = 0
    perfect_lines = 0

    with open(pred_file, 'r', encoding='utf-8') as pf, \
         open(truth_file, 'r', encoding='utf-8') as tf:

        for pred_line, truth_line in zip(pf, tf):
            # 只去除换行符
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

    # 定义三个模型的配置
    models = [
        {
            'name': 'transformer_s2_on_s2',
            'desc': 'S2训练 + S2测试',
            'pred': Path("<REPO_ROOT>/outputs/transformer_s2_on_s2/test.out.original"),
            'truth': Path("<REPO_ROOT>/data/raw_s2_on_s2/test.out.original")
        },
        {
            'name': 'transformer_s4_on_s2',
            'desc': 'S4训练 + S2测试',
            'pred': Path("<REPO_ROOT>/outputs/transformer_s4_on_s2/test.out.original"),
            'truth': Path("<REPO_ROOT>/data/raw_s2_on_s2/test.out.original")
        },
        {
            'name': 'transformer_s4_on_s4',
            'desc': 'S4训练 + S4测试',
            'pred': Path("<REPO_ROOT>/outputs/transformer_s4_on_s4/test.out.original"),
            'truth': Path("<REPO_ROOT>/outputs/transformer_s4_on_s4/test.out.original.truth")
        }
    ]

    print("=" * 90)
    print("三个 Transformer 模型综合对比 - Levenshtein 字符正确率分析")
    print("=" * 90)
    print()

    results = []

    for model in models:
        model_name = model['name']
        model_desc = model['desc']
        pred_file = model['pred']
        truth_file = model['truth']

        if not pred_file.exists():
            print(f"⚠️  警告: {model_name} 预测文件不存在，跳过")
            continue

        if not truth_file.exists():
            print(f"⚠️  警告: {model_name} 标准文件不存在，跳过")
            continue

        print(f"正在分析: {model_name} ({model_desc})")
        print("-" * 90)

        result = calculate_accuracy(pred_file, truth_file)

        print(f"  总字符数:        {result['total_chars']:,}")
        print(f"  正确字符数:      {result['correct_chars']:,}")
        print(f"  错误字符数:      {result['total_errors']:,}")
        print(f"  字符正确率:      {result['accuracy']:.4f}%")
        print()
        print(f"  总行数:          {result['line_count']:,}")
        print(f"  完美预测行数:    {result['perfect_lines']:,} ({result['perfect_lines']/result['line_count']*100:.2f}%)")
        print(f"  有错误行数:      {result['error_lines']:,} ({result['error_lines']/result['line_count']*100:.2f}%)")
        print()

        results.append({
            'model': model_name,
            'desc': model_desc,
            **result
        })

    # 排序并显示排名
    if len(results) >= 2:
        results.sort(key=lambda x: x['accuracy'], reverse=True)

        print("=" * 90)
        print("排名汇总（按字符正确率）")
        print("=" * 90)
        print(f"{'排名':<6} {'模型':<25} {'配置':<20} {'字符正确率':<14} {'完美行率':<12}")
        print("-" * 90)

        for idx, result in enumerate(results, 1):
            perfect_rate = result['perfect_lines'] / result['line_count'] * 100
            emoji = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉"
            print(f"{emoji} {idx:<4} {result['model']:<25} {result['desc']:<20} {result['accuracy']:>11.4f}%  {perfect_rate:>10.2f}%")

        print()

        # 性能对比
        print("=" * 90)
        print("关键发现")
        print("=" * 90)

        best = results[0]
        worst = results[-1]

        print(f"\n🏆 最佳模型: {best['model']} ({best['desc']})")
        print(f"   字符正确率: {best['accuracy']:.4f}%")
        print(f"   完美预测行: {best['perfect_lines']:,} / {best['line_count']:,} ({best['perfect_lines']/best['line_count']*100:.2f}%)")
        print(f"   错误字符数: {best['total_errors']:,}")

        # S2测试集模型对比
        s2_models = [r for r in results if 's2_on_s2' in r['model'] or 's4_on_s2' in r['model']]
        if len(s2_models) >= 2:
            s2_models.sort(key=lambda x: x['accuracy'], reverse=True)
            print(f"\n📊 S2测试集对比:")
            print(f"   最佳: {s2_models[0]['model']} - {s2_models[0]['accuracy']:.4f}%")
            print(f"   对比: {s2_models[1]['model']} - {s2_models[1]['accuracy']:.4f}%")
            print(f"   差距: {s2_models[0]['accuracy'] - s2_models[1]['accuracy']:.4f} 个百分点")

        # S4模型对比
        s4_model = [r for r in results if 's4_on_s4' in r['model']]
        if s4_model:
            print(f"\n🔬 S4测试集性能:")
            print(f"   {s4_model[0]['model']} - {s4_model[0]['accuracy']:.4f}%")
            print(f"   完美预测行率: {s4_model[0]['perfect_lines']/s4_model[0]['line_count']*100:.2f}%")
            print(f"   测试规模: {s4_model[0]['line_count']:,} 行 (S2的 {s4_model[0]['line_count']/10869:.1f}x)")

        print()

if __name__ == "__main__":
    main()
