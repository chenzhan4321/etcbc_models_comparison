#!/usr/bin/env python3
"""
对比prediction.out.final和truth.out.final文件
将所有标签和字母都作为独立单位，计算编辑距离和正确率
"""

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

def levenshtein_distance_with_details(s1, s2):
    """计算编辑距离并返回详细的操作序列"""
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
    
    # 回溯找出操作序列
    operations = []
    i, j = m, n
    while i > 0 or j > 0:
        if i == 0:
            operations.append(('insert', None, s2[j-1], j-1))
            j -= 1
        elif j == 0:
            operations.append(('delete', s1[i-1], None, i-1))
            i -= 1
        elif s1[i-1] == s2[j-1]:
            operations.append(('match', s1[i-1], s2[j-1], i-1))
            i -= 1
            j -= 1
        else:
            min_val = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
            if dp[i-1][j-1] == min_val:
                operations.append(('replace', s1[i-1], s2[j-1], i-1))
                i -= 1
                j -= 1
            elif dp[i-1][j] == min_val:
                operations.append(('delete', s1[i-1], None, i-1))
                i -= 1
            else:
                operations.append(('insert', None, s2[j-1], j-1))
                j -= 1
    
    operations.reverse()
    return dp[m][n], operations

def format_unit(unit):
    """格式化单位显示"""
    if isinstance(unit, int):
        return f"[{unit}]"
    else:
        return f"'{unit}'"

def analyze_line(pred_units, truth_units, line_num):
    """分析一行的详细对比"""
    distance, operations = levenshtein_distance_with_details(pred_units, truth_units)
    
    print(f"\n第{line_num}行详细分析:")
    print(f"预测单位数: {len(pred_units)}")
    print(f"真实单位数: {len(truth_units)}")
    print(f"编辑距离: {distance}")
    
    # 显示对齐
    print("\n对齐详情:")
    pred_aligned = []
    truth_aligned = []
    symbols = []
    
    for op, pred_val, truth_val, _ in operations:
        if op == 'match':
            pred_aligned.append(format_unit(pred_val))
            truth_aligned.append(format_unit(truth_val))
            symbols.append("✓")
        elif op == 'replace':
            pred_aligned.append(format_unit(pred_val))
            truth_aligned.append(format_unit(truth_val))
            symbols.append("✗")
        elif op == 'delete':
            pred_aligned.append(format_unit(pred_val))
            truth_aligned.append("---")
            symbols.append("D")
        elif op == 'insert':
            pred_aligned.append("---")
            truth_aligned.append(format_unit(truth_val))
            symbols.append("I")
    
    # 分批显示（每次10个单位）
    batch_size = 10
    for i in range(0, len(pred_aligned), batch_size):
        batch_end = min(i + batch_size, len(pred_aligned))
        # 修复位置显示，确保准确反映实际元素范围
        if batch_end - i == 1:
            print(f"\n位置 {i}:")
        else:
            print(f"\n位置 {i}-{batch_end-1}:")
        print("预测: " + " ".join(pred_aligned[i:batch_end]))
        print("真实: " + " ".join(truth_aligned[i:batch_end]))
        print("状态: " + "   ".join(symbols[i:batch_end]))
    
    # 统计错误类型
    error_counts = {'replace': 0, 'delete': 0, 'insert': 0}
    for op, _, _, _ in operations:
        if op in error_counts:
            error_counts[op] += 1
    
    print(f"\n错误统计:")
    print(f"  替换错误: {error_counts['replace']}")
    print(f"  删除错误: {error_counts['delete']}")
    print(f"  插入错误: {error_counts['insert']}")
    print(f"  总错误数: {distance}")
    
    return distance

def calculate_overall_accuracy(pred_file, truth_file):
    """计算总体正确率"""
    
    total_units = 0
    total_errors = 0
    line_errors = []
    
    with open(pred_file, 'r', encoding='utf-8') as pf, \
         open(truth_file, 'r', encoding='utf-8') as tf:
        
        for line_num, (pred_line, truth_line) in enumerate(zip(pf, tf), 1):
            pred_units = parse_out_final_line(pred_line)
            truth_units = parse_out_final_line(truth_line)
            
            if not truth_units:
                continue
            
            # 计算这一行的编辑距离
            distance, _ = levenshtein_distance_with_details(pred_units, truth_units)
            
            total_units += len(truth_units)
            total_errors += distance
            
            if distance > 0:
                line_errors.append({
                    'line': line_num,
                    'pred_len': len(pred_units),
                    'truth_len': len(truth_units),
                    'distance': distance
                })
    
    accuracy = (total_units - total_errors) / total_units * 100 if total_units > 0 else 0
    
    return {
        'total_units': total_units,
        'total_errors': total_errors,
        'correct_units': total_units - total_errors,
        'accuracy': accuracy,
        'line_errors': line_errors
    }

def main():
    """主函数"""
    pred_file = "test.out.final"
    truth_file = "test.out.final.truth"
    
    print("=" * 60)
    print("叙利亚文形态分析 - 单位级别正确率分析")
    print("=" * 60)
    
    # 首先读取第731行和第622行
    with open(pred_file, 'r', encoding='utf-8') as pf, \
         open(truth_file, 'r', encoding='utf-8') as tf:
        
        pred_lines = pf.readlines()
        truth_lines = tf.readlines()
    
    # 分析第731行
    pred_731 = parse_out_final_line(pred_lines[730])  # 0-indexed
    truth_731 = parse_out_final_line(truth_lines[730])
    
    print("\n" + "="*50)
    print("第731行分析")
    print("="*50)
    print(f"预测.final: {pred_lines[730].strip()}")
    print(f"真实.final: {truth_lines[730].strip()}")
    print(f"预测单位: {pred_731}")
    print(f"真实单位: {truth_731}")
    analyze_line(pred_731, truth_731, 731)
    
    # 分析第622行
    pred_622 = parse_out_final_line(pred_lines[621])
    truth_622 = parse_out_final_line(truth_lines[621])
    
    print("\n" + "="*50)
    print("第622行分析")
    print("="*50)
    print(f"预测.final: {pred_lines[621].strip()}")
    print(f"真实.final: {truth_lines[621].strip()}")
    print(f"预测单位: {pred_622}")
    print(f"真实单位: {truth_622}")
    analyze_line(pred_622, truth_622, 622)
    
    # 计算总体正确率
    print("\n" + "="*50)
    print("总体统计")
    print("="*50)
    
    results = calculate_overall_accuracy(pred_file, truth_file)
    
    print(f"\n总体结果:")
    print(f"  总单位数: {results['total_units']:,}")
    print(f"  正确单位数: {results['correct_units']:,}")
    print(f"  错误单位数: {results['total_errors']:,}")
    print(f"  单位正确率: {results['accuracy']:.2f}%")
    
    # 显示错误分布
    if results['line_errors']:
        error_dist = {}
        for err in results['line_errors']:
            dist = err['distance']
            error_dist[dist] = error_dist.get(dist, 0) + 1
        
        print(f"\n错误行数: {len(results['line_errors'])}")
        print("编辑距离分布:")
        for dist in sorted(error_dist.keys())[:10]:
            print(f"  距离{dist}: {error_dist[dist]}行")
    
    # 保存结果
    import json
    with open('unit_accuracy_report.json', 'w', encoding='utf-8') as f:
        # 只保存统计信息，不保存详细的行错误
        save_results = {
            'total_units': results['total_units'],
            'total_errors': results['total_errors'],
            'correct_units': results['correct_units'],
            'accuracy': results['accuracy'],
            'total_error_lines': len(results['line_errors'])
        }
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: unit_accuracy_report.json")

if __name__ == "__main__":
    main()