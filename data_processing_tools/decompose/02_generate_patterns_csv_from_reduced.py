#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从当前目录下的.reduced文件生成patterns.csv
基于叙利亚文形态分析的模式提取规则

模式提取规则：
- 字母：[A-Z<>] 
- 其他都是 pattern
- 特殊规则：
  - "(" 右侧一个字母算在 "(" 内作为符号
  - ":" 右侧的一到两个小写字母都算在 ":" 内作为符号
- 符号在字母的前后出现
"""

import csv
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import glob


def is_letter(char):
    """判断字符是否为叙利亚文字母"""
    return char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ<>'


def extract_patterns_from_word(word):
    """
    从一个词中提取字母和模式
    
    返回: (letters, patterns)
    - letters: 字母列表
    - patterns: 模式列表，长度为 len(letters) + 1
    """
    letters = []
    patterns = []
    current_pattern = ""
    i = 0
    
    while i < len(word):
        char = word[i]
        
        if is_letter(char):
            # 遇到字母，保存前面的pattern
            patterns.append(current_pattern)
            letters.append(char)
            current_pattern = ""
            i += 1
        elif char == '(':
            # "(" 右侧一个字母算在 "(" 内作为符号
            current_pattern += char
            i += 1
            # 检查右侧是否有字母
            if i < len(word) and is_letter(word[i]):
                current_pattern += word[i]
                i += 1
        elif char == ':':
            # ":" 右侧的一到两个小写字母都算在 ":" 内作为符号
            current_pattern += char
            i += 1
            # 收集后面的小写字母（最多两个）
            collected = 0
            while i < len(word) and word[i].islower() and collected < 2:
                current_pattern += word[i]
                i += 1
                collected += 1
        else:
            # 普通符号
            current_pattern += char
            i += 1
    
    # 最后的pattern
    patterns.append(current_pattern)
    
    return letters, patterns


def process_reduced_file(file_path):
    """
    处理一个.reduced文件，提取所有模式
    
    返回: Counter对象，包含所有模式的计数
    """
    print(f"处理文件: {file_path.name}")
    pattern_counts = Counter()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line_num, line in enumerate(tqdm(lines, desc=f"处理 {file_path.name}"), 1):
        line = line.strip()
        if not line:
            continue
        
        # 分割成单词
        words = line.split()
        
        for word in words:
            if not word:
                continue
            
            try:
                letters, patterns = extract_patterns_from_word(word)
                
                # 验证提取结果
                if len(patterns) != len(letters) + 1:
                    print(f"⚠️ 行{line_num}，词'{word}': 模式数量不匹配 ({len(patterns)} patterns, {len(letters)} letters)")
                    continue
                
                # 统计所有模式
                for pattern in patterns:
                    pattern_counts[pattern] += 1
                    
            except Exception as e:
                print(f"❌ 行{line_num}，词'{word}': 处理出错 - {e}")
                continue
    
    return pattern_counts


def generate_patterns_csv():
    """
    生成patterns.csv文件
    """
    print("=" * 60)
    print("开始生成 patterns.csv")
    print("=" * 60)
    
    # 在当前目录查找.reduced文件
    current_dir = Path(".")
    reduced_files = list(current_dir.glob("*.out.reduced"))
    
    if not reduced_files:
        print("❌ 当前目录下未找到.reduced文件")
        print("请确保当前目录包含以下文件:")
        print("  - train.out.reduced")
        print("  - val.out.reduced") 
        print("  - test.out.reduced")
        return None, 0
    
    print(f"找到 {len(reduced_files)} 个.reduced文件:")
    for f in reduced_files:
        print(f"  - {f.name}")
    
    # 收集所有模式
    all_pattern_counts = Counter()
    
    for file_path in reduced_files:
        pattern_counts = process_reduced_file(file_path)
        all_pattern_counts.update(pattern_counts)
    
    print(f"\n✅ 总共发现 {len(all_pattern_counts)} 种不同的模式")
    print(f"✅ 总计模式出现次数: {sum(all_pattern_counts.values())}")
    
    # 按出现频率排序
    sorted_patterns = all_pattern_counts.most_common()
    
    # 生成CSV文件到当前目录
    output_file = current_dir / "patterns.csv"
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        # 写入标题行
        writer.writerow(['label', 'pattern', 'count'])
        
        # 写入数据行，按标签编号（0开始）
        for label, (pattern, count) in enumerate(sorted_patterns):
            writer.writerow([label, pattern, count])
        
        # 添加一个特殊的条目用于未知pattern
        # 使用下一个可用的label号
        unknown_label = len(sorted_patterns)
        writer.writerow([unknown_label, '<UNKNOWN>', 0])
        print(f"\n📌 添加了未知pattern条目: label={unknown_label}, pattern='<UNKNOWN>'")
    
    print(f"\n✅ 成功生成: {output_file}")
    
    # 显示前10个最频繁的模式
    print("\n📊 前10个最频繁的模式:")
    print("-" * 40)
    for i, (pattern, count) in enumerate(sorted_patterns[:10]):
        percentage = (count / sum(all_pattern_counts.values())) * 100
        pattern_display = f"'{pattern}'" if pattern else "''"
        print(f"{i:2d}. {pattern_display:<15} {count:8d} ({percentage:5.2f}%)")
    
    # 生成统计文件
    stats_file = current_dir / "pattern_statistics.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("Pattern\tCount\tPercentage\n")
        total_count = sum(all_pattern_counts.values())
        for pattern, count in sorted_patterns:
            percentage = (count / total_count) * 100
            f.write(f"{pattern}\t{count}\t{percentage:.4f}\n")
    
    print(f"✅ 成功生成统计文件: {stats_file}")
    
    return output_file, len(all_pattern_counts)


if __name__ == "__main__":
    try:
        output_file, pattern_count = generate_patterns_csv()
        if output_file:
            print(f"\n🎉 完成！生成了包含 {pattern_count} 种模式的 patterns.csv")
            print(f"📁 文件位置: {output_file.absolute()}")
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        raise