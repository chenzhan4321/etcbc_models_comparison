#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将.reduced文件转换为.final文件
支持两种输出格式：
1. .final格式：纯数字ID序列（字符ID和标签ID交替）
2. .out.final格式：标签ID和字母交替（保留原始字母）

重要格式说明：
- 每行以空格开头
- 每个字母前都有一个标签（没有pattern时用0）
- 格式：[空格]标签字母标签字母... 标签字母标签字母...

Pattern提取规则：
- '(' 右侧第一个大写字母或'<'、'>'包含在 '(' 内算作一个pattern
- ':' 右侧的 d/p/dp 都包含在 ':' 内算作一个pattern
"""

import csv
from pathlib import Path
from tqdm import tqdm
import argparse

def is_letter(char):
    """判断字符是否为叙利亚文字母"""
    return char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ<>'

def load_pattern_to_label_mapping(csv_file):
    """
    加载patterns.csv文件，建立pattern到label的映射
    返回：pattern_to_label字典, unknown_label, pattern_counts
    """
    pattern_to_label = {}
    pattern_counts = {}  # 存储每个pattern的预设count
    unknown_label = None
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = int(row['label'])
            pattern = row['pattern']
            count = int(row['count']) if row['count'] else 0
            pattern_to_label[pattern] = label
            pattern_counts[pattern] = count
            # 记录<UNKNOWN>的标签号
            if pattern == '<UNKNOWN>':
                unknown_label = label
                print("\n" + "⚠️" * 20)
                print("⚠️ 警报：发现<UNKNOWN> pattern！")
                print(f"⚠️ Label: {label}, Count: {count}")
                print("⚠️ 处理时将对未知pattern发出警告")
                print("⚠️" * 20 + "\n")
    
    print(f"✅ 加载了{len(pattern_to_label)}个pattern到label的映射")
    if unknown_label is not None:
        print(f"✅ 未知pattern将使用标签: {unknown_label}")
    
    # 空pattern应该映射到0
    if '' not in pattern_to_label:
        pattern_to_label[''] = 0
        pattern_counts[''] = 0
    
    # 显示前20个最常用的pattern
    sorted_patterns = sorted([(p, c, pattern_to_label[p]) for p, c in pattern_counts.items() if p != ''], 
                            key=lambda x: x[1], reverse=True)
    
    print("\n" + "="*60)
    print("📊 前20个最常用Pattern:")
    print("="*60)
    print(f"{'排名':<6} {'标签':<8} {'Pattern':<20} {'预设次数':<12}")
    print("-"*60)
    for i, (pattern, count, label) in enumerate(sorted_patterns[:20], 1):
        display_pattern = pattern if len(pattern) <= 15 else pattern[:12] + '...'
        print(f"{i:<6} {label:<8} {display_pattern:<20} {count:>10,}次")
    
    # 显示后20个最少用的pattern
    print("\n" + "="*60)
    print("📊 后20个最少用Pattern:")
    print("="*60)
    print(f"{'排名':<6} {'标签':<8} {'Pattern':<20} {'预设次数':<12}")
    print("-"*60)
    bottom_20 = sorted_patterns[-20:]
    for i, (pattern, count, label) in enumerate(bottom_20, 1):
        display_pattern = pattern if pattern != '<UNKNOWN>' else '⚠️ UNKNOWN ⚠️'
        if len(display_pattern) > 15 and display_pattern != '⚠️ UNKNOWN ⚠️':
            display_pattern = display_pattern[:12] + '...'
        print(f"{i:<6} {label:<8} {display_pattern:<20} {count:>10,}次")
    
    print("="*60 + "\n")
    
    return pattern_to_label, unknown_label, pattern_counts

def char_to_id(char):
    """
    将字符转换为ID
    基于CLAUDE.md中的字符映射系统
    """
    char_map = {
        ' ': 0, '>': 1, 'B': 2, 'G': 3, 'D': 4, 'H': 5, 'W': 6, 'Z': 7,
        'X': 8, 'V': 9, 'J': 10, 'K': 11, 'L': 12, 'M': 13, 'N': 14, 'S': 15,
        '<': 16, 'P': 17, 'Y': 18, 'Q': 19, 'R': 20, 'C': 21, 'T': 22,
        '^': 23, '#': 24, 'A': 25
    }
    return char_map.get(char, -1)  # 返回-1表示未知字符

def convert_word_to_out_final(word, pattern_to_label, unknown_label=None, line_num=None, word_idx=None, silent_unknown=False):
    """
    将一个单词转换为.out.final格式
    格式：标签字母标签字母...标签（每个字母前后都有标签）
    
    例如：W-B-H -> 0W1B1H0
    
    参数：
    - line_num: 当前行号（用于错误报告）
    - word_idx: 当前单词索引（用于错误报告）
    """
    letters = []
    patterns = []
    current_pattern = ""
    i = 0
    
    while i < len(word):
        char = word[i]
        
        # 检查是否是独立的字母（不在'('后面）
        if is_letter(char) and (i == 0 or word[i-1] != '('):
            # 保存之前的pattern
            patterns.append(current_pattern)
            current_pattern = ""
            # 保存字母
            letters.append(char)
            i += 1
        elif char == '(':
            # '(' 开始一个可能包含字母的pattern
            current_pattern += char
            i += 1
            # 查看后面是否有字母
            if i < len(word):
                next_char = word[i]
                if next_char.isupper() or next_char in '<>':
                    # 这个字母是pattern的一部分
                    current_pattern += next_char
                    i += 1
                    # 继续收集直到遇到真正的字母或到达末尾
                    while i < len(word):
                        if is_letter(word[i]) and word[i-1] != '(':
                            # 遇到独立的字母，停止
                            break
                        if word[i] == '(' and i + 1 < len(word) and (word[i+1].isupper() or word[i+1] in '<>'):
                            # 遇到另一个'(字母'模式，停止
                            break
                        current_pattern += word[i]
                        i += 1
        elif char == ':':
            # ':' 右侧的 d/p/dp 都包含在 ':' 内
            current_pattern += char
            i += 1
            while i < len(word) and word[i].lower() in 'dp':
                current_pattern += word[i]
                i += 1
        else:
            # 普通符号
            current_pattern += char
            i += 1
    
    # 最后的pattern
    patterns.append(current_pattern)
    
    # 确保pattern数量正确
    if len(patterns) != len(letters) + 1:
        print(f"⚠️ Pattern数量不匹配：{len(patterns)} patterns, {len(letters)} letters")
    
    # 构建结果
    result = []
    for i, letter in enumerate(letters):
        # 添加字母前的标签
        if i < len(patterns):
            pattern = patterns[i]
            if pattern and pattern not in pattern_to_label:
                if unknown_label is not None:
                    location_info = ""
                    if line_num is not None:
                        location_info += f"第{line_num}行"
                    if word_idx is not None:
                        location_info += f" 第{word_idx+1}个单词"
                    if location_info:
                        location_info = f" [{location_info}]"
                    print(f"⚠️ 警告{location_info}：未知pattern '{pattern}'，使用<UNKNOWN>标签 {unknown_label}")
                    print(f"   原始单词: {word}")
                    label = unknown_label
                else:
                    location_info = ""
                    if line_num is not None:
                        location_info += f"第{line_num}行"
                    if word_idx is not None:
                        location_info += f" 第{word_idx+1}个单词"
                    if location_info:
                        location_info = f" [{location_info}]"
                    print(f"❌ 错误{location_info}：在patterns.csv中未找到pattern '{pattern}'的映射")
                    print(f"   原始单词: {word}")
                    print(f"   请确保<UNKNOWN>已在patterns.csv中定义，使用默认标签0继续")
                    label = 0  # 使用默认标签0继续处理
            else:
                label = pattern_to_label.get(pattern, 0)
            result.append(str(label))
        else:
            result.append('0')
        # 添加字母
        result.append(letter)
    
    # 添加最后的标签
    if len(patterns) > len(letters):
        last_pattern = patterns[-1]
        if last_pattern and last_pattern not in pattern_to_label:
            if unknown_label is not None:
                location_info = ""
                if line_num is not None:
                    location_info += f"第{line_num}行"
                if word_idx is not None:
                    location_info += f" 第{word_idx+1}个单词"
                if location_info:
                    location_info = f" [{location_info}]"
                print(f"⚠️ 警告{location_info}：未知pattern '{last_pattern}'（末尾），使用<UNKNOWN>标签 {unknown_label}")
                print(f"   原始单词: {word}")
                last_label = unknown_label
            else:
                location_info = ""
                if line_num is not None:
                    location_info += f"第{line_num}行"
                if word_idx is not None:
                    location_info += f" 第{word_idx+1}个单词"
                if location_info:
                    location_info = f" [{location_info}]"
                print(f"❌ 错误{location_info}：在patterns.csv中未找到pattern '{last_pattern}'（末尾）的映射")
                print(f"   原始单词: {word}")
                print(f"   请确保<UNKNOWN>已在patterns.csv中定义，使用默认标签0继续")
                last_label = 0  # 使用默认标签0继续处理
        else:
            last_label = pattern_to_label.get(last_pattern, 0)
        result.append(str(last_label))
    else:
        result.append('0')
    
    return ''.join(result)

def extract_pattern_before_position(word, pos, pattern_to_label):
    """提取位置pos之前的pattern"""
    if pos == 0:
        return ''
    return extract_pattern_between_positions(word, 0, pos, pattern_to_label)

def extract_pattern_between_positions(word, start, end, pattern_to_label):
    """提取start到end之间的pattern"""
    if start >= end:
        return ''
    
    substring = word[start:end]
    
    # 如果整个substring在pattern_to_label中，直接返回
    if substring in pattern_to_label:
        return substring
    
    # 否则，尝试按照特殊规则解析
    i = 0
    result_pattern = ""
    
    while i < len(substring):
        char = substring[i]
        
        if char == '(':
            # 特殊处理：'(' 右侧第一个大写字母或'<'、'>'包含在 '(' 内
            pattern = char
            i += 1
            if i < len(substring):
                next_char = substring[i]
                if next_char.isupper() or next_char in '<>':
                    pattern += next_char
                    i += 1
            result_pattern += pattern
        elif char == ':':
            # 特殊处理：':' 右侧的 d/p/dp 都包含在 ':' 内
            pattern = char
            i += 1
            while i < len(substring) and substring[i].lower() in 'dp':
                pattern += substring[i]
                i += 1
            result_pattern += pattern
        else:
            # 普通字符
            result_pattern += char
            i += 1
    
    return result_pattern

def convert_word_to_final(word, pattern_to_label, unknown_label=None, line_num=None, word_idx=None, silent_unknown=False):
    """
    将一个单词转换为.final格式（纯数字）
    格式：标签ID 字符ID 标签ID 字符ID...标签ID
    
    参数：
    - line_num: 当前行号（用于错误报告）
    - word_idx: 当前单词索引（用于错误报告）
    """
    letters = []
    patterns = []
    current_pattern = ""
    i = 0
    
    while i < len(word):
        char = word[i]
        
        # 检查是否是独立的字母（不在'('后面）
        if is_letter(char) and (i == 0 or word[i-1] != '('):
            # 保存之前的pattern
            patterns.append(current_pattern)
            current_pattern = ""
            # 保存字母
            letters.append(char)
            i += 1
        elif char == '(':
            # '(' 开始一个可能包含字母的pattern
            current_pattern += char
            i += 1
            # 查看后面是否有字母
            if i < len(word):
                next_char = word[i]
                if next_char.isupper() or next_char in '<>':
                    # 这个字母是pattern的一部分
                    current_pattern += next_char
                    i += 1
                    # 继续收集直到遇到真正的字母或到达末尾
                    while i < len(word):
                        if is_letter(word[i]) and word[i-1] != '(':
                            # 遇到独立的字母，停止
                            break
                        if word[i] == '(' and i + 1 < len(word) and (word[i+1].isupper() or word[i+1] in '<>'):
                            # 遇到另一个'(字母'模式，停止
                            break
                        current_pattern += word[i]
                        i += 1
        elif char == ':':
            # ':' 右侧的 d/p/dp 都包含在 ':' 内
            current_pattern += char
            i += 1
            while i < len(word) and word[i].lower() in 'dp':
                current_pattern += word[i]
                i += 1
        else:
            # 普通符号
            current_pattern += char
            i += 1
    
    # 最后的pattern
    patterns.append(current_pattern)
    
    # 构建结果
    result = []
    for i, letter in enumerate(letters):
        # 添加字母前的标签
        if i < len(patterns):
            pattern = patterns[i]
            if pattern and pattern not in pattern_to_label:
                if unknown_label is not None:
                    location_info = ""
                    if line_num is not None:
                        location_info += f"第{line_num}行"
                    if word_idx is not None:
                        location_info += f" 第{word_idx+1}个单词"
                    if location_info:
                        location_info = f" [{location_info}]"
                    print(f"⚠️ 警告{location_info}：未知pattern '{pattern}'，使用<UNKNOWN>标签 {unknown_label}")
                    print(f"   原始单词: {word}")
                    label = unknown_label
                else:
                    location_info = ""
                    if line_num is not None:
                        location_info += f"第{line_num}行"
                    if word_idx is not None:
                        location_info += f" 第{word_idx+1}个单词"
                    if location_info:
                        location_info = f" [{location_info}]"
                    print(f"❌ 错误{location_info}：在patterns.csv中未找到pattern '{pattern}'的映射")
                    print(f"   原始单词: {word}")
                    print(f"   请确保<UNKNOWN>已在patterns.csv中定义，使用默认标签0继续")
                    label = 0  # 使用默认标签0继续处理
            else:
                label = pattern_to_label.get(pattern, 0)
            result.append(str(label))
        else:
            result.append('0')
        # 添加字符ID
        char_id = char_to_id(letter)
        if char_id == -1:
            location_info = ""
            if line_num is not None:
                location_info += f"第{line_num}行"
            if word_idx is not None:
                location_info += f" 第{word_idx+1}个单词"
            if location_info:
                location_info = f" [{location_info}]"
            print(f"❌ 错误{location_info}：未知字符 '{letter}'")
            print(f"   原始单词: {word}")
            print(f"   使用默认0继续处理")
            char_id = 0  # 使用默认值继续
        result.append(str(char_id))
    
    # 添加最后的标签
    if len(patterns) > len(letters):
        last_pattern = patterns[-1]
        if last_pattern and last_pattern not in pattern_to_label:
            if unknown_label is not None:
                location_info = ""
                if line_num is not None:
                    location_info += f"第{line_num}行"
                if word_idx is not None:
                    location_info += f" 第{word_idx+1}个单词"
                if location_info:
                    location_info = f" [{location_info}]"
                print(f"⚠️ 警告{location_info}：未知pattern '{last_pattern}'（末尾），使用<UNKNOWN>标签 {unknown_label}")
                print(f"   原始单词: {word}")
                last_label = unknown_label
            else:
                location_info = ""
                if line_num is not None:
                    location_info += f"第{line_num}行"
                if word_idx is not None:
                    location_info += f" 第{word_idx+1}个单词"
                if location_info:
                    location_info = f" [{location_info}]"
                print(f"❌ 错误{location_info}：在patterns.csv中未找到pattern '{last_pattern}'（末尾）的映射")
                print(f"   原始单词: {word}")
                print(f"   请确保<UNKNOWN>已在patterns.csv中定义，使用默认标签0继续")
                last_label = 0  # 使用默认标签0继续处理
        else:
            last_label = pattern_to_label.get(last_pattern, 0)
        result.append(str(last_label))
    else:
        result.append('0')
    
    return ''.join(result)

def convert_reduced_to_final(reduced_file, final_file, pattern_to_label, unknown_label, output_format='mixed'):
    """
    将整个reduced文件转换为final文件
    
    参数：
    - reduced_file: 输入的.reduced文件路径
    - final_file: 输出的.final或.out.final文件路径
    - pattern_to_label: pattern到标签的映射
    - output_format: 输出格式（'numeric'或'mixed'）
    
    返回：
    - pattern_usage: pattern使用统计字典
    """
    print(f"\n📝 转换文件: {reduced_file} -> {final_file}")
    print(f"   输出格式: {'纯数字(.final)' if output_format == 'numeric' else '字母+标签(.out.final)'}")
    
    # 统计pattern使用次数
    pattern_usage = {}
    
    with open(reduced_file, 'r', encoding='utf-8') as infile, \
         open(final_file, 'w', encoding='utf-8') as outfile:
        
        lines = infile.readlines()
        for line_num, line in enumerate(tqdm(lines, desc="转换进度"), 1):
            line = line.strip()
            if not line:
                outfile.write('\n')
                continue
            
            # 分割成单词
            words = line.split()
            converted_words = []
            
            for word_idx, word in enumerate(words):
                # 转换每个单词
                if output_format == 'mixed':
                    # .out.final格式：标签字母标签字母...
                    converted = convert_word_to_out_final(word, pattern_to_label, unknown_label, line_num, word_idx)
                else:
                    # .final格式：纯数字
                    converted = convert_word_to_final(word, pattern_to_label, unknown_label, line_num, word_idx)
                converted_words.append(converted)
                
                # 统计pattern使用
                patterns_in_word = extract_patterns_from_word(word)
                for pattern in patterns_in_word:
                    if pattern:  # 不统计空pattern
                        # 如果pattern不在pattern_to_label中，统一计入<UNKNOWN>
                        if pattern not in pattern_to_label:
                            pattern_usage['<UNKNOWN>'] = pattern_usage.get('<UNKNOWN>', 0) + 1
                        else:
                            pattern_usage[pattern] = pattern_usage.get(pattern, 0) + 1
            
            # 写入转换后的行（注意：每行以空格开头）
            outfile.write(' ' + ' '.join(converted_words) + '\n')
    
    print(f"✅ 转换完成！")
    return pattern_usage

def extract_patterns_from_word(word):
    """
    从单词中提取所有pattern
    返回：pattern列表
    """
    patterns = []
    current_pattern = ""
    i = 0
    
    while i < len(word):
        char = word[i]
        
        # 检查是否是独立的字母（不在'('后面）
        if is_letter(char) and (i == 0 or word[i-1] != '('):
            # 保存之前的pattern
            patterns.append(current_pattern)
            current_pattern = ""
            i += 1
        elif char == '(':
            # '(' 开始一个可能包含字母的pattern
            current_pattern += char
            i += 1
            # 查看后面是否有字母
            if i < len(word):
                next_char = word[i]
                if next_char.isupper() or next_char in '<>':
                    # 这个字母是pattern的一部分
                    current_pattern += next_char
                    i += 1
                    # 继续收集直到遇到真正的字母或到达末尾
                    while i < len(word):
                        if is_letter(word[i]) and word[i-1] != '(':
                            # 遇到独立的字母，停止
                            break
                        if word[i] == '(' and i + 1 < len(word) and (word[i+1].isupper() or word[i+1] in '<>'):
                            # 遇到另一个'(字母'模式，停止
                            break
                        current_pattern += word[i]
                        i += 1
        elif char == ':':
            # ':' 右侧的 d/p/dp 都包含在 ':' 内
            current_pattern += char
            i += 1
            while i < len(word) and word[i].lower() in 'dp':
                current_pattern += word[i]
                i += 1
        else:
            # 普通符号
            current_pattern += char
            i += 1
    
    # 最后的pattern
    patterns.append(current_pattern)
    return patterns

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='将.reduced文件转换为.final或.out.final文件')
    parser.add_argument('--format', choices=['numeric', 'mixed'], default='mixed',
                        help='输出格式：numeric(纯数字.final)或mixed(字母+标签.out.final)')
    args = parser.parse_args()
    
    # 使用当前目录
    data_dir = Path('.')
    
    # 加载pattern到label的映射
    patterns_csv = data_dir / 'patterns.csv'
    if not patterns_csv.exists():
        raise FileNotFoundError(f"❌ 错误：在当前目录下未找到patterns.csv文件")
    pattern_to_label, unknown_label, pattern_counts = load_pattern_to_label_mapping(patterns_csv)
    
    # 根据输出格式决定文件后缀
    if args.format == 'numeric':
        # 生成.final文件（纯数字格式）
        files_to_convert = [
            ('train.out.reduced', 'train.final'),
            ('val.out.reduced', 'val.final'),
            ('test.out.reduced', 'test.final')
        ]
    else:
        # 生成.out.final文件（字母+标签格式）
        files_to_convert = [
            ('train.out.reduced', 'train.out.final'),
            ('val.out.reduced', 'val.out.final'),
            ('test.out.reduced', 'test.out.final')
        ]
    
    print("\n" + "=" * 60)
    print(f"开始转换.reduced文件到{'final' if args.format == 'numeric' else 'out.final'}格式")
    print("=" * 60)
    
    # 用于存储每个文件的pattern统计
    all_file_stats = {}
    
    for reduced_filename, final_filename in files_to_convert:
        reduced_path = data_dir / reduced_filename
        final_path = data_dir / final_filename
        
        if not reduced_path.exists():
            print(f"❌ 文件不存在: {reduced_path}")
            continue
        
        try:
            pattern_usage = convert_reduced_to_final(reduced_path, final_path, pattern_to_label, unknown_label, args.format)
            all_file_stats[reduced_filename] = pattern_usage
        except Exception as e:
            print(f"❌ 转换错误: {e}")
            print(f"   文件: {reduced_filename}")
            print(f"   继续处理其他文件...")
            continue  # 继续处理其他文件而不是停止
    
    print("\n" + "=" * 60)
    print("所有文件转换完成！")
    print("=" * 60)
    
    # 显示pattern使用统计
    if all_file_stats:
        print("\n" + "=" * 60)
        print("Pattern使用统计")
        print("=" * 60)
        
        for filename, pattern_usage in all_file_stats.items():
            print(f"\n📄 {filename}:")
            
            # 初始化所有patterns.csv中定义的pattern的计数为0
            complete_usage = {}
            for pattern in pattern_to_label.keys():
                complete_usage[pattern] = pattern_usage.get(pattern, 0)
            
            # 按使用次数排序
            sorted_patterns = sorted(complete_usage.items(), key=lambda x: x[1], reverse=True)
            
            # 显示所有pattern的使用情况
            print("\n  所有Pattern使用统计（包括未使用的）:")
            print(f"  {'Label':<8} {'Pattern':<20} {'Count':<12}")
            print("  " + "-" * 45)
            
            # 分批显示：使用过的和未使用的
            used_patterns = [(p, c) for p, c in sorted_patterns if c > 0]
            unused_patterns = [(p, c) for p, c in sorted_patterns if c == 0]
            
            # 显示使用过的pattern
            for pattern, count in used_patterns:
                label = pattern_to_label.get(pattern, '?')
                display_pattern = pattern
                if pattern == '<UNKNOWN>':
                    display_pattern = '⚠️ <UNKNOWN> ⚠️'
                elif len(pattern) > 15:
                    display_pattern = pattern[:12] + '...'
                print(f"  [{label:3}]     {display_pattern:<20} {count:>10,}次")
            
            # 显示统计摘要
            print("\n  " + "=" * 45)
            print(f"  使用过的pattern: {len(used_patterns)}种")
            print(f"  未使用的pattern: {len(unused_patterns)}种")
            print(f"  patterns.csv中总计: {len(pattern_to_label)}种")
            total_uses = sum(complete_usage.values())
            print(f"  总使用次数: {total_uses:,}次")
            
            # 如果有<UNKNOWN>的使用，特别提醒
            if '<UNKNOWN>' in complete_usage and complete_usage['<UNKNOWN>'] > 0:
                print(f"\n  ⚠️ 警告：发现{complete_usage['<UNKNOWN>']}个未知pattern被映射到<UNKNOWN>！")
            
            # 不显示未使用的pattern列表，只显示统计数字
    
    # 验证转换结果
    print("\n验证示例（前3行）:")
    for _, final_filename in files_to_convert[:1]:  # 只显示第一个文件的示例
        final_path = data_dir / final_filename
        if final_path.exists():
            print(f"\n{final_filename}:")
            with open(final_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 3:
                        break
                    # 限制显示长度
                    display_line = line.rstrip()
                    if len(display_line) > 100:
                        display_line = display_line[:100] + "..."
                    print(f"  行{i+1}: {display_line}")

if __name__ == "__main__":
    main()