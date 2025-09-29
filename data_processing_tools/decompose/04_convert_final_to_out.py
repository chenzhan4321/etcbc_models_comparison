#!/usr/bin/env python3
"""
从.out.final格式文件提取标签，生成对应的.out格式文件
.out.final格式: 空格开头，标签字母标签字母...标签，如" 0W1B1H0 0L0J0N0"
.out格式: 只包含标签数字，如"0 1 1 0 0 0"

注意：
- .out.final格式中每个字母前都有标签，字母后的标签属于下一个字母（或词尾）
- .out格式只包含每个字母对应的标签（字母前的标签）
"""

import re
import sys
import os
from pathlib import Path
from tqdm import tqdm

def is_letter(char):
    """判断字符是否为叙利亚文字母"""
    return char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ<>'

def extract_labels_from_out_final_word(word):
    """
    从.out.final格式的单词中提取标签序列
    输入：'0W1B1H0' 
    输出：[0, 1, 1, 0]（提取每个字母前的标签 + 词尾标签）
    
    重要：包含词尾标签，因为它们包含重要的形态信息
    """
    labels = []
    i = 0
    
    while i < len(word):
        # 读取标签
        label_str = ""
        while i < len(word) and word[i].isdigit():
            label_str += word[i]
            i += 1
        
        if label_str:
            label = int(label_str)
            labels.append(label)  # 添加所有标签，包括词尾
            
            # 如果后面有字母，跳过它
            if i < len(word) and is_letter(word[i]):
                i += 1
    
    return labels

def extract_labels_from_out_final_line(line):
    """
    从.out.final格式的行中提取标签序列
    例如: " 0W1B1H0 0L0J0N0" -> [0, 1, 1, 0, 0, 0]
    """
    # 移除行首空格
    line = line.strip()
    if not line:
        return []
    
    # 分割成单词
    words = line.split()
    all_labels = []
    
    for word in words:
        labels = extract_labels_from_out_final_word(word)
        all_labels.extend(labels)
    
    return all_labels

def convert_out_final_to_out(input_file, output_file):
    """
    将.out.final格式文件转换为.out格式文件
    """
    print(f"\n📝 转换 {Path(input_file).name} -> {Path(output_file).name}")
    
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"❌ 错误：输入文件 {input_path.name} 不存在")
        return False
    
    try:
        with open(input_file, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()
        
        with open(output_file, 'w', encoding='utf-8') as fout:
            for line in tqdm(lines, desc="转换进度"):
                # 提取标签
                labels = extract_labels_from_out_final_line(line)
                
                # 写入.out格式（空格分隔的标签）
                if labels:
                    labels_str = ' '.join(map(str, labels))
                    fout.write(labels_str + '\n')
                else:
                    fout.write('\n')
        
        print(f"✅ 转换完成：{Path(output_file).name}")
        return True
        
    except Exception as e:
        print(f"❌ 转换过程中出错：{e}")
        return False

def verify_conversion(out_final_file, out_file, in_file, num_lines=5):
    """
    验证转换结果的正确性
    包含词尾标签时，标签数 = 字母数 + 词数
    """
    print(f"\n验证转换结果（前{num_lines}行）：")
    print("=" * 60)
    
    try:
        with open(in_file, 'r', encoding='utf-8') as f_in, \
             open(out_file, 'r', encoding='utf-8') as f_out:
            
            for i in range(num_lines):
                in_line = f_in.readline().strip()
                out_line = f_out.readline().strip()
                
                if not in_line:
                    break
                
                # 统计字母数量和词数
                in_letters = sum(1 for c in in_line if c != ' ')
                in_words = len(in_line.split())
                expected_labels = in_letters + in_words  # 每个字母+每个词尾
                
                # 统计标签数量（.out文件中的数字）
                out_labels = len(out_line.split()) if out_line else 0
                
                print(f"\n行 {i+1}:")
                print(f"  .in文件: {in_letters}个字母, {in_words}个词")
                print(f"  .out文件: {out_labels}个标签")
                print(f"  期望: {expected_labels}个标签（字母+词尾）")
                
                if expected_labels == out_labels:
                    print(f"  ✅ 数量正确")
                else:
                    print(f"  ❌ 数量不匹配！差异: {abs(expected_labels - out_labels)}")
                    if len(in_line) > 50:
                        in_line = in_line[:50] + "..."
                    if len(out_line) > 50:
                        out_line = out_line[:50] + "..."
                    print(f"  .in内容: {in_line}")
                    print(f"  .out内容: {out_line}")
    except Exception as e:
        print(f"验证时出错：{e}")

def main():
    """主函数"""
    # 使用当前目录
    data_dir = Path('.')
    
    # 定义要转换的文件
    files_to_convert = [
        ('test.out.final', 'test.out'),
        ('train.out.final', 'train.out'), 
        ('val.out.final', 'val.out')
    ]
    
    print("\n" + "=" * 60)
    print("开始转换.out.final文件到.out格式")
    print("=" * 60)
    
    success_count = 0
    for input_filename, output_filename in files_to_convert:
        input_path = data_dir / input_filename
        output_path = data_dir / output_filename
        
        if not input_path.exists():
            print(f"❌ 文件不存在: {input_filename}")
            continue
            
        if convert_out_final_to_out(input_path, output_path):
            success_count += 1
            
            # 验证转换结果
            in_filename = output_filename.replace('.out', '.in')
            in_path = data_dir / in_filename
            if in_path.exists():
                verify_conversion(input_path, output_path, in_path, num_lines=3)
    
    print("\n" + "=" * 60)
    print(f"转换完成：{success_count}/{len(files_to_convert)} 个文件成功转换")
    print("=" * 60)
    
    # 显示示例
    print("\n转换示例（test.out的前3行）:")
    test_out = data_dir / 'test.out'
    if test_out.exists():
        with open(test_out, 'r', encoding='utf-8') as f:
            for i in range(3):
                line = f.readline().strip()
                if line:
                    # 限制显示长度
                    if len(line) > 100:
                        line = line[:100] + "..."
                    print(f"  行{i+1}: {line}")

if __name__ == '__main__':
    main()