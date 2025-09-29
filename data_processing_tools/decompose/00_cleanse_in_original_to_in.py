#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_cleanse_in.py

清理.in文件脚本
功能：
1. 移除文件中的#、^、"这三个符号
2. 在每一行前面添加空格
3. 处理train.in、val.in、test.in三个文件

用法: python 01_cleanse_in.py
"""

import os
import sys
from pathlib import Path

def cleanse_line(line):
    """
    清理单行文本
    - 移除#、^、"三个符号
    - 在行首添加空格（如果没有的话）
    """
    # 移除指定的三个符号
    cleaned_line = line.replace('#', '').replace('^', '').replace('"', '')
    
    # 在行首添加空格（如果行不为空且不是以空格开头）
    if cleaned_line.strip() and not cleaned_line.startswith(' '):
        cleaned_line = ' ' + cleaned_line
    
    return cleaned_line

def cleanse_file(input_file_path, output_file_path=None):
    """
    清理指定的.in文件
    
    参数:
    input_file_path: 输入文件路径
    output_file_path: 输出文件路径，如果为None则覆盖原文件
    """
    try:
        # 读取原文件
        with open(input_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 处理每一行
        cleaned_lines = []
        for line in lines:
            cleaned_line = cleanse_line(line)
            cleaned_lines.append(cleaned_line)
        
        # 确定输出文件路径
        if output_file_path is None:
            output_file_path = input_file_path
        
        # 写入清理后的内容
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.writelines(cleaned_lines)
        
        print(f"✅ 成功清理文件: {input_file_path}")
        if output_file_path != input_file_path:
            print(f"   输出到: {output_file_path}")
        
        return True
        
    except FileNotFoundError:
        print(f"❌ 文件不存在: {input_file_path}")
        return False
    except Exception as e:
        print(f"❌ 处理文件时出错 {input_file_path}: {e}")
        return False

def main():
    """
    主函数：处理三个.in文件
    """
    print("🧹 开始清理.in文件...")
    print("清理规则:")
    print("  1. 移除符号: #、^、\"")
    print("  2. 在每行前添加空格")
    print()
    
    # 定义要处理的文件（从.original.txt读取，生成.in文件）
    current_dir = Path(__file__).parent
    files_to_process = [
        ('train.in.original', 'train.in'),
        ('val.in.original', 'val.in'), 
        ('test.in.original', 'test.in')
    ]
    
    success_count = 0
    total_files = len(files_to_process)
    
    # 处理每个文件
    for input_name, output_name in files_to_process:
        input_path = current_dir / input_name
        output_path = current_dir / output_name
        print(f"处理文件: {input_name} -> {output_name}")
        if cleanse_file(input_path, output_path):
            success_count += 1
        print()
    
    # 总结
    print(f"📊 清理完成!")
    print(f"成功处理: {success_count}/{total_files} 个文件")
    
    if success_count == total_files:
        print("✅ 所有文件清理成功!")
    else:
        print("⚠️  部分文件处理失败，请检查上述错误信息")
    
    return success_count == total_files

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)