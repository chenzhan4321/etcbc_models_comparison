#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
符号对移除工具
处理.out文件，移除指定符号对中的第一个符号
支持的符号对：!! @@ ]]
"""

import sys
import os
import re


def reduce_symbol_pairs(text):
    """
    移除文本中符号对的第一个符号
    处理形如 !...! @...@ ]...] 的符号对
    
    Args:
        text (str): 输入文本
        
    Returns:
        str: 处理后的文本
    """
    # 处理 !...! 符号对，移除第一个 !
    # 使用非贪婪匹配，匹配从第一个!到下一个!的内容
    text = re.sub(r'!([^!]*?)!', r'\1!', text)
    
    # 处理 @...@ 符号对，移除第一个 @
    text = re.sub(r'@([^@]*?)@', r'\1@', text)
    
    # 处理 ]...] 符号对，移除第一个 ]
    text = re.sub(r']([^\]]*?)]', r'\1]', text)
    
    return text


def process_file(input_file, output_file):
    """
    处理单个文件
    
    Args:
        input_file (str): 输入文件路径
        output_file (str): 输出文件路径
        
    Returns:
        bool: 处理是否成功
    """
    if not os.path.exists(input_file):
        print(f"错误：文件 {input_file} 不存在")
        return False
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 处理符号对
        reduced_content = reduce_symbol_pairs(content)
        
        # 写入输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(reduced_content)
        
        print(f"处理完成：{input_file} -> {output_file}")
        return True
        
    except Exception as e:
        print(f"处理文件时出错：{e}")
        return False


def main():
    """主函数 - 自动处理所有.out.original.txt文件"""
    # 定义要处理的文件
    files_to_process = [
        ('train.out.original', 'train.out.reduced'),
        ('val.out.original', 'val.out.reduced'),
        ('test.out.original', 'test.out.reduced')
    ]
    
    print("开始处理所有.out.original.txt文件...")
    
    success_count = 0
    for input_file, output_file in files_to_process:
        if process_file(input_file, output_file):
            success_count += 1
    
    print(f"\n处理完成：{success_count}/{len(files_to_process)} 个文件成功处理")
    
    if success_count == len(files_to_process):
        print("✅ 所有文件处理成功！")
    else:
        print("⚠️ 部分文件处理失败")


if __name__ == "__main__":
    main()