#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
反向工程脚本：从.out和.in文件直接还原.out.reduced和.out.final文件

处理流程（在内存中完成）：
1. .out + .in → 内存中的标签字母交替格式
2. 使用patterns.csv将标签转为pattern → 生成.out.reduced文件
3. 直接使用标签数字 → 生成.out.final文件

文件格式说明：
- .out.reduced: 字母和符号pattern交替，如 W-B-HLJN（符号用pattern表示）
- .out.final: 字母和数字交替，如 0W1B1H0L0J0N0（数字是原始标签值）
"""

import csv
import sys
from pathlib import Path


def load_label_to_pattern_mapping(csv_file):
    """
    加载patterns.csv文件，建立label到pattern的映射
    返回：label_to_pattern字典
    """
    label_to_pattern = {}
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = int(row['label'])
            pattern = row['pattern']
            label_to_pattern[label] = pattern
    
    print(f"✅ 加载了{len(label_to_pattern)}个label到pattern的映射")
    return label_to_pattern


def process_line_reduced(in_line, out_line, label_to_pattern):
    """
    处理单行数据，直接从.in和.out生成.out.reduced格式
    """
    if not in_line.strip():
        return '\n'
    
    # 分割字母和标签
    in_words = in_line.strip().split()
    out_labels = out_line.strip().split() if out_line.strip() else []
    
    # 转换为整数标签
    labels = [int(label) for label in out_labels]
    label_idx = 0
    
    reduced_words = []
    
    for word in in_words:
        reduced_word = []
        
        # 处理每个字母及其前面的标签
        for char in word:
            if label_idx < len(labels):
                label = labels[label_idx]
                pattern = label_to_pattern.get(label, '')
                reduced_word.append(pattern)
                reduced_word.append(char)
                label_idx += 1
            else:
                reduced_word.append(char)
        
        # 词尾标签
        if label_idx < len(labels):
            label = labels[label_idx]
            pattern = label_to_pattern.get(label, '')
            reduced_word.append(pattern)
            label_idx += 1
        
        reduced_words.append(''.join(reduced_word))
    
    # 合并成一行
    line = ' '.join(reduced_words)
    
    return line + '\n'


def process_line_final(in_line, out_line):
    """
    处理单行数据，直接从.in和.out生成.out.final格式
    """
    if not in_line.strip():
        return '\n'
    
    # 分割字母和标签
    in_words = in_line.strip().split()
    out_labels = out_line.strip().split() if out_line.strip() else []
    
    # 转换为整数标签
    labels = [int(label) for label in out_labels]
    label_idx = 0
    
    final_words = []
    
    for word in in_words:
        final_word = []
        
        # 处理每个字母及其前面的标签
        for char in word:
            if label_idx < len(labels):
                label = labels[label_idx]
                final_word.append(str(label))
                final_word.append(char)
                label_idx += 1
            else:
                final_word.append('0')  # 默认标签为0
                final_word.append(char)
        
        # 词尾标签
        if label_idx < len(labels):
            label = labels[label_idx]
            final_word.append(str(label))
            label_idx += 1
        else:
            final_word.append('0')  # 默认标签为0
        
        final_words.append(''.join(final_word))
    
    # 合并成一行，在行首添加空格
    line = ' ' + ' '.join(final_words)
    
    return line + '\n'


def restore_files(in_file, out_file, patterns_csv, generate_final=True):
    """
    主函数：直接从.in和.out生成.out.reduced和.out.final
    
    参数：
        in_file: 输入文件路径 (.in)
        out_file: 输出文件路径 (.out)
        patterns_csv: patterns映射文件路径
        generate_final: 是否生成.final文件（默认True）
    """
    print("\n" + "=" * 60)
    print("开始反向工程：还原.out.reduced和.out.final文件")
    print("=" * 60)
    
    # 设置输出文件路径
    base_name = in_file.stem  # 如 'test'
    output_dir = in_file.parent
    out_reduced_file = output_dir / f"{base_name}.out.reduced"
    out_final_file = output_dir / f"{base_name}.out.final"
    
    try:
        # 加载pattern映射
        label_to_pattern = load_label_to_pattern_mapping(patterns_csv)
        
        print(f"\n处理文件:")
        print(f"  输入: {in_file.name}, {out_file.name}")
        print(f"  输出: {out_reduced_file.name}")
        if generate_final:
            print(f"        {out_final_file.name}")
        
        # 处理文件
        with open(in_file, 'r', encoding='utf-8') as f_in, \
             open(out_file, 'r', encoding='utf-8') as f_out:
            
            in_lines = f_in.readlines()
            out_lines = f_out.readlines()
            
            if len(in_lines) != len(out_lines):
                print(f"⚠️ 警告：行数不匹配 - .in有{len(in_lines)}行，.out有{len(out_lines)}行")
            
            # 打开输出文件
            with open(out_reduced_file, 'w', encoding='utf-8') as f_reduced:
                if generate_final:
                    with open(out_final_file, 'w', encoding='utf-8') as f_final:
                        # 逐行处理
                        for line_num, (in_line, out_line) in enumerate(zip(in_lines, out_lines), 1):
                            # 生成.reduced格式
                            reduced_line = process_line_reduced(in_line, out_line, label_to_pattern)
                            f_reduced.write(reduced_line)
                            
                            # 生成.final格式
                            final_line = process_line_final(in_line, out_line)
                            f_final.write(final_line)
                            
                            # 每1000行显示进度
                            if line_num % 1000 == 0:
                                print(f"  已处理 {line_num} 行...")
                else:
                    # 只生成.reduced格式
                    for line_num, (in_line, out_line) in enumerate(zip(in_lines, out_lines), 1):
                        reduced_line = process_line_reduced(in_line, out_line, label_to_pattern)
                        f_reduced.write(reduced_line)
                        
                        if line_num % 1000 == 0:
                            print(f"  已处理 {line_num} 行...")
        
        print("\n" + "=" * 60)
        print(f"✅ 反向工程完成！")
        print(f"📁 生成的文件：")
        print(f"   - {out_reduced_file.name}")
        if generate_final:
            print(f"   - {out_final_file.name}")
        print("=" * 60)
        
        # 显示示例
        print("\n还原结果示例（前3行）：")
        print("\n📄 .out.reduced格式：")
        with open(out_reduced_file, 'r', encoding='utf-8') as f:
            for i in range(3):
                line = f.readline().strip()
                if line:
                    if len(line) > 80:
                        line = line[:80] + "..."
                    print(f"  行{i+1}: {line}")
        
        if generate_final:
            print("\n📄 .out.final格式：")
            with open(out_final_file, 'r', encoding='utf-8') as f:
                for i in range(3):
                    line = f.readline().strip()
                    if line:
                        if len(line) > 80:
                            line = line[:80] + "..."
                        print(f"  行{i+1}: {line}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 反向工程失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主程序入口"""
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description='从.in和.out文件生成.reduced和.final格式文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
文件格式说明：
  .out.reduced: 字母和符号pattern交替，如 W-B-HLJN
  .out.final:   字母和数字交替，如 0W1B1H0L0J0N0

示例用法：
  python restore_reduced.py                    # 处理test文件，生成两种格式
  python restore_reduced.py --prefix train     # 处理train文件
  python restore_reduced.py --no-final         # 只生成.reduced格式
        """
    )
    
    parser.add_argument('--prefix', type=str, default='test',
                       help='文件前缀（默认: test）')
    parser.add_argument('--no-final', action='store_true',
                       help='只生成.reduced格式，不生成.final格式')
    parser.add_argument('--data-dir', type=str, default='.',
                       help='数据文件目录（默认: 当前目录）')
    
    args = parser.parse_args()
    
    # 设置文件路径
    restore_dir = Path(args.data_dir)
    
    # 检查必要文件是否存在
    required_files = {
        f'{args.prefix}.in': restore_dir / f'{args.prefix}.in',
        f'{args.prefix}.out': restore_dir / f'{args.prefix}.out',
        'patterns.csv': restore_dir / 'patterns.csv'
    }
    
    print("检查必要文件...")
    for name, path in required_files.items():
        if not path.exists():
            print(f"❌ 错误：找不到必要文件 {name}")
            print(f"   期望路径：{path}")
            return
        print(f"✅ 找到 {name}")
    
    # 执行反向工程
    restore_files(
        required_files[f'{args.prefix}.in'],
        required_files[f'{args.prefix}.out'],
        required_files['patterns.csv'],
        generate_final=not args.no_final
    )


if __name__ == "__main__":
    main()