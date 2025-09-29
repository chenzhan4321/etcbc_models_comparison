#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Syriac Morphological Analysis 2D Accuracy Analysis System
Unified version - Supports simple and enhanced modes
"""

import os
import json
import argparse
from collections import defaultdict
from typing import Dict, Set, List

class AccuracyAnalyzer:
    """2D Accuracy Analyzer"""
    
    def __init__(self, data_dir: str = ".", mode: str = "simple"):
        """
        Initialize analyzer
        
        Args:
            data_dir: Directory containing data files
            mode: Run mode ("simple" or "enhanced")
        """
        self.data_dir = data_dir
        self.mode = mode
        self.train_in_forms = set()
        self.train_out_forms = set()
        self.patterns = {}
        self.test_data = []
        
    def load_word_forms(self, filename: str, save_to: str = None) -> Set[str]:
        """Load all word forms from file"""
        word_forms = set()
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.strip().split()
                word_forms.update(words)
        
        # 增强模式：保存词形文件
        if self.mode == "enhanced" and save_to:
            output_path = os.path.join(self.data_dir, save_to)
            with open(output_path, 'w', encoding='utf-8') as f:
                for word in sorted(word_forms):
                    f.write(word + '\n')
            print(f"  Saved word forms to {save_to}")
        
        return word_forms
    
    def load_train_forms(self):
        """Load training set word forms"""
        if self.mode == "enhanced":
            print("\n" + "="*60)
            print("Step 1: Preparing training set word forms")
            print("="*60)
            self.train_in_forms = self.load_word_forms('train.in', 'train.in.forms')
            print(f"  Training input forms: {len(self.train_in_forms)} unique")
            self.train_out_forms = self.load_word_forms('train.out', 'train.out.forms')
            print(f"  Training output forms: {len(self.train_out_forms)} unique")
        else:
            # 简单模式：静默加载
            self.train_in_forms = self.load_word_forms('train.in')
            self.train_out_forms = self.load_word_forms('train.out')
    
    def learn_patterns(self):
        """Learn input-output mapping patterns (enhanced mode only)"""
        if self.mode != "enhanced":
            return
            
        print("\n" + "="*60)
        print("Step 2: Learning input-output mapping patterns")
        print("="*60)
        
        # 加载训练数据
        train_in_lines = []
        train_out_lines = []
        
        with open(os.path.join(self.data_dir, 'train.in'), 'r', encoding='utf-8') as f:
            for line in f:
                train_in_lines.append(line.strip().split())
        
        with open(os.path.join(self.data_dir, 'train.out'), 'r', encoding='utf-8') as f:
            for line in f:
                train_out_lines.append(line.strip().split())
        
        # 学习映射
        pattern_mapping = defaultdict(lambda: defaultdict(int))
        
        for i in range(min(len(train_in_lines), len(train_out_lines))):
            in_words = train_in_lines[i]
            out_words = train_out_lines[i]
            
            for in_word in in_words:
                for out_word in out_words:
                    pattern_mapping[in_word][out_word] += 1
        
        # 转换格式
        self.patterns = {}
        patterns_for_json = {}
        
        for in_word, out_mapping in pattern_mapping.items():
            sorted_outputs = sorted(out_mapping.items(), key=lambda x: x[1], reverse=True)
            self.patterns[in_word] = sorted_outputs
            
            # 准备JSON格式
            output_list = []
            for out_word, freq in sorted_outputs:
                output_list.extend([out_word, freq])
            patterns_for_json[in_word] = output_list
        
        # 保存模式
        with open(os.path.join(self.data_dir, 'pattern.json'), 'w', encoding='utf-8') as f:
            json.dump(patterns_for_json, f, ensure_ascii=False, indent=2)
        
        print(f"  Learned mapping patterns for {len(self.patterns)} input words")
        print(f"  Saved to pattern.json")
        
        # 显示示例
        print("\n  Pattern examples:")
        for i, (in_word, outputs) in enumerate(list(self.patterns.items())[:3]):
            top_outputs = outputs[:3]
            output_str = ", ".join([f"{w}({f})" for w, f in top_outputs])
            print(f"    {in_word} → {output_str}")
    
    def load_test_data(self):
        """Load and align test data"""
        if self.mode == "enhanced":
            print("\n" + "="*60)
            print("Step 3: Loading test data")
            print("="*60)
        
        # 加载三个文件
        test_in_lines = []
        test_pred_lines = []
        test_out_lines = []
        
        with open(os.path.join(self.data_dir, 'test.in'), 'r', encoding='utf-8') as f:
            for line in f:
                test_in_lines.append(line.strip().split())
        
        with open(os.path.join(self.data_dir, 'test.prediction'), 'r', encoding='utf-8') as f:
            for line in f:
                test_pred_lines.append(line.strip().split())
        
        with open(os.path.join(self.data_dir, 'test.out'), 'r', encoding='utf-8') as f:
            for line in f:
                test_out_lines.append(line.strip().split())
        
        # 验证行数
        if not (len(test_in_lines) == len(test_pred_lines) == len(test_out_lines)):
            print(f"Error: File line counts do not match!")
            print(f"  test.in: {len(test_in_lines)} lines")
            print(f"  test.prediction: {len(test_pred_lines)} lines")
            print(f"  test.out: {len(test_out_lines)} lines")
            return False
        
        # 对齐数据
        self.test_data = []
        for line_idx in range(len(test_in_lines)):
            in_words = test_in_lines[line_idx]
            pred_words = test_pred_lines[line_idx]
            out_words = test_out_lines[line_idx]
            
            min_len = min(len(in_words), len(pred_words), len(out_words))
            
            for word_idx in range(min_len):
                self.test_data.append({
                    'input': in_words[word_idx],
                    'predicted': pred_words[word_idx],
                    'truth': out_words[word_idx],
                    'line': line_idx,
                    'position': word_idx
                })
        
        if self.mode == "enhanced":
            print(f"  Loaded {len(test_in_lines)} lines of test data")
            print(f"  {len(self.test_data)} words after alignment")
        
        return True
    
    def calculate_2d_accuracy(self):
        """Calculate 2D accuracy matrix"""
        if self.mode == "enhanced":
            print("\n" + "="*60)
            print("Step 4: Calculating 2D accuracy matrix")
            print("="*60)
        
        # 初始化统计
        combos = {
            'Combo 1 (Input✓ Output✓)': {'correct': 0, 'total': 0, 'examples': []},
            'Combo 2 (Input✓ Output✗)': {'correct': 0, 'total': 0, 'examples': []},
            'Combo 3 (Input✗ Output✓)': {'correct': 0, 'total': 0, 'examples': []},
            'Combo 4 (Input✗ Output✗)': {'correct': 0, 'total': 0, 'examples': []}
        }
        
        # 分析每个词
        for item in self.test_data:
            input_word = item['input']
            predicted = item['predicted']
            truth = item['truth']
            
            # 判断seen/unseen
            input_seen = input_word in self.train_in_forms
            output_seen = truth in self.train_out_forms
            
            # 判断正确性
            is_correct = (predicted == truth)
            
            # 确定组合
            if input_seen and output_seen:
                combo_key = 'Combo 1 (Input✓ Output✓)'
            elif input_seen and not output_seen:
                combo_key = 'Combo 2 (Input✓ Output✗)'
            elif not input_seen and output_seen:
                combo_key = 'Combo 3 (Input✗ Output✓)'
            else:
                combo_key = 'Combo 4 (Input✗ Output✗)'
            
            # 更新统计
            combos[combo_key]['total'] += 1
            if is_correct:
                combos[combo_key]['correct'] += 1
            
            # 增强模式：保存示例
            if self.mode == "enhanced" and len(combos[combo_key]['examples']) < 10:
                combos[combo_key]['examples'].append({
                    'input': input_word,
                    'predicted': predicted,
                    'truth': truth,
                    'correct': is_correct
                })
        
        # 显示结果
        print("\n" + "="*60)
        print("2D Accuracy Analysis Results")
        print("="*60)
        
        overall_correct = 0
        overall_total = 0
        
        for combo_name, stats in combos.items():
            correct = stats['correct']
            total = stats['total']
            accuracy = (correct / total * 100) if total > 0 else 0
            
            overall_correct += correct
            overall_total += total
            
            print(f"\n{combo_name}:")
            print(f"  Samples: {total:,}")
            print(f"  Correct: {correct:,}")
            print(f"  Accuracy: {accuracy:.2f}%")
            
            # 增强模式：显示示例
            if self.mode == "enhanced" and stats['examples']:
                print("  Examples:")
                for i, ex in enumerate(stats['examples'][:3], 1):
                    status = "✓" if ex['correct'] else "✗"
                    print(f"    {i}. {ex['input']} → {ex['predicted']} (Truth: {ex['truth']}) {status}")
        
        # 总体统计
        overall_accuracy = (overall_correct / overall_total * 100) if overall_total > 0 else 0
        print(f"\nOverall:")
        print(f"  Total samples: {overall_total:,}")
        print(f"  Total correct: {overall_correct:,}")
        print(f"  Overall accuracy: {overall_accuracy:.2f}%")
        
        # 保存结果
        output_file = 'accuracy_2d_analysis.txt' if self.mode == "enhanced" else 'accuracy_2d_results.txt'
        result_path = os.path.join(self.data_dir, output_file)
        
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write("2D Accuracy Analysis Results\n")
            f.write("=" * 60 + "\n\n")
            
            for combo_name, stats in combos.items():
                correct = stats['correct']
                total = stats['total']
                accuracy = (correct / total * 100) if total > 0 else 0
                
                f.write(f"{combo_name}:\n")
                f.write(f"  Samples: {total:,}\n")
                f.write(f"  Correct: {correct:,}\n")
                f.write(f"  Accuracy: {accuracy:.2f}%\n\n")
                
                # 增强模式：写入示例
                if self.mode == "enhanced" and stats['examples']:
                    f.write("  Examples:\n")
                    for i, ex in enumerate(stats['examples'], 1):
                        status = "正确" if ex['correct'] else "错误"
                        f.write(f"    {i}. 输入: {ex['input']}\n")
                        f.write(f"       预测: {ex['predicted']}\n")
                        f.write(f"       答案: {ex['truth']}\n")
                        f.write(f"       状态: {status}\n\n")
            
            f.write(f"总体:\n")
            f.write(f"  Total samples: {overall_total:,}\n")
            f.write(f"  Total correct: {overall_correct:,}\n")
            f.write(f"  Overall accuracy: {overall_accuracy:.2f}%\n")
        
        print(f"\nResults saved to {output_file}")
        
        return combos
    
    def analyze_patterns(self):
        """Analyze pattern prediction effectiveness (enhanced mode only)"""
        if self.mode != "enhanced" or not self.patterns:
            return
            
        print("\n" + "="*60)
        print("Step 5: Analyzing pattern prediction effectiveness")
        print("="*60)
        
        # 找出组合2的样本
        combo2_samples = []
        for item in self.test_data:
            if (item['input'] in self.train_in_forms and 
                item['truth'] not in self.train_out_forms):
                combo2_samples.append(item)
        
        if not combo2_samples:
            print("  No Combo 2 samples found")
            return
        
        # 分析模式预测
        pattern_exact = 0
        pattern_in_list = 0
        no_pattern = 0
        
        for sample in combo2_samples:
            input_word = sample['input']
            truth = sample['truth']
            
            if input_word in self.patterns:
                outputs = self.patterns[input_word]
                if outputs:
                    if outputs[0][0] == truth:
                        pattern_exact += 1
                    if any(out == truth for out, _ in outputs):
                        pattern_in_list += 1
            else:
                no_pattern += 1
        
        print(f"\nCombo 2 Analysis (Input seen but output unseen):")
        print(f"  Total samples: {len(combo2_samples)}")
        print(f"  Pattern available: {len(combo2_samples) - no_pattern}")
        print(f"  No pattern available: {no_pattern}")
        
        if len(combo2_samples) - no_pattern > 0:
            print(f"\n  Pattern prediction effectiveness:")
            print(f"    Top prediction matches: {pattern_exact} ({pattern_exact/len(combo2_samples)*100:.1f}%)")
            print(f"    In prediction list: {pattern_in_list} ({pattern_in_list/len(combo2_samples)*100:.1f}%)")
        
        # 显示示例
        print(f"\n  Example analysis:")
        for i, sample in enumerate(combo2_samples[:5], 1):
            print(f"    {i}. {sample['input']} → Predicted: {sample['predicted']}, Truth: {sample['truth']}")
            
            if sample['input'] in self.patterns:
                outputs = self.patterns[sample['input']][:3]
                pattern_str = ", ".join([f"{w}({f})" for w, f in outputs])
                print(f"       Pattern predictions: {pattern_str}")
    
    def check_files(self):
        """Check required files"""
        required_files = [
            'train.in', 'train.out',
            'test.in', 'test.prediction', 'test.out'
        ]
        
        missing = []
        for file in required_files:
            if not os.path.exists(os.path.join(self.data_dir, file)):
                missing.append(file)
        
        if missing:
            print("Error: Missing required files:")
            for file in missing:
                print(f"  - {file}")
            return False
        return True
    
    def run(self):
        """Run complete analysis workflow"""
        if self.mode == "enhanced":
            print("\n" + "="*60)
            print("Syriac Morphological Analysis 2D Accuracy System - Enhanced Mode")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("Syriac Morphological Analysis 2D Accuracy System - Simple Mode")
            print("="*60)
        
        # 检查文件
        if not self.check_files():
            return None
        
        # 执行分析
        self.load_train_forms()
        
        if self.mode == "enhanced":
            self.learn_patterns()
        
        if not self.load_test_data():
            return None
        
        combos = self.calculate_2d_accuracy()
        
        if self.mode == "enhanced":
            self.analyze_patterns()
        
        print("\n" + "="*60)
        print("Analysis Complete!")
        print("="*60)
        
        return combos

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Syriac Morphological Analysis 2D Accuracy Analysis System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Run modes:
  simple   - Simple mode: Quick basic accuracy calculation
  enhanced - Enhanced mode: Includes pattern learning and detailed analysis

Required files:
  train.in         - Training set input
  train.out        - Training set output
  test.in          - Test set input
  test.prediction  - Model predictions
  test.out         - Test set ground truth

Output files:
  Simple mode: accuracy_2d_results.txt
  Enhanced mode: accuracy_2d_analysis.txt, pattern.json, *.forms

Examples:
  python accuracy_analyzer.py              # Default simple mode
  python accuracy_analyzer.py --enhanced   # Enhanced mode
  python accuracy_analyzer.py --data-dir ../data/  # Specify data directory
        """
    )
    
    parser.add_argument('--enhanced', action='store_true',
                       help='Use enhanced mode (includes pattern learning and detailed analysis)')
    parser.add_argument('--data-dir', type=str, default='.',
                       help='Directory containing data files (default: current directory)')
    
    args = parser.parse_args()
    
    # 确定模式
    mode = "enhanced" if args.enhanced else "simple"
    
    # 创建分析器并运行
    analyzer = AccuracyAnalyzer(data_dir=args.data_dir, mode=mode)
    analyzer.run()

if __name__ == "__main__":
    main()