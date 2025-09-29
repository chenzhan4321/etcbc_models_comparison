#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Z-Score Analysis for Word vs Point Accuracy
Calculates how many standard deviations a point accuracy is from the expected value
given the current word-level accuracy.
"""

import math
import argparse
from typing import Tuple

def calculate_z_score(word_accuracy: float, point_accuracy: float, avg_points_per_word: float = 7.0) -> Tuple[float, float, float]:
    """
    Calculate z-score for point accuracy given word accuracy.
    
    Under independence assumption:
    - If word accuracy is p_word
    - Average points per word is n
    - Then expected point accuracy is: p_point = p_word^(1/n)
    - Variance of point accuracy: var = p_point(1-p_point) / (total_points)
    
    Args:
        word_accuracy: Word-level accuracy (0-1)
        point_accuracy: Point-level accuracy (0-1)
        avg_points_per_word: Average number of points per word (default 7)
    
    Returns:
        Tuple of (expected_point_accuracy, std_deviation, z_score)
    """
    # Calculate expected point accuracy under independence
    expected_point_acc = word_accuracy ** (1 / avg_points_per_word)
    
    # For large sample approximation
    # Assuming ~76000 words with ~7 points each = ~532000 points
    total_points = 76083 * avg_points_per_word
    
    # Standard error for binomial proportion
    variance = expected_point_acc * (1 - expected_point_acc) / total_points
    std_dev = math.sqrt(variance)
    
    # Calculate z-score
    z_score = (point_accuracy - expected_point_acc) / std_dev
    
    return expected_point_acc, std_dev, z_score

def analyze_accuracy_relationship(word_accuracy=0.8533):
    """
    Analyze the relationship between word and point accuracy from test results.
    """
    print("\n" + "="*60)
    print("Z-Score Analysis: Word Accuracy vs Point Accuracy")
    print("="*60)
    
    # Use provided word accuracy or default
    # word_accuracy = 0.8533  # 85.33% overall accuracy
    
    # Common point accuracy values to test
    test_point_accuracies = [0.970, 0.975, 0.980, 0.983, 0.985, 0.990]
    
    print(f"\nGiven word accuracy: {word_accuracy:.2%}")
    print(f"Assuming average {7} points per word")
    print("\n" + "-"*60)
    
    # Calculate expected point accuracy
    expected, _, _ = calculate_z_score(word_accuracy, 0.98, 7.0)
    print(f"Expected point accuracy (under independence): {expected:.2%}")
    print("-"*60)
    
    print("\nZ-Score Analysis for Different Point Accuracies:")
    print("-"*60)
    print(f"{'Point Acc':<12} {'Expected':<12} {'Std Dev':<12} {'Z-Score':<12} {'Significance':<20}")
    print("-"*60)
    
    for point_acc in test_point_accuracies:
        expected, std_dev, z = calculate_z_score(word_accuracy, point_acc, 7.0)
        
        # Determine significance level
        if abs(z) < 1.96:
            sig = "Not significant"
        elif abs(z) < 2.58:
            sig = "p < 0.05 *"
        elif abs(z) < 3.29:
            sig = "p < 0.01 **"
        else:
            sig = "p < 0.001 ***"
            
        print(f"{point_acc:<12.1%} {expected:<12.4f} {std_dev:<12.6f} {z:<+12.2f} {sig:<20}")
    
    print("-"*60)
    print("\nInterpretation:")
    print("- Positive z-score: Point accuracy is HIGHER than expected")
    print("- Negative z-score: Point accuracy is LOWER than expected")
    print("- |z| > 1.96: Statistically significant at p < 0.05")
    print("- |z| > 2.58: Statistically significant at p < 0.01")
    print("- |z| > 3.29: Statistically significant at p < 0.001")
    
    # Analyze what point accuracy corresponds to specific z-scores
    print("\n" + "="*60)
    print("Reverse Analysis: What point accuracy for given z-scores?")
    print("="*60)
    
    target_z_scores = [-3, -2, -1, 0, 1, 2, 3]
    expected_base, std_dev_base, _ = calculate_z_score(word_accuracy, 0.98, 7.0)
    
    print(f"\n{'Z-Score':<10} {'Point Accuracy':<15} {'Interpretation':<30}")
    print("-"*60)
    
    for z in target_z_scores:
        point_acc = expected_base + z * std_dev_base
        interpretation = ""
        if z == 0:
            interpretation = "Expected (independence)"
        elif z > 0:
            interpretation = f"{z} std dev above expected"
        else:
            interpretation = f"{abs(z)} std dev below expected"
            
        print(f"{z:<+10.1f} {point_acc:<15.4%} {interpretation:<30}")
    
    print("-"*60)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Z-Score Analysis for Word vs Point Accuracy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python z_score_analysis.py                      # Run full analysis
  python z_score_analysis.py --word 0.85 --point 0.983  # Custom values
  python z_score_analysis.py --word 0.85 --point 0.983 --ppw 8  # Custom points per word

The z-score tells you how many standard deviations the observed point accuracy
is from the expected value under the independence assumption.
        """
    )
    
    parser.add_argument('--word', type=float, default=0.8533,
                       help='Word-level accuracy (default: 0.8533 from test)')
    parser.add_argument('--point', type=float, default=None,
                       help='Point-level accuracy to analyze')
    parser.add_argument('--ppw', type=float, default=7.0,
                       help='Average points per word (default: 7.0)')
    
    args = parser.parse_args()
    
    if args.point is not None:
        # Single calculation mode
        expected, std_dev, z_score = calculate_z_score(args.word, args.point, args.ppw)
        
        print("\n" + "="*60)
        print("Z-Score Calculation")
        print("="*60)
        print(f"Word accuracy:     {args.word:.2%}")
        print(f"Point accuracy:    {args.point:.2%}")
        print(f"Points per word:   {args.ppw:.1f}")
        print("-"*60)
        print(f"Expected point acc: {expected:.4%}")
        print(f"Observed point acc: {args.point:.4%}")
        print(f"Difference:        {(args.point - expected):.4%}")
        print(f"Standard deviation: {std_dev:.6f}")
        print(f"Z-score:           {z_score:+.2f}")
        print("-"*60)
        
        if abs(z_score) < 1.96:
            print("Result: Not statistically significant (p > 0.05)")
        elif abs(z_score) < 2.58:
            print("Result: Statistically significant (p < 0.05)")
        elif abs(z_score) < 3.29:
            print("Result: Highly significant (p < 0.01)")
        else:
            print("Result: Extremely significant (p < 0.001)")
            
        if z_score > 0:
            print(f"Point accuracy is {z_score:.2f} standard deviations ABOVE expected")
        else:
            print(f"Point accuracy is {abs(z_score):.2f} standard deviations BELOW expected")
    else:
        # Full analysis mode
        analyze_accuracy_relationship(args.word)

if __name__ == "__main__":
    main()