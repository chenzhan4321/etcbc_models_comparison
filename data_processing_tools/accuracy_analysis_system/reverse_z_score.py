#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reverse Z-Score Analysis
Calculate z-score for the difference between actual and expected word accuracy
given a point accuracy
"""

import math
import argparse

def calculate_reverse_z_score(point_accuracy: float, actual_word_accuracy: float, 
                             avg_points_per_word: float = 7.0, 
                             total_words: int = 76083) -> tuple:
    """
    Calculate z-score for the difference between actual and expected word accuracy.
    
    Args:
        point_accuracy: Observed point-level accuracy (0-1)
        actual_word_accuracy: Observed word-level accuracy (0-1)
        avg_points_per_word: Average number of points per word (default 7)
        total_words: Total number of words in test set (default 76083)
    
    Returns:
        Tuple of (expected_word_acc, std_dev, z_score)
    """
    # Calculate expected word accuracy under independence
    expected_word_acc = point_accuracy ** avg_points_per_word
    
    # Calculate standard error for word accuracy
    # Using binomial approximation for word-level predictions
    variance = expected_word_acc * (1 - expected_word_acc) / total_words
    std_dev = math.sqrt(variance)
    
    # Calculate z-score
    z_score = (actual_word_accuracy - expected_word_acc) / std_dev
    
    return expected_word_acc, std_dev, z_score

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Calculate z-score for reverse accuracy analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This calculates how many standard deviations the actual word accuracy
differs from the expected word accuracy given a point accuracy.

Example:
  python reverse_z_score.py --point 0.9814 --word 0.8876
        """
    )
    
    parser.add_argument('--point', type=float, default=0.9814,
                       help='Point-level accuracy (default: 0.9814)')
    parser.add_argument('--word', type=float, default=0.8876,
                       help='Actual word-level accuracy (default: 0.8876)')
    parser.add_argument('--ppw', type=float, default=7.0,
                       help='Average points per word (default: 7.0)')
    parser.add_argument('--total', type=int, default=76083,
                       help='Total number of words (default: 76083)')
    
    args = parser.parse_args()
    
    expected, std_dev, z_score = calculate_reverse_z_score(
        args.point, args.word, args.ppw, args.total
    )
    
    print("\n" + "="*60)
    print("Reverse Z-Score Analysis")
    print("="*60)
    print("Input:")
    print(f"  Point accuracy:      {args.point:.4%}")
    print(f"  Actual word acc:     {args.word:.4%}")
    print(f"  Points per word:     {args.ppw:.1f}")
    print(f"  Total words:         {args.total:,}")
    
    print("\n" + "-"*60)
    print("Analysis:")
    print(f"  Expected word acc:   {expected:.4%} (from independence)")
    print(f"  Actual word acc:     {args.word:.4%}")
    print(f"  Difference:          {(args.word - expected):.4%}")
    print(f"  Standard deviation:  {std_dev:.6f}")
    print(f"  Z-score:            {z_score:+.2f}")
    
    print("\n" + "-"*60)
    print("Interpretation:")
    
    # Statistical significance
    if abs(z_score) < 1.96:
        sig_level = "Not statistically significant (p > 0.05)"
    elif abs(z_score) < 2.58:
        sig_level = "Statistically significant (p < 0.05)"
    elif abs(z_score) < 3.29:
        sig_level = "Highly significant (p < 0.01)"
    else:
        sig_level = "Extremely significant (p < 0.001)"
    
    print(f"  Significance:        {sig_level}")
    
    if z_score > 0:
        print(f"\n→ Actual word accuracy is {z_score:.2f} standard deviations")
        print(f"  HIGHER than expected from point accuracy")
        print(f"→ This indicates NEGATIVE correlation of errors within words")
        print(f"  (When one point is correct, others are more likely correct)")
    else:
        print(f"\n→ Actual word accuracy is {abs(z_score):.2f} standard deviations")
        print(f"  LOWER than expected from point accuracy")
        print(f"→ This indicates POSITIVE correlation of errors within words")
        print(f"  (Errors tend to cluster within the same words)")
    
    print("\n" + "="*60)
    
    # Additional analysis: confidence interval
    print("95% Confidence Interval for Expected Word Accuracy:")
    print("-"*60)
    lower_bound = expected - 1.96 * std_dev
    upper_bound = expected + 1.96 * std_dev
    print(f"  Lower bound:         {lower_bound:.4%}")
    print(f"  Expected:            {expected:.4%}")
    print(f"  Upper bound:         {upper_bound:.4%}")
    print(f"  Actual:              {args.word:.4%}", end="")
    
    if args.word < lower_bound:
        print(" (below interval)")
    elif args.word > upper_bound:
        print(" (above interval)")
    else:
        print(" (within interval)")
    
    print("="*60)

if __name__ == "__main__":
    main()