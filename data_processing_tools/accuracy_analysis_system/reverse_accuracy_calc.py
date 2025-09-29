#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reverse Accuracy Calculator
Calculate word accuracy from point accuracy under independence assumption
"""

import math
import argparse

def point_to_word_accuracy(point_accuracy: float, avg_points_per_word: float = 7.0) -> float:
    """
    Calculate expected word accuracy from point accuracy.
    
    Under independence assumption:
    - If point accuracy is p_point
    - Average points per word is n
    - Then word accuracy is: p_word = p_point^n
    
    Args:
        point_accuracy: Point-level accuracy (0-1)
        avg_points_per_word: Average number of points per word (default 7)
    
    Returns:
        Expected word-level accuracy
    """
    word_accuracy = point_accuracy ** avg_points_per_word
    return word_accuracy

def analyze_reverse_calculation():
    """
    Analyze the relationship from point accuracy to word accuracy.
    """
    print("\n" + "="*60)
    print("Reverse Calculation: Point Accuracy → Word Accuracy")
    print("="*60)
    
    # Test different point accuracies
    point_accuracies = [0.95, 0.96, 0.97, 0.975, 0.98, 0.9814, 0.985, 0.99, 0.995]
    points_per_word_values = [5, 6, 7, 8, 9]
    
    print("\nAssuming independence between points:")
    print("Word Accuracy = (Point Accuracy)^(Points per Word)")
    
    # Main table for n=7
    print("\n" + "="*60)
    print("For average 7 points per word:")
    print("-"*60)
    print(f"{'Point Acc':<12} {'Word Acc':<12} {'Interpretation':<36}")
    print("-"*60)
    
    for point_acc in point_accuracies:
        word_acc = point_to_word_accuracy(point_acc, 7.0)
        
        # Special highlighting for 98.14%
        if point_acc == 0.9814:
            interpretation = "← Your current point accuracy"
            print(f"{point_acc:<12.2%} {word_acc:<12.2%} {interpretation:<36}")
        else:
            interpretation = ""
            print(f"{point_acc:<12.2%} {word_acc:<12.2%} {interpretation:<36}")
    
    print("-"*60)
    
    # Sensitivity analysis
    print("\n" + "="*60)
    print("Sensitivity Analysis for 98.14% Point Accuracy:")
    print("-"*60)
    print(f"{'Points/Word':<12} {'Word Acc':<12} {'Note':<36}")
    print("-"*60)
    
    for ppw in points_per_word_values:
        word_acc = point_to_word_accuracy(0.9814, ppw)
        if ppw == 7:
            note = "← Default assumption"
        elif ppw < 7:
            note = f"Shorter words ({ppw} points avg)"
        else:
            note = f"Longer words ({ppw} points avg)"
        print(f"{ppw:<12} {word_acc:<12.2%} {note:<36}")
    
    print("-"*60)
    
    # Compare with actual
    print("\n" + "="*60)
    print("Comparison with Actual Results:")
    print("="*60)
    print(f"Point accuracy:                   98.14%")
    print(f"Expected word acc (7 pts/word):   {point_to_word_accuracy(0.9814, 7.0):.2%}")
    print(f"Actual word accuracy:             88.76%")
    print(f"Difference:                       {88.76 - point_to_word_accuracy(0.9814, 7.0)*100:.2f}%")
    print("-"*60)
    
    if 88.76 > point_to_word_accuracy(0.9814, 7.0)*100:
        print("→ Actual word accuracy is HIGHER than expected")
        print("→ This suggests errors are MORE correlated within words")
        print("→ When model makes an error, it tends to get multiple points wrong")
    else:
        print("→ Actual word accuracy is LOWER than expected")
        print("→ This suggests errors are LESS correlated within words")
        print("→ Model tends to make isolated point errors")
    
    # Find what point accuracy would give 88.76% word accuracy
    print("\n" + "="*60)
    print("Reverse Engineering: What point acc gives 88.76% word acc?")
    print("-"*60)
    
    target_word_acc = 0.8876
    for ppw in [5, 6, 7, 8, 9]:
        required_point_acc = target_word_acc ** (1/ppw)
        print(f"{ppw} points/word: {required_point_acc:.4%} point accuracy needed")
    
    print("-"*60)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Calculate word accuracy from point accuracy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python reverse_accuracy_calc.py                    # Full analysis
  python reverse_accuracy_calc.py --point 0.9814     # Specific point accuracy
  python reverse_accuracy_calc.py --point 0.9814 --ppw 8  # Custom points per word

This calculator assumes independence between point predictions within a word.
The actual correlation structure may differ significantly.
        """
    )
    
    parser.add_argument('--point', type=float, default=None,
                       help='Point-level accuracy to convert')
    parser.add_argument('--ppw', type=float, default=7.0,
                       help='Average points per word (default: 7.0)')
    
    args = parser.parse_args()
    
    if args.point is not None:
        # Single calculation mode
        word_acc = point_to_word_accuracy(args.point, args.ppw)
        
        print("\n" + "="*60)
        print("Point to Word Accuracy Conversion")
        print("="*60)
        print(f"Point accuracy:    {args.point:.4%}")
        print(f"Points per word:   {args.ppw:.1f}")
        print("-"*60)
        print(f"Expected word accuracy: {word_acc:.4%}")
        print("-"*60)
        print(f"\nFormula: Word Acc = (Point Acc)^(Points per Word)")
        print(f"         {word_acc:.4%} = {args.point:.4%}^{args.ppw:.1f}")
        
        # Compare with known actual if using default values
        if args.point == 0.9814 and args.ppw == 7.0:
            print("\n" + "="*60)
            print("Comparison with Actual:")
            print("-"*60)
            print(f"Expected word acc: {word_acc:.2%}")
            print(f"Actual word acc:   88.76%")
            print(f"Difference:        {88.76 - word_acc*100:+.2f}%")
            print("-"*60)
            print("The actual is higher than expected, suggesting")
            print("errors are correlated within words.")
    else:
        # Full analysis mode
        analyze_reverse_calculation()

if __name__ == "__main__":
    main()