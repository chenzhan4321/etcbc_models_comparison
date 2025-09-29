# Syriac Morphological Analysis - 2D Accuracy Analysis System

## 🎯 System Overview

A unified analysis tool that evaluates model performance across different conditions using a 2D matrix approach. Supports both simple mode (quick analysis) and enhanced mode (deep analysis) for comprehensive model evaluation.

## 📊 Core Concept

The system builds a 2×2 analysis matrix based on two dimensions:
- **Dimension 1**: Whether the test input word was seen in the training set (seen/unseen)
- **Dimension 2**: Whether the correct answer was seen in the training set (seen/unseen)

This produces 4 combinations:

| Combination | Input | Output | Meaning |
|-------------|-------|--------|---------|
| Combination 1 | ✓ Seen | ✓ Seen | Tests memorization ability |
| Combination 2 | ✓ Seen | ✗ Unseen | Tests creative generation ability |
| Combination 3 | ✗ Unseen | ✓ Seen | Tests generalization ability |
| Combination 4 | ✗ Unseen | ✗ Unseen | Tests true understanding ability |

## 📁 System Files

```
accuracy_analysis_system/
├── accuracy_analyzer.py      # Unified analysis tool (main script)
├── z_score_analysis.py       # Z-score analysis for point vs word accuracy
├── reverse_accuracy_calc.py  # Reverse accuracy calculations
├── reverse_z_score.py        # Reverse z-score calculations
├── pattern.json             # Learned input-output mapping patterns
├── *.forms                  # Word form files (generated in enhanced mode)
└── README.md               # This documentation
```

## 🚀 Quick Start

### Required Files

You need 5 input files:
```
train.in         # Training set input
train.out        # Training set output
test.in          # Test set input
test.prediction  # Model prediction results
test.out         # Test set ground truth
```

### Running Analysis

#### Simple Mode (Default, recommended for first-time users)
```bash
python accuracy_analyzer.py
```
- Quickly calculates 2D accuracy matrix
- Generates `accuracy_2d_results.txt`
- Fast execution, minimal output

#### Enhanced Mode (Deep analysis)
```bash
python accuracy_analyzer.py --enhanced
```
- Learns input→output mapping patterns
- Provides detailed error examples
- Analyzes pattern prediction effectiveness
- Generates multiple analysis files
- More comprehensive but slower

### Advanced Usage

#### Specify Data Directory
```bash
python accuracy_analyzer.py --data-dir ../data/
python accuracy_analyzer.py --enhanced --data-dir ../data/
```

#### Command Line Arguments
```bash
python accuracy_analyzer.py [-h] [--enhanced] [--data-dir DATA_DIR]

optional arguments:
  -h, --help           show help message and exit
  --enhanced           enable enhanced mode for detailed analysis
  --data-dir DATA_DIR  directory containing data files (default: current)
```

## 📈 Simple Mode vs Enhanced Mode

| Feature | Simple Mode | Enhanced Mode |
|---------|-------------|---------------|
| 2D Accuracy Matrix | ✅ | ✅ |
| Execution Speed | Fast | Slower |
| Generate Word Forms | ❌ | ✅ |
| Learn Mapping Patterns | ❌ | ✅ |
| Error Example Analysis | ❌ | ✅ |
| Pattern Prediction Analysis | ❌ | ✅ |
| Output Files | 1 file | 4+ files |
| Memory Usage | Low | Higher |

## 📊 Output Files

### Simple Mode Output
- `accuracy_2d_results.txt` - 2D accuracy results summary

### Enhanced Mode Output
- `accuracy_2d_analysis.txt` - Detailed analysis results with examples
- `pattern.json` - Learned input→output mapping patterns
- `train.in.forms` - All unique input word forms from training
- `train.out.forms` - All unique output morphological forms from training

## 🔍 Results Interpretation

### Accuracy Matrix Example
```
Combination 1 (Input✓ Output✓): 90.5%  # High → Good memorization
Combination 2 (Input✓ Output✗): 65.3%  # High → Good creativity
Combination 3 (Input✗ Output✓): 72.1%  # High → Good generalization
Combination 4 (Input✗ Output✗): 41.2%  # High → True rule understanding
```

### Ideal Model Characteristics
- **Combination 1 high accuracy**: Basic requirement (memorization)
- **Combinations 2-4 close to Combination 1**: True rule understanding
- **Combination 4 much lower than others**: Only memorization, poor generalization

## 🔧 Additional Analysis Tools

### Z-Score Analysis
```bash
python z_score_analysis.py --word-accuracy 0.85 --point-accuracy 0.92
```
Analyzes how many standard deviations point accuracy is from expected value given word accuracy.

### Reverse Accuracy Calculations
```bash
python reverse_accuracy_calc.py --target-point-accuracy 0.90
```
Calculates required word accuracy to achieve target point accuracy.

### Reverse Z-Score Analysis
```bash
python reverse_z_score.py --word-accuracy 0.85 --target-z-score 2.0
```
Calculates required point accuracy to achieve target z-score.

## 💡 Usage Recommendations

1. **First-time use**: Run simple mode to quickly understand model performance
2. **Deep analysis**: Use enhanced mode to examine detailed errors and patterns
3. **Model comparison**: Save result files from different models for comparison
4. **Model debugging**: Focus on the lowest accuracy combination for targeted improvements
5. **Performance monitoring**: Track accuracy improvements across training iterations

## 📝 Data Format Requirements

All files are space-separated text files:
```
# train.in / test.in (Syriac text input)
WQM MNWX W>ZL
BCLM> DHW> DJN

# train.out / test.out / test.prediction (morphological analysis)
W/Q/M MN/WX W>/ZL
B/CLM> D/HW> D/J/N
```

### File Specifications
- **Encoding**: UTF-8
- **Format**: Space-separated tokens per line
- **Line count**: test.in, test.prediction, and test.out must have identical line counts
- **Content**: Syriac characters for input, morphological segmentation for output

## 🛠 Troubleshooting

| Problem | Solution |
|---------|----------|
| File line count mismatch | Ensure test.in, test.prediction, test.out have same number of lines |
| Encoding errors | Ensure all files are UTF-8 encoded |
| Memory issues | Use simple mode or process files in batches |
| Missing files | Check all 5 required files exist in specified directory |
| Empty results | Verify file formats match expected structure |

## 📚 Technical Details

### Algorithm Overview
1. **Load Training Data**: Extract unique word forms from training set
2. **Classify Test Cases**: Categorize each test case into 4 combinations
3. **Calculate Accuracy**: Compute accuracy for each combination
4. **Pattern Learning** (Enhanced): Learn input→output mappings
5. **Error Analysis** (Enhanced): Analyze prediction errors by category

### Performance Characteristics
- **Time Complexity**: O(n) for simple mode, O(n²) for enhanced mode
- **Space Complexity**: O(k) where k is vocabulary size
- **Scalability**: Handles datasets up to millions of examples

### Statistical Methods
- Uses independence assumption for z-score calculations
- Applies Levenshtein distance for error analysis
- Implements pattern frequency analysis for mapping learning

## 🤝 Integration with Main System

This tool integrates with the main Syriac morphological analysis pipeline:

```bash
# 1. Train model
python train.py --model_type transformer

# 2. Generate predictions
python train.py --model_type transformer --mode test

# 3. Analyze accuracy
python data_processing_tools/accuracy_analysis_system/accuracy_analyzer.py --enhanced
```

## 🔬 Research Applications

- **Model comparison**: Compare different neural architectures
- **Ablation studies**: Analyze impact of training data size
- **Error analysis**: Identify systematic model weaknesses
- **Generalization studies**: Measure out-of-vocabulary performance
- **Pattern analysis**: Understand morphological rule learning

---

For more information about the complete data processing pipeline, see the main `data_processing_tools/README.md`.