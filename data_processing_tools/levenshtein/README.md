# Syriac Morphological Analysis - Levenshtein Distance Analysis

## 🎯 Overview

The Levenshtein analysis tool provides **character-level and unit-level accuracy analysis** using edit distance algorithms. This tool is specifically designed for fine-grained evaluation of morphological analysis predictions by comparing each character and morphological label as independent units.

## 📊 Core Concept

Unlike word-level accuracy that treats entire words as correct/incorrect, this tool:
- **Treats each character and label as an independent unit**
- **Calculates edit distance** between prediction and ground truth sequences
- **Provides detailed operation breakdowns** (insertions, deletions, substitutions)
- **Generates unit-level accuracy metrics** for precise error analysis

## 📁 System Files

```
levenshtein/
├── compare_final_units.py        # Main unit-level comparison tool
├── test.out.final.truth          # Ground truth data for testing
├── unit_accuracy_report.json     # Generated accuracy report
├── tested/                       # Test results and validation data
│   ├── various test files...
└── README.md                    # This documentation
```

## 🚀 Quick Start

### Basic Unit Comparison

```bash
# Compare prediction with ground truth
python compare_final_units.py

# With custom files
python compare_final_units.py --prediction model_output.out.final --truth ground_truth.out.final
```

### Required File Format

Input files must be in `.out.final` format (from decompose pipeline):
```
# Each line contains alternating labels and characters
0W1Q2M 0B1C2L
3A4B5C 6D7E8F
```

## 🔧 Tool Functionality

### Main Analysis Script (`compare_final_units.py`)

**Purpose**: Performs comprehensive unit-level accuracy analysis using Levenshtein distance.

**Features**:
- **Unit-level parsing**: Extracts all characters and labels as independent units
- **Edit distance calculation**: Computes minimum operations needed for correction
- **Operation analysis**: Breaks down insertions, deletions, and substitutions
- **Accuracy metrics**: Provides overall and per-category accuracy statistics

**Algorithm Flow**:
1. **Parse Input**: Extract units (labels + characters) from `.out.final` files
2. **Align Sequences**: Use dynamic programming for optimal alignment
3. **Calculate Distance**: Compute edit operations and accuracy
4. **Generate Report**: Output detailed statistics and examples

### File Format Parsing

The tool parses `.out.final` format where each line contains:
```
0W1Q2M 0B1C2L
│└┼┼┼ │└┼┼┼
│ │││ │ │││
│ │││ │ │└── Character 'L'
│ │││ │ └─── Label '2'
│ │││ └───── Character 'B'
│ ││└─────── Label '0'
│ │└──────── Character 'M'
│ └────────── Label '2'
└──────────── Character 'Q', Label '1', Character 'W', Label '0'
```

## 📊 Analysis Features

### Unit-Level Accuracy

**Character Accuracy**:
- Individual character prediction accuracy
- Character-level error patterns
- Position-specific accuracy analysis

**Label Accuracy**:
- Morphological label prediction accuracy
- Label distribution analysis
- Pattern-specific performance

**Combined Unit Accuracy**:
- Overall unit-level performance (characters + labels)
- Sequence-level error analysis
- Edit operation statistics

### Edit Distance Analysis

**Operation Types**:
- **Insertions**: Extra units in prediction
- **Deletions**: Missing units from prediction
- **Substitutions**: Incorrect unit replacements
- **Matches**: Correctly predicted units

**Statistical Breakdown**:
```json
{
  "total_units": 15420,
  "correct_units": 13876,
  "accuracy": 0.8998,
  "edit_operations": {
    "insertions": 234,
    "deletions": 187,
    "substitutions": 1123,
    "matches": 13876
  }
}
```

## 💡 Usage Examples

### Basic Comparison
```bash
python compare_final_units.py
```

**Expected Output**:
```
Unit-level accuracy analysis:
Total units compared: 15420
Correct units: 13876
Unit accuracy: 89.98%

Edit distance breakdown:
- Insertions: 234 (1.52%)
- Deletions: 187 (1.21%)
- Substitutions: 1123 (7.28%)
- Matches: 13876 (89.98%)

Average edit distance per sequence: 2.47
```

### Custom File Analysis
```bash
python compare_final_units.py \
  --prediction outputs/lstm_predictions.out.final \
  --truth data/test.out.final
```

### Detailed Error Analysis
```bash
python compare_final_units.py --detailed --output-errors errors.txt
```

## 🔍 Output Analysis

### Accuracy Report (`unit_accuracy_report.json`)

Generated automatically after analysis:
```json
{
  "total_sequences": 10869,
  "total_units": 154382,
  "correct_units": 138947,
  "unit_accuracy": 0.8998,
  "sequence_accuracy": 0.7234,
  "average_edit_distance": 2.47,
  "edit_operations": {
    "insertions": 2341,
    "deletions": 1876,
    "substitutions": 11218,
    "matches": 138947
  },
  "error_distribution": {
    "character_errors": 5629,
    "label_errors": 7806,
    "mixed_errors": 1900
  }
}
```

### Error Pattern Analysis

**Common Error Types**:
1. **Character Confusion**: Similar characters misclassified
2. **Label Confusion**: Related morphological labels swapped
3. **Sequence Alignment**: Length mismatches causing cascading errors
4. **Pattern Boundaries**: Errors at morpheme boundaries

**Example Error Output**:
```
Sequence 1247: Edit distance = 4
Ground truth: [0, 'W', 1, 'Q', 2, 'M']
Prediction:   [0, 'W', 1, 'R', 2, 'M', 3, 'N']
Operations:   [M, M, M, S, M, M, I, I]
             Match Match Match Subst Match Match Insert Insert
```

## 🛠 Advanced Features

### Command Line Arguments

```bash
python compare_final_units.py [-h] [--prediction PRED] [--truth TRUTH]
                              [--detailed] [--output-errors FILE]
                              [--min-edit-distance N] [--max-sequences N]

optional arguments:
  -h, --help            show help message and exit
  --prediction PRED     prediction file path (.out.final format)
  --truth TRUTH         ground truth file path (.out.final format)
  --detailed            enable detailed error analysis
  --output-errors FILE  save detailed errors to file
  --min-edit-distance N only analyze sequences with edit distance >= N
  --max-sequences N     limit analysis to first N sequences
```

### Filtering and Analysis Options

**Error Threshold Filtering**:
```bash
# Only analyze sequences with edit distance >= 3
python compare_final_units.py --min-edit-distance 3
```

**Sample Analysis**:
```bash
# Analyze first 1000 sequences only
python compare_final_units.py --max-sequences 1000
```

**Detailed Error Export**:
```bash
# Export all errors to file for further analysis
python compare_final_units.py --detailed --output-errors detailed_errors.txt
```

## 📈 Integration with Other Tools

### With Accuracy Analysis System
```bash
# 1. Run 2D accuracy analysis
cd ../accuracy_analysis_system/
python accuracy_analyzer.py --enhanced

# 2. Run unit-level analysis
cd ../levenshtein/
python compare_final_units.py --detailed

# 3. Compare results for comprehensive evaluation
```

### With Decompose Pipeline
```bash
# 1. Process through decompose pipeline
cd ../decompose/
python 01_out_original_reducer.py
python 03_convert_reduced_to_final.py

# 2. Analyze with Levenshtein tool
cd ../levenshtein/
python compare_final_units.py
```

## 🔬 Research Applications

### Model Comparison Studies
- **Architecture Evaluation**: Compare different neural network architectures
- **Training Progress**: Track unit-level accuracy improvements over epochs
- **Hyperparameter Tuning**: Identify optimal configurations for unit-level performance

### Error Analysis Research
- **Systematic Errors**: Identify recurring patterns in model predictions
- **Linguistic Analysis**: Study morphological complexity impacts on accuracy
- **Character-Level Patterns**: Analyze character-specific prediction difficulties

### Performance Benchmarking
- **Baseline Establishment**: Set unit-level accuracy benchmarks
- **Cross-Language Studies**: Adapt analysis for other morphologically rich languages
- **Fine-tuning Evaluation**: Measure improvements from domain adaptation

## 🧪 Validation and Testing

### Test Data Validation
```bash
# Validate test files in tested/ directory
cd tested/
python ../compare_final_units.py --prediction test_pred.out.final --truth test_truth.out.final
```

### Unit Test Suite
```bash
# Run unit tests for parsing and analysis functions
python -m pytest test_levenshtein.py

# Test with sample data
python test_sample_analysis.py
```

### Performance Benchmarking
```bash
# Benchmark analysis speed with large datasets
time python compare_final_units.py --max-sequences 50000
```

## 💡 Best Practices

### File Preparation
1. **Ensure Consistent Format**: All files must be in `.out.final` format
2. **Validate Line Counts**: Prediction and truth files must have identical line counts
3. **Check Encoding**: Use UTF-8 encoding for all input files

### Analysis Strategy
1. **Start with Sample**: Use `--max-sequences` for initial exploration
2. **Focus on Errors**: Use `--min-edit-distance` to study problematic cases
3. **Export Details**: Use `--output-errors` for manual error inspection

### Interpretation Guidelines
1. **Unit vs Sequence Accuracy**: Unit accuracy is typically higher than sequence accuracy
2. **Error Distribution**: Focus on substitution errors as they indicate systematic issues
3. **Edit Distance Patterns**: Low average edit distance suggests good model performance

## 🔧 Troubleshooting

| Problem | Symptoms | Solution |
|---------|----------|----------|
| Parsing Errors | Invalid format warnings | Verify `.out.final` format compliance |
| Line Count Mismatch | Analysis stops early | Check file line counts match exactly |
| Memory Issues | Process killed/slow | Use `--max-sequences` to limit analysis |
| Empty Results | No output generated | Verify input files exist and are readable |

### Debug Mode
```bash
# Enable verbose debugging
python compare_final_units.py --debug --max-sequences 10
```

## 📚 Technical Implementation

### Algorithm Complexity
- **Time Complexity**: O(mn) where m, n are sequence lengths
- **Space Complexity**: O(mn) for dynamic programming table
- **Scalability**: Handles sequences up to 10,000 units efficiently

### Edit Distance Algorithm
Uses optimized dynamic programming with:
- **Traceback mechanism** for operation identification
- **Memory optimization** for large sequences
- **Early termination** for maximum distance thresholds

---

For complete data processing workflow, see the main `data_processing_tools/README.md`.
For higher-level accuracy analysis, see `accuracy_analysis_system/README.md`.
For data format preparation, see `decompose/README.md`.