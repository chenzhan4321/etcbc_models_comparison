# Syriac Morphological Analysis - Data Decomposition Pipeline

## 🎯 Overview

The decompose pipeline is the **core data processing system** that transforms raw model outputs into the final standardized format used for evaluation. This 4-stage pipeline handles morphological pattern extraction, labeling, and format standardization for Syriac text analysis.

## 🔄 Complete Pipeline Flow

```
Raw Model Output (.out.original)
            ↓
    [01] Symbol Reduction
            ↓
    Simplified Format (.out.reduced)
            ↓
    [02] Pattern Extraction
            ↓
    Pattern Database (patterns.csv)
            ↓
    [03] Pattern → Label Conversion
            ↓
    Numerical Format (.out.final)
            ↓
    [04] Label → Symbol Conversion
            ↓
    Standard Output (.out)
```

## 📁 Pipeline Components

```
decompose/
├── 00_cleanse_in_original_to_in.py    # [Optional] Input cleaning
├── 01_out_original_reducer.py         # [Stage 1] Symbol pair reduction
├── 02_generate_patterns_csv_from_reduced.py  # [Stage 2] Pattern extraction
├── 03_convert_reduced_to_final.py     # [Stage 3] Pattern labeling
├── 04_convert_final_to_out.py         # [Stage 4] Final conversion
├── patterns.csv                       # Pattern-to-label mapping database
└── README.md                         # This documentation
```

## 🚀 Quick Start

### Complete Pipeline Execution

```bash
# Navigate to decompose directory
cd data_processing_tools/decompose/

# Execute full pipeline in order
python 01_out_original_reducer.py
python 02_generate_patterns_csv_from_reduced.py
python 03_convert_reduced_to_final.py
python 04_convert_final_to_out.py
```

### Individual Stage Execution

```bash
# Stage 1: Reduce symbol pairs
python 01_out_original_reducer.py --input model_output.out.original

# Stage 2: Extract patterns (after stage 1)
python 02_generate_patterns_csv_from_reduced.py

# Stage 3: Convert to final format
python 03_convert_reduced_to_final.py --mode final

# Stage 4: Generate standard output
python 04_convert_final_to_out.py
```

## 🔧 Detailed Stage Documentation

### Stage 1: Symbol Pair Reduction (`01_out_original_reducer.py`)

**Purpose**: Removes redundant symbol pairs from raw model output to simplify subsequent processing.

**Input**: `.out.original` files
**Output**: `.out.reduced` files

**Supported Symbol Pairs**:
- `!...!` → `...!` (removes first `!`)
- `@...@` → `...@` (removes first `@`)
- `]...]` → `...]` (removes first `]`)

**Example Transformation**:
```
Input:  !ABC!DEF @GHI@ ]JKL]
Output: ABC!DEF GHI@ JKL]
```

**Usage**:
```bash
python 01_out_original_reducer.py

# Processes automatically:
# train.out.original → train.out.reduced
# val.out.original → val.out.reduced
# test.out.original → test.out.reduced
```

### Stage 2: Pattern Extraction (`02_generate_patterns_csv_from_reduced.py`)

**Purpose**: Analyzes `.out.reduced` files to extract morphological patterns and creates a comprehensive pattern database.

**Input**: `.out.reduced` files
**Output**: `patterns.csv` (pattern database)

**Pattern Extraction Rules**:
1. **Parentheses patterns**: `(` followed by uppercase letter or `<>`
2. **Colon patterns**: `:` followed by `d`, `p`, or `dp`
3. **Special markers**: Various morphological indicators

**patterns.csv Format** (`label,pattern,count`; label 0 = the empty/null
pattern, labels 1–328 = observed registers ordered by descending frequency,
label 329 = the reserved `<UNKNOWN>` fallback):
```csv
label,pattern,count
0,,3066705
1,-,404156
2,/,165508
329,<UNKNOWN>,0
```

**Usage**:
```bash
python 02_generate_patterns_csv_from_reduced.py

# Automatically processes all .out.reduced files
# Generates patterns.csv with frequency counts
```

### Stage 3: Pattern to Label Conversion (`03_convert_reduced_to_final.py`)

**Purpose**: Converts morphological patterns to numerical labels using the pattern database, creating the final numerical representation.

**Input**:
- `.out.reduced` files
- `patterns.csv` (pattern database)

**Output**: `.out.final` files

**Output Formats**:
1. **Pure Final** (`--mode final`): Character ID + Label ID alternating
2. **Out Final** (`--mode out-final`): Label + Character alternating (preserves original letters)

**Example Conversion**:
```
Input (.reduced):  W(A)Q:d M(B)N:p
Pattern Database:  (A)→1, :d→2, (B)→3, :p→4
Output (.final):   W1Q2 M3N4
```

**Usage**:
```bash
# Pure final format (default)
python 03_convert_reduced_to_final.py

# Alternative format preserving characters
python 03_convert_reduced_to_final.py --mode out-final

# Custom input file
python 03_convert_reduced_to_final.py --input custom.out.reduced
```

**Command Line Arguments**:
```bash
python 03_convert_reduced_to_final.py [-h] [--mode {final,out-final}] [--input INPUT]

optional arguments:
  -h, --help            show help message and exit
  --mode {final,out-final}  output format mode (default: final)
  --input INPUT         input .reduced file path
```

### Stage 4: Final to Standard Conversion (`04_convert_final_to_out.py`)

**Purpose**: Converts numerical `.out.final` format back to symbolic representation for evaluation and analysis.

**Input**: `.out.final` files
**Output**: `.out` files (standard format)

**Conversion Process**:
1. Reads numerical label sequences
2. Maps labels back to morphological symbols
3. Reconstructs standard morphological analysis format
4. Handles special cases and unknown patterns

**Usage**:
```bash
python 04_convert_final_to_out.py

# Automatically processes:
# train.out.final → train.out
# val.out.final → val.out
# test.out.final → test.out
```

## 📊 File Format Specifications

### .out.original Format
Raw model output with morphological annotations:
```
W!ABC!Q@DEF@ M]GHI]N
B!XYZ!C@UVW@ L]STU]M
```

### .out.reduced Format
Simplified with reduced symbol pairs:
```
WABC!QDEF@ MGHI]N
BXYZ!CUVW@ LSTU]M
```

### patterns.csv Format
Pattern database with labels and frequencies (`label,pattern,count`; label 0 is
the null pattern, label 329 is the reserved `<UNKNOWN>` fallback):
```csv
label,pattern,count
0,,3066705
1,-,404156
2,/,165508
329,<UNKNOWN>,0
```

### .out.final Format
Numerical representation with character-label pairs:
```
W1Q3 M2N
B1C3 L2M
```

### .out Format
Final symbolic morphological analysis:
```
W/Q M/N
B/C L/M
```

## 🛠 Advanced Configuration

### Pattern Extraction Customization

Modify pattern extraction rules in `02_generate_patterns_csv_from_reduced.py`:

```python
# Custom pattern extraction rules
def extract_patterns_custom(text):
    patterns = []
    # Add your custom extraction logic here
    return patterns
```

### Label Assignment Strategy

Label assignment in `patterns.csv`:
- **Label 0**: the empty/null pattern (no boundary marker)
- **Labels 1–328**: observed register patterns, assigned by descending frequency
- **Label 329**: the reserved `<UNKNOWN>` fallback (training count 0; never predicted)

### Unknown Pattern Handling

Configure unknown pattern behavior in `03_convert_reduced_to_final.py`:
```python
# Options for unknown patterns:
UNKNOWN_STRATEGIES = {
    'zero': 0,           # Assign label 0
    'skip': None,        # Skip unknown patterns
    'error': 'raise',    # Raise error on unknown
    'warn': 'warn'       # Log warning and continue
}
```

## 🔍 Quality Assurance

### Validation Checks

The pipeline includes built-in validation:

1. **File Existence**: Verifies all required input files exist
2. **Format Validation**: Checks file format compliance
3. **Line Count Consistency**: Ensures consistent line counts across stages
4. **Pattern Coverage**: Validates all patterns have labels
5. **Character Preservation**: Verifies character sequences are preserved

### Error Detection

Common error patterns and solutions:

| Error Type | Symptoms | Solution |
|------------|----------|----------|
| Missing Patterns | `<UNKNOWN>` in output | Update patterns.csv or re-run stage 2 |
| Line Count Mismatch | Processing stops | Check for empty lines or encoding issues |
| Format Errors | Parsing failures | Validate input file format |
| Label Conflicts | Inconsistent labeling | Regenerate patterns.csv |

### Debugging Tools

```bash
# Check pattern coverage
python -c "
import csv
with open('patterns.csv') as f:
    patterns = list(csv.DictReader(f))
    print(f'Total patterns: {len(patterns)}')
    unknown = [p for p in patterns if p['pattern'] == '<UNKNOWN>']
    print(f'Unknown patterns: {len(unknown)}')
"

# Validate file line counts
wc -l *.out.reduced *.out.final *.out
```

## 🎯 Integration with Main System

### Training Pipeline Integration

```bash
# 1. Train model (generates .out.original)
python train.py --model_type transformer

# 2. Process through decompose pipeline
cd data_processing_tools/decompose/
python 01_out_original_reducer.py
python 02_generate_patterns_csv_from_reduced.py
python 03_convert_reduced_to_final.py
python 04_convert_final_to_out.py

# 3. Analyze results
cd ../accuracy_analysis_system/
python accuracy_analyzer.py --enhanced
```

### Batch Processing

For multiple model outputs:
```bash
# Process multiple models
for model in transformer mdlm bert; do
    echo "Processing $model..."
    cp outputs/${model}_predictions.out.original train.out.original
    python 01_out_original_reducer.py
    python 03_convert_reduced_to_final.py
    python 04_convert_final_to_out.py
    mv train.out outputs/${model}_final.out
done
```

## ⚡ Performance Optimization

### Memory Usage
- **Large Files**: Process in chunks for files >100MB
- **Pattern Database**: Cache patterns.csv in memory
- **Parallel Processing**: Use multiprocessing for multiple files

### Speed Optimization
```bash
# Fast processing for development
python 01_out_original_reducer.py --quick-mode
python 03_convert_reduced_to_final.py --fast --mode final
```

## 🧪 Testing and Validation

### Unit Tests
```bash
# Run pipeline tests
python -m pytest test_decompose_pipeline.py

# Test individual stages
python -m pytest test_stage_01.py
python -m pytest test_stage_03.py
```

### Sample Data Validation
```bash
# Validate with sample data
python 01_out_original_reducer.py --input sample.out.original --output sample.out.reduced
python 03_convert_reduced_to_final.py --input sample.out.reduced
```

## 🔬 Research Applications

- **Morphological Analysis**: Study pattern distribution across languages
- **Model Comparison**: Standardized evaluation across different architectures
- **Pattern Evolution**: Track pattern usage changes during training
- **Cross-lingual Studies**: Adapt pipeline for other Semitic languages

## 🤝 Contributing

### Adding New Stages
1. Follow naming convention: `XX_stage_name.py`
2. Implement standard interface: input/output file handling
3. Add validation and error handling
4. Update this documentation

### Pattern Extraction Rules
1. Add new rules to `02_generate_patterns_csv_from_reduced.py`
2. Test with diverse input data
3. Update pattern database validation
4. Document new pattern types

---

For integration with other tools, see the main `data_processing_tools/README.md`.
For accuracy analysis of processed data, see `accuracy_analysis_system/README.md`.
For distance-based analysis, see `levenshtein/README.md`.