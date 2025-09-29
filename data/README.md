# Data Directory Structure

This directory contains the Syriac morphological analysis datasets organized into two clear configurations:

## 📁 Directory Layout

```
data/
├── s2_on_s2/              # S2-on-S2 Configuration (Standard)
│   ├── train.in/out       # 50,649 lines, 309 classes
│   ├── val.in/out         # 10,964 lines
│   ├── test.in/out        # 10,869 lines
│   └── patterns.csv       # S2 pattern mappings
├── s4_on_s2/              # S4-on-S2 Configuration (Extended Training)
│   ├── train.in/out       # 97,048 lines, ~350 classes
│   ├── val.in/out         # 10,964 lines (same as S2)
│   ├── test.in/out        # 10,869 lines (same as S2)
│   └── patterns.csv       # S4 pattern mappings
└── raw_s4_on_s2/       # [Legacy] Original S4 dataset location
└── raw_s2_on_s2/       # [Legacy] Original S2 dataset location
```

## ⚠️ Dataset Configurations

### S2-on-S2 (Standard Configuration)
- **Training**: 50,649 lines from S2 dataset
- **Validation**: 10,964 lines from S2 dataset
- **Testing**: 10,869 lines from S2 dataset
- **Classes**: 309 morphological classes
- **Use Case**: Standard baseline, fair comparison

### S4-on-S2 (Extended Training Configuration)
- **Training**: 97,048 lines from S4 dataset (larger training set)
- **Validation**: 10,964 lines from S2 dataset (consistent evaluation)
- **Testing**: 10,869 lines from S2 dataset (consistent evaluation)
- **Classes**: ~350 morphological classes in training, 309 in evaluation
- **Use Case**: Extended training data with consistent evaluation metrics

## 🚀 Usage

### Training Commands

```bash
# S2-on-S2 Configuration
python train.py --model_type transformer --data_dir data/s2_on_s2 --epochs 50

# S4-on-S2 Configuration
python train.py --model_type transformer --data_dir data/s4_on_s2 --epochs 50
```

### Key Benefits

1. **Consistent Evaluation**: Both configurations use identical validation and test sets
2. **Fair Comparison**: Results can be directly compared across configurations
3. **Clear Separation**: No confusion about which dataset is being used
4. **Reproducible**: Each configuration is self-contained

## 📊 File Formats

- **`.in` files**: Input Syriac character sequences
- **`.out` files**: Morphological tag sequences (space-separated integers)
- **`patterns.csv`**: Mapping from morphological patterns to integer labels

## 🔧 Verification

```bash
# Check training set sizes
wc -l data/s2_on_s2/train.in data/s4_on_s2/train.in
# Expected: 50,649 and 97,048 lines respectively

# Verify test sets are identical
diff data/s2_on_s2/test.in data/s4_on_s2/test.in
# Should show no differences
```