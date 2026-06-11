# Data Directory Structure

This directory contains the Classical Syriac morphological restoration datasets,
organized into the two training regimes used in the paper. **Both regimes share
the identical fixed S2 validation and test sets**, so results are directly
comparable.

## 📁 Directory Layout

```
data/
├── s2_on_s2/              # Standard regime: train S2, test S2
│   ├── train.in/out       # 50,649 lines
│   ├── val.in/out         # 10,964 lines
│   ├── test.in/out        # 10,869 lines
│   └── patterns.csv       # register inventory (330 rows, labels 0–329)
├── s4_on_s2/              # Scaled regime: train S2+S3+S4, test S2
│   ├── train.in/out       # 97,048 lines
│   ├── val.in/out         # 10,964 lines (identical to s2_on_s2)
│   ├── test.in/out        # 10,869 lines (identical to s2_on_s2)
│   └── patterns.csv       # register inventory (330 rows, identical file)
├── raw_s2_on_s2/          # restored-surface ground truth (.out.original) for s2_on_s2
└── raw_s4_on_s2/          # restored-surface ground truth (.out.original) for s4_on_s2
```

## ⚠️ Dataset Configurations

### s2_on_s2 (standard / in-distribution)
- **Training**: 50,649 lines from the ETCBC S2 corpus
- **Validation / Test**: 10,964 / 10,869 lines from S2
- **Use case**: the in-distribution baseline regime of the paper

### s4_on_s2 (scaled / + out-of-distribution expansion)
- **Training**: 97,048 lines from the combined S2+S3+S4 corpus
- **Validation / Test**: the *same* 10,964 / 10,869 S2 lines as `s2_on_s2`
- **Use case**: the scaled regime — isolates the effect of adding heterogeneous
  training data while keeping the evaluation target fixed

### Label space (`patterns.csv`)

Both configurations carry the **same 330-row register inventory** (one header +
labels 0–329), used identically at training and evaluation time:

- **label 0** — the empty/null pattern (no boundary marker);
- **labels 1–328** — the register patterns observed in the S2+S3+S4 training
  split, ordered by descending frequency;
- **label 329** — the reserved `<UNKNOWN>` fallback (training count 0; never
  predicted). Out-of-vocabulary patterns that surface only at test time map here,
  outside the bijection, to avoid test-set leakage.

There is **no train-vs-test class mismatch**: the classification vocabulary is
fixed by `patterns.csv` within each configuration.

## 🚀 Usage

```bash
# standard regime
python train.py --model_type transformer --data_dir data/s2_on_s2

# scaled regime
python train.py --model_type transformer --data_dir data/s4_on_s2
```

## 📊 File Formats

- **`.in` files**: input Syriac consonantal character sequences
- **`.out` files**: per-position integer label sequences (space-separated)
- **`raw_*/.out.original`**: the restored ETCBC surface string — the
  representation on which the paper's CER is computed
- **`patterns.csv`**: `label,pattern,count` mapping from register patterns to
  integer labels

## 🔧 Verification

```bash
# Check training set sizes
wc -l data/s2_on_s2/train.in data/s4_on_s2/train.in
# Expected: 50,649 and 97,048 lines respectively

# Verify the two regimes share identical test sets
diff data/s2_on_s2/test.in data/s4_on_s2/test.in
# Should show no differences

# Verify the register inventory (330 rows + header)
wc -l data/s2_on_s2/patterns.csv   # 331
```
