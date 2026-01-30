# Levenshtein Analysis - Executive Summary

## Quick Results

### Model Rankings

| Rank | Model | Accuracy | Perfect Lines | Recommendation |
|------|-------|----------|---------------|----------------|
| 🥇 | transformer_s4_on_s4 | 96.54% | 50.42% | Use for S4 test data |
| 🥈 | transformer_s2_on_s2 | 96.00% | 46.18% | **Use for S2 test data** ⭐ |
| 🥉 | transformer_s4_on_s2 | 95.92% | 45.16% | Not recommended |

## Key Findings

### 1. Dataset Quality > Size
- **S2 training** (50K lines) outperforms **S4 training** (97K lines) on S2 test set
- Performance gap: 0.074 percentage points
- **Conclusion**: Smaller, cleaner data is more effective

### 2. Distribution Matching Matters
- Best results when training and test data match
- Cross-dataset performance degrades slightly
- Native test set performance: S4_on_S4 (96.54%) > S2_on_S2 (96.00%)

### 3. All Models Perform Well
- All models achieve >95.9% character accuracy
- Perfect prediction rates range from 45-50%
- Transformer architecture is highly effective for Syriac morphology

## Recommendations

### For Production Use (S2 Test Set)
✅ **Use: transformer_s2_on_s2**
- Highest S2 accuracy: 96.00%
- Best perfect line rate: 46.18%
- Lower training cost (half the data)
- Simpler pipeline

### For S4 Test Set
✅ **Use: transformer_s4_on_s4**
- Highest overall accuracy: 96.54%
- Best perfect line rate: 50.42%

## Detailed Report
See `LEVENSHTEIN_ANALYSIS_REPORT.md` for full analysis.

## Reproduce Results
```bash
cd data_processing_tools/levenshtein
python compare_all_three_transformers.py
```
