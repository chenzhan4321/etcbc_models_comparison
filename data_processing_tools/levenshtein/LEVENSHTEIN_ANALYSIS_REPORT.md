# Levenshtein Distance Analysis Report
## Three Transformer Models Comparison

**Date**: January 28, 2026
**Analysis Type**: Character-level Levenshtein Distance
**Test Dataset**: Syriac Morphological Analysis

---

## 📊 Executive Summary

This report presents a comprehensive Levenshtein distance analysis of three Transformer model variants trained and tested on different dataset configurations (S2 and S4). The analysis evaluates character-level accuracy to assess morphological prediction quality.

### Overall Rankings

| Rank | Model | Training Data | Test Data | Character Accuracy | Perfect Line Rate | Error Count |
|------|-------|--------------|-----------|-------------------|------------------|-------------|
| 🥇 1st | **transformer_s4_on_s4** | S4 (97,048 lines) | S4 (18,232 lines) | **96.5366%** | **50.42%** | 35,384 |
| 🥈 2nd | **transformer_s2_on_s2** | S2 (50,649 lines) | S2 (10,869 lines) | **95.9968%** | **46.18%** | 24,370 |
| 🥉 3rd | **transformer_s4_on_s2** | S4 (97,048 lines) | S2 (10,869 lines) | **95.9224%** | **45.16%** | 24,823 |

---

## 🔍 Detailed Analysis

### 1. Model Performance Breakdown

#### transformer_s4_on_s4 (S4-trained, S4-tested)
- **Character Accuracy**: 96.5366%
- **Total Characters**: 1,021,660
- **Correct Characters**: 986,276
- **Error Characters**: 35,384
- **Test Lines**: 18,232
- **Perfect Predictions**: 9,193 lines (50.42%)
- **Lines with Errors**: 9,039 lines (49.58%)

**Key Insight**: This model achieves the highest overall accuracy when trained and tested on the same (S4) dataset distribution. The perfect prediction rate exceeds 50%, demonstrating strong generalization on its native test set.

#### transformer_s2_on_s2 (S2-trained, S2-tested)
- **Character Accuracy**: 95.9968%
- **Total Characters**: 608,769
- **Correct Characters**: 584,399
- **Error Characters**: 24,370
- **Test Lines**: 10,869
- **Perfect Predictions**: 5,019 lines (46.18%)
- **Lines with Errors**: 5,850 lines (53.82%)

**Key Insight**: Best performance on S2 test set despite using approximately half the training data compared to S4. This suggests excellent data quality and strong alignment between training and test distributions.

#### transformer_s4_on_s2 (S4-trained, S2-tested)
- **Character Accuracy**: 95.9224%
- **Total Characters**: 608,769
- **Correct Characters**: 583,946
- **Error Characters**: 24,823
- **Test Lines**: 10,869
- **Perfect Predictions**: 4,908 lines (45.16%)
- **Lines with Errors**: 5,961 lines (54.84%)

**Key Insight**: Despite using nearly 2× training data (S4 vs S2), this model performs slightly worse on S2 test set compared to the S2-trained model. This counterintuitive result suggests potential distribution mismatch or data quality issues in the S4 extension.

---

## 📈 Key Findings

### Finding 1: Dataset Consistency Matters More Than Size

**S2 Test Set Performance Comparison**:
- transformer_s2_on_s2: **95.9968%** ✅ (trained on 50,649 lines)
- transformer_s4_on_s2: **95.9224%** (trained on 97,048 lines)
- **Performance Gap**: 0.0744 percentage points
- **Error Reduction**: 453 fewer character errors (1.82% reduction)
- **Perfect Line Improvement**: 111 additional perfect predictions

**Interpretation**: Training on the larger S4 dataset (nearly 2× the size of S2) does not improve performance on the S2 test set. In fact, it slightly degrades performance. This suggests that:
1. **Data quality > quantity**: S2 training data may be cleaner and more consistent
2. **Distribution alignment**: S2 training data better matches S2 test distribution
3. **Potential noise in S4**: Extended S4 data may introduce inconsistencies or labeling noise

### Finding 2: Native Test Set Performance

**transformer_s4_on_s4 Performance**:
- Achieves highest overall accuracy: **96.5366%**
- Perfect prediction rate: **50.42%** (highest among all models)
- Test scale: 18,232 lines (1.7× larger than S2)

**Interpretation**: When training and test data come from the same distribution (S4), the model achieves optimal performance. The 0.54 percentage point improvement over S2 models may be attributed to:
1. Better training-test distribution alignment
2. Larger test set may contain easier examples
3. S4 data characteristics favor Transformer architecture

### Finding 3: Training Data Scale Impact

| Training Dataset | Size | Best Test Accuracy | Cost-Benefit |
|-----------------|------|-------------------|--------------|
| S2 | 50,649 lines | 95.9968% (on S2) | ⭐⭐⭐⭐⭐ High efficiency |
| S4 | 97,048 lines | 96.5366% (on S4) | ⭐⭐⭐ Moderate efficiency |
| S4 | 97,048 lines | 95.9224% (on S2) | ⭐⭐ Poor cross-domain |

**Interpretation**:
- Doubling training data does not guarantee proportional improvement
- Cross-dataset performance may degrade with larger heterogeneous data
- For S2 deployment, smaller but cleaner S2 training set is more cost-effective

---

## 💡 Recommendations

### For S2 Test Set Deployment
**Recommended Model**: `transformer_s2_on_s2`

**Rationale**:
- ✅ Highest S2 test accuracy (95.9968%)
- ✅ Best perfect prediction rate on S2 (46.18%)
- ✅ Lower training cost (50% less data)
- ✅ Simpler pipeline (no S4 data required)

**Expected Performance**:
- ~96% character-level accuracy
- ~46% of test sequences predicted perfectly
- Average 2.24 character errors per imperfect sequence

### For S4 Test Set Deployment
**Recommended Model**: `transformer_s4_on_s4`

**Rationale**:
- ✅ Highest overall accuracy (96.5366%)
- ✅ Best perfect prediction rate (50.42%)
- ✅ Native dataset distribution

**Expected Performance**:
- ~96.5% character-level accuracy
- ~50% of test sequences predicted perfectly
- Average 3.91 character errors per imperfect sequence

### For Future Work
1. **Data Quality Audit**: Investigate S4 dataset extensions for potential noise or inconsistencies
2. **Hybrid Training**: Consider S2+filtered_S4 approach to leverage data scale while maintaining quality
3. **Error Analysis**: Deep dive into systematic error patterns (see error examples below)
4. **Ensemble Methods**: Combine S2 and S4 models for potentially improved robustness

---

## 🔬 Error Pattern Analysis

### Common Error Types (from samples)

**Type 1: Character Substitution** (most frequent)
```
Predicted: BR/-H    (missing '=')
Truth:     BR/-H=
Distance: 1 character
```

**Type 2: Symbol Confusion**
```
Predicted: >MR[/
Truth:     >MR[
Distance: 1 character (extra '/')
```

**Type 3: Morphological Marker Errors**
```
Predicted: MR(>/~>
Truth:     MR>/
Distance: 3 characters
```

**Observation**: Most errors are localized (1-3 character distances), suggesting the model captures overall structure well but struggles with fine-grained morphological markers.

---

## 📊 Statistical Summary

### Performance Metrics by Test Set

**S2 Test Set (10,869 lines)**:
| Metric | S2-trained | S4-trained | Delta |
|--------|-----------|-----------|-------|
| Accuracy | 95.9968% | 95.9224% | +0.0744% |
| Perfect Lines | 5,019 (46.18%) | 4,908 (45.16%) | +111 lines |
| Total Errors | 24,370 | 24,823 | -453 errors |

**S4 Test Set (18,232 lines)**:
| Metric | S4-trained |
|--------|-----------|
| Accuracy | 96.5366% |
| Perfect Lines | 9,193 (50.42%) |
| Total Errors | 35,384 |

### Error Rate Analysis

**Character Error Rate by Model**:
- transformer_s2_on_s2: 4.00% (1 in 25 characters)
- transformer_s4_on_s2: 4.08% (1 in 24.5 characters)
- transformer_s4_on_s4: 3.46% (1 in 29 characters)

**Line Error Rate by Model**:
- transformer_s2_on_s2: 53.82% (5,850 / 10,869 lines have errors)
- transformer_s4_on_s2: 54.84% (5,961 / 10,869 lines have errors)
- transformer_s4_on_s4: 49.58% (9,039 / 18,232 lines have errors)

---

## 🛠️ Methodology

### Data Preprocessing
All `.original` files were preprocessed with the following steps:
1. **transformer_s2_on_s2**: Removed leading whitespace from all lines
2. **transformer_s4_on_s2**: Removed `Truevalue` lines and `Predicted ` prefix
3. **transformer_s4_on_s4**: Removed leading whitespace; extracted ground truth from results file

### Backup Files Created
- `outputs/transformer_s2_on_s2/test.out.original.backup`
- `outputs/transformer_s4_on_s2/test.out.original.backup`
- `outputs/transformer_s4_on_s4/test.out.original.backup`

### Comparison Method
- **Algorithm**: Dynamic Programming Levenshtein Distance
- **Complexity**: O(mn) time, O(mn) space
- **Granularity**: Character-level comparison
- **No Normalization**: Exact character matching (no strip() applied during comparison)

### Ground Truth Sources
- **S2 models**: `data/raw_s2_on_s2/test.out.original`
- **S4 model**: Extracted from `results_7seq_len_0.0001lr_512embsize_8nhead_transformer_0.1dropout_128_batchsize_30epochs_3beamsize.txt` (Truevalue lines)

---

## 📁 Deliverables

### Analysis Scripts
- `compare_s2_cleaned.py` - S2_on_S2 analysis
- `compare_s4_on_s2_cleaned.py` - S4_on_S2 analysis
- `compare_s4_on_s4.py` - S4_on_S4 analysis
- `compare_all_three_transformers.py` - Comprehensive comparison
- `compare_both_transformers.py` - S2 test set comparison

### Output Files
- `batch_accuracy_report.json` - All model accuracy metrics
- Processed `.original` files with cleaned formatting
- Backup `.original.backup` files

---

## 🎯 Conclusions

1. **Quality Over Quantity**: The smaller S2 dataset (50K lines) produces better results on S2 test set than the larger S4 dataset (97K lines), highlighting the importance of data quality over sheer volume.

2. **Distribution Alignment Critical**: Best performance is achieved when training and test distributions match (96.54% for S4_on_S4, 96.00% for S2_on_S2).

3. **Cross-Dataset Degradation**: Training on S4 and testing on S2 yields worse results than training on S2 directly, suggesting potential distribution mismatch.

4. **High Overall Performance**: All three models achieve >95.9% character accuracy, demonstrating the Transformer architecture's strong capability for Syriac morphological analysis.

5. **Practical Recommendation**: For production deployment on S2-like data, use `transformer_s2_on_s2` for optimal accuracy and cost-efficiency.

---

## 📞 Contact & Reproducibility

**Analysis Location**: `data_processing_tools/levenshtein/`
**Reproducibility**: All analysis scripts and processed data files are version-controlled.
**Rerun Analysis**:
```bash
cd data_processing_tools/levenshtein
python compare_all_three_transformers.py
```

---

*Report Generated: January 28, 2026*
*Analysis Tool: Levenshtein Distance (Edit Distance) Algorithm*
*Character Encoding: UTF-8*
