# MDLM canonical CER — both regimes, 5 seeds (reviewer reference)

Authoritative canonical CER for the headline MDLM, on **both data regimes**, recomputed with one
identical, calibrated pipeline. The two regimes are the **same model** (identical architecture and
training recipe); they differ **only** in the training data.

## The two models are identical (only the training data differs)
| hyperparameter | S4-on-S2 | S2-on-S2 |
|---|---|---|
| d_model | 768 | 768 |
| num_layers | 10 | 10 |
| num_heads | 6 | 6 |
| dropout | 0.23 | 0.23 |
| learning_rate | 5e-05 | 5e-05 |
| diffusion steps (T) | 3 | 3 |
| batch_size | 16 | 16 |
| num_epochs | 250 | 250 |
| **training data** | **S2 + S4** | **S2 only** |
| test set | pure-S2 held-out (10,869 lines) | the same pure-S2 set |

`data/s4_on_s2/test.out` and `data/s2_on_s2/test.out` are byte-identical, i.e. both regimes are scored
on exactly the same held-out test set; the comparison isolates the effect of adding S4 to the
training data.

## Metric
Canonical character-level micro-CER on the restored ETCBC surface string:
`CER = Σ_i Levenshtein(pred_i.original, gt_i.original) / Σ_i len(gt_i.original)`, computed with
`eval_cer.py` (integer label prediction → inject patterns → `restore_to_original.restore_line` →
per-line Levenshtein). Identical metric to every other model in the paper. The restored ground truth
(`test.GT.out.original`, 10,869 lines) was regenerated here and **validated**: scoring the
encoder-only / MDLM predictions reproduces the established headline numbers bit-for-bit (MDLM S4 seed42
= 3.3489 %, matching the recorded 3.349 %).

## Results
**S4-on-S2 (headline; train on S2+S4):**
| seed | 42 | 43 | 46 | 48 | 49 | **mean ± SD** |
|---|---|---|---|---|---|---|
| CER | 3.349% | 3.420% | 3.365% | 3.404% | 3.376% | **3.383% ± 0.029** ; bootstrap 95% CI **[3.302, 3.467]** |

**S2-on-S2 (train on S2 only):**
| seed | 42 | 43 | 46 | 48 | 49 | **mean ± SD** |
|---|---|---|---|---|---|---|
| CER | 3.687% | 3.725% | 3.638% | 3.650% | 3.735% | **3.687% ± 0.043** ; bootstrap 95% CI **[3.603, 3.775]** |

(Bootstrap = resample the 10,869 test lines, averaging the five seeds per line, 2,000 resamples, RNG 20260606 — the paper's per-cell method; it reproduces the S4 headline CI [3.302, 3.467] exactly.)

## Reading
- **Positive transfer.** Adding S4 to the training data *improves* accuracy on the pure-S2 test set
  (3.687% → 3.383%, −0.30 pp). The generative encoder–decoder shows the opposite (negative) transfer
  (4.00% → 4.08%). Discretized models benefit from more data; the generative baseline does not.
- The S2 number (3.687%) is essentially tied with the 2-step variant (3.637%; 0.05 pp, within seed
  noise), consistent with the diffusion-step ablation (step count has no material effect).

## Files
- `s4_on_s2/mdlm_pred_s{42,43,46,48,49}.out` — integer predictions (source: `mdlm_better_result_after_submission_…_s4-on-s2_5seeds`, Feb run).
- `s2_on_s2/mdlm_pred_s{42,43,46,48,49}.out` — integer predictions (source: `mdlm_train_20260608_165413_s{N}`).
- `test.GT.out.original` — restored ground-truth surface string (shared by both regimes).
- Re-score: `eval_cer.py --pred <pred> --infile data/s2_on_s2/test.in --patterns data/s2_on_s2/patterns.csv --gt test.GT.out.original`.
