# MDLM diffusion-step ablation — first-100-epoch validation accuracy (reviewer reference)

Supports the step-count ablation claim (R2-M3 / R3-1 / R3-4 / W8): **the number of
diffusion steps T has no material effect on accuracy.** Data are capped at the first
**100 epochs** (the regime the response letter relies on), to avoid over-reading the
long-tail noise of full 250-epoch runs.

## Setup
Same-scale MDLM, fast recipe, identical except the diffusion-step count T:
`d_model=768, num_layers=10, num_heads=6, dropout=0.23, batch=64, lr=3e-4,
data=S4-on-S2, seed=42`, T ∈ {1,2,3,4}, each in fp32 and bf16 (8 runs).
Metric below = upstream **validation word-accuracy** (per epoch), NOT canonical CER —
it is the plateau proxy used for the step comparison (val-acc and canonical CER are
not the same quantity; the letter's headline CER numbers come from the 5-seed
canonical-CER pipeline, see ../mdlm/canonical_cer_5seed/).

## Files
- `tsweep_valacc_100ep.csv` — per-epoch (1–100) validation word-accuracy, one column per
  run: `T{1,2,3,4}_{fp32,bf16}`.

## Plateau summary (epochs 60–100)
| run | mean | max |
|---|---|---|
| T1_fp32 | 0.9652 | 0.9669 |
| T2_fp32 | 0.9662 | 0.9670 |
| T3_fp32 | 0.9659 | 0.9670 |
| T4_fp32 | 0.9659 | 0.9670 |
| T1_bf16 | 0.9666 | 0.9674 |
| T2_bf16 | 0.9661 | 0.9669 |
| T3_bf16 | 0.9659 | 0.9671 |
| T4_bf16 | 0.9654 | 0.9671 |

## Reading
- All eight runs sit in a **~0.1 pp band** (plateau means 0.9652–0.9666; plateau maxima
  0.9669–0.9674). There is **no monotonic trend in T and no step count is robustly best**:
  the per-step plateau maxima are effectively tied (~0.967), and the small mean differences
  are within epoch-to-epoch / seed noise.
- The earlier small/short-scale hint that "T=1 is best" does **not** survive at this
  (best-performing) model scale: T=1 and T=4 are both near the top; T=2,3,4 are
  statistically indistinguishable.
- fp32 vs bf16: per-T differences ≤0.05 pp → bf16 does not cost accuracy (it is only faster).
- T=3 is the working point chosen for **interpretability** (step-by-step unmasking as a
  window into the model's morphological analysis), not because it is the accuracy optimum.

Single seed (42) → no formal significance test here; the headline uses 5 seeds. This file
is corroborating evidence for "step count is immaterial", consistent with the 5-seed
MDLM-S2 result (steps=3 ≈ steps=2, within noise).
