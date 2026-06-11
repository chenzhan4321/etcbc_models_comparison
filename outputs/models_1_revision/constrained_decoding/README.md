# Constrained decoding for the encoder–decoder (reviewer reference, R2-M2 / R2-M4)

Does forcing the encoder–decoder to respect the consonantal skeleton at decode time help?
Answer: **no — a hard skeleton constraint roughly doubles the error.** An informative negative
result: the generative model does not benefit from hard structural constraints, which supports the
discretized design (the discretized models encode that structure implicitly and robustly).

## Metric (canonical CER, identical formula to every other model)
`compute_cer` = `Σ_i Levenshtein(pred_i, ref_i) / Σ_i len(ref_i)` (micro, char-level) on the
restored **`test.out.original`** surface strings (10,869 test lines) — the same `cer_micro` as
`analysis/three_model_string_error.py`.

**Why the unconstrained number is ~4.7%, not the headline 4.00%:** this ablation runs on the
**`etcbc_update` re-implementation** `SyriacSeq2SeqModel` (`models/seq2seq.py`) with **greedy**
decoding (`model.generate` = beam-width 1). Greedy + re-impl ≈ 4.7%; the published headline 4.00%
uses the **upstream `ssi_morphology` pipeline with beam = 3**. Same canonical CER metric — the gap
is decoder (greedy vs beam) + pipeline, not metric. The constrained experiment is a within-decoder
comparison (greedy, same model, constraint ON vs OFF), so the conclusion is unaffected by the offset.

## Constraint mechanism
`models/seq2seq.py:generate_constrained` forces the output skeleton (source consonants + spaces, same
order) and only lets the model choose freely on the inserted morphological markers; EOS is allowed
only once the skeleton is exhausted. Here 13 of 40 target-vocabulary tokens are free markers.

## Results — reported set = the main-table 5 seeds {42, 43, 46, 48, 49}
| seed | UNCONSTRAINED CER (greedy) | CONSTRAINED CER (greedy + skeleton) |
|---|---|---|
| 42 | 4.5791% | 8.0687% |
| 43 | 4.8478% | 9.2723% |
| 46 | 4.8362% | 9.4652% |
| 48 | 4.8989% | 8.4817% |
| 49 | 4.6647% | 8.2110% |
| **mean ± SD** | **4.77% ± 0.14** | **8.70% ± 0.63** |

→ The hard skeleton constraint **increases** CER from 4.77% to 8.70% (≈ +3.9 pp, ~1.8×). The
constraint roughly doubles the error in every single seed (constrained 8.0–9.5% vs unconstrained
4.6–4.9%), so the negative result is robust.

## Caveats (disclosed to reviewer)
- Decoder is **greedy (beam = 1)**, model is the `etcbc_update` re-implementation — so the
  unconstrained baseline here (≈4.7%) is *not* the headline beam=3 upstream 4.00%. There is **no
  beam=3 constrained run** (`generate_constrained` is greedy-only); a constrained beam search was not
  implemented. Do not compare 8.70% (greedy-constrained) against 4.00% (beam3-unconstrained) — that
  would double-confound decoder and constraint. The valid comparison is within-greedy: 4.77 → 8.70.
- An earlier run set `cons_s2_s*.log` reported CONSTRAINED CER 47–58% — a **buggy constraint
  implementation, superseded** by the `cons2_*` logs here. Do **not** cite the `cons_*` numbers.

## Files
- `cons2_s2_s{42,43,46,48,49}.log` — one line each: `UNCONSTRAINED CER | CONSTRAINED CER`.
- `eval_seq2seq_constrained.py` — the evaluation script (greedy unconstrained vs greedy + skeleton).
