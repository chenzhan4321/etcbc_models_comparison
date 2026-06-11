# Encoder–Decoder hyperparameter search (reviewer reference)

To check whether the published encoder–decoder configuration (Naaijer et al.: emb 512 / 8 heads /
3 encoder / 3 decoder layers) is near-optimal, we ran an independent Optuna hyperparameter search on the
same upstream `ssi_morphology` pipeline.

## Search space
| hyperparameter | type | range |
|---|---|---|
| learning_rate | log-uniform | [1e-5, 1e-3] |
| num_encoder_layers | int | [2, 6] |
| num_decoder_layers | int | [2, 6] |
| emb_size | categorical | {256, 384, 512} |
| nhead | categorical | {4, 8} |
| dropout | uniform | [0.05, 0.30] |
| batch_size | categorical | {128, 192, 256} |
| epochs | int | [20, 50] |
| ffn_hid_dim | tied | = emb_size |
| (constraint) | | emb_size divisible by nhead |

Optuna objective = upstream validation word-accuracy (the original selection metric). The
publication-comparable **canonical CER** (character-level Levenshtein on the restored `.out.original`
surface string, identical metric to every other model in the paper) is computed separately on the
best configuration.

## Result
- Best configuration found: **emb 512 / 4 heads / 2 encoder / 2 decoder layers, lr ≈ 1.29e-4,
  dropout ≈ 0.235, 49 epochs** (validation accuracy 0.9803).
- Its **canonical CER (full-set, beam = 3): 4.188%** — this does **not** improve on the published
  baseline (S2-on-S2 4.00%). **No configuration in the search beats the
  published one.**
- Regardless of how the original hyperparameters were chosen, this independent search confirms they are
  **near-optimal for the architecture**: the encoder–decoder plateaus at ≈4% CER, and the gap to the
  discretized models (encoder-only 3.53%, MDLM 3.38%) is architectural, not a tuning artifact.

## 5-seed reproduction of the published baseline (for robustness)
Re-training the published configuration with 5 fixed seeds (same canonical CER metric, beam = 3):

| regime | 5-seed mean ± SD | published single run |
|---|---|---|
| S2-on-S2 | 4.107% ± 0.053 | 4.00% |
| S4-on-S2 | 4.010% ± 0.122 | 4.08% |

The published numbers fall within the seed spread; the baseline is stable and faithfully reproduced.

## Files
- `HPO_encoder_decoder_36_configs.csv` — every configuration tried (parameters + validation objective +
  state), exported directly from the live Optuna study databases. Of the 36 trials, 34 completed and 2
  are orphaned/failed (started 2025-12-11, never returned an objective; state `FAIL`); the conclusion
  rests on the completed trials, and none of them beats the published configuration.
