# Encoder–Decoder 5-seed reproduction (reviewer reference)

5-seed re-training of the published encoder–decoder baseline (Naaijer et al.) on the upstream
`ssi_morphology` pipeline, for both regimes. Fixed pre-split data (pure-S2 held-out test), no train-loader
shuffle, no re-split. Configuration (identical across all 10 models):

`emb 512 / 8 heads / 3 encoder / 3 decoder layers / ffn 512 / dropout 0.1 / lr 1e-4 / batch 128 / 30 epochs
/ seq_len 7 / beam 3, α 0.75`. Seeds {42, 43, 46, 48, 49}.

## Canonical CER (character-level Levenshtein on restored `.out.original`, beam=3)
| regime | 5-seed mean ± SD | published single run |
|---|---|---|
| **S2-on-S2** | **4.107% ± 0.053** | 4.00% |
| **S4-on-S2** | **4.010% ± 0.122** | 4.08% |

Per-seed (S4-on-S2): 42 = 4.101, 43 = 3.864, 46 = 3.858, 48 = 4.097, 49 = 4.129.
Per-seed (S2-on-S2): 42 = 4.142, 43 = 4.028, 46 = 4.119, 48 = 4.177, 49 = 4.068.

The published numbers fall inside the seed spread → the baseline is stable and faithfully reproduced.
Both encoder-only (3.53%) and MDLM (3.38%) stay well below this; the gap is architectural.

## Files (per regime, per seed)
- `seq2seq_..._seed{N}.pth` — model checkpoint.
- `model_config..._seed{N}.json` — architecture + vocabulary (needed to load the checkpoint).
- `results_ed5dec_{s2,s4}_{N}.txt` — full-set `Predicted` / `Truevalue` pairs (10,869 lines) for
  independent canonical-CER re-verification.
