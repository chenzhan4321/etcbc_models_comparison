# Results Archive — index and paper cross-reference

This directory holds the trained-model outputs and result reports for the paper.
Two layers live here:

- **`models_1_revision/`** — the revision's **canonical 5-seed archive**: one folder
  per model, holding the test-set **predictions** (`.out` / `.out.original`) and the
  result JSONs from which every table in the paper is recomputed. Large model
  weights (`*.pt` / `*.pth`) are intentionally excluded; the numbers are reproducible
  from the predictions alone (see `../analysis/`).
- **per-configuration report directories** (the long, self-documenting names such as
  `encoder_only_100ep_bs128_lr3e-4_d512_l4_h16_d0.25_5seeds/`) — the original
  per-run output trees referenced in the **Supplementary Methods §S2**.

## Naming reconciliation (Supplementary Methods ⇄ repository)

The Supplementary Methods describe an idealised tree; the repository's actual paths
are listed here so a reviewer following the Supplementary can locate everything.

| Supplementary Methods wording | In this repository |
|---|---|
| the `report/` results tree (§S2.1) | this `outputs/` directory |
| `data_post_processing/` (§S4.4, §S7) | `../data_processing_tools/` |
| `report/mdlm_250ep_d768_l10_h6_dr0.23_lr5e-05_steps3_5seeds/` | `mdlm_better_result_after_submission_d768_l10_h6_dr0.23_lr5e-05_steps3_s4-on-s2_5seeds/` |

## Where each reported result lives

| Paper item | Path under `outputs/` |
|---|---|
| **Encoder–decoder** baseline (single published run, §S2.1) | `encoder_decoder_30ep_bs128_lr1e-4_emb512_h8_d0.1_b3_s7/` |
| Encoder–decoder, 5-seed reproduction | `models_1_revision/encoder_decoder_5seed/` |
| **Encoder-only** (5 seeds) | `encoder_only_100ep_bs128_lr3e-4_d512_l4_h16_d0.25_5seeds/` and `models_1_revision/encoder_only/` |
| **MDLM, headline** (steps = 3, 5 seeds) | `mdlm_better_result_after_submission_d768_l10_h6_dr0.23_lr5e-05_steps3_s4-on-s2_5seeds/` and `models_1_revision/mdlm/` |
| MDLM, steps = 2 (5 seeds) | `mdlm_200ep_bs16_lr1e-4_d768_l5_h4_d0.25_steps2_5seeds/` |
| **BiLSTM-CRF / Encoder+CRF / BERT** baselines | `models_1_revision/bilstm_crf/`, `models_1_revision/encoder_crf/`, `models_1_revision/bert/` |
| **Encoder–decoder HPO** (36 configurations) | `models_1_revision/hpo_encoder_decoder/` — see below |
| Constrained-decoding runs (R2-M2) | `models_1_revision/constrained_decoding/` |
| Diffusion-step sweep (§S4.8) | `models_1_revision/tsweep_step_ablation_100ep/` |
| **Under-trained case study** (T = 2, §S5) | `case_study_mdlm_50ep_bs16_lr3e-4_d384_l6_h4_d0.25_steps2_s52/` |

`models_1_revision/MANIFEST_models.txt` maps each model × dataset × seed back to its
original source run.

## The encoder–decoder HPO report (36 configurations)

`models_1_revision/hpo_encoder_decoder/` contains:

- `HPO_encoder_decoder_36_configs.csv` — every configuration the independent Optuna
  search tried (hyperparameters + validation objective + trial state), exported
  directly from the Optuna study databases. Of the 36 sampled trials, 34 completed
  and 2 failed; **no completed configuration beats the published baseline** (best
  canonical CER 4.188 %, beam = 3).
- `README.md` — the search space, the result, and the 5-seed reproduction of the
  published configuration (S2-on-S2 4.107 % ± 0.053 / S4-on-S2 4.010 % ± 0.122).

This is the evidence behind the manuscript's claim that the published encoder–decoder
hyperparameters are near-optimal for the architecture (response to R2-M2).
