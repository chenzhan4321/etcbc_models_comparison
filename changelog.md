# Changelog

All notable changes to this repository. Format: `[date] version: summary`.

## [2026-06-11] 13.0.0: Sync repository to the accepted-revision paper

Aligned the public, reviewer-facing repository with the final revised manuscript
and response letter. **No experimental numbers were changed** — only the
documentation was brought into line with the canonical results already on disk.

- **README rewritten to the paper's口径.**
  - Headline is now **MDLM (3 diffusion steps) = 3.383 %** CER on S4-on-S2
    (was the older 3.42 %, 2-step result).
  - Replaced the 3-model table with the full **6-model, 5-seed** comparison
    (added BiLSTM-CRF, Encoder+CRF, BERT-from-scratch) with 95 % bootstrap CIs,
    matching the manuscript's main results table exactly.
  - Fixed the swapped dataset-regime column labels (`s2_on_s2` = standard /
    in-distribution; `s4_on_s2` = scaled / +OOD; both test on the same fixed S2
    set).
  - Corrected the label-space description: **330 rows** = label 0 (null) +
    labels 1–328 (observed registers, frequency-extracted) + label 329
    (`<UNKNOWN>` reserved fallback); previous "309 / ~350 classes" wording removed.
  - Corrected the MDLM production configuration to the headline run
    (`d768 / L10 / H6 / dropout 0.23 / lr 5e-5 / 250 ep / steps 3`, ≈ 55.8 M
    params); the old `L5 / H4 / steps 2 / 200 ep` block was the superseded config.
  - Reframed the contribution as *structure-first* (discretization is the decisive
    accuracy step; the MDLM's value is **interpretability**, not a CER margin),
    consistent with the calibrated claims in the revision.
  - README now focuses on the **6 study models**; the exploratory architectures
    (Mamba / RWKV-7 / RetNet / Switch / BiMamba) are retained in the code but
    explicitly noted as *not part of this study*.
  - Documented the metric (CER on `.out.original`), the bootstrap protocol
    (2,000 resamples, RNG `20260606`), the sliding-window non-independence caveat,
    and the no-seed-exclusion disclosure.

- **Reproduction artifacts now tracked.**
  - Added `analysis/` (cluster_F bootstrap/paired-test scripts + `SUMMARY.md`,
    cluster_A divergence/seen-unseen/position analyses, the three-model error and
    per-wrong-word correction-effort scripts) — the code that regenerates every
    paper number.
  - Added the encoder–decoder baseline + fairness suite: `models/seq2seq.py`,
    `train_seq2seq.py`, `hpo_seq2seq.py` (matched Optuna HPO, R2-M2),
    `dump_encdec_hpo_trials.py` (HPO transparency), `beam_eval_seq2seq.py`,
    `eval_seq2seq_cer.py`, and `eval_seq2seq_constrained.py` (the constrained-
    decoding negative result).
  - Added the lightweight 5-seed prediction archive under
    `outputs/models_1_revision/` (predictions + result JSONs + HPO records);
    large model weights (`*.pt` / `*.pth`) remain untracked — every table is
    reproducible from the predictions alone.

- **CRF baselines wired in.** `train.py` gains `--use_crf`; `models/{transformer,
  bert,lstm}.py` gain an optional CRF head. Added `pytorch-crf` to
  `requirements.txt`.

- **Out-of-scope material kept out of the public repo.** Local-only exploratory
  scripts, scratch experiment directories, re-downloadable raw data, and HPC job
  launchers that are not part of this study are excluded from version control. The
  study's scope is strictly the ETCBC S2 / S4 stages.

## [2026-02] 12.x: MDLM redesign + 5-seed reproducibility

- Redesigned the MDLM as a discrete masked-diffusion model with a freeze-on-commit
  (no-remasking) inference policy; added five-seed experiment outputs and the
  statistical-analysis pipeline.

## earlier: initial benchmark

- Encoder-only and encoder–decoder benchmarks on the ETCBC S2 / S4 Syriac corpus;
  data-processing (discretization) pipeline and multi-architecture trainer.
