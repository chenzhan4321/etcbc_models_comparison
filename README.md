# A Structure-First Paradigm for Classical Syriac Morphological Parsing

> **Reviewer quick pointers (revision v3.1).** The two analyses added in this
> revision live here:
> - **Q3 (segment-level block bootstrap):** `analysis/block_bootstrap_v31.py`
>   → output `analysis/block_bootstrap_v31.json`
> - **Q4 (Table 3, five-seed mean ± SD):** `analysis/per_word_editdistance_5seed.py`
>   → output `analysis/per_word_editdistance_5seed.json`
>   (note: `analysis/per_word_editdistance.py` *without* the `_5seed` suffix is the
>   older single-seed version kept for provenance)
> - The frozen, training-only label inventory: `data/raw_s4_on_s2/patterns.csv`
> - Five-seed prediction archives: `outputs/` (see `outputs/models_1_revision/`)
>
> If directory browsing is temporarily unavailable on the anonymized mirror,
> the files above are reachable by direct path, and the full-repo ZIP download
> always works.

Reference implementation and reproduction package for the paper

> **A Structure-First Paradigm for Morphological Parsing: Synthesizing Discrete
> Representation and Diffusion Model.**

The work reformulates ETCBC-style Classical Syriac morphological *restoration*
from open-ended string generation into a **fixed-length classification** problem,
and shows that this *structure-first* reframing — not raw data volume — is what
lets a family of discretized models, including a **Masked Diffusion Language
Model (MDLM)**, clearly outperform the prior generative encoder–decoder baseline
on the identical target test set.

This repository contains the data-processing pipeline, the model code, the
five-seed prediction archive, and the statistical-analysis scripts needed to
reproduce every number in the paper.

---

## Headline result

On the fixed **S2 test set** (10,869 lines), training on the scaled
**S2+S3+S4** corpus and scoring the restored ETCBC surface string
(`.out.original`) with a character-level Levenshtein CER:

- **MDLM (3 diffusion steps): CER = 3.383 %** [95 % bootstrap CI 3.302, 3.467] —
  the lowest error among all architectures evaluated, a **≈16 % relative
  reduction** over the generative encoder–decoder baseline (4.010 %).

The MDLM's value, however, is **not** framed as an accuracy margin (it is a close
sibling of the encoder-only classifier). Its distinctive contribution is
**interpretability**: the multi-step denoising externalises a step-by-step,
inspectable morphological analysis. The decisive *accuracy* result is at the
framework level — **every discretized model beats the generative baseline**, and
that gap is what the discretization buys.

---

## Main comparison (paper Table, 5-seed mean CER %, 95 % bootstrap CI)

All models are scored on the **same** fixed S2 test set with **one identical
canonical metric** (per-line character Levenshtein on the restored ETCBC surface
string, micro-averaged over 10,869 lines, then over 5 seeds). The encoder–decoder
is the published generative baseline; every other system shares the matched
discrete register-pattern representation, so each comparison varies a single
factor.

| Model | Paradigm | train **S2** | train **S2+S3+S4** | Input violation |
|---|---|---|---|---|
| *Encoder–decoder* (generative baseline) | seq2seq | 4.107 [4.01, 4.20] | 4.010 [3.92, 4.10] | ~0.15 % |
| BiLSTM-CRF | recurrent + CRF | 3.957 [3.87, 4.05] | 3.688 [3.61, 3.77] | 0.00 % |
| BERT (from scratch) | token classifier | 4.227 [4.14, 4.31] | 3.709 [3.63, 3.79] | 0.00 % |
| Encoder+CRF | transformer + CRF | 3.926 [3.84, 4.01] | 3.534 [3.45, 3.62] | 0.00 % |
| Encoder-only | discriminative | 3.876 [3.79, 3.96] | 3.528 [3.45, 3.61] | 0.00 % |
| **MDLM (steps = 3)** | **iterative denoising** | **3.687 [3.60, 3.78]** | **3.383 [3.30, 3.47]** | **0.00 %** |

Notes:
- **Two training regimes, one test set.** *train S2* = the standard in-distribution
  setting (`s2_on_s2`); *train S2+S3+S4* = the scaled setting that adds the
  out-of-distribution expansion (`s4_on_s2`). Both evaluate on the identical fixed
  S2 test set, so the columns are directly comparable.
- Adding the heterogeneous S4 data helps **every** architecture on the target —
  marginally for the generative baseline (4.107→4.010, within seed noise),
  substantially for the discretized models.
- **Input violation** is the fraction of lines whose consonantal skeleton is
  corrupted. Discretized models predict labels on a fixed-length canvas and so
  *cannot* insert/delete characters (0.00 % by construction); the generative
  baseline regenerates the text and occasionally corrupts it.
- **Encoder+CRF (3.534) ≈ Encoder-only (3.528):** an explicit CRF label-dependency
  head reaches the encoder-only level but does not close the gap to the MDLM, so
  the MDLM's behaviour is not explained by generic label-dependency modelling.
- Across-seed SD is 0.03–0.14 pp; per-seed values, paired tests and the bootstrap
  protocol are in `analysis/cluster_F/` and the paper's Supplementary Methods.

---

## The structure-first pipeline

The ETCBC encoding linearises text *and* its morphological tagging into a single
symbol-rich string. The discretization pipeline maps this onto a fixed-length,
one-to-one canvas so that classification architectures (and the MDLM) become
applicable at all.

```
ETCBC .out.original surface string
   │  data_processing_tools/decompose/01_out_original_reducer.py   (syntactic reduction)
   ▼
.out.reduced
   │  02_generate_patterns_csv_from_reduced.py   (induce the register inventory → patterns.csv)
   │  03_convert_reduced_to_final.py             (atomic discretization: pattern → integer id)
   │  04_convert_final_to_out.py                 (per-position label sequence)
   ▼
fixed-length aligned (x_t, y_t) canvas  ──►  model  ──►  integer labels
   │  -01_restore_reduced.py  +  restore to .out.original
   ▼
restored ETCBC surface string  ──►  character-level Levenshtein CER
```

### The register inventory (`patterns.csv`)

`data/s2_on_s2/patterns.csv` and `data/s4_on_s2/patterns.csv` each have **330
rows** (plus a header):

- **label 0** = the empty / null pattern (no boundary marker);
- **labels 1–328** = the morphological register patterns **observed in the
  S2+S3+S4 training split**, ordered by descending frequency (`328` is the largest
  *observed* label index);
- **label 329** = a reserved `<UNKNOWN>` fallback (training count 0; never
  predicted by the classifier).

The inventory is **frequency-extracted from the corpus, not hand-designed**. The
mapping Φ is bijective only on the *observed* register types; any out-of-vocabulary
combination that surfaces only at test time maps to `<UNKNOWN>`, **outside** the
bijection. We deliberately do not pre-encode unseen patterns (that would leak
test-set information into the label space). The distribution is **steeply
long-tailed** — a small number of high-frequency patterns dominate, while the rare
combinations sit in the far tail and unseen ones fall back to `<UNKNOWN>`.

---

## Datasets

| Configuration | Train | Test | Train lines | Path |
|---|---|---|---|---|
| **s2_on_s2** (standard / in-distribution) | S2 | S2 (fixed) | 50,649 | `data/s2_on_s2/` |
| **s4_on_s2** (scaled / + OOD expansion) | S2+S3+S4 | S2 (fixed) | 97,048 | `data/s4_on_s2/` |

Both configurations use the **identical** validation (10,964) and **test
(10,869)** sets drawn from S2, so the two regimes are scored on the same target.
The `data/raw_s{2,4}_on_s2/` directories hold the restored-surface ground truth
(`test.out.original`) against which CER is computed.

> **Scope.** This study uses only the ETCBC stages **S2** and **S4** (= S2+S3+S4).
> The stage nomenclature is the ETCBC project's historical annotation milestones.

The underlying ETCBC Classical Syriac database is curated by the Eep Talstra
Centre for Bible and Computer and is available at
<https://github.com/ETCBC/ssi_morphology>.

---

## Models in this study

Six systems, all scored on the same metric:

1. **Encoder–decoder** (`models/seq2seq.py`, `train_seq2seq.py`) — the published
   generative seq2seq baseline. The published configuration reproduces the upstream
   single-run result (4.0032 %) bit-for-bit, and our five-seed reproduction gives
   4.107 % (train S2) / 4.010 % (train S2+S3+S4) — the values in the table above. An
   independent matched Optuna search (`hpo_seq2seq.py`, 36 configurations) finds no
   configuration that beats the published one (best 4.188 %, beam = 3), confirming
   the published hyperparameters are near-optimal. A **constrained-decoding**
   variant (`src/eval_seq2seq_constrained.py`) that hard-masks the decoder does *not*
   help — under matched greedy decoding CER rises 4.77 % → 8.70 % — because ETCBC
   restoration is non-monotonic (only 11.6 % of gold restorations are identical to
   the input, 26.0 % are pure-insertion supersequences, 62.5 % are neither).
2. **BiLSTM-CRF** — `train.py --model_type lstm --use_crf`.
3. **BERT (from scratch)** — `train.py --model_type bert` (no Syriac pretraining).
4. **Encoder+CRF** — `train.py --model_type transformer --use_crf`.
5. **Encoder-only** — `train.py --model_type transformer`.
6. **MDLM** (`models/mdlm.py`) — the masked discrete diffusion model.

**MDLM production configuration** (the headline 3.383 % run): `d_model = 768`,
`num_layers = 10`, `num_heads = 6`, `dropout = 0.23`, `lr = 5e-5`, `250` epochs,
**3 diffusion steps**; ≈ 55.8 M parameters, vocabulary 40, 329 label classes. The
MDLM uses a **freeze-on-commit, no-remasking** inference policy: at each of the
T steps it commits the most confident still-masked slots and never revises them,
so the benefit lies in the *order* of resolution. The number of diffusion steps is
**not an accuracy lever** (a step sweep is flat across T = 1…4 within seed noise);
T = 3 is an interpretability-motivated working point.

> Additional exploratory architectures (Mamba / BiMamba / RetNet / Switch /
> RWKV-7) remain in `models/cutting_edge.py` for completeness. **They are not part
> of this study** and are not reported in the paper.

---

## Evaluation metric (CER)

CER is the **per-line character-level Levenshtein distance** between the restored
ETCBC surface string (`.out.original`) and the ground truth, divided by the
ground-truth length, micro-averaged over the 10,869 test lines and then over five
seeds. It is computed on the surface string a philologist actually reads — **not**
on the raw label-digit sequence — so every counted edit is one keystroke-level
correction, making CER a literal proxy for restoration effort.

- **Bootstrap CI:** 2,000 percentile-bootstrap resamples of the test lines, fixed
  RNG seed `20260606`, percentiles [2.5, 97.5].
- **Across-seed SD** is reported separately (training stochasticity).
- The test lines are overlapping sliding windows over shared verses and are **not
  independent**, so the line-level intervals/p-values are anti-conservative and are
  treated as a **lower bound** on uncertainty.
- **No seed was excluded by any criterion** (no 2-sigma / outlier-rejection code
  exists anywhere in the repository); the reported 5-seed means equal the
  full, unfiltered means exactly.

Reproduce the canonical numbers:

```bash
# 5-seed means + 95% bootstrap CI + paired comparisons (reproduces the JSON bit-for-bit)
python analysis/cluster_F/bootstrap_ci.py

# three-model error overlap + per-wrong-word correction effort (paper §4.4 / Supp §S6)
python analysis/three_model_string_error.py
python analysis/per_word_editdistance.py
```

---

## Repository layout

```
.
├── README.md                       # this file
├── changelog.md                    # version history
├── requirements.txt                # Python dependencies
├── HPO_README.md                   # hyperparameter-search component guide
│
├── train.py                        # unified trainer (transformer / bert / lstm / mdlm; --use_crf)
├── train_hpo.py                    # Optuna HPO for the discretized models
├── train_seq2seq.py                # encoder–decoder baseline training
├── hpo_seq2seq.py                  # matched Optuna HPO for the encoder–decoder (R2-M2)
│                                   #   (the four trainers stay at root: the Supplementary
│                                   #    documents `python train.py ...` as the repro command)
│
├── src/                            # evaluation & utility scripts (run from the repo root,
│   │                               #  e.g. `python src/eval_cer.py`)
│   ├── eval_cer.py / batch_cer.py  # canonical CER for label-model predictions
│   ├── eval_seq2seq_cer.py         # canonical CER for seq2seq predictions
│   ├── beam_eval_seq2seq.py        # beam-search re-evaluation of seq2seq checkpoints
│   ├── eval_seq2seq_constrained.py # constrained-decoding negative result (R2-M2)
│   ├── dump_encdec_hpo_trials.py   # exports every enc–dec HPO trial → CSV (transparency)
│   ├── collect_results.py          # 5-seed aggregation → markdown tables
│   └── download_ssi_data.py        # fetch the upstream ETCBC SSI data
│
├── models/                         # model definitions
│   ├── seq2seq.py                  # encoder–decoder baseline
│   ├── transformer.py              # encoder-only (+ optional CRF head)
│   ├── bert.py                     # from-scratch BERT token classifier (+ optional CRF)
│   ├── lstm.py                     # BiLSTM (+ optional CRF head)
│   ├── mdlm.py                     # masked diffusion language model
│   └── ...                         # core/, components/, model_factory.py, cutting_edge.py
│
├── data/
│   ├── s2_on_s2/                   # standard regime  (+ patterns.csv, 330 rows)
│   ├── s4_on_s2/                   # scaled regime    (+ patterns.csv, 330 rows)
│   └── raw_s{2,4}_on_s2/           # restored-surface ground truth (.out.original)
│
├── data_processing_tools/
│   └── decompose/                  # the 4-stage discretization pipeline + inverse restore
│
├── analysis/                       # statistical analysis backing the paper numbers
│   ├── cluster_F/                  # CER + bootstrap CI + paired tests (R2-M5/M6) + SUMMARY.md
│   ├── cluster_A/                  # divergence / seen-unseen / position-distance buckets
│   ├── three_model_string_error.py # discrete-vs-generative error overlap (§4.4 / Supp §S6)
│   └── per_word_editdistance.py    # per-wrong-word correction effort
│
└── outputs/                        # results archive (see outputs/README.md for the
    │                               #  paper-cross-reference index)
    ├── models_1_revision/          # 5-seed prediction archive (one folder per model)
    │   ├── encoder_decoder_5seed/  bilstm_crf/  bert/  encoder_crf/  encoder_only/  mdlm/
    │   ├── hpo_encoder_decoder/    # matched HPO study (36 configs) + report
    │   ├── constrained_decoding/   # constrained-decoding runs
    │   ├── tsweep_step_ablation_100ep/  # diffusion-step sweep
    │   └── MANIFEST_models.txt     # seed → source-run provenance map
    └── <config-named dirs>/        # per-configuration result reports (Supplementary §S2)
```

> **Model weights** (`*.pt` / `*.pth`) are intentionally **not** committed — every
> table in the paper is reproducible directly from the on-disk **predictions**
> (`.out` / `.out.original`) and result JSONs under `outputs/models_1_revision/`.

---

## Quick start

```bash
# environment (uv recommended; pip works too)
uv pip install -r requirements.txt      # or: pip install -r requirements.txt

# train the MDLM production configuration (headline 3.383%)
python train.py --model_type mdlm --data_dir data/s4_on_s2 \
    --d_model 768 --num_layers 10 --num_heads 6 --dropout 0.23 \
    --learning_rate 5e-5 --num_epochs 250 --num_timesteps 3 --seed 42

# encoder-only / Encoder+CRF / BiLSTM-CRF / BERT
python train.py --model_type transformer --data_dir data/s4_on_s2 --seed 42
python train.py --model_type transformer --use_crf --data_dir data/s4_on_s2 --seed 42
python train.py --model_type lstm        --use_crf --data_dir data/s4_on_s2 --seed 42
python train.py --model_type bert                  --data_dir data/s4_on_s2 --seed 42

# encoder–decoder generative baseline
python train_seq2seq.py --data_dir data/s2_seq2seq --seed 42
```

---

## Reproducibility

- **Environment:** Python 3.10, PyTorch 2.x, CUDA 11.8; runs reported on an NVIDIA
  RTX 4090D (24 GB). CRF baselines additionally require `pytorch-crf`.
- **Total compute** for the full revision (all 6 models × 5 seeds × 2 regimes, the
  matched encoder–decoder HPO, the diffusion-step sweep, and the constrained-decoding
  runs): on the order of **700 GPU-hours**.
- Every metric is regenerated from the committed predictions by the `analysis/`
  scripts; `analysis/cluster_F/SUMMARY.md` documents the bit-for-bit reproduction.

## Data & code availability

- **Data:** ETCBC Classical Syriac database — <https://github.com/ETCBC/ssi_morphology>.
- **Code (anonymous review mirror):** <https://anonymous.4open.science/r/mdlm>.
- Stable links to permanent public repositories will be provided upon publication.

## Acknowledgements

Syriac morphological data from the Eep Talstra Centre for Bible and Computer
(ETCBC). We thank Constantijn Sikkel for valuable insights on the ETCBC encoding.
