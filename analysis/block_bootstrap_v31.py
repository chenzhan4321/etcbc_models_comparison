#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Segment-level (block) bootstrap robustness check (reviewer v3.1, Question 3).

The 10,869 S4-on-S2 test lines are 7-word stride-1 sliding windows. Window
continuity (line i's words 2..7 == line i+1's words 1..6) breaks exactly where
the underlying contiguous text span ends: the test set decomposes into 120
contiguous, non-overlapping text segments (mean ~90.6 windows). All
within-segment dependence -- including windows sharing a verse -- is contained
inside a segment; segments share no text. Resampling whole segments with
replacement is therefore a block bootstrap at a granularity COARSER than
verse-level (a segment spans several verses), i.e. more conservative than the
verse-level check the reviewer asks for.

For the three models (MDLM steps=3 / encoder-only / encoder--decoder beam=3,
all S4-on-S2, 5 seeds each, restored .out.original), we recompute:
  (a) each model's 5-seed CER with a 95% percentile CI, line-level (published
      protocol: 2000 reps, RNG 20260606) vs segment-level (same reps/RNG);
  (b) the MDLM - encoder-only paired difference CI, line-level vs segment-level.

Statistic: micro CER = sum(d_i)/sum(g_i) * 100 with d_i = per-line char
Levenshtein averaged over the 5 seeds, g_i = GT length (same as ed_bootstrap.py
and analysis/recompute_baselines_5seed.py).

Run: uv run --with levenshtein --with numpy python analysis/block_bootstrap_v31.py
"""

import os
import glob
import json

import numpy as np
import Levenshtein as Lev

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
RAW = os.path.join(REPO, "data", "raw_s4_on_s2")
OUT = os.path.dirname(os.path.abspath(__file__))
BOOT_REPS, BOOT_SEED = 2000, 20260606

MDLM_DIR = os.path.join(
    REPO, "outputs",
    "mdlm_better_result_after_submission_d768_l10_h6_dr0.23_lr5e-05_steps3_s4-on-s2_5seeds")
ENC_DIR = os.path.join(
    REPO, "outputs",
    "encoder_only_100ep_bs128_lr3e-4_d512_l4_h16_d0.25_5seeds", "s4_on_s2")
ED_DIR = os.path.join(
    REPO, "outputs", "models_1_revision", "encoder_decoder_5seed", "s4_on_s2")
ED_SEEDS = [42, 43, 46, 48, 49]


def rd(p):
    with open(p, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f]


GT = rd(os.path.join(RAW, "test.out.original"))
N = len(GT)
assert N == 10869, N
G = np.array([len(g.strip()) for g in GT], float)

# ---- segment reconstruction from window continuity of test.in ----
IN = [l.split() for l in rd(os.path.join(RAW, "test.in"))]
assert len(IN) == N
block_id = np.zeros(N, dtype=int)
bid = 0
for i in range(1, N):
    if not (len(IN[i]) == len(IN[i - 1]) and IN[i - 1][1:] == IN[i][:-1]):
        bid += 1
    block_id[i] = bid
NBLK = bid + 1
blocks = [np.where(block_id == b)[0] for b in range(NBLK)]
assert NBLK == 120, NBLK


# ---- load predictions ----
def mdlm_preds():
    runs = sorted(glob.glob(os.path.join(MDLM_DIR, "mdlm_train_*")))
    assert len(runs) == 5
    out = []
    for r in runs:
        f = glob.glob(os.path.join(
            r, "results", "restore_and_levenshtein", "mdlm_predictions_*.out.original"))
        assert len(f) == 1
        out.append(rd(f[0]))
    return out


def enc_preds():
    runs = sorted(glob.glob(os.path.join(ENC_DIR, "transformer_train_*_seed*")))
    assert len(runs) == 5
    out = []
    for r in runs:
        f = glob.glob(os.path.join(
            r, "results", "restore_and_levenshtein", "transformer_predictions_*.out.original"))
        assert len(f) == 1
        out.append(rd(f[0]))
    return out


def ed_preds():
    out = []
    for s in ED_SEEDS:
        p = os.path.join(ED_DIR, f"results_ed5dec_s4_{s}.txt")
        preds, trues = [], []
        with open(p, encoding="utf-8") as f:
            for ln in f:
                ln = ln.rstrip("\n")
                if ln.startswith("Predicted "):
                    preds.append(ln[len("Predicted "):])
                elif ln.startswith("Truevalue "):
                    trues.append(ln[len("Truevalue "):])
        assert len(preds) == N and len(trues) == N
        assert all(trues[i].strip() == GT[i].strip() for i in range(N)), s
        out.append(preds)
    return out


def dbar(pred_seeds):
    """Per-line char Levenshtein distance, averaged over the 5 seeds."""
    D = np.zeros((len(pred_seeds), N))
    for k, pred in enumerate(pred_seeds):
        assert len(pred) == N
        D[k] = [Lev.distance(pred[i].strip(), GT[i].strip()) for i in range(N)]
    return D.mean(0), D


def boot_ci(stat_fn):
    """2000-rep percentile bootstrap of stat_fn over LINE resampling."""
    rng = np.random.default_rng(BOOT_SEED)
    idx = np.arange(N)
    bs = np.empty(BOOT_REPS)
    for b in range(BOOT_REPS):
        sm = rng.choice(idx, N, replace=True)
        bs[b] = stat_fn(sm)
    return np.percentile(bs, [2.5, 97.5])


def boot_ci_block(stat_fn):
    """2000-rep percentile bootstrap resampling the 120 SEGMENTS with replacement."""
    rng = np.random.default_rng(BOOT_SEED)
    bs = np.empty(BOOT_REPS)
    for b in range(BOOT_REPS):
        chosen = rng.choice(NBLK, NBLK, replace=True)
        sm = np.concatenate([blocks[c] for c in chosen])
        bs[b] = stat_fn(sm)
    return np.percentile(bs, [2.5, 97.5])


def main():
    print(f"segments: {NBLK}, sizes mean {N/NBLK:.1f} "
          f"min {min(len(b) for b in blocks)} max {max(len(b) for b in blocks)}")
    res = {"n_lines": N, "n_segments": NBLK, "boot_reps": BOOT_REPS,
           "rng_seed": BOOT_SEED, "models": {}, "paired": {}}

    db = {}
    for name, loader in [("MDLM", mdlm_preds), ("ENC", enc_preds), ("ED", ed_preds)]:
        d, D = dbar(loader())
        db[name] = d
        cer = d.sum() / G.sum() * 100
        seed_cer = D.sum(1) / G.sum() * 100
        line_ci = boot_ci(lambda sm, d=d: d[sm].sum() / G[sm].sum() * 100)
        blk_ci = boot_ci_block(lambda sm, d=d: d[sm].sum() / G[sm].sum() * 100)
        res["models"][name] = {
            "cer_5seed_mean": cer,
            "seed_cers": seed_cer.tolist(),
            "across_seed_sd": float(seed_cer.std(ddof=1)),
            "ci95_line": line_ci.tolist(),
            "ci95_segment_block": blk_ci.tolist(),
        }
        print(f"{name}: CER={cer:.3f}%  line CI=[{line_ci[0]:.3f},{line_ci[1]:.3f}]  "
              f"segment CI=[{blk_ci[0]:.3f},{blk_ci[1]:.3f}]  seed SD={seed_cer.std(ddof=1):.3f}")

    # paired MDLM - ENC difference (micro CER difference on identical lines)
    diff = db["MDLM"] - db["ENC"]
    delta = diff.sum() / G.sum() * 100
    line_ci = boot_ci(lambda sm: diff[sm].sum() / G[sm].sum() * 100)
    blk_ci = boot_ci_block(lambda sm: diff[sm].sum() / G[sm].sum() * 100)
    res["paired"]["MDLM_minus_ENC"] = {
        "delta_pp": delta, "ci95_line": line_ci.tolist(),
        "ci95_segment_block": blk_ci.tolist(),
    }
    print(f"MDLM-ENC: delta={delta:.3f}pp  line CI=[{line_ci[0]:.3f},{line_ci[1]:.3f}]  "
          f"segment CI=[{blk_ci[0]:.3f},{blk_ci[1]:.3f}]")

    # paired ENC - ED (discretized family vs generative baseline, closest pair)
    diff2 = db["ENC"] - db["ED"]
    delta2 = diff2.sum() / G.sum() * 100
    line_ci2 = boot_ci(lambda sm: diff2[sm].sum() / G[sm].sum() * 100)
    blk_ci2 = boot_ci_block(lambda sm: diff2[sm].sum() / G[sm].sum() * 100)
    res["paired"]["ENC_minus_ED"] = {
        "delta_pp": delta2, "ci95_line": line_ci2.tolist(),
        "ci95_segment_block": blk_ci2.tolist(),
    }
    print(f"ENC-ED:  delta={delta2:.3f}pp  line CI=[{line_ci2[0]:.3f},{line_ci2[1]:.3f}]  "
          f"segment CI=[{blk_ci2[0]:.3f},{blk_ci2[1]:.3f}]")

    with open(os.path.join(OUT, "block_bootstrap_v31.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("written: analysis/block_bootstrap_v31.json")


if __name__ == "__main__":
    main()
