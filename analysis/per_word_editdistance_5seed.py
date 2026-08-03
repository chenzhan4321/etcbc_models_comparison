#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""5-seed version of per_word_editdistance.py (reviewer v3.1, Question 4).

For EACH of the five seeds of the three models (MDLM steps=3 / encoder-only /
encoder-decoder beam=3, all S4-on-S2, restored .out.original), compute the
per-error-word character Levenshtein statistics of tab:error_cost:

    # wrong words, mean edits, >=4-char tail %, max, unseen mean, seen mean

then aggregate mean +/- sample SD (ddof=1) across the five seeds per model.

Word-slot alignment logic is IDENTICAL to per_word_editdistance.py /
three_model_string_error.py: whitespace-split tokens, positional alignment,
missing pred slot -> "" (distance == len(gt word)).

Seed sets (authoritative_numbers.md): MDLM {42,43,46,48,49} (5 run dirs),
encoder-only {42,49,50,51,52}, encoder-decoder {42,43,46,48,49}.

Run: uv run --with levenshtein --with numpy python analysis/per_word_editdistance_5seed.py
"""

import os
import glob
import json
import statistics as st

import Levenshtein as Lev

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
RAW = os.path.join(REPO, "data", "raw_s4_on_s2")
OUT = os.path.dirname(os.path.abspath(__file__))

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
GT_WORDS = [g.split() for g in GT]
GT_WORD_TOTAL = sum(len(w) for w in GT_WORDS)
assert GT_WORD_TOTAL == 76083, GT_WORD_TOTAL

TEST_IN = rd(os.path.join(RAW, "test.in"))
TRAIN_IN_FORMS = set()
with open(os.path.join(RAW, "train.in"), encoding="utf-8") as f:
    for line in f:
        TRAIN_IN_FORMS.update(line.split())

SEEN_MASK = []
_seen_slots = _unseen_slots = 0
for li in range(N):
    _in = TEST_IN[li].split()
    _gw = GT_WORDS[li]
    assert len(_in) == len(_gw), (li, len(_in), len(_gw))
    row = [(t in TRAIN_IN_FORMS) for t in _in]
    SEEN_MASK.append(row)
    _seen_slots += sum(row)
    _unseen_slots += len(row) - sum(row)
assert _seen_slots == 67168, _seen_slots
assert _unseen_slots == 8915, _unseen_slots


# ---------------------------------------------------------------- loaders
def mdlm_runs():
    runs = sorted(glob.glob(os.path.join(MDLM_DIR, "mdlm_train_*")))
    assert len(runs) == 5, runs
    out = {}
    for r in runs:
        f = glob.glob(os.path.join(
            r, "results", "restore_and_levenshtein", "mdlm_predictions_*.out.original"))
        assert len(f) == 1, (r, f)
        out[os.path.basename(r)] = rd(f[0])
    return out


def enc_runs():
    runs = sorted(glob.glob(os.path.join(ENC_DIR, "transformer_train_*_seed*")))
    assert len(runs) == 5, runs
    out = {}
    for r in runs:
        f = glob.glob(os.path.join(
            r, "results", "restore_and_levenshtein", "transformer_predictions_*.out.original"))
        assert len(f) == 1, (r, f)
        out[os.path.basename(r)] = rd(f[0])
    return out


def ed_runs():
    out = {}
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
        assert len(preds) == N and len(trues) == N, (s, len(preds), len(trues))
        # the Truevalue stream must be the same GT, token-for-token
        for i in range(N):
            assert trues[i].split() == GT_WORDS[i], (s, i)
        out[f"seed{s}"] = preds
    return out


# ---------------------------------------------------------------- stats
def seed_stats(pred):
    dists, seen_d, unseen_d = [], [], []
    for li in range(N):
        gw = GT_WORDS[li]
        pw = pred[li].split()
        mask = SEEN_MASK[li]
        for i in range(len(gw)):
            g = gw[i]
            p = pw[i] if i < len(pw) else ""
            if p != g:
                d = Lev.distance(p, g)
                dists.append(d)
                (seen_d if mask[i] else unseen_d).append(d)
    n = len(dists)
    return {
        "n_error_words": n,
        "mean_edits": st.mean(dists),
        "ge4_pct": 100 * sum(1 for d in dists if d >= 4) / n,
        "max_edits": max(dists),
        "seen_mean": st.mean(seen_d),
        "unseen_mean": st.mean(unseen_d),
    }


def aggregate(per_seed):
    keys = ["n_error_words", "mean_edits", "ge4_pct", "max_edits",
            "seen_mean", "unseen_mean"]
    agg = {}
    for k in keys:
        vals = [d[k] for d in per_seed.values()]
        agg[k] = {"mean": st.mean(vals), "sd": st.stdev(vals),
                  "min": min(vals), "max": max(vals)}
    return agg


def main():
    res = {"n_lines": N, "gt_total_words": GT_WORD_TOTAL, "models": {}}
    for name, runs in [("MDLM", mdlm_runs()), ("ENC", enc_runs()), ("ED", ed_runs())]:
        per_seed = {}
        for run_id, pred in runs.items():
            assert len(pred) == N, (name, run_id, len(pred))
            per_seed[run_id] = seed_stats(pred)
        res["models"][name] = {"per_seed": per_seed, "aggregate": aggregate(per_seed)}
        print(f"\n=== {name} ===")
        for rid, d in per_seed.items():
            print(f"  {rid}: nErr={d['n_error_words']} mean={d['mean_edits']:.3f} "
                  f"ge4={d['ge4_pct']:.2f}% max={d['max_edits']} "
                  f"unseen={d['unseen_mean']:.3f} seen={d['seen_mean']:.3f}")
        a = res["models"][name]["aggregate"]
        print(f"  AGG  nErr={a['n_error_words']['mean']:.0f}+/-{a['n_error_words']['sd']:.0f} "
              f"mean={a['mean_edits']['mean']:.3f}+/-{a['mean_edits']['sd']:.3f} "
              f"ge4={a['ge4_pct']['mean']:.2f}+/-{a['ge4_pct']['sd']:.2f}% "
              f"max={a['max_edits']['mean']:.1f}+/-{a['max_edits']['sd']:.1f} "
              f"unseen={a['unseen_mean']['mean']:.3f}+/-{a['unseen_mean']['sd']:.3f} "
              f"seen={a['seen_mean']['mean']:.3f}+/-{a['seen_mean']['sd']:.3f}")

    with open(os.path.join(OUT, "per_word_editdistance_5seed.json"), "w",
              encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("\nwritten: analysis/per_word_editdistance_5seed.json")


if __name__ == "__main__":
    main()
