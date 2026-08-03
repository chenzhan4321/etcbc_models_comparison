#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Three-model FINAL-STRING error comparison for the ETCBC morphological-restoration
task (S4-on-S2 split), serving the rewrite of the §4.6 error-analysis subsection.

Why string-level (and not label-level):
  * encoder-only and MDLM emit a per-character DISCRETE label sequence;
  * encoder-decoder (seq2seq) emits a CHARACTER sequence.
  These do not live in the same label space, so the only common ground on which
  all THREE models can be compared is the restored final string `.out.original`
  (the consonant+boundary-marker string the philologist actually reads).
  We therefore avoid the position-level "fix/break" framing (valid only for the
  two label-space models that share an identical input grid) entirely.

Models (all S4-on-S2, restored .out.original, 10869 lines):
  * MDLM (T=3)     : per-seed restored string; primary = seed42; family = 5 seeds.
  * encoder-only   : per-seed restored string; primary = seed42; family = 5 seeds.
  * encoder-decoder: single model (s7, beam=3), restored string.

Because the encoder-decoder is single-seed, the PRIMARY 3-way comparison uses one
representative model per family (seed42 for the two seeded families, whose CER is
within 0.03 pt of the family mean), so all three are single models compared on the
SAME 10869 lines. Family-mean CER is reported alongside for context.

Outputs a JSON digest next to this script; the markdown report is written by hand
from these numbers.
"""

import os
import json
import statistics as st
from collections import Counter, defaultdict

import Levenshtein as Lev

REPO = "<REPO_ROOT>"
RAW = os.path.join(REPO, "data", "raw_s4_on_s2")
OUT = os.path.dirname(os.path.abspath(__file__))

MDLM_DIR = os.path.join(
    REPO, "outputs",
    "mdlm_better_result_after_submission_d768_l10_h6_dr0.23_lr5e-05_steps3_s4-on-s2_5seeds")
MDLM_SEEDS = {
    "42": "mdlm_train_20260209_213751", "43": "mdlm_train_20260209_213754",
    "46": "mdlm_train_20260209_213822", "48": "mdlm_train_20260209_213839",
    "49": "mdlm_train_20260209_213845"}
ENC_DIR = os.path.join(
    REPO, "outputs",
    "encoder_only_100ep_bs128_lr3e-4_d512_l4_h16_d0.25_5seeds", "s4_on_s2")
ENC_SEEDS = {
    "42": "transformer_train_20260127_233759_seed42",
    "49": "transformer_train_20260127_235126_seed49",
    "50": "transformer_train_20260127_235126_seed50",
    "51": "transformer_train_20260127_235146_seed51",
    "52": "transformer_train_20260127_235149_seed52"}
ED_PRED = os.path.join(
    REPO, "outputs",
    "encoder_decoder_30ep_bs128_lr1e-4_emb512_h8_d0.1_b3_s7",
    "s4_on_s2", "results", "test.out.original")


# ----------------------------------------------------------------------------
def rd(p):
    with open(p, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f]


def _find(d, pfx, suf):
    for f in os.listdir(d):
        if f.startswith(pfx) and f.endswith(suf):
            return os.path.join(d, f)
    raise FileNotFoundError(f"{pfx}*{suf} in {d}")


def mdlm_path(seed):
    return _find(os.path.join(MDLM_DIR, MDLM_SEEDS[seed], "results",
                              "restore_and_levenshtein"),
                 "mdlm_predictions_", ".out.original")


def enc_path(seed):
    return _find(os.path.join(ENC_DIR, ENC_SEEDS[seed], "results",
                              "restore_and_levenshtein"),
                 "transformer_predictions_", ".out.original")


GT = rd(os.path.join(RAW, "test.out.original"))
N = len(GT)
assert N == 10869, N


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def line_lev(pred):
    return [Lev.distance(p, g) for p, g in zip(pred, GT)]


def cer_micro(pred):
    tot = sum(len(g) for g in GT)
    err = sum(line_lev(pred))
    return 100.0 * err / tot, err, tot


def exact_line_acc(pred):
    return sum(1 for p, g in zip(pred, GT) if p == g) / N


def words(s):
    # split on whitespace; the boundary markers are attached INSIDE tokens
    return s.split()


def word_sets():
    """Per-line list of GT words; total GT word count (positional)."""
    return [words(g) for g in GT]


GT_WORDS = word_sets()
GT_WORD_TOTAL = sum(len(w) for w in GT_WORDS)


def exact_word_acc(pred):
    """Positional word accuracy: token i of pred line vs token i of GT line.

    Only count positions that exist in BOTH (length-aligned prefix); extra/missing
    tokens count as wrong.  This mirrors how a philologist would score word slots.
    """
    corr = 0
    tot = 0
    for p, gw in zip(pred, GT_WORDS):
        pw = words(p)
        tot += len(gw)
        for i in range(len(gw)):
            if i < len(pw) and pw[i] == gw[i]:
                corr += 1
    return corr / tot, corr, tot


def line_error_set(pred):
    return set(i for i in range(N) if pred[i] != GT[i])


def word_error_set(pred):
    """Set of (line, word_index) where the positional GT word is not reproduced."""
    s = set()
    for li, (p, gw) in enumerate(zip(pred, GT_WORDS)):
        pw = words(p)
        for i in range(len(gw)):
            if i >= len(pw) or pw[i] != gw[i]:
                s.add((li, i))
    return s


def editop_breakdown(pred):
    """Aggregate insert/delete/replace counts over all lines (pred relative to GT).

    Levenshtein.editops(a, b) transforms a->b; we use a=GT, b=pred so the ops are
    described as turning the truth INTO the prediction:
       insert  = a char present in pred but not GT  (model ADDED a char)
       delete  = a char present in GT  but not pred (model DROPPED a char)
       replace = substitution.
    """
    ins = dele = rep = 0
    for p, g in zip(pred, GT):
        for op, _, _ in Lev.editops(g, p):
            if op == "insert":
                ins += 1
            elif op == "delete":
                dele += 1
            else:
                rep += 1
    tot = ins + dele + rep
    return {"insert": ins, "delete": dele, "replace": rep, "total": tot,
            "insert_pct": 100*ins/tot, "delete_pct": 100*dele/tot,
            "replace_pct": 100*rep/tot}


def length_collapse_stats(pred):
    """Generation-specific pathologies: length mismatch vs GT.

    len_short = pred shorter than GT (dropped material), len_long = pred longer.
    Also count lines with >50% length deviation (catastrophic length collapse).
    """
    short = long = same = 0
    big_dev = 0
    abs_len_diff = 0
    for p, g in zip(pred, GT):
        d = len(p) - len(g)
        abs_len_diff += abs(d)
        if d < 0:
            short += 1
        elif d > 0:
            long += 1
        else:
            same += 1
        if len(g) > 0 and abs(d) / len(g) > 0.5:
            big_dev += 1
    return {"lines_pred_shorter": short, "lines_pred_longer": long,
            "lines_same_length": same, "lines_len_dev_gt_50pct": big_dev,
            "mean_abs_len_diff_chars": abs_len_diff / N}


# ----------------------------------------------------------------------------
# seen / unseen INPUT word forms (using train.in surface forms, like cluster_A)
# ----------------------------------------------------------------------------
def load_train_in_forms():
    s = set()
    with open(os.path.join(RAW, "train.in"), encoding="utf-8") as f:
        for line in f:
            s.update(line.split())
    return s


def load_test_in():
    return rd(os.path.join(RAW, "test.in"))


def seen_unseen_word_acc(pred):
    """Word-slot accuracy split by whether the INPUT surface token was seen in
    train.in.  We align test.in tokens (which the GT word slots correspond to) to
    GT word slots and to pred word slots positionally.

    test.in tokens are the de-marked input; GT .out.original tokens are the marked
    output. They share token COUNT per line (boundary markers are intra-token, not
    new tokens). We verify count equality and skip the rare line where it fails.
    """
    train_in = load_train_in_forms()
    test_in = load_test_in()
    seen_tot = seen_corr = uns_tot = uns_corr = 0
    skipped = 0
    for li in range(N):
        in_toks = test_in[li].split()
        gw = GT_WORDS[li]
        pw = words(pred[li])
        if len(in_toks) != len(gw):
            skipped += 1
            continue
        for i in range(len(gw)):
            is_seen = in_toks[i] in train_in
            ok = (i < len(pw) and pw[i] == gw[i])
            if is_seen:
                seen_tot += 1
                seen_corr += ok
            else:
                uns_tot += 1
                uns_corr += ok
    return {"seen_word_acc": seen_corr / seen_tot if seen_tot else None,
            "unseen_word_acc": uns_corr / uns_tot if uns_tot else None,
            "seen_total": seen_tot, "unseen_total": uns_tot,
            "lines_skipped_token_mismatch": skipped}


# ----------------------------------------------------------------------------
# length buckets for error rate
# ----------------------------------------------------------------------------
def error_by_line_length(pred, edges=(0, 20, 40, 60, 80, 120, 10**9)):
    buckets = defaultdict(lambda: {"n": 0, "err_chars": 0, "gt_chars": 0,
                                   "wrong_lines": 0})
    for p, g in zip(pred, GT):
        L = len(g)
        for k in range(len(edges) - 1):
            if edges[k] <= L < edges[k + 1]:
                b = buckets[k]
                b["n"] += 1
                b["err_chars"] += Lev.distance(p, g)
                b["gt_chars"] += L
                b["wrong_lines"] += (p != g)
                break
    out = []
    for k in sorted(buckets):
        b = buckets[k]
        out.append({
            "range": f"[{edges[k]},{edges[k+1] if edges[k+1] < 10**9 else 'inf'})",
            "n_lines": b["n"],
            "cer_pct": 100 * b["err_chars"] / b["gt_chars"] if b["gt_chars"] else 0,
            "line_error_rate_pct": 100 * b["wrong_lines"] / b["n"] if b["n"] else 0})
    return out


# ----------------------------------------------------------------------------
def main():
    preds = {
        "MDLM": rd(mdlm_path("42")),
        "ENC": rd(enc_path("42")),
        "ED": rd(ED_PRED),
    }
    for k, v in preds.items():
        assert len(v) == N, (k, len(v))

    res = {"n_lines": N, "gt_total_chars": sum(len(g) for g in GT),
           "gt_total_words": GT_WORD_TOTAL}

    # ---- 1. overall metrics (primary single-model + family mean) -----------
    overall = {}
    for k, p in preds.items():
        c, err, tot = cer_micro(p)
        ewa, wc, wt = exact_word_acc(p)
        overall[k] = {"cer_pct": c, "err_chars": err, "exact_line_acc": exact_line_acc(p),
                      "exact_word_acc": ewa, "word_total": wt,
                      "editops": editop_breakdown(p),
                      "length": length_collapse_stats(p)}
    # family means
    fam = {}
    mdlm_cers = [cer_micro(rd(mdlm_path(s)))[0] for s in MDLM_SEEDS]
    enc_cers = [cer_micro(rd(enc_path(s)))[0] for s in ENC_SEEDS]
    fam["MDLM_family_cer_mean"] = st.mean(mdlm_cers)
    fam["MDLM_family_cer_std"] = st.pstdev(mdlm_cers)
    fam["ENC_family_cer_mean"] = st.mean(enc_cers)
    fam["ENC_family_cer_std"] = st.pstdev(enc_cers)
    fam["ED_single_cer"] = overall["ED"]["cer_pct"]
    res["overall_primary_seed42"] = overall
    res["family_cer"] = fam

    # ---- 2. error overlap (line + word) ------------------------------------
    le = {k: line_error_set(p) for k, p in preds.items()}
    we = {k: word_error_set(p) for k, p in preds.items()}

    def overlap(sets):
        M, E, D = sets["MDLM"], sets["ENC"], sets["ED"]
        return {
            "n_err_MDLM": len(M), "n_err_ENC": len(E), "n_err_ED": len(D),
            "all_three": len(M & E & D),
            "MDLM_and_ENC": len(M & E), "MDLM_and_ED": len(M & D),
            "ENC_and_ED": len(E & D),
            "MDLM_only": len(M - E - D), "ENC_only": len(E - M - D),
            "ED_only": len(D - M - E),
            "union": len(M | E | D)}
    res["line_error_overlap"] = overlap(le)
    res["word_error_overlap"] = overlap(we)

    # ---- 3. seen / unseen --------------------------------------------------
    res["seen_unseen"] = {k: seen_unseen_word_acc(p) for k, p in preds.items()}

    # ---- 4. error by line length -------------------------------------------
    res["error_by_length"] = {k: error_by_line_length(p) for k, p in preds.items()}

    # ---- 5. word length of errored words (defs. difficulty) ----------------
    # mean GT word length among word-error slots, per model
    wl = {}
    for k in preds:
        lens = []
        for (li, i) in we[k]:
            if i < len(GT_WORDS[li]):
                lens.append(len(GT_WORDS[li][i]))
        wl[k] = {"mean_errored_word_len": st.mean(lens) if lens else None,
                 "median_errored_word_len": st.median(lens) if lens else None,
                 "n_errored_words": len(lens)}
    # baseline mean word length for reference
    all_wl = [len(w) for line in GT_WORDS for w in line]
    wl["_corpus_mean_word_len"] = st.mean(all_wl)
    res["errored_word_length"] = wl

    with open(os.path.join(OUT, "three_model_string_error.json"), "w",
              encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    # ---- console digest ----
    print("=== OVERALL (primary = seed42 single models; ED = s7 single) ===")
    print(f"{'model':6s} {'CER%':>7s} {'exLine%':>8s} {'exWord%':>8s}"
          f" {'ins%':>6s} {'del%':>6s} {'rep%':>6s}")
    for k in ["MDLM", "ENC", "ED"]:
        o = overall[k]; e = o["editops"]
        print(f"{k:6s} {o['cer_pct']:7.3f} {100*o['exact_line_acc']:8.2f}"
              f" {100*o['exact_word_acc']:8.2f}"
              f" {e['insert_pct']:6.1f} {e['delete_pct']:6.1f} {e['replace_pct']:6.1f}")
    print(f"\nfamily CER: MDLM {fam['MDLM_family_cer_mean']:.3f}±{fam['MDLM_family_cer_std']:.3f}"
          f"  ENC {fam['ENC_family_cer_mean']:.3f}±{fam['ENC_family_cer_std']:.3f}"
          f"  ED {fam['ED_single_cer']:.3f}")

    print("\n=== LINE-ERROR OVERLAP ===")
    o = res["line_error_overlap"]
    for k in ["n_err_MDLM", "n_err_ENC", "n_err_ED", "all_three",
              "MDLM_and_ENC", "MDLM_and_ED", "ENC_and_ED",
              "MDLM_only", "ENC_only", "ED_only", "union"]:
        print(f"  {k:14s} {o[k]}")

    print("\n=== WORD-ERROR OVERLAP ===")
    o = res["word_error_overlap"]
    for k in ["n_err_MDLM", "n_err_ENC", "n_err_ED", "all_three",
              "MDLM_and_ENC", "MDLM_and_ED", "ENC_and_ED",
              "MDLM_only", "ENC_only", "ED_only", "union"]:
        print(f"  {k:14s} {o[k]}")

    print("\n=== SEEN / UNSEEN word acc ===")
    for k in ["MDLM", "ENC", "ED"]:
        s = res["seen_unseen"][k]
        print(f"  {k:6s} seen={100*s['seen_word_acc']:.2f}% unseen={100*s['unseen_word_acc']:.2f}%"
              f"  (gap={100*(s['seen_word_acc']-s['unseen_word_acc']):.2f}pt)"
              f"  seen_n={s['seen_total']} unseen_n={s['unseen_total']} skip={s['lines_skipped_token_mismatch']}")

    print("\n=== LENGTH STATS (generation pathologies) ===")
    for k in ["MDLM", "ENC", "ED"]:
        l = overall[k]["length"]
        print(f"  {k:6s} shorter={l['lines_pred_shorter']} longer={l['lines_pred_longer']}"
              f" same={l['lines_same_length']} >50%dev={l['lines_len_dev_gt_50pct']}"
              f" mean|dlen|={l['mean_abs_len_diff_chars']:.3f}")

    print("\n=== CER by line length ===")
    for k in ["MDLM", "ENC", "ED"]:
        print(f"  {k}:")
        for b in res["error_by_length"][k]:
            print(f"    {b['range']:12s} n={b['n_lines']:5d}"
                  f" CER={b['cer_pct']:6.3f}%  lineErr={b['line_error_rate_pct']:6.2f}%")

    print("\n=== errored-word length ===")
    print(f"  corpus mean word len = {wl['_corpus_mean_word_len']:.2f}")
    for k in ["MDLM", "ENC", "ED"]:
        print(f"  {k:6s} mean errored-word len={wl[k]['mean_errored_word_len']:.2f}"
              f"  n={wl[k]['n_errored_words']}")


if __name__ == "__main__":
    main()
