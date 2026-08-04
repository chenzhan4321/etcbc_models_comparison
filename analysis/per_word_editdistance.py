#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NOTE: legacy SINGLE-SEED analysis, kept for provenance. The five-seed version
used for Table 3 of the revised paper (mean +/- SD over all seeds) is
analysis/per_word_editdistance_5seed.py.

Per-word CHARACTER-level edit-distance distribution among ERROR words, for the
three models (MDLM seed42 / encoder-only seed42 / encoder-decoder s7), all
S4-on-S2, restored .out.original.

Goal: support the claim "encoder-decoder makes FEWER wrong words, but once a word
is wrong it tends to be wrong by MANY characters (一错错一堆)" — i.e. the mean
per-error-word character Levenshtein distance is higher for the seq2seq model.

We REUSE the exact word-slot alignment from three_model_string_error.py:
  * a "word" is a whitespace-split token (boundary markers are intra-token);
  * word slot i on a line is aligned positionally to GT word slot i;
  * a slot is an ERROR if the pred token is missing (pred line has < i+1 tokens)
    or differs from the GT token.
For each ERROR slot we compute Lev.distance(pred_word, gt_word), where pred_word
is "" when the slot is missing in the prediction (so a dropped word counts as a
distance == len(gt_word), which is correct: the philologist must supply the whole
word). distance > 0 by construction for every error slot.

Outputs:
  * analysis/per_word_editdistance.json   (machine digest)
  * (markdown report written separately)

Run: uv run python analysis/per_word_editdistance.py [--sample]
"""

import os
import sys
import json
import statistics as st
from collections import Counter

import Levenshtein as Lev

REPO = "<REPO_ROOT>"
RAW = os.path.join(REPO, "data", "raw_s4_on_s2")
OUT = os.path.dirname(os.path.abspath(__file__))

MDLM_DIR = os.path.join(
    REPO, "outputs",
    "mdlm_better_result_after_submission_d768_l10_h6_dr0.23_lr5e-05_steps3_s4-on-s2_5seeds")
MDLM_SEEDS = {"42": "mdlm_train_20260209_213751"}
ENC_DIR = os.path.join(
    REPO, "outputs",
    "encoder_only_100ep_bs128_lr3e-4_d512_l4_h16_d0.25_5seeds", "s4_on_s2")
ENC_SEEDS = {"42": "transformer_train_20260127_233759_seed42"}
ED_PRED = os.path.join(
    REPO, "outputs",
    "encoder_decoder_30ep_bs128_lr1e-4_emb512_h8_d0.1_b3_s7",
    "s4_on_s2", "results", "test.out.original")


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


def words(s):
    return s.split()


GT = rd(os.path.join(RAW, "test.out.original"))
N = len(GT)
assert N == 10869, N
GT_WORDS = [words(g) for g in GT]
GT_WORD_TOTAL = sum(len(w) for w in GT_WORDS)
assert GT_WORD_TOTAL == 76083, GT_WORD_TOTAL


# ----------------------------------------------------------------------------
# seen / unseen INPUT surface forms, reused VERBATIM from
# three_model_string_error.seen_unseen_word_acc: a GT word slot is "seen" iff the
# corresponding test.in surface token was present anywhere in train.in.
# test.in tokens align positionally to GT word slots (boundary markers are
# intra-token, so token COUNTS match; the published run has 0 skipped lines).
# ----------------------------------------------------------------------------
def load_train_in_forms():
    s = set()
    with open(os.path.join(RAW, "train.in"), encoding="utf-8") as f:
        for line in f:
            s.update(line.split())
    return s


TEST_IN = rd(os.path.join(RAW, "test.in"))
TRAIN_IN_FORMS = load_train_in_forms()
# Per-line list of booleans: is GT word slot i "seen"?  None => line skipped for
# token-count mismatch (mirrors the research script; expected 0 such lines).
SEEN_MASK = []
_skipped_lines = 0
_seen_slots = _unseen_slots = 0
for _li in range(N):
    _in = TEST_IN[_li].split()
    _gw = GT_WORDS[_li]
    if len(_in) != len(_gw):
        SEEN_MASK.append(None)
        _skipped_lines += 1
    else:
        row = [(_in[_i] in TRAIN_IN_FORMS) for _i in range(len(_gw))]
        SEEN_MASK.append(row)
        _seen_slots += sum(row)
        _unseen_slots += len(row) - sum(row)
assert _skipped_lines == 0, _skipped_lines       # matches lines_skipped_token_mismatch=0
assert _seen_slots == 67168, _seen_slots          # matches three_model_string_error.json
assert _unseen_slots == 8915, _unseen_slots       # matches three_model_string_error.json


def error_word_distances(pred):
    """Yield Lev.distance(pred_word, gt_word) for every ERROR word slot.

    Positional alignment identical to three_model_string_error.word_error_set:
    slot i is an error if i >= len(pred_words) (missing) or pred_words[i] != gt.
    For missing slots, pred_word := "" so distance == len(gt_word).
    """
    dists = []
    for li in range(N):
        gw = GT_WORDS[li]
        pw = words(pred[li])
        for i in range(len(gw)):
            g = gw[i]
            p = pw[i] if i < len(pw) else ""
            if p != g:
                dists.append(Lev.distance(p, g))
    return dists


def error_word_distances_seen_unseen(pred):
    """Same as error_word_distances but split into (seen, unseen) error-word
    distance lists using the train.in-membership SEEN_MASK."""
    seen, unseen = [], []
    for li in range(N):
        mask = SEEN_MASK[li]
        if mask is None:
            continue
        gw = GT_WORDS[li]
        pw = words(pred[li])
        for i in range(len(gw)):
            g = gw[i]
            p = pw[i] if i < len(pw) else ""
            if p != g:
                d = Lev.distance(p, g)
                (seen if mask[i] else unseen).append(d)
    return seen, unseen


def distribution(dists):
    n = len(dists)
    c = Counter(dists)
    # bucket: exactly 1, 2, 3, 4+, and the long tail >=5
    b1 = c.get(1, 0)
    b2 = c.get(2, 0)
    b3 = c.get(3, 0)
    b4plus = sum(v for k, v in c.items() if k >= 4)
    b5plus = sum(v for k, v in c.items() if k >= 5)
    return {
        "n_error_words": n,
        "mean_char_editdist": st.mean(dists),
        "median_char_editdist": st.median(dists),
        "stdev_char_editdist": st.pstdev(dists),
        "max_char_editdist": max(dists),
        "p90_char_editdist": _quantile(dists, 0.90),
        "p95_char_editdist": _quantile(dists, 0.95),
        "p99_char_editdist": _quantile(dists, 0.99),
        "dist_eq1": b1, "dist_eq2": b2, "dist_eq3": b3,
        "dist_ge4": b4plus, "dist_ge5": b5plus,
        "dist_eq1_pct": 100 * b1 / n,
        "dist_eq2_pct": 100 * b2 / n,
        "dist_eq3_pct": 100 * b3 / n,
        "dist_ge4_pct": 100 * b4plus / n,
        "dist_ge5_pct": 100 * b5plus / n,
        "total_error_chars": sum(dists),
        # full histogram (sorted) for transparency
        "histogram": {str(k): c[k] for k in sorted(c)},
    }


def _quantile(xs, q):
    """Nearest-rank percentile on a sorted copy (integers in, value out)."""
    s = sorted(xs)
    if not s:
        return None
    idx = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[idx]


def sample_check(preds):
    """Show the first error words found for each model to verify alignment."""
    print("=== SAMPLE CHECK (first error word slots; verify alignment) ===")
    for k, pred in preds.items():
        print(f"\n--- {k} ---")
        shown = 0
        for li in range(N):
            gw = GT_WORDS[li]
            pw = words(pred[li])
            for i in range(len(gw)):
                g = gw[i]
                p = pw[i] if i < len(pw) else "<MISSING>"
                if (pw[i] if i < len(pw) else "") != g:
                    d = Lev.distance(p if p != "<MISSING>" else "", g)
                    print(f"  line {li} slot {i}: GT={g!r} PRED={p!r} dist={d}")
                    shown += 1
                    break  # one per line for readability
            if shown >= 8:
                break


def main():
    preds = {
        "MDLM": rd(mdlm_path("42")),
        "ENC": rd(enc_path("42")),
        "ED": rd(ED_PRED),
    }
    for k, v in preds.items():
        assert len(v) == N, (k, len(v))

    if "--sample" in sys.argv:
        sample_check(preds)
        return

    res = {
        "n_lines": N,
        "gt_total_words": GT_WORD_TOTAL,
        "note": ("Per-error-word character Levenshtein distance. Alignment reuses "
                 "three_model_string_error.py word-slot logic (whitespace split, "
                 "positional). Missing pred slot -> empty string -> dist==len(GT word)."),
        "models": {},
    }
    res["seen_unseen_slot_totals"] = {"seen": _seen_slots, "unseen": _unseen_slots,
                                      "lines_skipped_token_mismatch": _skipped_lines}
    res["seen_unseen_note"] = (
        "seen = test.in surface token present in train.in (else unseen); reused "
        "from three_model_string_error.seen_unseen_word_acc. Error point = manual "
        "correction effort proxy; this split shows whether unseen errors also cost "
        "MORE per word, not just occur more often.")
    expected_n = {"MDLM": 8807, "ENC": 9603, "ED": 8947}
    for k, pred in preds.items():
        dists = error_word_distances(pred)
        d = distribution(dists)
        # cross-check error-word count against the published JSON
        d["expected_n_from_string_error_json"] = expected_n[k]
        d["matches_expected"] = (d["n_error_words"] == expected_n[k])
        # seen / unseen split
        seen_d, unseen_d = error_word_distances_seen_unseen(pred)
        d["seen"] = distribution(seen_d)
        d["unseen"] = distribution(unseen_d)
        # consistency: seen+unseen error words must equal the all-words count
        assert d["seen"]["n_error_words"] + d["unseen"]["n_error_words"] == d["n_error_words"]
        res["models"][k] = d

    with open(os.path.join(OUT, "per_word_editdistance.json"), "w",
              encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    # ---- console digest ----
    print("=== PER-ERROR-WORD CHARACTER EDIT DISTANCE ===")
    print(f"{'model':6s} {'nErr':>6s} {'mean':>6s} {'med':>4s} {'std':>5s}"
          f" {'max':>4s} {'p95':>4s} {'=1%':>6s} {'=2%':>6s} {'=3%':>6s}"
          f" {'>=4%':>6s} {'>=5%':>6s}")
    for k in ["MDLM", "ENC", "ED"]:
        d = res["models"][k]
        flag = "" if d["matches_expected"] else f" (!= {d['expected_n_from_string_error_json']})"
        print(f"{k:6s} {d['n_error_words']:6d} {d['mean_char_editdist']:6.3f}"
              f" {d['median_char_editdist']:4.1f} {d['stdev_char_editdist']:5.2f}"
              f" {d['max_char_editdist']:4d} {d['p95_char_editdist']:4d}"
              f" {d['dist_eq1_pct']:6.2f} {d['dist_eq2_pct']:6.2f}"
              f" {d['dist_eq3_pct']:6.2f} {d['dist_ge4_pct']:6.2f}"
              f" {d['dist_ge5_pct']:6.2f}{flag}")
    print("\nClaim check (一错错一堆): ED mean per-error-word dist vs MDLM/ENC:")
    m = {k: res["models"][k]["mean_char_editdist"] for k in ["MDLM", "ENC", "ED"]}
    print(f"  MDLM={m['MDLM']:.3f}  ENC={m['ENC']:.3f}  ED={m['ED']:.3f}")
    print(f"  ED higher than MDLM? {m['ED'] > m['MDLM']}   ED higher than ENC? {m['ED'] > m['ENC']}")

    # ---- seen / unseen digest ----
    print("\n=== SEEN vs UNSEEN error words (per-error-word char edit distance) ===")
    print(f"{'model':6s} {'grp':>6s} {'nErr':>6s} {'mean':>6s} {'med':>4s}"
          f" {'std':>5s} {'=1%':>6s} {'=2%':>6s} {'=3%':>6s} {'>=4%':>6s}")
    for k in ["MDLM", "ENC", "ED"]:
        for grp in ["seen", "unseen"]:
            d = res["models"][k][grp]
            print(f"{k:6s} {grp:>6s} {d['n_error_words']:6d}"
                  f" {d['mean_char_editdist']:6.3f} {d['median_char_editdist']:4.1f}"
                  f" {d['stdev_char_editdist']:5.2f}"
                  f" {d['dist_eq1_pct']:6.2f} {d['dist_eq2_pct']:6.2f}"
                  f" {d['dist_eq3_pct']:6.2f} {d['dist_ge4_pct']:6.2f}")
    print("\nunseen costs MORE chars per error word than seen?")
    for k in ["MDLM", "ENC", "ED"]:
        s = res["models"][k]["seen"]["mean_char_editdist"]
        u = res["models"][k]["unseen"]["mean_char_editdist"]
        print(f"  {k:6s} seen={s:.3f} unseen={u:.3f}  delta=+{u-s:.3f}  unseen>seen? {u > s}")


if __name__ == "__main__":
    main()
