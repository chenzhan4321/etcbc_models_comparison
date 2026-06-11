#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cluster A / statistical honesty : bootstrap CI for the headline claims.

We bootstrap over LINES (the natural i.i.d. unit; words within a line are
correlated, so resampling lines is the conservative choice) to put confidence
intervals on:
  * the unseen-word exact-accuracy gap (MDLM - encoder), shared seed 42;
  * the seen-word gap (should be much smaller -> shows the gain is generalisation);
  * the line-level (sequence) accuracy gap, shared seed 42.

We also report a paired bootstrap p-value (fraction of resamples where the gap
flips sign), which is an honest two-sided significance proxy.
"""

import os
import json
import random

import lib_load as L

OUT = os.path.dirname(os.path.abspath(__file__))
SEED_RNG = 20260606
N_BOOT = 2000
ENC_SEED = "42"
MDLM_SEED = "42"


def build_units():
    """Per line, collect (unseen_word_outcomes, seen_word_outcomes, line_outcome)
    for both models, so a single line resample updates everything coherently."""
    gt = L.load_gt_labels()
    ti = L.load_test_in()
    train = L.load_train_forms()
    e = L.load_labels("enc", ENC_SEED)
    m = L.load_labels("mdlm", MDLM_SEED)

    lines = []
    for i in range(len(gt)):
        gw = L.split_words(ti[i], gt[i])
        ew = L.split_words(ti[i], e[i])
        mw = L.split_words(ti[i], m[i])
        seen_e = seen_m = seen_n = 0
        uns_e = uns_m = uns_n = 0
        for (w, gl), (_, el), (_, ml) in zip(gw, ew, mw):
            unseen = w not in train
            ec = (gl == el)
            mc = (gl == ml)
            if unseen:
                uns_n += 1
                uns_e += ec
                uns_m += mc
            else:
                seen_n += 1
                seen_e += ec
                seen_m += mc
        line_e = (gt[i] == e[i])
        line_m = (gt[i] == m[i])
        lines.append({
            "seen_n": seen_n, "seen_e": seen_e, "seen_m": seen_m,
            "uns_n": uns_n, "uns_e": uns_e, "uns_m": uns_m,
            "line_e": int(line_e), "line_m": int(line_m),
        })
    return lines


def agg(sample):
    se_n = sum(l["seen_n"] for l in sample)
    un_n = sum(l["uns_n"] for l in sample)
    n = len(sample)
    seen_e = sum(l["seen_e"] for l in sample) / se_n
    seen_m = sum(l["seen_m"] for l in sample) / se_n
    uns_e = sum(l["uns_e"] for l in sample) / un_n
    uns_m = sum(l["uns_m"] for l in sample) / un_n
    line_e = sum(l["line_e"] for l in sample) / n
    line_m = sum(l["line_m"] for l in sample) / n
    return {
        "seen_gap": seen_m - seen_e,
        "unseen_gap": uns_m - uns_e,
        "line_gap": line_m - line_e,
    }


def ci(vals, lo=2.5, hi=97.5):
    s = sorted(vals)
    n = len(s)
    return s[int(lo / 100 * n)], s[int(hi / 100 * n)]


def main():
    lines = build_units()
    point = agg(lines)

    rng = random.Random(SEED_RNG)
    n = len(lines)
    boot = {"seen_gap": [], "unseen_gap": [], "line_gap": []}
    for _ in range(N_BOOT):
        sample = [lines[rng.randrange(n)] for _ in range(n)]
        a = agg(sample)
        for k in boot:
            boot[k].append(a[k])

    res = {"seed_pair": f"enc{ENC_SEED}/mdlm{MDLM_SEED}", "n_boot": N_BOOT,
           "bootstrap_unit": "line", "point": point}
    for k in boot:
        lo, hi = ci(boot[k])
        # fraction of resamples with gap <= 0 (one-sided sign-flip rate)
        flip = sum(1 for v in boot[k] if v <= 0) / N_BOOT
        res[k] = {"point": point[k], "ci95_lo": lo, "ci95_hi": hi,
                  "p_signflip_two_sided": min(1.0, 2 * flip)}

    with open(os.path.join(OUT, "bootstrap_ci.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print(f"Bootstrap ({N_BOOT} line-resamples), seed pair enc{ENC_SEED}/mdlm{MDLM_SEED}:")
    for k in ["seen_gap", "unseen_gap", "line_gap"]:
        r = res[k]
        print(f"  {k:11s}: {r['point']:+.4f}  95% CI [{r['ci95_lo']:+.4f}, {r['ci95_hi']:+.4f}]"
              f"  p(signflip)={r['p_signflip_two_sided']:.3f}")


if __name__ == "__main__":
    main()
