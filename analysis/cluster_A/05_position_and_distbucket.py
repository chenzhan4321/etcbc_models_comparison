#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cluster A / supporting evidence for the "non-local" claim + edit-distance buckets.

(1) Where in the word do the corrections happen?  If MDLM's net corrections are
    concentrated at WORD-FINAL / WORD-INITIAL positions (where the morphological
    boundary depends on the neighbouring token), that supports a non-local /
    boundary-disambiguation mechanism rather than per-character memorisation.
    We compute, on shared seed 42, the within-word relative position of every
    cell MDLM fixes (mdlm-correct & enc-wrong) vs breaks, normalised to
    {initial, internal, final, single-char-word}.

(2) Edit-distance bucket taxonomy (reusing the case-study md buckets:
    dist=0 / 1-5 / 6-10 / >10) on the RESTORED .reduced strings, comparing the
    two models line-by-line, and how MDLM redistributes lines across buckets
    relative to the encoder (the "where do the gains land" view).
"""

import os
import json
from collections import Counter

import lib_load as L

OUT = os.path.dirname(os.path.abspath(__file__))
SEED = "42"


def levenshtein(a, b):
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * lb
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[lb]


def position_analysis():
    gt = L.load_gt_labels()
    ti = L.load_test_in()
    e = L.load_labels("enc", SEED)
    m = L.load_labels("mdlm", SEED)

    fix = Counter()
    brk = Counter()

    def classify(idx, wlen):
        if wlen == 1:
            return "single-char word"
        if idx == 0:
            return "word-initial"
        if idx == wlen - 1:
            return "word-final"
        return "word-internal"

    for i in range(len(gt)):
        gw = L.split_words(ti[i], gt[i])
        ew = L.split_words(ti[i], e[i])
        mw = L.split_words(ti[i], m[i])
        for (w, gl), (_, el), (_, ml) in zip(gw, ew, mw):
            wlen = len(gl)
            for idx in range(wlen):
                ec = (el[idx] == gl[idx])
                mc = (ml[idx] == gl[idx])
                pos = classify(idx, wlen)
                if mc and not ec:
                    fix[pos] += 1
                elif ec and not mc:
                    brk[pos] += 1

    cats = ["word-initial", "word-internal", "word-final", "single-char word"]
    table = []
    for c in cats:
        f, b = fix[c], brk[c]
        table.append({"position": c, "mdlm_fixes": f, "mdlm_breaks": b,
                      "net": f - b, "fix_break_ratio": (f / b) if b else None})
    return {"seed": SEED,
            "total_fixes": sum(fix.values()),
            "total_breaks": sum(brk.values()),
            "by_position": table}


def dist_bucket_analysis():
    gtr = L.load_gt_restored("reduced")
    er = L.load_restored("enc", SEED, "reduced")
    mr = L.load_restored("mdlm", SEED, "reduced")

    def bucket(d):
        if d == 0:
            return "dist=0 (exact)"
        if d <= 5:
            return "dist 1-5 (minor)"
        if d <= 10:
            return "dist 6-10 (moderate)"
        return "dist >10 (large)"

    enc_b = Counter()
    md_b = Counter()
    # transition matrix: encoder bucket -> mdlm bucket
    trans = Counter()
    for g, ee, mm in zip(gtr, er, mr):
        de = levenshtein(g, ee)
        dm = levenshtein(g, mm)
        be, bm = bucket(de), bucket(dm)
        enc_b[be] += 1
        md_b[bm] += 1
        trans[(be, bm)] += 1

    order = ["dist=0 (exact)", "dist 1-5 (minor)", "dist 6-10 (moderate)", "dist >10 (large)"]
    return {
        "seed": SEED,
        "encoder_buckets": {k: enc_b[k] for k in order},
        "mdlm_buckets": {k: md_b[k] for k in order},
        "transition_enc_to_mdlm": {f"{a} -> {b}": trans[(a, b)]
                                   for a in order for b in order if trans[(a, b)]},
        "n_moved_to_exact": sum(trans[(a, "dist=0 (exact)")]
                                for a in order if a != "dist=0 (exact)"),
        "n_left_exact": sum(trans[("dist=0 (exact)", b)]
                            for b in order if b != "dist=0 (exact)"),
    }


def main():
    res = {"position": position_analysis(), "dist_bucket": dist_bucket_analysis()}
    with open(os.path.join(OUT, "position_distbucket.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    p = res["position"]
    print(f"== Within-word position of corrections (seed{SEED}) ==")
    print(f"  total fixes={p['total_fixes']}  total breaks={p['total_breaks']}")
    for r in p["by_position"]:
        ratio = f"{r['fix_break_ratio']:.2f}" if r["fix_break_ratio"] else "n/a"
        print(f"  {r['position']:18s} fixes={r['mdlm_fixes']:5d} breaks={r['mdlm_breaks']:5d}"
              f" net={r['net']:+5d} ratio={ratio}")

    d = res["dist_bucket"]
    print(f"\n== Edit-distance buckets on restored .reduced (seed{SEED}) ==")
    print(f"  {'bucket':22s} {'encoder':>9s} {'mdlm':>9s}")
    for k in d["encoder_buckets"]:
        print(f"  {k:22s} {d['encoder_buckets'][k]:9d} {d['mdlm_buckets'][k]:9d}")
    print(f"  lines moved INTO exact by MDLM: {d['n_moved_to_exact']}")
    print(f"  lines MDLM lost FROM exact    : {d['n_left_exact']}")


if __name__ == "__main__":
    main()
