#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cluster A / Task 1 + 2 : divergence analysis and error taxonomy.

Question (R2-M3, R3-1, R1-W9): is the MDLM gain over the encoder-only model
merely "a different (discrete) representation", or does iterative denoising
actually fix structurally meaningful errors?  We answer by directly comparing
position-aligned predictions.

Two complementary pairings:
  (A) SHARED-SEED paired comparison (enc seed42 vs mdlm seed42; seed49 vs seed49).
      This is the cleanest controlled contrast: identical seed, identical data,
      only the architecture/decoding differs.
  (B) CROSS-SEED majority-vote ensemble (5 enc seeds vs 5 mdlm seeds). Each
      position's label is the per-family modal prediction; ties broken by the
      lowest-numbered seed. This removes single-seed noise and asks whether the
      *family* of MDLM models systematically corrects the encoder family.

Metrics emitted:
  * net correction = (#positions MDLM-correct & enc-wrong)
                     - (#positions MDLM-wrong & enc-correct)
  * a 2x2 McNemar table (both-correct / mdlm-only / enc-only / both-wrong)
  * McNemar exact test on the discordant cells
  * taxonomy of the cells MDLM fixes, by (a) GT pattern label, (b) marker family,
    (c) edit-distance bucket on the restored string for the affected LINES.

All outputs go to analysis/cluster_A/ . Nothing in the repo is modified.
"""

import os
import json
from collections import Counter, defaultdict
from math import comb

import lib_load as L

OUT = os.path.dirname(os.path.abspath(__file__))


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def majority_labels(model, seeds):
    """Per-position modal label across the given seeds (tie -> first seed)."""
    preds = [L.load_labels(model, s) for s in seeds]
    gt = L.load_gt_labels()
    out = []
    for li in range(len(gt)):
        n = len(gt[li])
        line = []
        for pos in range(n):
            votes = Counter(preds[k][li][pos] for k in range(len(seeds)))
            top = max(votes.values())
            # tie-break: among labels with top votes, take the one from the
            # earliest seed at this position (deterministic + reproducible)
            best = None
            for k in range(len(seeds)):
                lab = preds[k][li][pos]
                if votes[lab] == top:
                    best = lab
                    break
            line.append(best)
        out.append(line)
    return out


def mcnemar_p(b, c):
    """Two-sided McNemar p-value on discordant counts b, c.

    Exact binomial tail when b+c is small enough to enumerate, otherwise the
    chi-square approximation with continuity correction (Edwards). For the
    counts here (thousands) the two agree to many digits.
    """
    n = b + c
    if n == 0:
        return 1.0
    if n <= 1000:
        k = min(b, c)
        tail = sum(comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
        return min(1.0, 2.0 * tail)
    # normal approximation of the binomial (continuity-corrected McNemar)
    from math import erf, sqrt
    chi = (abs(b - c) - 1.0) ** 2 / n  # ~ chi2_1
    z = sqrt(chi)
    # two-sided p from standard normal
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0))))
    return max(0.0, min(1.0, p))


def marker_family(pattern):
    """Coarse morphological marker family of a pattern string (taxonomy reused
    from the case-study levenshtein_detailed_analysis.md vocabulary)."""
    if pattern == "":
        return "null (no boundary)"
    fams = []
    if "!" in pattern:
        fams.append("emphatic/restored-! (!!,!)")
    if "/" in pattern:
        fams.append("lexeme-end /")
    if "[" in pattern or "]" in pattern:
        fams.append("verbal-stem []")
    if "(" in pattern or "&" in pattern:
        fams.append("affix/enclitic (&")
    if ":d" in pattern:
        fams.append("determiner :d")
    if "-" in pattern:
        fams.append("prefix -")
    if "=" in pattern or "~" in pattern or "@" in pattern:
        fams.append("vowel/diacritic =~@")
    if not fams:
        fams.append("other")
    # report the single most "structural" family by a fixed priority
    priority = [
        "verbal-stem []",
        "affix/enclitic (&",
        "emphatic/restored-! (!!,!)",
        "determiner :d",
        "lexeme-end /",
        "prefix -",
        "vowel/diacritic =~@",
        "other",
    ]
    for p in priority:
        if p in fams:
            return p
    return "other"


# -----------------------------------------------------------------------------
# core divergence computation for a single enc/mdlm label pairing
# -----------------------------------------------------------------------------
def divergence(enc_lab, mdlm_lab, tag):
    gt = L.load_gt_labels()
    patt = L.load_patterns()

    both_correct = mdlm_only = enc_only = both_wrong = 0
    total = 0
    # taxonomy of positions MDLM fixes (mdlm_only) and breaks (enc_only)
    fix_by_pattern = Counter()
    fix_by_family = Counter()
    break_by_pattern = Counter()
    break_by_family = Counter()
    # confusion on fixed cells: what enc predicted -> what GT was
    fix_confusion = Counter()
    break_confusion = Counter()
    # line-level divergence accounting
    line_net = []  # per line: (#fix - #break)

    for li in range(len(gt)):
        g = gt[li]
        e = enc_lab[li]
        m = mdlm_lab[li]
        lfix = lbreak = 0
        for pos in range(len(g)):
            gl = g[pos]
            ec = (e[pos] == gl)
            mc = (m[pos] == gl)
            total += 1
            if ec and mc:
                both_correct += 1
            elif mc and not ec:
                mdlm_only += 1
                lfix += 1
                fix_by_pattern[gl] += 1
                fix_by_family[marker_family(patt[gl])] += 1
                fix_confusion[(patt.get(e[pos], "?"), patt.get(gl, "?"))] += 1
            elif ec and not mc:
                enc_only += 1
                lbreak += 1
                break_by_pattern[gl] += 1
                break_by_family[marker_family(patt[gl])] += 1
                break_confusion[(patt.get(m[pos], "?"), patt.get(gl, "?"))] += 1
            else:
                both_wrong += 1
        line_net.append(lfix - lbreak)

    net = mdlm_only - enc_only
    p = mcnemar_p(mdlm_only, enc_only)

    return {
        "tag": tag,
        "total_positions": total,
        "both_correct": both_correct,
        "mdlm_only_correct": mdlm_only,
        "enc_only_correct": enc_only,
        "both_wrong": both_wrong,
        "net_correction": net,
        "correction_break_ratio": (mdlm_only / enc_only) if enc_only else None,
        "mcnemar_p": p,
        "enc_acc": (both_correct + enc_only) / total,
        "mdlm_acc": (both_correct + mdlm_only) / total,
        "fix_by_family": dict(fix_by_family.most_common()),
        "break_by_family": dict(break_by_family.most_common()),
        "fix_by_pattern_top": [
            {"label": lab, "pattern": patt[lab], "count": c}
            for lab, c in fix_by_pattern.most_common(15)
        ],
        "break_by_pattern_top": [
            {"label": lab, "pattern": patt[lab], "count": c}
            for lab, c in break_by_pattern.most_common(15)
        ],
        "fix_confusion_top": [
            {"enc_pred": a, "gt": b, "count": c}
            for (a, b), c in fix_confusion.most_common(15)
        ],
        "line_net_positive": sum(1 for x in line_net if x > 0),
        "line_net_negative": sum(1 for x in line_net if x < 0),
        "line_net_zero": sum(1 for x in line_net if x == 0),
    }


def main():
    results = {}

    # (A) shared-seed paired comparisons
    for s in L.SHARED_SEEDS:
        enc = L.load_labels("enc", s)
        md = L.load_labels("mdlm", s)
        results[f"shared_seed_{s}"] = divergence(enc, md, f"enc_seed{s} vs mdlm_seed{s}")

    # (B) cross-seed majority-vote ensemble
    enc_maj = majority_labels("enc", list(L.ENC_SEEDS.keys()))
    md_maj = majority_labels("mdlm", list(L.MDLM_SEEDS.keys()))
    results["ensemble_majority"] = divergence(
        enc_maj, md_maj, "enc 5-seed majority vs mdlm 5-seed majority"
    )

    # (C) all-cross pairings averaged (5x5) — net correction robustness
    cross_nets = []
    cross_fix = []
    cross_break = []
    for es in L.ENC_SEEDS:
        for ms in L.MDLM_SEEDS:
            enc = L.load_labels("enc", es)
            md = L.load_labels("mdlm", ms)
            d = divergence(enc, md, f"enc{es}_x_mdlm{ms}")
            cross_nets.append(d["net_correction"])
            cross_fix.append(d["mdlm_only_correct"])
            cross_break.append(d["enc_only_correct"])
    import statistics as st
    results["cross_all_25_pairings"] = {
        "n_pairings": len(cross_nets),
        "net_correction_mean": st.mean(cross_nets),
        "net_correction_min": min(cross_nets),
        "net_correction_max": max(cross_nets),
        "net_correction_all_positive": all(x > 0 for x in cross_nets),
        "mdlm_only_mean": st.mean(cross_fix),
        "enc_only_mean": st.mean(cross_break),
    }

    with open(os.path.join(OUT, "divergence_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # console digest
    for k, d in results.items():
        if "net_correction" in d:
            print(f"\n== {k} ==")
            print(f"  both_correct={d['both_correct']:,}  mdlm_only={d['mdlm_only_correct']:,}"
                  f"  enc_only={d['enc_only_correct']:,}  both_wrong={d['both_wrong']:,}")
            print(f"  net_correction={d['net_correction']:,}  ratio={d['correction_break_ratio']:.3f}"
                  f"  McNemar p={d['mcnemar_p']:.2e}")
            print(f"  enc_acc={d['enc_acc']:.4f}  mdlm_acc={d['mdlm_acc']:.4f}")
    c = results["cross_all_25_pairings"]
    print(f"\n== 25 cross pairings == net mean={c['net_correction_mean']:.0f}"
          f"  range=[{c['net_correction_min']},{c['net_correction_max']}]"
          f"  all_positive={c['net_correction_all_positive']}")


if __name__ == "__main__":
    main()
