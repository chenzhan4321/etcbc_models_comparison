#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cluster A / Task 1 (line-level divergence) + Task 4 (seen vs unseen, rare-pattern).

Why line-level matters: the character-level accuracy gap between MDLM and the
encoder is tiny (~0.1-0.2 pts), but the SEQUENCE/line-level gap is ~3 pts.
That asymmetry is itself the headline evidence for "global coherence": iterative
denoising turns near-misses into fully correct sequences. We quantify:

  1. line-level McNemar (exact line right/wrong) on shared seeds + ensemble.
  2. "MDLM rescues a line the encoder got wrong" vs the reverse, and HOW the
     rescued lines differ (how many positions had to flip).
  3. seen vs unseen WORD analysis. A test word is "unseen" if its surface input
     string never appears in train.in. We measure per-model WORD-level exact
     accuracy (all labels of the word correct) split by seen/unseen, because the
     reviewer asks specifically about generalisation to novel forms.
  4. rare-pattern analysis: bucket GT label positions by training frequency rank
     (from patterns.csv) and compare per-model recall of each frequency band.
"""

import os
import json
from collections import defaultdict, Counter
from math import erf, sqrt, comb

import lib_load as L

OUT = os.path.dirname(os.path.abspath(__file__))


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    if n <= 1000:
        k = min(b, c)
        tail = sum(comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
        return min(1.0, 2.0 * tail)
    chi = (abs(b - c) - 1.0) ** 2 / n
    z = sqrt(chi)
    return max(0.0, min(1.0, 2.0 * (1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0))))))


def line_correct_vector(model, seed):
    gt = L.load_gt_labels()
    pl = L.load_labels(model, seed)
    return [g == p for g, p in zip(gt, pl)]


def line_divergence(enc_seed, mdlm_seed, tag):
    gt = L.load_gt_labels()
    e = L.load_labels("enc", enc_seed)
    m = L.load_labels("mdlm", mdlm_seed)
    both = mo = eo = neither = 0
    rescue_flip_counts = []  # for lines MDLM gets right & enc wrong: #positions enc was wrong
    for g, el, ml in zip(gt, e, m):
        ec = (g == el)
        mc = (g == ml)
        if ec and mc:
            both += 1
        elif mc and not ec:
            mo += 1
            rescue_flip_counts.append(sum(1 for a, b in zip(g, el) if a != b))
        elif ec and not mc:
            eo += 1
        else:
            neither += 1
    tot = len(gt)
    import statistics as st
    return {
        "tag": tag,
        "total_lines": tot,
        "both_correct_lines": both,
        "mdlm_only_correct_lines": mo,
        "enc_only_correct_lines": eo,
        "neither_lines": neither,
        "net_line_rescue": mo - eo,
        "rescue_break_ratio": (mo / eo) if eo else None,
        "line_mcnemar_p": mcnemar_p(mo, eo),
        "enc_line_acc": (both + eo) / tot,
        "mdlm_line_acc": (both + mo) / tot,
        "median_positions_enc_wrong_on_rescued_lines": (
            st.median(rescue_flip_counts) if rescue_flip_counts else None),
        "mean_positions_enc_wrong_on_rescued_lines": (
            st.mean(rescue_flip_counts) if rescue_flip_counts else None),
    }


# -----------------------------------------------------------------------------
# seen vs unseen WORD analysis
# -----------------------------------------------------------------------------
def seen_unseen(model_seeds_enc, model_seeds_mdlm):
    """Word-level exact accuracy split by seen/unseen input surface form.

    We aggregate over seeds by averaging per-seed accuracies (so each family is
    represented fairly). Words are the space-delimited tokens of test.in.
    """
    train_in = L.load_train_forms()
    test_in = L.load_test_in()
    gt = L.load_gt_labels()

    # pre-split GT words once
    gt_words = [L.split_words(test_in[i], gt[i]) for i in range(len(gt))]

    def per_model(model, seed):
        pl = L.load_labels(model, seed)
        seen_tot = seen_corr = uns_tot = uns_corr = 0
        for i in range(len(gt)):
            pw = L.split_words(test_in[i], pl[i])
            gw = gt_words[i]
            # both splits use the same input chars => same word boundaries
            for (gword, glabs), (_, plabs) in zip(gw, pw):
                is_seen = gword in train_in
                ok = (glabs == plabs)
                if is_seen:
                    seen_tot += 1
                    seen_corr += ok
                else:
                    uns_tot += 1
                    uns_corr += ok
        return seen_corr, seen_tot, uns_corr, uns_tot

    res = {}
    for fam, seeds in [("enc", model_seeds_enc), ("mdlm", model_seeds_mdlm)]:
        seen_accs, uns_accs = [], []
        seen_tot = uns_tot = 0
        for s in seeds:
            sc, stot, uc, utot = per_model(fam, s)
            seen_accs.append(sc / stot)
            uns_accs.append(uc / utot)
            seen_tot, uns_tot = stot, utot
        import statistics as st
        res[fam] = {
            "seen_word_acc_mean": st.mean(seen_accs),
            "seen_word_acc_std": st.pstdev(seen_accs),
            "unseen_word_acc_mean": st.mean(uns_accs),
            "unseen_word_acc_std": st.pstdev(uns_accs),
            "seen_word_total": seen_tot,
            "unseen_word_total": uns_tot,
        }
    res["unseen_gap_mdlm_minus_enc"] = (
        res["mdlm"]["unseen_word_acc_mean"] - res["enc"]["unseen_word_acc_mean"])
    res["seen_gap_mdlm_minus_enc"] = (
        res["mdlm"]["seen_word_acc_mean"] - res["enc"]["seen_word_acc_mean"])
    return res


# -----------------------------------------------------------------------------
# rare-pattern recall (per GT label frequency band)
# -----------------------------------------------------------------------------
def rare_pattern_recall(enc_seeds, mdlm_seeds):
    counts, rank = L.load_pattern_counts()
    patt = L.load_patterns()
    gt = L.load_gt_labels()

    # frequency bands by training count
    def band(lab):
        c = counts.get(lab, 0)
        if lab == 0:
            return "0:null label"
        if c >= 10000:
            return "1:very-common (>=1e4)"
        if c >= 1000:
            return "2:common (1e3-1e4)"
        if c >= 100:
            return "3:mid (1e2-1e3)"
        if c >= 10:
            return "4:rare (10-1e2)"
        return "5:very-rare (<10)"

    # per-band GT support
    band_support = Counter()
    for g in gt:
        for lab in g:
            band_support[band(lab)] += 1

    def per_model_band_recall(model, seeds):
        # recall of each band = fraction of GT positions in that band the model
        # labels correctly, averaged across seeds
        import statistics as st
        per_seed = []
        for s in seeds:
            pl = L.load_labels(model, s)
            corr = Counter()
            for g, p in zip(gt, pl):
                for gl, pl_ in zip(g, p):
                    if gl == pl_:
                        corr[band(gl)] += 1
            per_seed.append({b: corr[b] / band_support[b] for b in band_support})
        out = {}
        for b in band_support:
            vals = [d[b] for d in per_seed]
            out[b] = {"recall_mean": st.mean(vals), "recall_std": st.pstdev(vals)}
        return out

    enc_b = per_model_band_recall("enc", enc_seeds)
    md_b = per_model_band_recall("mdlm", mdlm_seeds)

    bands = sorted(band_support)
    table = []
    for b in bands:
        table.append({
            "band": b,
            "gt_support": band_support[b],
            "enc_recall": enc_b[b]["recall_mean"],
            "enc_std": enc_b[b]["recall_std"],
            "mdlm_recall": md_b[b]["recall_mean"],
            "mdlm_std": md_b[b]["recall_std"],
            "mdlm_minus_enc": md_b[b]["recall_mean"] - enc_b[b]["recall_mean"],
        })

    # Also: per individual rare label (count<100) recall, top by support
    rare_labels = [lab for lab in counts if 0 < counts[lab] < 1000]
    rare_support = Counter()
    for g in gt:
        for lab in g:
            if lab in rare_labels:
                rare_support[lab] += 1

    def per_label_recall(model, seeds, labels):
        import statistics as st
        per_seed_corr = defaultdict(list)
        for s in seeds:
            pl = L.load_labels(model, s)
            corr = Counter()
            for g, p in zip(gt, pl):
                for gl, pl_ in zip(g, p):
                    if gl in labels and gl == pl_:
                        corr[gl] += 1
            for lab in labels:
                per_seed_corr[lab].append(corr[lab])
        return {lab: st.mean(per_seed_corr[lab]) for lab in labels}

    enc_lab = per_label_recall("enc", enc_seeds, set(rare_labels))
    md_lab = per_label_recall("mdlm", mdlm_seeds, set(rare_labels))
    per_label = []
    for lab, sup in rare_support.most_common(25):
        if sup == 0:
            continue
        per_label.append({
            "label": lab,
            "pattern": patt[lab],
            "train_count": counts[lab],
            "freq_rank": rank[lab],
            "gt_support_in_test": sup,
            "enc_recall": enc_lab[lab] / sup,
            "mdlm_recall": md_lab[lab] / sup,
            "mdlm_minus_enc": (md_lab[lab] - enc_lab[lab]) / sup,
        })

    return {"band_table": table, "per_rare_label": per_label}


def main():
    res = {}
    for s in L.SHARED_SEEDS:
        res[f"line_shared_seed_{s}"] = line_divergence(s, s, f"enc{s} vs mdlm{s}")

    res["seen_unseen"] = seen_unseen(list(L.ENC_SEEDS), list(L.MDLM_SEEDS))
    res["rare_pattern"] = rare_pattern_recall(list(L.ENC_SEEDS), list(L.MDLM_SEEDS))

    with open(os.path.join(OUT, "line_and_seen_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    for s in L.SHARED_SEEDS:
        d = res[f"line_shared_seed_{s}"]
        print(f"\n== LINE seed{s} ==  mdlm_only_lines={d['mdlm_only_correct_lines']}"
              f"  enc_only_lines={d['enc_only_correct_lines']}"
              f"  net={d['net_line_rescue']}  ratio={d['rescue_break_ratio']:.3f}"
              f"  p={d['line_mcnemar_p']:.2e}")
        print(f"   enc_line_acc={d['enc_line_acc']:.4f}  mdlm_line_acc={d['mdlm_line_acc']:.4f}"
              f"  median enc-wrong-pos on rescued={d['median_positions_enc_wrong_on_rescued_lines']}")

    su = res["seen_unseen"]
    print("\n== SEEN/UNSEEN word exact acc ==")
    for fam in ["enc", "mdlm"]:
        print(f"  {fam}: seen={su[fam]['seen_word_acc_mean']:.4f}"
              f"  unseen={su[fam]['unseen_word_acc_mean']:.4f}"
              f"  (seen_n={su[fam]['seen_word_total']:,} unseen_n={su[fam]['unseen_word_total']:,})")
    print(f"  unseen gap (mdlm-enc) = {su['unseen_gap_mdlm_minus_enc']:.4f}")
    print(f"  seen   gap (mdlm-enc) = {su['seen_gap_mdlm_minus_enc']:.4f}")

    print("\n== RARE-PATTERN band recall ==")
    for r in res["rare_pattern"]["band_table"]:
        print(f"  {r['band']:24s} sup={r['gt_support']:7d}"
              f"  enc={r['enc_recall']:.4f}  mdlm={r['mdlm_recall']:.4f}"
              f"  d={r['mdlm_minus_enc']:+.4f}")


if __name__ == "__main__":
    main()
