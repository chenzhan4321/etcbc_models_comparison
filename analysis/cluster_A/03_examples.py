#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cluster A / Task 3 : concrete non-local / long-range dependency examples.

We want lines where MDLM is exactly correct and the encoder is wrong, AND the
encoder's error is plausibly a NON-LOCAL one — i.e. the encoder predicted a
locally-licensed but globally-wrong label, while MDLM (iterative denoising over
the whole sequence) resolved it.

Operational proxies for "non-local / global" rather than "memorise the n-gram":
  * The UNSEEN-word filter: the affected word's surface form never appears in
    train.in, so neither model can have memorised its label tuple. Getting it
    right requires generalising from sequence context.
  * The error is at the END or START of a word/clause and the disambiguation
    requires the neighbouring word (e.g. a determiner / enclitic boundary that
    depends on whether the following token is a noun vs verb).
  * We rank candidate lines by (a) encoder makes >=1 error on the word, (b) the
    word is unseen, (c) the error involves a structural marker family
    ([] verbal stem, (& affix, ! restored), and surface-show the comparison.

For maximum reproducibility we use shared seed 42 for both models, and confirm
the same correction also holds under the 5-seed majority vote (robustness flag).
"""

import os
import json
from collections import Counter

import lib_load as L

OUT = os.path.dirname(os.path.abspath(__file__))
SEED = "42"

STRUCTURAL_CHARS = set("[]()&!:d")  # markers that encode non-local structure


def restored_word_view(in_line, labels, patterns):
    """Render a line as space-joined restored tokens: char with its pattern
    appended, so the morphological segmentation is visible (mirrors .reduced)."""
    words = L.split_words(in_line, labels)
    rendered = []
    for word, labs in words:
        s = []
        for ch, lab in zip(word, labs):
            s.append(ch + patterns[lab])
        rendered.append("".join(s))
    return " ".join(rendered)


def main():
    gt = L.load_gt_labels()
    test_in = L.load_test_in()
    patt = L.load_patterns()
    train_in = L.load_train_forms()

    e = L.load_labels("enc", SEED)
    m = L.load_labels("mdlm", SEED)

    # majority for robustness flag
    def majority(model, seeds):
        preds = [L.load_labels(model, s) for s in seeds]
        out = []
        for li in range(len(gt)):
            line = []
            for pos in range(len(gt[li])):
                votes = Counter(preds[k][li][pos] for k in range(len(seeds)))
                top = max(votes.values())
                best = next(preds[k][li][pos] for k in range(len(seeds))
                            if votes[preds[k][li][pos]] == top)
                line.append(best)
            out.append(line)
        return out

    e_maj = majority("enc", list(L.ENC_SEEDS))
    m_maj = majority("mdlm", list(L.MDLM_SEEDS))

    candidates = []
    for li in range(len(gt)):
        g, el, ml = gt[li], e[li], m[li]
        if ml != g:        # MDLM must be perfectly correct on this line
            continue
        if el == g:        # encoder must be wrong on this line
            continue
        # locate encoder errors and characterise them
        gwords = L.split_words(test_in[li], g)
        ewords = L.split_words(test_in[li], el)
        err_words = []
        unseen_err = False
        structural_err = False
        for (gw, gl), (ew, eln) in zip(gwords, ewords):
            if gl != eln:
                is_unseen = gw not in train_in
                # structural if the differing label's pattern uses a structural char
                struct = False
                for a, b in zip(gl, eln):
                    if a != b:
                        if any(ch in STRUCTURAL_CHARS for ch in patt[a] + patt[b]):
                            struct = True
                err_words.append({
                    "word": gw,
                    "gt_labels": gl,
                    "enc_labels": eln,
                    "unseen": is_unseen,
                    "structural": struct,
                })
                if is_unseen:
                    unseen_err = True
                if struct:
                    structural_err = True
        n_err_pos = sum(1 for a, b in zip(g, el) if a != b)
        # prefer few-position errors (so the surface diff is interpretable) but
        # require unseen + structural for the "non-local" story
        robust = (m_maj[li] == g and e_maj[li] != g)
        candidates.append({
            "line": li,
            "n_err_pos": n_err_pos,
            "unseen_err": unseen_err,
            "structural_err": structural_err,
            "robust_under_majority": robust,
            "n_err_words": len(err_words),
            "err_words": err_words,
        })

    # rank: robust, unseen, structural, small but >0 error, single error word
    def score(c):
        return (
            c["robust_under_majority"],
            c["unseen_err"],
            c["structural_err"],
            c["n_err_words"] == 1,
            -c["n_err_pos"],          # fewer position errors = cleaner example
        )

    candidates.sort(key=score, reverse=True)

    # build human-readable example records for the top hits, de-duplicating by
    # the set of error words so the showcased cases are genuinely distinct
    # phenomena (the corpus uses overlapping sliding windows, so the same word
    # recurs across consecutive lines).
    examples = []
    seen_err_signatures = set()
    for c in candidates:
        if len(examples) >= 8:
            break
        sig = tuple(sorted(w["word"] for w in c["err_words"]))
        if sig in seen_err_signatures:
            continue
        seen_err_signatures.add(sig)
        li = c["line"]
        rec = {
            "line": li,
            "robust_under_5seed_majority": c["robust_under_majority"],
            "unseen_error_word": c["unseen_err"],
            "structural_marker_involved": c["structural_err"],
            "n_encoder_error_positions": c["n_err_pos"],
            "input_consonantal": test_in[li].strip(),
            "GT_restored": L.load_gt_restored("reduced")[li],
            "MDLM_restored": L.load_restored("mdlm", SEED, "reduced")[li],
            "encoder_restored": L.load_restored("enc", SEED, "reduced")[li],
            "GT_segmentation": restored_word_view(test_in[li], gt[li], patt),
            "MDLM_segmentation": restored_word_view(test_in[li], m[li], patt),
            "encoder_segmentation": restored_word_view(test_in[li], e[li], patt),
            "error_words_detail": c["err_words"],
        }
        examples.append(rec)

    summary = {
        "seed": SEED,
        "n_lines_mdlm_right_enc_wrong": sum(1 for c in candidates),
        "n_with_unseen_error_word": sum(1 for c in candidates if c["unseen_err"]),
        "n_with_structural_marker": sum(1 for c in candidates if c["structural_err"]),
        "n_robust_under_majority": sum(1 for c in candidates if c["robust_under_majority"]),
        "examples": examples,
    }
    with open(os.path.join(OUT, "examples.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"lines MDLM-right & enc-wrong (seed{SEED}): {summary['n_lines_mdlm_right_enc_wrong']}")
    print(f"  with unseen error word: {summary['n_with_unseen_error_word']}")
    print(f"  with structural marker: {summary['n_with_structural_marker']}")
    print(f"  robust under 5-seed majority: {summary['n_robust_under_majority']}")
    print("\nTop examples:")
    for ex in examples[:6]:
        print(f"\n--- line {ex['line']} (errpos={ex['n_encoder_error_positions']},"
              f" unseen={ex['unseen_error_word']}, struct={ex['structural_marker_involved']},"
              f" robust={ex['robust_under_5seed_majority']}) ---")
        print(f"  IN : {ex['input_consonantal']}")
        print(f"  GT : {ex['GT_restored']}")
        print(f"  MD : {ex['MDLM_restored']}")
        print(f"  EN : {ex['encoder_restored']}")
        for w in ex["error_words_detail"]:
            print(f"     word={w['word']} unseen={w['unseen']} struct={w['structural']}"
                  f" GT={w['gt_labels']} EN={w['enc_labels']}")


if __name__ == "__main__":
    main()
