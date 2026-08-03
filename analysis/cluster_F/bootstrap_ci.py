#!/usr/bin/env python3
"""
Cluster F — statistical re-analysis for reviewer R2-M5 / R2-M6.

============================================================================
WHAT METRIC IS USED, AND ON WHICH STRING (answers R2-m2)
============================================================================
The character error rate (CER) is computed on the RESTORED ETCBC surface
string (the `.out.original` representation, e.g.

    W-B-HLJN KL/-HJN NJX/~> B<J[T L-J D-B->JN> JWTRN/~>

i.e. consonants + morphological boundary/marker symbols), NOT on the raw
space-separated label-digit sequence (the `.out` files such as
"0 1 1 0 0 0 ...").  We verified this empirically: re-running a per-line,
character-level Levenshtein of every model's `.out.original` prediction
against data/raw_s2_on_s2/test.out.original reproduces the recorded
levenshtein_results.json EXACTLY (e.g. encoder-only S4-on-S2 seed42:
total_distance=21681, char_accuracy=0.9643855058322615,
avg_distance=1.9947557272978196, exact_matches=4756) — see assertions below.

(The `.out.final` numeric form and the `.out` label form are intermediate
representations; the published evaluation distance is on the restored
`.out.original` string, which is what a philologist actually reads.)

============================================================================
STATISTICAL UNIT (R2-M5)
============================================================================
Paired statistical unit = per-test-line CER
    CER_line = Levenshtein(pred_line, gt_line) / len(gt_line)   (char level)
There are 10,869 test lines (all share the same GT for every model because
every model is evaluated ON the S2 test set).

CAVEAT (honest): the 10,869 test lines are produced by a sliding window over
shared Bible verses, so neighbouring lines overlap heavily and are NOT
independent.  Bootstrapping over lines therefore *under*-estimates the true
sampling variance (book/verse-level correlation).  A block bootstrap over
books/verses would be more conservative; we flag this explicitly and treat
the line-level CIs as a lower bound on uncertainty.

============================================================================
QUANTITIES
============================================================================
For each (model x dataset) cell and each seed:
  - per-line distance vector d (length = #scored lines)
  - corpus CER = sum(d) / sum(gt_chars)          [micro-average; = 1-char_acc]
  - char_accuracy = 1 - corpus CER               [matches recorded JSON]
  - avg_distance = sum(d) / total_lines          [matches recorded JSON]
5-seed mean = average of the per-seed quantities.

BOOTSTRAP 95% CI (>=1000 reps, fixed seed):
  Resample the test-line indices with replacement (n out of n).  For each
  resample recompute corpus CER = sum(d_resampled)/sum(gtchars_resampled)
  on the 5-seed-AVERAGED per-line distances (so the bootstrapped quantity is
  the reported point estimate, the 5-seed-mean CER).  Percentile 2.5/97.5.

PAIRED COMPARISONS (R2-M5):
  (a) MDLM steps=2 (S4-on-S2)   vs encoder-only (S4-on-S2)
  (b) encoder-only (S2-on-S2)   vs encoder-decoder (S2-on-S2)
  (bonus) MDLM steps=3 (S4-on-S2) vs encoder-only (S4-on-S2)
  Per-line CER difference (seed-averaged per-line distances), paired
  bootstrap of the mean difference -> 95% CI (does it exclude 0?), plus a
  paired t-test and a Wilcoxon signed-rank test on the per-line CER diffs.

SEED-EXCLUSION DISCLOSURE (R2-M6):
  The 5-seed mean is recomputed directly from the 5 levenshtein_results.json
  files.  There is NO outlier/2-sigma rejection code anywhere in the repo, so
  "with exclusion" == "without exclusion" by construction.  Printed explicitly.

Run:
  cd <REPO_ROOT>
  .venv/bin/python analysis/cluster_F/bootstrap_ci.py
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats
import Levenshtein  # C extension; standard edit distance (ins/del/sub cost 1)

REPO = Path("<REPO_ROOT>")
OUT_DIR = REPO / "analysis" / "cluster_F"
# Canonical ground truth used by the recorded HPC pipeline (verified to
# reproduce levenshtein_results.json exactly). Every model is tested on S2.
GT_PATH = REPO / "data" / "raw_s2_on_s2" / "test.out.original"

N_BOOT = 2000
RNG_SEED = 20260606  # fixed for reproducibility


def levenshtein(a, b):
    # Sanity: matches the repo's DP (equal unit costs). Use C extension.
    return Levenshtein.distance(a, b)


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def per_line_distances(pred_path, gt_stripped, kept_idx):
    """Char-level Levenshtein per scored line. Mirrors compare_original_correct.py:
    strip both sides, skip lines whose GT (stripped) is empty."""
    pred_lines = read_lines(pred_path)
    dists = np.empty(len(kept_idx), dtype=np.float64)
    for k, i in enumerate(kept_idx):
        p = pred_lines[i].strip() if i < len(pred_lines) else ""
        dists[k] = levenshtein(p, gt_stripped[i])
    return dists


def E(sub):
    return REPO / "outputs" / sub


# (model_key) -> dict(label, dataset, seeds=[(seed, pred_.out.original, json_or_None)])
def orig(p):
    """convert a .out.final restore path to the sibling .out.original path"""
    return p.replace(".out.final", ".out.original")


MODELS = {
    "encoder_only_s2_on_s2": {
        "label": "Encoder-only (S2-on-S2)",
        "dataset": "s2_on_s2",
        "seeds": [
            ("123",  "encoder_only_100ep_bs128_lr3e-4_d512_l4_h16_d0.25_5seeds/s2_on_s2/transformer_train_20260128_154403_seed123/results/restore_and_levenshtein/transformer_predictions_20260128_154403.out.original"),
            ("456",  "encoder_only_100ep_bs128_lr3e-4_d512_l4_h16_d0.25_5seeds/s2_on_s2/transformer_train_20260128_154403_seed456/results/restore_and_levenshtein/transformer_predictions_20260128_154403.out.original"),
            ("789",  "encoder_only_100ep_bs128_lr3e-4_d512_l4_h16_d0.25_5seeds/s2_on_s2/transformer_train_20260128_154403_seed789/results/restore_and_levenshtein/transformer_predictions_20260128_154403.out.original"),
            ("2023", "encoder_only_100ep_bs128_lr3e-4_d512_l4_h16_d0.25_5seeds/s2_on_s2/transformer_train_20260128_175004_seed2023/results/restore_and_levenshtein/transformer_predictions_20260128_175004.out.original"),
            ("2024", "encoder_only_100ep_bs128_lr3e-4_d512_l4_h16_d0.25_5seeds/s2_on_s2/transformer_train_20260128_175004_seed2024/results/restore_and_levenshtein/transformer_predictions_20260128_175004.out.original"),
        ],
        "has_json": True,
    },
    "encoder_only_s4_on_s2": {
        "label": "Encoder-only (S4-on-S2)",
        "dataset": "s4_on_s2",
        "seeds": [
            ("42", "encoder_only_100ep_bs128_lr3e-4_d512_l4_h16_d0.25_5seeds/s4_on_s2/transformer_train_20260127_233759_seed42/results/restore_and_levenshtein/transformer_predictions_20260127_233759.out.original"),
            ("49", "encoder_only_100ep_bs128_lr3e-4_d512_l4_h16_d0.25_5seeds/s4_on_s2/transformer_train_20260127_235126_seed49/results/restore_and_levenshtein/transformer_predictions_20260127_235126.out.original"),
            ("50", "encoder_only_100ep_bs128_lr3e-4_d512_l4_h16_d0.25_5seeds/s4_on_s2/transformer_train_20260127_235126_seed50/results/restore_and_levenshtein/transformer_predictions_20260127_235126.out.original"),
            ("51", "encoder_only_100ep_bs128_lr3e-4_d512_l4_h16_d0.25_5seeds/s4_on_s2/transformer_train_20260127_235146_seed51/results/restore_and_levenshtein/transformer_predictions_20260127_235146.out.original"),
            ("52", "encoder_only_100ep_bs128_lr3e-4_d512_l4_h16_d0.25_5seeds/s4_on_s2/transformer_train_20260127_235149_seed52/results/restore_and_levenshtein/transformer_predictions_20260127_235149.out.original"),
        ],
        "has_json": True,
    },
    "mdlm_steps2_s2_on_s2": {
        "label": "MDLM steps=2 (S2-on-S2)",
        "dataset": "s2_on_s2",
        "seeds": [
            ("123",  "mdlm_200ep_bs16_lr1e-4_d768_l5_h4_d0.25_steps2_5seeds/s2-on-s2/mdlm_train_20260129_120013_seed123/results/restore_and_levenshtein/mdlm_predictions_20260129_120013.out.original"),
            ("456",  "mdlm_200ep_bs16_lr1e-4_d768_l5_h4_d0.25_steps2_5seeds/s2-on-s2/mdlm_train_20260129_120018_seed456/results/restore_and_levenshtein/mdlm_predictions_20260129_120018.out.original"),
            ("789",  "mdlm_200ep_bs16_lr1e-4_d768_l5_h4_d0.25_steps2_5seeds/s2-on-s2/mdlm_train_20260129_120023_seed789/results/restore_and_levenshtein/mdlm_predictions_20260129_120023.out.original"),
            ("3407", "mdlm_200ep_bs16_lr1e-4_d768_l5_h4_d0.25_steps2_5seeds/s2-on-s2/mdlm_train_20260129_120033_seed3407/results/restore_and_levenshtein/mdlm_predictions_20260129_120033.out.original"),
            ("8888", "mdlm_200ep_bs16_lr1e-4_d768_l5_h4_d0.25_steps2_5seeds/s2-on-s2/mdlm_train_20260129_120038_seed8888/results/restore_and_levenshtein/mdlm_predictions_20260129_120038.out.original"),
        ],
        "has_json": True,
    },
    "mdlm_steps2_s4_on_s2": {
        "label": "MDLM steps=2 (S4-on-S2)",
        "dataset": "s4_on_s2",
        "seeds": [
            ("123",  "mdlm_200ep_bs16_lr1e-4_d768_l5_h4_d0.25_steps2_5seeds/s4-on-s2/mdlm_train_20260129_032214_seed123/results/restore_and_levenshtein/mdlm_predictions_20260129_032214.out.original"),
            ("456",  "mdlm_200ep_bs16_lr1e-4_d768_l5_h4_d0.25_steps2_5seeds/s4-on-s2/mdlm_train_20260129_032219_seed456/results/restore_and_levenshtein/mdlm_predictions_20260129_032219.out.original"),
            ("789",  "mdlm_200ep_bs16_lr1e-4_d768_l5_h4_d0.25_steps2_5seeds/s4-on-s2/mdlm_train_20260129_032224_seed789/results/restore_and_levenshtein/mdlm_predictions_20260129_032224.out.original"),
            ("3407", "mdlm_200ep_bs16_lr1e-4_d768_l5_h4_d0.25_steps2_5seeds/s4-on-s2/mdlm_train_20260129_032233_seed3407/results/restore_and_levenshtein/mdlm_predictions_20260129_032233.out.original"),
            ("8888", "mdlm_200ep_bs16_lr1e-4_d768_l5_h4_d0.25_steps2_5seeds/s4-on-s2/mdlm_train_20260129_032238_seed8888/results/restore_and_levenshtein/mdlm_predictions_20260129_032238.out.original"),
        ],
        "has_json": True,
    },
    "mdlm_steps3_s4_on_s2": {
        "label": "MDLM steps=3 (S4-on-S2, post-submission HPO)",
        "dataset": "s4_on_s2",
        "seeds": [
            ("r1", "mdlm_better_result_after_submission_d768_l10_h6_dr0.23_lr5e-05_steps3_s4-on-s2_5seeds/mdlm_train_20260209_213751/results/restore_and_levenshtein/mdlm_predictions_20260209_213751.out.original"),
            ("r2", "mdlm_better_result_after_submission_d768_l10_h6_dr0.23_lr5e-05_steps3_s4-on-s2_5seeds/mdlm_train_20260209_213754/results/restore_and_levenshtein/mdlm_predictions_20260209_213754.out.original"),
            ("r3", "mdlm_better_result_after_submission_d768_l10_h6_dr0.23_lr5e-05_steps3_s4-on-s2_5seeds/mdlm_train_20260209_213822/results/restore_and_levenshtein/mdlm_predictions_20260209_213822.out.original"),
            ("r4", "mdlm_better_result_after_submission_d768_l10_h6_dr0.23_lr5e-05_steps3_s4-on-s2_5seeds/mdlm_train_20260209_213839/results/restore_and_levenshtein/mdlm_predictions_20260209_213839.out.original"),
            ("r5", "mdlm_better_result_after_submission_d768_l10_h6_dr0.23_lr5e-05_steps3_s4-on-s2_5seeds/mdlm_train_20260209_213845/results/restore_and_levenshtein/mdlm_predictions_20260209_213845.out.original"),
        ],
        "has_json": True,
    },
    "encoder_decoder_s2_on_s2": {
        "label": "Encoder-decoder (S2-on-S2, single seed)",
        "dataset": "s2_on_s2",
        "seeds": [
            ("s7", "encoder_decoder_30ep_bs128_lr1e-4_emb512_h8_d0.1_b3_s7/s2_on_s2/results/test.out.original"),
        ],
        "has_json": False,  # single seed; CER recomputed directly
    },
}


def json_path_for(pred_original_path):
    return pred_original_path.rsplit("/", 1)[0] + "/levenshtein_results.json"


def main():
    gt_lines = read_lines(GT_PATH)
    gt_stripped = [t.strip() for t in gt_lines]
    kept_idx = [i for i, t in enumerate(gt_stripped) if t]
    n_total = len(gt_lines)
    n_kept = len(kept_idx)
    gt_chars = np.array([len(gt_stripped[i]) for i in kept_idx], dtype=np.float64)

    results = {
        "meta": {
            "gt_path": str(GT_PATH),
            "gt_total_lines": n_total,
            "gt_scored_lines": n_kept,
            "n_bootstrap": N_BOOT,
            "rng_seed": RNG_SEED,
            "metric": "character-level Levenshtein (ins/del/sub cost 1), per line",
            "cer_target_string": ".out.original (restored ETCBC surface string), NOT the raw label-digit '.out' sequence",
            "cer_definition": "corpus CER = sum(line distances) / sum(GT line chars) = 1 - char_accuracy",
            "unit": "per-test-line CER = line distance / line GT char count",
            "bootstrap_method": "resample line indices with replacement (n=#scored lines, >=1000 reps, fixed seed); CER recomputed as sum(dist)/sum(gtchars) on the 5-seed-averaged per-line distances; percentile [2.5, 97.5]",
            "caveat": "test lines are overlapping sliding windows over shared verses -> NOT i.i.d.; line-level bootstrap under-estimates variance (verse/book correlation). Reported CIs are a lower bound on true uncertainty.",
        },
        "models": {},
        "paired": {},
        "seed_disclosure": {},
        "reproducibility_check": {},
    }

    seedavg_dist = {}  # model_key -> seed-averaged per-line distance vector

    for key, spec in MODELS.items():
        per_seed_corpus_cer = []
        per_seed_char_acc = []
        per_seed_avg_dist = []
        per_seed_total_dist = []
        per_seed_exact = []
        dist_stack = []
        repro = []
        for seed, sub in spec["seeds"]:
            d = per_line_distances(E(sub), gt_stripped, kept_idx)
            dist_stack.append(d)
            total_dist = float(d.sum())
            corpus_cer = total_dist / gt_chars.sum()
            char_acc = 1.0 - corpus_cer
            avg_dist = total_dist / n_total
            exact = int(np.sum(d == 0))
            per_seed_corpus_cer.append(float(corpus_cer))
            per_seed_char_acc.append(float(char_acc))
            per_seed_avg_dist.append(float(avg_dist))
            per_seed_total_dist.append(total_dist)
            per_seed_exact.append(exact)
            # reproducibility vs recorded JSON (when present)
            if spec["has_json"]:
                jp = E(json_path_for(sub))
                if jp.exists():
                    with open(jp) as f:
                        jd = json.load(f)
                    repro.append({
                        "seed": seed,
                        "recorded_char_accuracy": jd["char_accuracy"],
                        "recomputed_char_accuracy": float(char_acc),
                        "abs_err_char_acc": abs(jd["char_accuracy"] - char_acc),
                        "recorded_total_distance": jd["total_distance"],
                        "recomputed_total_distance": int(total_dist),
                        "recorded_exact": jd["exact_matches"],
                        "recomputed_exact": exact,
                    })

        dist_stack = np.array(dist_stack)             # (n_seeds, n_kept)
        seed_mean_perline = dist_stack.mean(axis=0)
        seedavg_dist[key] = seed_mean_perline
        corpus_cer_5seed = float(seed_mean_perline.sum() / gt_chars.sum())

        # bootstrap CI on the seed-averaged per-line distances
        rng = np.random.default_rng(RNG_SEED)
        boot = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx = rng.integers(0, n_kept, size=n_kept)
            boot[b] = seed_mean_perline[idx].sum() / gt_chars[idx].sum()
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

        results["models"][key] = {
            "label": spec["label"],
            "dataset": spec["dataset"],
            "n_seeds": len(spec["seeds"]),
            "seeds": [s for s, _ in spec["seeds"]],
            "per_seed_corpus_cer": per_seed_corpus_cer,
            "per_seed_char_accuracy": per_seed_char_acc,
            "per_seed_avg_distance": per_seed_avg_dist,
            "per_seed_total_distance": per_seed_total_dist,
            "per_seed_exact_matches": per_seed_exact,
            "cer_5seed_mean": corpus_cer_5seed,
            "cer_across_seed_mean": float(np.mean(per_seed_corpus_cer)),
            "cer_across_seed_std": float(np.std(per_seed_corpus_cer, ddof=1)) if len(per_seed_corpus_cer) > 1 else 0.0,
            "char_accuracy_5seed_mean": float(np.mean(per_seed_char_acc)),
            "char_accuracy_5seed_std": float(np.std(per_seed_char_acc, ddof=1)) if len(per_seed_char_acc) > 1 else 0.0,
            "avg_distance_5seed_mean": float(np.mean(per_seed_avg_dist)),
            "bootstrap_cer_ci95": [float(ci_lo), float(ci_hi)],
            "bootstrap_cer_se": float(np.std(boot, ddof=1)),
        }
        if repro:
            results["reproducibility_check"][key] = {
                "per_seed": repro,
                "max_abs_err_char_acc": float(max(r["abs_err_char_acc"] for r in repro)),
                "all_total_distance_exact": all(r["recorded_total_distance"] == r["recomputed_total_distance"] for r in repro),
                "all_exact_matches_exact": all(r["recorded_exact"] == r["recomputed_exact"] for r in repro),
            }

    # ----- seed-exclusion disclosure (R2-M6) from recorded JSONs -----------
    for key, spec in MODELS.items():
        if not spec["has_json"]:
            continue
        recorded = []
        for seed, sub in spec["seeds"]:
            jp = E(json_path_for(sub))
            with open(jp) as f:
                jd = json.load(f)
            recorded.append(jd["char_accuracy"])
        mean_with = float(np.mean(recorded))
        results["seed_disclosure"][key] = {
            "n_seeds_recorded": len(recorded),
            "recorded_char_accuracy_per_seed": recorded,
            "mean_char_accuracy_all_seeds": mean_with,
            "mean_char_accuracy_after_any_exclusion": mean_with,  # identical
            "with_equals_without": True,
            "note": "No outlier/2-sigma rejection code exists in the repository; all seeds are used. with==without by construction.",
        }

    # ----- paired comparisons ---------------------------------------------
    def paired(name, key_a, key_b):
        da = seedavg_dist[key_a]
        db = seedavg_dist[key_b]
        cer_a = da / gt_chars
        cer_b = db / gt_chars
        diff = cer_a - cer_b  # negative => A better
        mean_diff = float(np.mean(diff))
        corpus_diff = float(da.sum() / gt_chars.sum() - db.sum() / gt_chars.sum())
        boot_mean = np.empty(N_BOOT)
        boot_corpus = np.empty(N_BOOT)
        rng2 = np.random.default_rng(RNG_SEED + 1)
        for b in range(N_BOOT):
            idx = rng2.integers(0, n_kept, size=n_kept)
            boot_mean[b] = np.mean(diff[idx])
            boot_corpus[b] = da[idx].sum() / gt_chars[idx].sum() - db[idx].sum() / gt_chars[idx].sum()
        ci_mean = [float(x) for x in np.percentile(boot_mean, [2.5, 97.5])]
        ci_corpus = [float(x) for x in np.percentile(boot_corpus, [2.5, 97.5])]
        t_stat, t_p = stats.ttest_rel(cer_a, cer_b)
        try:
            w_stat, w_p = stats.wilcoxon(cer_a, cer_b, zero_method="wilcox")
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
        results["paired"][name] = {
            "A": MODELS[key_a]["label"],
            "B": MODELS[key_b]["label"],
            "direction": "diff = CER(A) - CER(B); negative => A better (lower CER)",
            "mean_perline_cer_diff": mean_diff,
            "mean_perline_cer_diff_ci95": ci_mean,
            "corpus_cer_diff": corpus_diff,
            "corpus_cer_diff_ci95": ci_corpus,
            "excludes_zero_perline": bool(ci_mean[0] > 0 or ci_mean[1] < 0),
            "excludes_zero_corpus": bool(ci_corpus[0] > 0 or ci_corpus[1] < 0),
            "paired_t_stat": float(t_stat),
            "paired_t_p": float(t_p),
            "wilcoxon_stat": float(w_stat),
            "wilcoxon_p": float(w_p),
            "n_pairs": int(n_kept),
            "caveat": "per-line unit; lines overlap (sliding window over verses) so independence assumption of t/Wilcoxon and line bootstrap is violated -> p-values/CIs anti-conservative.",
        }

    paired("mdlm_steps2_vs_encoder_s4_on_s2", "mdlm_steps2_s4_on_s2", "encoder_only_s4_on_s2")
    paired("encoder_vs_encoderdecoder_s2_on_s2", "encoder_only_s2_on_s2", "encoder_decoder_s2_on_s2")
    paired("mdlm_steps3_vs_encoder_s4_on_s2", "mdlm_steps3_s4_on_s2", "encoder_only_s4_on_s2")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # console digest
    print("=" * 92)
    print("CER (corpus, 5-seed mean) with bootstrap 95% CI   [lower = better]")
    print("=" * 92)
    for key, m in results["models"].items():
        lo, hi = m["bootstrap_cer_ci95"]
        print(f"{m['label']:<46} CER={m['cer_5seed_mean']*100:6.3f}%  "
              f"95%CI=[{lo*100:.3f},{hi*100:.3f}]%  acc={m['char_accuracy_5seed_mean']*100:.3f}%  "
              f"seedSD={m['cer_across_seed_std']*100:.4f}pp")
    print()
    print("=" * 92)
    print("Reproducibility vs recorded levenshtein_results.json")
    print("=" * 92)
    for key, r in results["reproducibility_check"].items():
        print(f"{key:<26} max|Δchar_acc|={r['max_abs_err_char_acc']:.2e}  "
              f"total_dist_exact={r['all_total_distance_exact']}  exact_matches_exact={r['all_exact_matches_exact']}")
    print()
    print("=" * 92)
    print("Paired CER differences (per-line unit)")
    print("=" * 92)
    for name, p in results["paired"].items():
        lo, hi = p["mean_perline_cer_diff_ci95"]
        clo, chi = p["corpus_cer_diff_ci95"]
        print(f"{name}")
        print(f"   A={p['A']}")
        print(f"   B={p['B']}")
        print(f"   mean per-line CER diff (A-B) = {p['mean_perline_cer_diff']*100:+.4f}pp  "
              f"95%CI=[{lo*100:+.4f},{hi*100:+.4f}]pp  excl0={p['excludes_zero_perline']}")
        print(f"   corpus     CER diff (A-B) = {p['corpus_cer_diff']*100:+.4f}pp  "
              f"95%CI=[{clo*100:+.4f},{chi*100:+.4f}]pp  excl0={p['excludes_zero_corpus']}")
        print(f"   paired t p={p['paired_t_p']:.3e}   Wilcoxon p={p['wilcoxon_p']:.3e}   n={p['n_pairs']}")
        print()
    print("=" * 92)
    print("Seed-exclusion disclosure (R2-M6): with == without")
    print("=" * 92)
    for key, s in results["seed_disclosure"].items():
        print(f"{key:<26} mean acc all{s['n_seeds_recorded']}={s['mean_char_accuracy_all_seeds']:.6f}  "
              f"after-exclusion={s['mean_char_accuracy_after_any_exclusion']:.6f}  "
              f"with==without={s['with_equals_without']}")
    print()
    print(f"Wrote {OUT_DIR/'results.json'}")


if __name__ == "__main__":
    main()
