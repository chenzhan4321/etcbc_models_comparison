#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用最新完整 5-seed(outputs/models_1_revision)重算所有 baseline 的 canonical CER
+ 2000-rep percentile bootstrap 95% CI + across-seed SD,口径与 cluster_F 完全一致:
  label .out --reconstruct_original--> .out.original surface 串
  --> 逐行 char-level Levenshtein vs data/raw_s2_on_s2/test.out.original
  --> corpus CER = Σdist/Σgt;5-seed 平均逐行距离上做 bootstrap(RNG 20260606)。
内置自校验:enc-only(预测已是 .out.original)应复现 cluster_F S2=3.876 / S4=3.528。
"""
import sys, os, glob
import numpy as np
ROOT = "<REPO_ROOT>"
sys.path.insert(0, ROOT)
from eval_cer import load_patterns, reconstruct_original, levenshtein, read_lines

IN = os.path.join(ROOT, "data/raw_s2_on_s2/test.in")
GT = read_lines(os.path.join(ROOT, "data/raw_s2_on_s2/test.out.original"))
PAT = load_patterns(os.path.join(ROOT, "data/raw_s2_on_s2/patterns.csv"))
BOOT_REPS, BOOT_SEED = 2000, 20260606
MD = os.path.join(ROOT, "outputs/models_1_revision")
GLEN = np.array([len(g.strip()) for g in GT], float)


def from_labels(pred_path):
    recon, mism = reconstruct_original(pred_path, IN, PAT)
    n = min(len(recon), len(GT))
    d = np.array([levenshtein(recon[i].strip(), GT[i].strip()) for i in range(n)], float)
    return d, GLEN[:n], mism


def from_original(orig_path):
    recon = read_lines(orig_path)
    n = min(len(recon), len(GT))
    d = np.array([levenshtein(recon[i].strip(), GT[i].strip()) for i in range(n)], float)
    return d, GLEN[:n], 0


def summ(label, Ds):
    L0 = min(len(x) for x in Ds)
    D = np.stack([x[:L0] for x in Ds]); g = GLEN[:L0]
    seed_cer = D.sum(1) / g.sum() * 100
    mean = float(seed_cer.mean())
    sd = float(seed_cer.std(ddof=1)) if len(Ds) > 1 else float('nan')
    dbar = D.mean(0); rng = np.random.default_rng(BOOT_SEED); idx = np.arange(L0)
    bs = np.empty(BOOT_REPS)
    for b in range(BOOT_REPS):
        sm = rng.choice(idx, L0, replace=True); bs[b] = dbar[sm].sum() / g[sm].sum() * 100
    lo, hi = np.percentile(bs, [2.5, 97.5])
    sds = f"{sd:.3f}" if sd == sd else "n/a"
    per = " ".join(f"{c:.3f}" for c in seed_cer)
    print(f"| {label} | {len(Ds)} | **{mean:.3f}%** | {sds} | [{lo:.3f}, {hi:.3f}] | {per} |")


print("=== 自校验 enc-only(.out.original,应复现 cluster_F 3.876 / 3.528)===")
print("| cell | n | CER mean | SD pp | 95% CI | per-seed |")
print("|---|---|---|---|---|---|")
for label, ds in [("enc-only S2", "s2_on_s2"), ("enc-only S4", "s4_on_s2")]:
    Ds = []
    for dd in sorted(glob.glob(f"{MD}/encoder_only/{ds}_seed*")):
        op = glob.glob(f"{dd}/*.out.original")
        if op:
            d, g, _ = from_original(op[0]); Ds.append(d)
    if Ds:
        summ(label, Ds)

print("\n=== baselines 完整 5-seed {42,49,50,51,52}(label→reconstruct)===")
print("| cell | n | CER mean | SD pp | 95% CI | per-seed |")
print("|---|---|---|---|---|---|")
SEEDS = [42, 49, 50, 51, 52]
for label, sub in [("BiLSTM-CRF", "bilstm_crf"), ("Encoder+CRF", "encoder_crf"), ("BERT", "bert")]:
    for dsl, ds in [("S2", "s2_on_s2"), ("S4", "s4_on_s2")]:
        Ds, mref = [], 0
        for s in SEEDS:
            pp = [p for p in glob.glob(f"{MD}/{sub}/{ds}_seed{s}/*predictions*.out")
                  if p.endswith(".out") and ".original" not in p]
            if pp:
                d, g, mism = from_labels(pp[0]); Ds.append(d); mref = max(mref, mism)
        if Ds:
            summ(f"{label} {dsl}", Ds)
            if mref:
                print(f"   (warn: {label} {dsl} 对齐失配行数 max={mref})")
