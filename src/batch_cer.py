#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_cer.py — 把一堆整数预测 .out(命名 <model>_<dataset>_s<seed>_<extra>.out)批量算 CER,
按 cell 聚合 5-seed 均值 + across-seed SD + 2000 次 bootstrap 95% CI,出 markdown 表。
复用 eval_cer 的重建链路(注入 patterns → restore_to_original → 逐行 Levenshtein),口径同 cluster_F。

用法: python batch_cer.py <preds_dir>   (默认 ./cer_preds)
"""
import glob, os, re, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from eval_cer import load_patterns, reconstruct_original, levenshtein, read_lines, DEFAULT_GT, DEFAULT_PATTERNS

IN = os.path.join(REPO, "data/raw_s2_on_s2/test.in")   # 所有模型都测 S2 测试集
BOOT_REPS, BOOT_SEED = 2000, 20260606


def cer_vectors(pred_path, in_path, patterns, gt_lines):
    recon, _ = reconstruct_original(pred_path, in_path, patterns)
    n = min(len(recon), len(gt_lines))
    d = np.array([levenshtein(recon[i].strip(), gt_lines[i].strip()) for i in range(n)], float)
    g = np.array([len(gt_lines[i].strip()) for i in range(n)], float)
    return d, g


def parse_tag(tag):
    # <model>_<sX_on_s2>_s<seed>_<extra>
    m = re.match(r'(.+?)_(s\d_on_s2)_s(\d+)_(.+)$', tag)
    if not m:
        return None
    model, ds, seed, extra = m.groups()
    if extra.startswith('T'):
        cell = f"MDLM {extra} {ds}"
    elif extra == 'crf':
        name = {'transformer': 'Encoder+CRF', 'lstm': 'BiLSTM-CRF'}.get(model, model)
        cell = f"{name} {ds}"
    else:
        name = {'bert': 'BERT'}.get(model, model)
        cell = f"{name} {ds}"
    return cell, seed


def main():
    preds_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "cer_preds")
    gt = read_lines(DEFAULT_GT)
    patterns = load_patterns(DEFAULT_PATTERNS)
    files = sorted(glob.glob(os.path.join(preds_dir, "*.out")))
    if not files:
        sys.exit(f"无预测文件: {preds_dir}")

    cells = {}
    for p in files:
        tag = os.path.basename(p)[:-4]
        pr = parse_tag(tag)
        if not pr:
            print(f"[skip 无法解析] {tag}"); continue
        cell, seed = pr
        d, g = cer_vectors(p, IN, patterns, gt)
        cells.setdefault(cell, []).append((seed, d, g))

    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    cell_dbar = {}
    for cell in sorted(cells):
        lst = cells[cell]
        L = min(len(d) for _, d, _ in lst)
        D = np.stack([d[:L] for _, d, _ in lst])         # [n_seed, n_line]
        g = lst[0][2][:L]
        seed_cer = D.sum(1) / g.sum() * 100
        mean = float(seed_cer.mean())
        sd = float(seed_cer.std(ddof=1)) if len(seed_cer) > 1 else float('nan')
        dbar = D.mean(0); cell_dbar[cell] = (dbar, g)
        idx = np.arange(L)
        boots = np.empty(BOOT_REPS)
        for b in range(BOOT_REPS):
            s = rng.choice(idx, L, replace=True)
            boots[b] = dbar[s].sum() / g[s].sum() * 100
        lo, hi = np.percentile(boots, [2.5, 97.5])
        seeds = ",".join(sorted(s for s, _, _ in lst))
        rows.append((cell, len(lst), seeds, mean, sd, lo, hi))

    print("\n# Batch CER (口径同 cluster_F/eval_cer;测试集 = S2 test 10869 行)\n")
    print("| Cell | n_seed | seeds | CER mean | across-seed SD | 95% bootstrap CI |")
    print("|---|---|---|---|---|---|")
    for cell, n, seeds, mean, sd, lo, hi in rows:
        sds = f"{sd:.3f}pp" if sd == sd else "—"
        print(f"| {cell} | {n} | {seeds} | **{mean:.3f}%** | {sds} | [{lo:.3f}, {hi:.3f}] |")

    out = os.path.join(REPO, "analysis", "batch_cer_table.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("# Batch CER\n\n| Cell | n_seed | seeds | CER mean | SD | 95% CI |\n|---|---|---|---|---|---|\n")
        for cell, n, seeds, mean, sd, lo, hi in rows:
            sds = f"{sd:.3f}pp" if sd == sd else "—"
            f.write(f"| {cell} | {n} | {seeds} | {mean:.3f}% | {sds} | [{lo:.3f}, {hi:.3f}] |\n")
    print(f"\n已写 {out}")


if __name__ == "__main__":
    main()
