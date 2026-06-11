#!/usr/bin/env python3
"""
collect_results.py — 汇总各模型的 CER(逐行字符级 Levenshtein,口径与 cluster_F 完全一致),
出 5-seed 均值 + across-seed SD + 2000 次 bootstrap 95% CI + 配对差,生成 markdown 表,
供填回复信的 [PENDING]。

口径(对齐 analysis/cluster_F/SUMMARY.md,已 bit-for-bit 复现 levenshtein_results.json):
  - CER 算在 restored ETCBC surface 串(.out.original)上,逐行 Levenshtein(pred,gt)/len(gt),
    corpus CER = Σ dist / Σ gt_chars = 1 - char_accuracy(micro-average)。
  - GT 统一为 S2 测试集 data/raw_s2_on_s2/test.out.original(S2-on-S2 与 S4-on-S2 都测 S2)。
  - bootstrap:对 5-seed 平均的逐行距离,有放回重采样测试行 2000 次,percentile [2.5,97.5],
    固定种子 20260606。诚实 caveat:测试行为滑窗、非独立 → CI 偏窄(下界)。

用法:
  python collect_results.py --selftest        # 自检:复现已知 run 的 total_distance
  python collect_results.py                    # 汇总所有已具备 .out.original 的 run → 表
注:B1 overlay 的 train.py 只产出整数 .out;需先经仓库 restore 管线生成 .out.original
   (restore_to_original.py,自带 self-test),collector 会自动纳入已 restore 的 run。
"""
import argparse, glob, json, os, re, sys
import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
GT = os.path.join(REPO, "data/raw_s2_on_s2/test.out.original")  # 统一 S2 GT
BOOT_REPS = 2000
BOOT_SEED = 20260606

# 各 cell → 该 cell 下每个 seed 的 .out.original 预测文件 glob(已有 5-seed run)
# B1 新模型 restore 后,把对应 glob 加进来即可(键即回复信表的行名)
CELLS = {
    "encoder-decoder S2":   "outputs/encoder_decoder_30ep_*/s2_on_s2/**/*.out.original",
    "encoder-only S2":      "outputs/encoder_only_*/s2_on_s2/**/*.out.original",
    "encoder-only S4":      "outputs/encoder_only_*/s4_on_s2/**/*.out.original",
    "MDLM steps2 S2":       "outputs/mdlm_200ep_*steps2*/s2-on-s2/**/*.out.original",
    "MDLM steps2 S4":       "outputs/mdlm_200ep_*steps2*/s4-on-s2/**/*.out.original",
    "MDLM steps3 S4":       "outputs/mdlm_better_*steps3*/**/*.out.original",
    # ---- B1/B2 待 restore 后启用(占位 glob;restore 产出 .out.original 后自动命中) ----
    "BiLSTM-CRF S2":        "outputs/b1ov/bilstmcrf_s2_on_s2_*/**/*.out.original",
    "BiLSTM-CRF S4":        "outputs/b1ov/bilstmcrf_s4_on_s2_*/**/*.out.original",
    "Encoder+CRF S2":       "outputs/b1ov/enccrf_s2_on_s2_*/**/*.out.original",
    "Encoder+CRF S4":       "outputs/b1ov/enccrf_s4_on_s2_*/**/*.out.original",
    "BERT S2":              "outputs/b1ov/bert_s2_on_s2_*/**/*.out.original",
    "BERT S4":              "outputs/b1ov/bert_s4_on_s2_*/**/*.out.original",
    "enc-dec tuned S2":     "outputs/b2*/encdec_*s2*/**/*.out.original",
}

def levenshtein(s1, s2):
    """标准编辑距离(插/删/替各 1),滚动数组。与 cluster_F/repo 口径一致。"""
    if len(s1) < len(s2): s1, s2 = s2, s1
    if not s2: return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        cur = [i + 1]
        for j, c2 in enumerate(s2):
            cur.append(min(prev[j + 1] + 1, cur[j] + 1, prev[j] + (c1 != c2)))
        prev = cur
    return prev[-1]

def read_lines(p):
    with open(p, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]

def per_line_dist(pred_path, gt_lines):
    """逐行距离向量 + 每行 GT 长度。pred 与 gt 按行对齐(同为 S2 测试集 10869 行)。"""
    pred = read_lines(pred_path)
    n = min(len(pred), len(gt_lines))
    d = np.array([levenshtein(pred[i].strip(), gt_lines[i].strip()) for i in range(n)], float)
    g = np.array([len(gt_lines[i].strip()) for i in range(n)], float)
    return d, g

def cell_stats(globpat, gt_lines):
    files = sorted(glob.glob(os.path.join(REPO, globpat), recursive=True))
    if not files:
        return None
    dists, gts = [], None
    for fp in files:
        d, g = per_line_dist(fp, gt_lines)
        dists.append(d); gts = g
    L = min(len(x) for x in dists)
    D = np.stack([x[:L] for x in dists])          # [n_seed, n_line]
    g = gts[:L]
    seed_cer = D.sum(1) / g.sum() * 100           # 每 seed corpus CER %
    mean_cer = float(seed_cer.mean())
    seed_sd = float(seed_cer.std(ddof=1)) if len(seed_cer) > 1 else float("nan")
    dbar = D.mean(0)                              # 5-seed 平均逐行距离
    rng = np.random.default_rng(BOOT_SEED)
    boots = np.empty(BOOT_REPS)
    idx = np.arange(L)
    for b in range(BOOT_REPS):
        s = rng.choice(idx, L, replace=True)
        boots[b] = dbar[s].sum() / g[s].sum() * 100
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(n_seed=len(files), mean_cer=mean_cer, seed_sd=seed_sd,
                ci=(float(lo), float(hi)), dbar=dbar, g=g, files=files)

def selftest(gt_lines):
    # 复现 cluster_F 记录值:encoder-only S4-on-S2 seed42 total_distance=21681, char_acc=0.96439
    cand = sorted(glob.glob(os.path.join(REPO,
        "outputs/encoder_only_*/s4_on_s2/*seed42*/**/*.out.original"), recursive=True))
    assert cand, "找不到 encoder-only s4 seed42 的 .out.original"
    d, g = per_line_dist(cand[0], gt_lines)
    total = int(d.sum()); char_acc = 1 - d.sum() / g.sum()
    print(f"[selftest] {cand[0].split('/outputs/')[-1]}")
    print(f"[selftest] total_distance={total} (期望 21681)  char_acc={char_acc:.10f} (期望 0.9643855058)")
    ok = (total == 21681) and abs(char_acc - 0.9643855058322615) < 1e-9
    print("[selftest]", "✅ PASS — CER 口径与 cluster_F 一致" if ok else "❌ FAIL — 检查 GT 路径/行对齐")
    return ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=os.path.join(REPO, "analysis/collected_cer.md"))
    args = ap.parse_args()
    if not os.path.exists(GT):
        sys.exit(f"缺 GT 文件: {GT}")
    gt = read_lines(GT)
    if args.selftest:
        sys.exit(0 if selftest(gt) else 1)

    rows, cells = [], {}
    for name, pat in CELLS.items():
        st = cell_stats(pat, gt)
        if st is None:
            rows.append(f"| {name} | — | _(待 restore/未完成)_ | — | — |")
            continue
        cells[name] = st
        rows.append(f"| {name} | {st['n_seed']} | **{st['mean_cer']:.3f}%** | "
                    f"[{st['ci'][0]:.3f}, {st['ci'][1]:.3f}] | "
                    f"{st['seed_sd']:.3f} pp |")

    # 配对差(同 GT/对齐):MDLM steps3 − encoder-only S4;Encoder+CRF − encoder-only 等
    def paired(a, b):
        if a not in cells or b not in cells: return None
        L = min(len(cells[a]['dbar']), len(cells[b]['dbar']))
        da, db, g = cells[a]['dbar'][:L], cells[b]['dbar'][:L], cells[a]['g'][:L]
        diff_line = (da - db)
        rng = np.random.default_rng(BOOT_SEED)
        bt = np.empty(BOOT_REPS); idx = np.arange(L)
        for k in range(BOOT_REPS):
            s = rng.choice(idx, L, replace=True)
            bt[k] = diff_line[s].sum() / g[s].sum() * 100
        lo, hi = np.percentile(bt, [2.5, 97.5])
        return diff_line.sum()/g.sum()*100, lo, hi
    pairs_md = []
    for a, b in [("MDLM steps3 S4","encoder-only S4"), ("Encoder+CRF S4","encoder-only S4"),
                 ("BiLSTM-CRF S4","encoder-only S4"), ("MDLM steps2 S4","encoder-only S4")]:
        r = paired(a, b)
        if r: pairs_md.append(f"| {a} − {b} | {r[0]:+.3f} pp | [{r[1]:+.3f}, {r[2]:+.3f}] | "
                              f"{'是' if (r[1]<0)==(r[2]<0) else '否(含0)'} |")

    md = ["# Collected CER table (auto, 口径对齐 cluster_F)",
          "", "| Model × dataset | seeds | CER (5-seed mean) | 95% bootstrap CI | seed SD |",
          "|---|---|---|---|---|", *rows, "",
          "## 配对差(A−B,负=A 更优)", "| 对比 | 均值差 | 95% CI | 排除0 |",
          "|---|---|---|---|", *pairs_md, "",
          "_caveat:测试行为滑窗、非独立,行级 bootstrap CI 偏窄,按下界看。_"]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md)); print(f"\n已写 {args.out}")

if __name__ == "__main__":
    main()
