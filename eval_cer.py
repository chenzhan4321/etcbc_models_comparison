#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_cer.py — 单 checkpoint 的整数预测 → CER 一条龙(口径与既有 pipeline bit-for-bit 一致)。

链路(三段都复用仓库既有、已校准的逻辑,不自创口径):
  1) 整数 .out(逐字符标签,与 .in 位置对齐)→ .out.reduced
     注入口径同 eval_two_approaches.py:对每个 (char, label),先放 char,再放 patterns[label]
     (label 0 → '' 空),最后 lstrip(' ')。
  2) .out.reduced + .in → .out.original:复用 restore_to_original.restore_line(已自带 self-test)。
  3) 逐行 Levenshtein(pred.original, GT.original)/ Σlen(GT) = corpus CER(同 cluster_F / collect_results)。

验证(--validate):重建 encoder-only s4 seed42 的 .out.original,应与既有文件逐行一致,
且对 GT=data/raw_s2_on_s2/test.out.original 给 total_distance=21681 / char_acc=0.9643855058。

用法:
  python eval_cer.py --validate
  python eval_cer.py --pred <int.out> --infile <test.in> --gt <test.out.original> [--patterns <patterns.csv>]
"""
import argparse, csv, glob, os, sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from restore_to_original import restore_line  # 复用已校准的 reduced→original 还原器

DEFAULT_PATTERNS = os.path.join(REPO, "data/raw_s2_on_s2/patterns.csv")
DEFAULT_GT = os.path.join(REPO, "data/raw_s2_on_s2/test.out.original")


def load_patterns(path):
    m = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m[int(row["label"])] = row["pattern"]
    return m


def inject(in_line, labels, patterns):
    """整数标签序列 → reduced 表层串(口径同 eval_two_approaches.py)。"""
    in_line = in_line.rstrip("\n")
    out = []
    for c, lbl in zip(in_line, labels):
        out.append(c)
        pat = patterns.get(lbl, "")
        if pat:
            out.append(pat)
    return "".join(out).lstrip(" ")


def reconstruct_original(pred_out_path, in_path, patterns):
    """int .out + .in → list[str] 的 .out.original。"""
    originals, mism = [], 0
    with open(pred_out_path, encoding="utf-8") as pf, open(in_path, encoding="utf-8") as inf:
        for pline, iline in zip(pf, inf):
            labels = [int(x) for x in pline.split()]
            # 对齐自检:标签数应 == .in 字符数(含前导空格)
            if len(labels) != len(iline.rstrip("\n")):
                mism += 1
            reduced = inject(iline, labels, patterns)
            originals.append(restore_line(reduced, iline))
    return originals, mism


def levenshtein(s1, s2):
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        cur = [i + 1]
        for j, c2 in enumerate(s2):
            cur.append(min(prev[j + 1] + 1, cur[j] + 1, prev[j] + (c1 != c2)))
        prev = cur
    return prev[-1]


def corpus_cer(pred_orig, gt_lines):
    n = min(len(pred_orig), len(gt_lines))
    dist = sum(levenshtein(pred_orig[i].strip(), gt_lines[i].strip()) for i in range(n))
    tot = sum(len(gt_lines[i].strip()) for i in range(n))
    return dist, tot, n


def read_lines(p):
    with open(p, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def validate():
    """重建 enc-only s4 seed42,逐行比对既有 .out.original + 复现 21681。"""
    enc_dir = os.path.join(REPO, "outputs",
                           "encoder_only_100ep_bs128_lr3e-4_d512_l4_h16_d0.25_5seeds", "s4_on_s2",
                           "transformer_train_20260127_233759_seed42")
    pred_out = glob.glob(os.path.join(enc_dir, "results", "transformer_predictions_*.out"))
    pred_out = [p for p in pred_out if p.endswith(".out")]
    assert pred_out, f"找不到 int .out: {enc_dir}/results"
    pred_out = pred_out[0]
    existing_orig = glob.glob(os.path.join(enc_dir, "results", "restore_and_levenshtein",
                                           "transformer_predictions_*.out.original"))
    assert existing_orig, "找不到既有 .out.original"
    existing = read_lines(existing_orig[0])

    # 试两个候选 .in(s4_on_s2 测的是 s2 测试集)
    cand_in = [os.path.join(REPO, "data/raw_s4_on_s2/test.in"),
               os.path.join(REPO, "data/raw_s2_on_s2/test.in")]
    patterns_cand = [os.path.join(REPO, "data/raw_s4_on_s2/patterns.csv"), DEFAULT_PATTERNS]
    gt = read_lines(DEFAULT_GT)

    best = None
    for inp in cand_in:
        if not os.path.exists(inp):
            continue
        for patp in patterns_cand:
            if not os.path.exists(patp):
                continue
            pats = load_patterns(patp)
            recon, mism = reconstruct_original(pred_out, inp, pats)
            same = sum(1 for a, b in zip(recon, existing) if a == b)
            dist, tot, n = corpus_cer(recon, gt)
            acc = 1 - dist / tot
            tag = f"in={os.path.basename(os.path.dirname(inp))} pat={os.path.basename(os.path.dirname(patp))}"
            print(f"[try {tag}] 逐行一致={same}/{len(existing)} 对齐失配={mism} "
                  f"total_dist={dist} char_acc={acc:.10f}")
            if best is None or same > best[0]:
                best = (same, dist, acc, tag)
    if best:
        same, dist, acc, tag = best
        ok_recon = (same == len(existing))
        ok_dist = (dist == 21681) and abs(acc - 0.9643855058322615) < 1e-9
        print(f"\n[validate] 最佳: {tag}  逐行一致={same}/{len(existing)}  total_dist={dist}")
        print("[validate] 重建 bit-identical:", "✅" if ok_recon else "❌",
              " | CER 复现 21681:", "✅" if ok_dist else "❌")
        return ok_recon and ok_dist
    print("[validate] ❌ 没有可用的 .in/patterns 候选")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--pred", help="整数预测 .out")
    ap.add_argument("--infile", help="对应 test.in")
    ap.add_argument("--gt", default=DEFAULT_GT)
    ap.add_argument("--patterns", default=DEFAULT_PATTERNS)
    ap.add_argument("--save-original", help="把重建的 .out.original 写到此路径(供 collect_results 汇总)")
    a = ap.parse_args()
    if a.validate:
        sys.exit(0 if validate() else 1)
    if not (a.pred and a.infile):
        sys.exit("需要 --pred 和 --infile(或 --validate)")
    pats = load_patterns(a.patterns)
    recon, mism = reconstruct_original(a.pred, a.infile, pats)
    if a.save_original:
        os.makedirs(os.path.dirname(os.path.abspath(a.save_original)), exist_ok=True)
        open(a.save_original, "w", encoding="utf-8").write("\n".join(recon) + "\n")
        print(f"已写 .out.original → {a.save_original}")
    gt = read_lines(a.gt)
    dist, tot, n = corpus_cer(recon, gt)
    print(f"lines={n} 对齐失配={mism} total_dist={dist} chars={tot} "
          f"CER={dist/tot*100:.4f}% char_acc={(1-dist/tot)*100:.4f}%")


if __name__ == "__main__":
    main()
