#!/usr/bin/env python3
"""canonical CER for seq2seq predictions —— 自包含, 与 eval_cer/MDLM 同口径.
CER = Σ levenshtein(pred_i.strip(), gt_i.strip()) / Σ len(gt_i.strip())  (corpus micro).
seq2seq 直接输出 .out.original 表层串, 无需 restore.

用法: python eval_seq2seq_cer.py --gt <GT文件> PRED1 [PRED2 ...]
"""
import argparse


def levenshtein(a, b):
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        ca = a[i - 1]
        cur = [i]
        for j in range(1, lb + 1):
            cost = 0 if ca == b[j - 1] else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[lb]


def read_lines(p):
    with open(p, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="GT surface 文件")
    ap.add_argument("preds", nargs="+")
    args = ap.parse_args()

    gt = read_lines(args.gt)
    while gt and gt[-1].strip() == "":
        gt.pop()
    print(f"# GT={args.gt}  ({len(gt)} lines)")
    vals = []
    for p in args.preds:
        pred = read_lines(p)
        while pred and pred[-1].strip() == "":
            pred.pop()
        if len(pred) != len(gt):
            print(f"{p}\tLEN_MISMATCH pred={len(pred)} gt={len(gt)}")
            continue
        dist = sum(levenshtein(pred[i].strip(), gt[i].strip()) for i in range(len(gt)))
        tot = sum(len(gt[i].strip()) for i in range(len(gt)))
        cer = dist / tot * 100.0
        vals.append(cer)
        print(f"{p.split('/')[-2]}\tCER={cer:.4f}%\tdist={dist}\ttot={tot}")
    if vals:
        m = sum(vals) / len(vals)
        sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
        print(f"# MEAN over {len(vals)}: {m:.4f}%  (SD {sd:.4f})")


if __name__ == "__main__":
    main()
