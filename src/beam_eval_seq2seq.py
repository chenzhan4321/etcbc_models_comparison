#!/usr/bin/env python3
"""beam_eval_seq2seq.py —— 用 beam search 重评已训练的 seq2seq checkpoint.
不重训：load best.pth(含 model_state + args) → 重建模型 → beam=3 解码 test →
canonical CER (Σlevenshtein/Σlen, 与 MDLM/原始 4.00 同尺).

对 {HPO-best, Martijn 配置} 用同一套 beam，内部完全可比；再对照原始 beam 4.0%。

用法:
  python beam_eval_seq2seq.py --ckpt checkpoints/<run_tag>/best.pth \
     --vocab results/<run_tag>/vocab.json \
     --test_in data/s2_seq2seq/test.in.original \
     --gt data/s2_seq2seq/test.out.original --beam 3 [--out pred.beam.txt]
"""
import argparse
import json
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from models.seq2seq import SyriacSeq2SeqModel, PAD_IDX, SOS_IDX, EOS_IDX  # noqa: E402
from train_seq2seq import encode_line, decode_ids, read_lines  # noqa: E402


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


@torch.no_grad()
def beam_decode_one(model, src_ids, device, beam_size, max_len, alpha=0.6):
    """单条序列 beam search；beam 作为 batch 维, 全程保持 beam_size 条 live 假设。"""
    src = torch.tensor([src_ids], dtype=torch.long, device=device)      # [1, L]
    src_pad = (src == PAD_IDX)                                          # [1, L]
    memory0 = model.encode(src, src_pad)                               # [1, L, E]
    L, E = memory0.size(1), memory0.size(2)
    memory = memory0.expand(beam_size, L, E).contiguous()
    mem_pad = src_pad.expand(beam_size, L).contiguous()

    seqs = torch.full((beam_size, 1), SOS_IDX, dtype=torch.long, device=device)
    scores = torch.full((beam_size,), -1e9, device=device)
    scores[0] = 0.0                                                     # 起步仅 beam0 有效
    completed = []                                                     # (norm_score, tokens)

    for _ in range(max_len - 1):
        cur_len = seqs.size(1)
        tgt_mask = torch.triu(torch.ones(cur_len, cur_len, dtype=torch.bool, device=device), diagonal=1)
        hidden = model.decode(seqs, memory, tgt_mask, None, mem_pad)    # [beam, cur_len, E]
        logp = torch.log_softmax(model.generator(hidden[:, -1, :]), dim=-1)  # [beam, V]
        V = logp.size(-1)
        total = (scores[:, None] + logp).reshape(-1)                   # [beam*V]
        topv, topi = total.topk(beam_size * 2)
        beam_idx = (topi // V)
        tok_idx = (topi % V)
        new_seqs, new_scores = [], []
        for c in range(topi.size(0)):
            b = beam_idx[c].item()
            t = tok_idx[c].item()
            seq = torch.cat([seqs[b], tok_idx[c:c + 1]])
            if t == EOS_IDX:
                completed.append((topv[c].item() / (seq.size(0) ** alpha), seq.tolist()))
            else:
                new_seqs.append(seq)
                new_scores.append(topv[c])
            if len(new_seqs) == beam_size:
                break
        if not new_seqs or len(completed) >= beam_size:
            break
        seqs = torch.stack(new_seqs)
        scores = torch.stack(new_scores)

    if completed:
        completed.sort(key=lambda x: x[0], reverse=True)
        return completed[0][1]
    bi = (scores / (seqs.size(1) ** alpha)).argmax().item()
    return seqs[bi].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--test_in", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--beam", type=int, default=3)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    a = ckpt["args"]
    with open(args.vocab, encoding="utf-8") as f:
        vj = json.load(f)
    src_vocab, tgt_vocab = vj["src_vocab"], vj["tgt_vocab"]
    tgt_idx2char = {int(i): c for c, i in tgt_vocab.items()}

    model = SyriacSeq2SeqModel(
        src_vocab_size=ckpt["src_vocab_size"], tgt_vocab_size=ckpt["tgt_vocab_size"],
        emb_size=a["emb_size"], nhead=a["nhead"],
        num_encoder_layers=a["enc_layers"], num_decoder_layers=a["dec_layers"],
        ffn_hid_dim=a["ffn"], dropout=a["dropout"], max_len=a["max_len"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    src_lines = read_lines(args.test_in)
    gt = read_lines(args.gt)
    while gt and gt[-1].strip() == "":
        gt.pop()
    while src_lines and src_lines[-1].strip() == "":
        src_lines.pop()

    preds = []
    for ln in src_lines:
        ids = encode_line(ln, src_vocab)
        if len(ids) > args.max_len:
            ids = ids[: args.max_len - 1] + [EOS_IDX]
        out_ids = beam_decode_one(model, ids, device, args.beam, args.max_len)
        preds.append(decode_ids(out_ids, tgt_idx2char))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(preds) + "\n")

    n = min(len(preds), len(gt))
    dist = sum(levenshtein(preds[i].strip(), gt[i].strip()) for i in range(n))
    tot = sum(len(gt[i].strip()) for i in range(n))
    print(f"{os.path.basename(os.path.dirname(args.ckpt))}\tbeam={args.beam}\tCER={dist/tot*100:.4f}%\tdist={dist}\tn={n}")


if __name__ == "__main__":
    main()
