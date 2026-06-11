#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_seq2seq_constrained.py — 对一个训练好的 seq2seq checkpoint,跑「不约束」与「约束」两种
解码,各算 CER。回应 R2-M2 / R2-M4(constrained-decoding encoder-decoder),并支撑
「约束后隔离离散化贡献」的三段分解。

约束解码逻辑见 models/seq2seq.py:generate_constrained(强制输出骨架字符=源端辅音+空格、
顺序一致,只在形态标记上自由,EOS 仅当骨架走完才允许)。

用法:
  python eval_seq2seq_constrained.py --ckpt checkpoints/seq2seq_s2_seq2seq_emb512_h8_seed42/best.pth \
      --vocab results/seq2seq_s2_seq2seq_emb512_h8_seed42/vocab.json --data_subdir s2_seq2seq
"""
import argparse, json, os, sys
import torch
from torch.utils.data import DataLoader

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from train_seq2seq import read_lines, decode_ids, compute_cer, collate_batch, Seq2SeqDataset
from models.seq2seq import SyriacSeq2SeqModel

SPECIALS = {"PAD", "SOS", "EOS", "UNK"}


def build_maps(src_vocab, tgt_vocab):
    """marker_mask[tgt_id]=True 表示自由形态标记(字符只在 tgt 出现);
       src_to_tgt[src_id]=同字符的 tgt_id(骨架字符),特殊/无对应= -1。"""
    src_real = {k for k in src_vocab if k not in SPECIALS}
    marker = torch.zeros(len(tgt_vocab), dtype=torch.bool)
    for ch, tid in tgt_vocab.items():
        if ch in SPECIALS:
            continue
        if ch not in src_real:               # 只在目标端出现 = 插入的形态标记
            marker[int(tid)] = True
    s2t = torch.full((len(src_vocab),), -1, dtype=torch.long)
    for ch, sid in src_vocab.items():
        if ch in SPECIALS:
            continue
        if ch in tgt_vocab:                   # 骨架字符(辅音/空格)映射到 tgt 词表
            s2t[int(sid)] = int(tgt_vocab[ch])
    return s2t, marker


@torch.no_grad()
def run(model, ds, device, idx2char, max_len, bs, constrained, s2t=None, marker=None):
    model.eval()
    loader = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=collate_batch)
    preds, refs = [], []
    for src, tgt in loader:
        src = src.to(device)
        if constrained:
            gen = model.generate_constrained(src, s2t.to(device), marker.to(device), max_len=max_len)
        else:
            gen = model.generate(src, max_len=max_len)
        preds.extend(decode_ids(r, idx2char) for r in gen.tolist())
        refs.extend(decode_ids(r, idx2char) for r in tgt.tolist())
    cer, acc = compute_cer(preds, refs)
    return cer, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--data_subdir", required=True)
    ap.add_argument("--in_suffix", default="in.original")
    ap.add_argument("--out_suffix", default="out.original")
    ap.add_argument("--emb_size", type=int, default=512)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--enc_layers", type=int, default=3)
    ap.add_argument("--dec_layers", type=int, default=3)
    ap.add_argument("--ffn", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=128)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vd = json.load(open(a.vocab, encoding="utf-8"))
    src_vocab, tgt_vocab = vd["src_vocab"], vd["tgt_vocab"]
    idx2char = {int(i): c for c, i in tgt_vocab.items()}

    model = SyriacSeq2SeqModel(
        src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab),
        emb_size=a.emb_size, nhead=a.nhead,
        num_encoder_layers=a.enc_layers, num_decoder_layers=a.dec_layers,
        ffn_hid_dim=a.ffn, dropout=a.dropout, max_len=a.max_len,
    ).to(device)
    sd = torch.load(a.ckpt, map_location=device)
    if isinstance(sd, dict):
        for _k in ("model_state", "model_state_dict", "model", "state_dict"):
            if _k in sd and isinstance(sd[_k], dict):
                sd = sd[_k]
                break
    model.load_state_dict(sd)

    dp = os.path.join(a.data_dir, a.data_subdir)
    test_src = read_lines(os.path.join(dp, f"test.{a.in_suffix}"))
    test_tgt = read_lines(os.path.join(dp, f"test.{a.out_suffix}"))
    ds = Seq2SeqDataset(test_src, test_tgt, src_vocab, tgt_vocab, a.max_len)
    s2t, marker = build_maps(src_vocab, tgt_vocab)

    cu, _ = run(model, ds, device, idx2char, a.max_len, a.batch_size, False)
    cc, _ = run(model, ds, device, idx2char, a.max_len, a.batch_size, True, s2t, marker)
    tag = os.path.basename(os.path.dirname(a.ckpt))
    print(f"[{tag}] UNCONSTRAINED CER={cu*100:.4f}%  |  CONSTRAINED CER={cc*100:.4f}%  "
          f"(markers={int(marker.sum())}/{len(tgt_vocab)}, lines={len(test_tgt)})")


if __name__ == "__main__":
    main()
