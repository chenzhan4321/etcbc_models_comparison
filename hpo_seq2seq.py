#!/usr/bin/env python3
"""
hpo_seq2seq.py — 对 seq2seq baseline 做 Optuna 超参搜索（回应 R2-M2 的公平性质疑）。

为什么需要（核心动机）：
    审稿人 R2-M2 指出原 encoder-decoder baseline 用的是旧固定超参，而 proposed
    模型（encoder-only / MDLM）做过系统调优，比较不公平。本脚本给 seq2seq baseline
    做【同等预算】的 Optuna 调优，使"调优后的强 baseline"与 proposed 模型同台竞争。

设计：
    - 复用 train_seq2seq.py 的数据/训练/评估部件（import，不改任何现有文件）。
    - 目标函数最小化【验证集 CER】（自回归解码，贴近真实评估目标，而非仅 val loss）。
    - 每个 trial 内按 epoch 上报 val_loss 给 MedianPruner，提前砍掉没希望的配置，省算力。
    - 搜索空间量级对齐 encoder-only/mdlm 的 HPO（维度/层数/ffn/dropout/lr/batch）。
    - 支持 --storage 持久化（sqlite），HPC 上可断点续跑、可多进程并行同一 study。
    - 重要：seq2seq 是独立脚本，不经过 train.py 的 apply_smart_config，
      因此 HPO 搜出的超参不会被偷偷覆盖（这正是 R2-M2 要的"真·同等调优"）。

典型用法（HPC）：
    cd $REPO && .venv/bin/python -u hpo_seq2seq.py \
        --data_subdir <s2_seq2seq 目录> --n_trials 40 --hpo_epochs 8 \
        --max_len 128 --storage sqlite:///checkpoints/hpo_seq2seq.db \
        --study_name seq2seq_s2

调优完成后，用 best_params 跑 train_seq2seq.py 做 5-seed 正式重训。
"""

import argparse
import json
import logging
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import optuna  # noqa: E402
from optuna.samplers import TPESampler  # noqa: E402
from optuna.pruners import MedianPruner  # noqa: E402

# 复用 train_seq2seq.py 的部件（不修改它）
from train_seq2seq import (  # noqa: E402
    read_lines,
    build_vocab,
    Seq2SeqDataset,
    collate_batch,
    run_epoch,
    evaluate_cer,
    set_seed,
    pick_device,
)
from models.seq2seq import SyriacSeq2SeqModel, PAD_IDX  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="seq2seq baseline 的 Optuna 超参搜索")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--data_subdir", required=True)
    p.add_argument("--in_suffix", default="in.original")
    p.add_argument("--out_suffix", default="out.original")
    p.add_argument("--n_trials", type=int, default=40)
    p.add_argument("--hpo_epochs", type=int, default=8, help="每个 trial 训练的轮数（短，用于排序配置）")
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force_device", default="auto")
    p.add_argument("--study_name", default="seq2seq_hpo")
    p.add_argument("--storage", default=None, help="如 sqlite:///checkpoints/hpo_seq2seq.db，便于断点续跑/并行")
    p.add_argument("--timeout", type=int, default=None, help="总秒数预算（可选）")
    p.add_argument("--wide", action="store_true",
                   help="扩大搜索空间（emb->1024, nhead->16, layers->8, ffn->4096, lr->2e-3, batch->256）")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    set_seed(args.seed)
    device = pick_device(args.force_device)
    logging.info("device=%s, n_trials=%d, hpo_epochs=%d", device, args.n_trials, args.hpo_epochs)

    # ---- 数据只读一次、词表只从训练集构建一次（所有 trial 共享，省 IO）----
    dp = os.path.join(args.data_dir, args.data_subdir)
    train_src = read_lines(os.path.join(dp, f"train.{args.in_suffix}"))
    train_tgt = read_lines(os.path.join(dp, f"train.{args.out_suffix}"))
    val_src = read_lines(os.path.join(dp, f"val.{args.in_suffix}"))
    val_tgt = read_lines(os.path.join(dp, f"val.{args.out_suffix}"))
    src_vocab = build_vocab(train_src)
    tgt_vocab = build_vocab(train_tgt)
    tgt_idx2char = {i: c for c, i in tgt_vocab.items()}
    train_ds = Seq2SeqDataset(train_src, train_tgt, src_vocab, tgt_vocab, args.max_len)
    val_ds = Seq2SeqDataset(val_src, val_tgt, src_vocab, tgt_vocab, args.max_len)
    logging.info("src_vocab=%d tgt_vocab=%d | train=%d val=%d",
                 len(src_vocab), len(tgt_vocab), len(train_ds), len(val_ds))

    def objective(trial: optuna.Trial) -> float:
        if args.wide:
            # --- 扩大搜索空间（向上扩；best 撞了 ffn/lr/layers 的上界）---
            emb_size = trial.suggest_categorical("emb_size", [256, 384, 512, 768, 1024])
            nhead = trial.suggest_categorical("nhead", [4, 8, 16])
            if emb_size % nhead != 0:
                raise optuna.TrialPruned()
            enc_layers = trial.suggest_int("enc_layers", 2, 8)
            dec_layers = trial.suggest_int("dec_layers", 2, 8)
            ffn = trial.suggest_categorical("ffn", [512, 1024, 2048, 4096])
            dropout = trial.suggest_float("dropout", 0.05, 0.3)
            lr = trial.suggest_float("lr", 5e-5, 2e-3, log=True)
            batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        else:
            # --- 搜索空间（量级对齐 proposed 模型的 HPO）---
            emb_size = trial.suggest_categorical("emb_size", [128, 256, 384, 512])
            nhead = trial.suggest_categorical("nhead", [4, 8])
            if emb_size % nhead != 0:
                # nhead 必须整除 emb_size；非法组合直接剪枝（理论上这些组合都可整除，防御性保留）
                raise optuna.TrialPruned()
            enc_layers = trial.suggest_int("enc_layers", 2, 6)
            dec_layers = trial.suggest_int("dec_layers", 2, 6)
            ffn = trial.suggest_categorical("ffn", [256, 512, 1024, 2048])
            dropout = trial.suggest_float("dropout", 0.1, 0.3)
            lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
            batch_size = trial.suggest_categorical("batch_size", [64, 128])

        # 每个 trial 用相同种子初始化，保证配置之间的比较只反映超参差异（控制混淆）
        set_seed(args.seed)
        model = SyriacSeq2SeqModel(
            src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab),
            emb_size=emb_size, nhead=nhead,
            num_encoder_layers=enc_layers, num_decoder_layers=dec_layers,
            ffn_hid_dim=ffn, dropout=dropout, max_len=args.max_len,
        ).to(device)
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

        for epoch in range(args.hpo_epochs):
            run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
            trial.report(val_loss, epoch)  # 给 pruner 的中间信号
            if trial.should_prune():
                raise optuna.TrialPruned()

        # 真正的优化目标：验证集 CER（自回归解码），比 val_loss 更贴近论文报告的指标
        cer, _, _, _ = evaluate_cer(model, val_ds, device, tgt_idx2char, args.max_len, batch_size)
        return cer

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=args.seed),
        pruner=MedianPruner(n_warmup_steps=2),
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
    )
    # catch=RuntimeError: 大配置 OOM/数值异常只让该 trial 失败, 不杀 worker
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout,
                   catch=(RuntimeError,))

    logging.info("==== HPO 完成 ====")
    logging.info("最佳 val CER = %.6f", study.best_value)
    logging.info("最佳超参 = %s", study.best_params)

    out = {
        "best_val_cer": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "data_subdir": args.data_subdir,
    }
    os.makedirs(os.path.join(REPO_ROOT, "results"), exist_ok=True)
    out_path = os.path.join(REPO_ROOT, "results", f"hpo_seq2seq_{args.study_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logging.info("结果已保存：%s", out_path)
    # 机器可读的一行，便于脚本接力跑 5-seed 正式训练
    print(f"BEST_PARAMS {json.dumps(study.best_params)}")
    print(f"BEST_VAL_CER {study.best_value:.6f}")


if __name__ == "__main__":
    main()
