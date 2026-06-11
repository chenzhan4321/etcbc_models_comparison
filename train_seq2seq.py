#!/usr/bin/env python3
"""
train_seq2seq.py — 字符级 encoder-decoder seq2seq baseline 的独立训练/评估脚本。

用途（为什么有这个脚本）：
    论文返修需要重新复现一个 encoder-decoder seq2seq baseline（辅音文本 → ETCBC
    形态串，逐字符翻译）。原始训练源码已丢失，这里重写一份干净、可复现、且所有
    超参都走 CLI 的脚本，方便后续被 Optuna 调优并重训。

设计原则：
    - 自包含：只依赖 models/seq2seq.py + torch/numpy（不改动任何现有文件）。
    - 可复现：固定 random / numpy / torch / cuda 随机种子。
    - 字符级 tokenize：从训练集构建 src/tgt 两份词表（含 PAD/SOS/EOS/UNK）。
    - 评估指标：CER（逐行 Levenshtein 字符级错误率），复现仓库
      data_processing_tools/levenshtein 的标准编辑距离逻辑。

数据格式（已验证，s2_on_s2 char→char 对）：
    --data_dir/--data_subdir 下放 {train,val,test}.{in_suffix,out_suffix}，
    每行一个 7-词窗口，src 行与 tgt 行一一对应，空格分词、字符级 token。

典型命令（真实 s2_on_s2 训练，参考原始架构）：
    cd $REPO && .venv/bin/python -u train_seq2seq.py \
        --data_subdir _s2_seq2seq --emb_size 512 --nhead 8 \
        --enc_layers 3 --dec_layers 3 --ffn 512 --dropout 0.1 \
        --lr 1e-4 --batch_size 128 --epochs 30 --patience 5 --max_len 128

s4_on_s2 数据准备 TODO（见文件末尾的详细说明）：
    s4_on_s2 目前没有现成的 *.original 文件；需要由 *.in + restore_to_original.py
    流程从 s4 输入恢复出 .original 形态串后，才能用同样的接口在此训练。
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# 让脚本无论从哪个 cwd 运行都能 import 到 models 包
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.seq2seq import (  # noqa: E402
    SyriacSeq2SeqModel,
    PAD_IDX,
    SOS_IDX,
    EOS_IDX,
    UNK_IDX,
    SPECIAL_TOKENS,
)


# =============================================================================
# 1. 词表构建与字符级 tokenize
# =============================================================================
def build_vocab(lines: List[str]) -> Dict[str, int]:
    """从训练集文本行构建字符级词表。

    为什么先放特殊 token：保证 PAD=0/SOS=1/EOS=2/UNK=3 的 id 固定（与原始
    model_config 的约定一致，0 还要当 padding_idx）。其余字符按“首次出现顺序”
    赋 id——确定性排序（不依赖 set 的哈希随机性），保证多次运行词表一致、可复现。
    """
    char2idx: Dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for line in lines:
        for ch in line:  # 注意：保留空格，空格本身是有意义的分词符
            if ch not in char2idx:
                char2idx[ch] = len(char2idx)
    return char2idx


def encode_line(line: str, char2idx: Dict[str, int], add_sos_eos: bool = True) -> List[int]:
    """把一行字符串转成 id 列表。未见字符 -> UNK。"""
    ids = [char2idx.get(ch, UNK_IDX) for ch in line]
    if add_sos_eos:
        ids = [SOS_IDX] + ids + [EOS_IDX]
    return ids


def decode_ids(ids: List[int], idx2char: Dict[int, str]) -> str:
    """把 id 列表还原成字符串，遇到 EOS 停止，跳过 SOS/PAD。

    为什么这样：评估时只关心 SOS 与第一个 EOS 之间的内容；EOS 之后的（PAD 或
    残留）一律丢弃，才能与 GT 形态串公平比较。
    """
    chars: List[str] = []
    for i in ids:
        if i == EOS_IDX:
            break
        if i in (PAD_IDX, SOS_IDX):
            continue
        chars.append(idx2char.get(i, ""))  # UNK 还原为空串（极少出现）
    return "".join(chars)


# =============================================================================
# 2. 数据集与 collate
# =============================================================================
def read_lines(path: str) -> List[str]:
    """逐行读入（去掉行尾换行，保留行内空格）。"""
    with open(path, "r", encoding="utf-8") as f:
        # 用 rstrip('\n') 而非 strip()，避免误删行首/行尾有意义的空白
        return [line.rstrip("\n") for line in f]


class Seq2SeqDataset(Dataset):
    """成对 (src_ids, tgt_ids) 数据集。"""

    def __init__(
        self,
        src_lines: List[str],
        tgt_lines: List[str],
        src_vocab: Dict[str, int],
        tgt_vocab: Dict[str, int],
        max_len: int,
    ):
        # 防御：src/tgt 行数若不一致（如末行缺换行导致计数差异），按较短的对齐
        n = min(len(src_lines), len(tgt_lines))
        if len(src_lines) != len(tgt_lines):
            logging.warning(
                "src/tgt 行数不一致 (%d vs %d)，按较短的 %d 行对齐",
                len(src_lines), len(tgt_lines), n,
            )
        self.samples: List[Tuple[List[int], List[int]]] = []
        n_truncated = 0
        for i in range(n):
            src_ids = encode_line(src_lines[i], src_vocab)
            tgt_ids = encode_line(tgt_lines[i], tgt_vocab)
            # 截断到 max_len（含 SOS/EOS）。为什么保留 EOS：位置编码与解码终止都
            # 依赖 EOS，截断时强制把最后一个位置设为 EOS，保证序列语义完整。
            if len(src_ids) > max_len:
                src_ids = src_ids[: max_len - 1] + [EOS_IDX]
                n_truncated += 1
            if len(tgt_ids) > max_len:
                tgt_ids = tgt_ids[: max_len - 1] + [EOS_IDX]
                n_truncated += 1
            self.samples.append((src_ids, tgt_ids))
        if n_truncated:
            logging.info("有 %d 条序列超过 max_len=%d 被截断", n_truncated, max_len)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src_ids, tgt_ids = self.samples[idx]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_batch(batch):
    """把一个 batch 的变长序列 padding 到等长。

    为什么用 PAD_IDX 填充：模型里的 padding mask 依赖 PAD_IDX 识别填充位，
    CrossEntropyLoss 的 ignore_index=PAD_IDX 也靠它跳过这些位置的 loss。
    """
    src_list, tgt_list = zip(*batch)
    src_pad = nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=PAD_IDX)
    tgt_pad = nn.utils.rnn.pad_sequence(tgt_list, batch_first=True, padding_value=PAD_IDX)
    return src_pad, tgt_pad


# =============================================================================
# 3. Levenshtein / CER（复现 data_processing_tools/levenshtein 的标准实现）
# =============================================================================
def levenshtein_distance(s1: str, s2: str) -> int:
    """标准编辑距离（插入/删除/替换各计 1）。

    与 data_processing_tools/levenshtein/compare_final_units.py 的 DP 逻辑一致，
    这里只取距离值（不需要回溯操作序列），用滚动数组省内存。
    """
    if len(s1) < len(s2):
        s1, s2 = s2, s1  # 保证 s2 较短，DP 行更短
    if len(s2) == 0:
        return len(s1)
    previous = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current = [i + 1]
        for j, c2 in enumerate(s2):
            insert = previous[j + 1] + 1
            delete = current[j] + 1
            substitute = previous[j] + (c1 != c2)
            current.append(min(insert, delete, substitute))
        previous = current
    return previous[-1]


def compute_cer(predictions: List[str], references: List[str]) -> Tuple[float, float]:
    """逐行计算字符级 CER。

    CER = sum(每行 Levenshtein(pred, ref)) / sum(每行 len(ref))
    这是语音/OCR/序列翻译里 CER 的标准定义（用 GT 字符总数做分母）。
    char_accuracy = 1 - CER（截断到 [0,1]）。
    """
    total_dist = 0
    total_ref_chars = 0
    for pred, ref in zip(predictions, references):
        total_dist += levenshtein_distance(pred, ref)
        total_ref_chars += len(ref)
    if total_ref_chars == 0:
        return 0.0, 1.0
    cer = total_dist / total_ref_chars
    char_acc = max(0.0, 1.0 - cer)
    return cer, char_acc


# =============================================================================
# 4. 训练 / 评估
# =============================================================================
def set_seed(seed: int):
    """固定所有随机源，保证可复现（科研要求）。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(force_device: str) -> torch.device:
    """设备选择：优先尊重 --force_device，否则自动检测 cuda > mps > cpu。"""
    if force_device and force_device != "auto":
        return torch.device(force_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, optimizer, device, train: bool, clip: float = 1.0):
    """跑一个 epoch。train=True 时做反向 + 梯度裁剪 + 更新。

    teacher forcing 的关键两行（为什么这样切片）：
        tgt_input = tgt[:, :-1]   # 解码器输入：去掉最后一个 token（右移效果）
        tgt_out   = tgt[:, 1:]    # 监督目标：去掉 SOS，错一位对齐预测
    即“用 1..t-1 的真值预测第 t 个 token”，配合模型内部的 causal mask 不偷看未来。
    """
    model.train() if train else model.eval()
    total_loss = 0.0
    total_tokens = 0
    torch.set_grad_enabled(train)
    try:
        for src, tgt in loader:
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            logits = model(src, tgt_input)  # [batch, seq, vocab]
            # 展平成 [batch*seq, vocab] 与 [batch*seq] 算 token 级交叉熵
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_out.reshape(-1),
            )
            if train:
                optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪：Transformer 训练偶有梯度爆炸，裁剪到 clip 范数更稳
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

            # 统计非 PAD token 数做加权平均，使 loss 不受 padding 量影响
            n_tok = (tgt_out != PAD_IDX).sum().item()
            total_loss += loss.item() * n_tok
            total_tokens += n_tok
    finally:
        torch.set_grad_enabled(True)
    return total_loss / max(1, total_tokens)


@torch.no_grad()
def evaluate_cer(model, dataset, device, tgt_idx2char, max_len, batch_size):
    """对数据集自回归解码，返回 (cer, char_acc, predictions, references)。"""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    predictions: List[str] = []
    references: List[str] = []
    for src, tgt in loader:
        src = src.to(device)
        gen = model.generate(src, max_len=max_len)  # [batch, gen_len]
        for row in gen.tolist():
            predictions.append(decode_ids(row, tgt_idx2char))
        for row in tgt.tolist():
            references.append(decode_ids(row, tgt_idx2char))
    cer, char_acc = compute_cer(predictions, references)
    return cer, char_acc, predictions, references


def setup_logging(log_path: str):
    """同时写 tqdm 风格的控制台 + 文件日志（科研要求：日志可查）。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main():
    parser = argparse.ArgumentParser(
        description="字符级 encoder-decoder seq2seq baseline 训练/评估（辅音文本→ETCBC 形态串）",
    )
    # --- 数据相关 ---
    parser.add_argument("--data_dir", default="./data", help="数据根目录")
    parser.add_argument("--data_subdir", required=True, help="数据子目录（data_dir 下）")
    parser.add_argument("--in_suffix", default="in.original", help="源端文件后缀")
    parser.add_argument("--out_suffix", default="out.original", help="目标端文件后缀")
    # --- 模型架构（全部走 CLI，便于 Optuna HPO）---
    parser.add_argument("--emb_size", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--enc_layers", type=int, default=3)
    parser.add_argument("--dec_layers", type=int, default=3)
    parser.add_argument("--ffn", type=int, default=512, help="前馈隐藏维度 ffn_hid_dim")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=128, help="序列最大长度（含 SOS/EOS）")
    # --- 优化 / 训练控制 ---
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5, help="early stopping 容忍轮数")
    parser.add_argument("--clip", type=float, default=1.0, help="梯度裁剪范数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_device", default="auto", help="cpu/cuda/mps/auto")
    # --- 输出 ---
    parser.add_argument("--save_dir", default=None, help="本次运行的输出根目录（默认按时间戳建）")
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device(args.force_device)

    # ---- 目录准备：checkpoints / logs / results 都放仓库根目录下 ----
    run_tag = f"seq2seq_{args.data_subdir}_emb{args.emb_size}_h{args.nhead}_seed{args.seed}"
    ckpt_dir = os.path.join(REPO_ROOT, "checkpoints", run_tag)
    log_dir = os.path.join(REPO_ROOT, "logs")
    results_dir = args.save_dir or os.path.join(REPO_ROOT, "results", run_tag)
    for d in (ckpt_dir, log_dir, results_dir):
        os.makedirs(d, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_tag}.log")
    setup_logging(log_path)

    logging.info("=== 配置 ===")
    logging.info("device=%s, args=%s", device, vars(args))

    # ---- 读数据 ----
    data_path = os.path.join(args.data_dir, args.data_subdir)
    paths = {}
    for split in ("train", "val", "test"):
        paths[(split, "in")] = os.path.join(data_path, f"{split}.{args.in_suffix}")
        paths[(split, "out")] = os.path.join(data_path, f"{split}.{args.out_suffix}")
    for k, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"缺少数据文件: {p}")

    train_src = read_lines(paths[("train", "in")])
    train_tgt = read_lines(paths[("train", "out")])
    val_src = read_lines(paths[("val", "in")])
    val_tgt = read_lines(paths[("val", "out")])
    test_src = read_lines(paths[("test", "in")])
    test_tgt = read_lines(paths[("test", "out")])

    # ---- 构建词表（只从训练集！防止泄漏验证/测试集信息）----
    src_vocab = build_vocab(train_src)
    tgt_vocab = build_vocab(train_tgt)
    tgt_idx2char = {i: c for c, i in tgt_vocab.items()}
    logging.info("src_vocab_size=%d, tgt_vocab_size=%d", len(src_vocab), len(tgt_vocab))

    vocab_path = os.path.join(results_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({"src_vocab": src_vocab, "tgt_vocab": tgt_vocab}, f, ensure_ascii=False, indent=2)

    # ---- 数据集 / DataLoader ----
    train_ds = Seq2SeqDataset(train_src, train_tgt, src_vocab, tgt_vocab, args.max_len)
    val_ds = Seq2SeqDataset(val_src, val_tgt, src_vocab, tgt_vocab, args.max_len)
    test_ds = Seq2SeqDataset(test_src, test_tgt, src_vocab, tgt_vocab, args.max_len)
    # 固定 generator 让 shuffle 也可复现
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_batch, generator=g,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    # ---- 建模 ----
    model = SyriacSeq2SeqModel(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        emb_size=args.emb_size,
        nhead=args.nhead,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        ffn_hid_dim=args.ffn,
        dropout=args.dropout,
        max_len=args.max_len,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logging.info("模型参数量: %d", n_params)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---- 训练 + early stopping ----
    best_val = math.inf
    best_epoch = -1
    epochs_no_improve = 0
    best_ckpt = os.path.join(ckpt_dir, "best.pth")
    t0 = time.time()

    try:
        from tqdm import tqdm  # 进度条（仓库约定）；没装也能降级
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    for epoch in range(1, args.epochs + 1):
        ep_iter = train_loader
        if use_tqdm:
            ep_iter = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", file=sys.stdout)
        train_loss = run_epoch(model, ep_iter, criterion, optimizer, device, train=True, clip=args.clip)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        logging.info("epoch %d | train_loss=%.4f | val_loss=%.4f", epoch, train_loss, val_loss)

        # early stopping 基于 val loss（论文常用、稳定）
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            # checkpoint 同时存权重与重建所需的超参/词表大小，便于断点续跑/部署
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "src_vocab_size": len(src_vocab),
                    "tgt_vocab_size": len(tgt_vocab),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                best_ckpt,
            )
            logging.info("  -> 新最佳 val_loss=%.4f，保存 checkpoint", val_loss)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                logging.info("early stopping：连续 %d 轮无提升，停止于 epoch %d", args.patience, epoch)
                break

    train_time = time.time() - t0
    logging.info("训练结束，用时 %.1fs，最佳 epoch=%d (val_loss=%.4f)", train_time, best_epoch, best_val)

    # ---- 加载最佳权重，在 test 上自回归解码评估 CER ----
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        logging.info("已加载最佳 checkpoint（epoch %d）做 test 评估", ckpt["epoch"])

    logging.info("开始 test 自回归解码评估 ...")
    cer, char_acc, predictions, references = evaluate_cer(
        model, test_ds, device, tgt_idx2char, args.max_len, args.batch_size,
    )
    logging.info("test CER=%.4f | char_accuracy=%.4f", cer, char_acc)

    # ---- 落盘预测与结果汇总 ----
    pred_path = os.path.join(results_dir, "test.pred.out.original")
    with open(pred_path, "w", encoding="utf-8") as f:
        for line in predictions:
            f.write(line + "\n")

    results = {
        "data_subdir": args.data_subdir,
        "test_cer": cer,
        "char_accuracy": char_acc,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "train_time_sec": train_time,
        "n_params": n_params,
        "src_vocab_size": len(src_vocab),
        "tgt_vocab_size": len(tgt_vocab),
        "device": str(device),
        "args": vars(args),
        "n_test": len(references),
    }
    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logging.info("结果已保存：%s", results_path)
    logging.info("预测已保存：%s", pred_path)
    logging.info("词表已保存：%s", vocab_path)
    logging.info("checkpoint：%s", best_ckpt)
    # 给 stdout 一行机器可读的关键指标，方便 Optuna 抓取
    print(f"FINAL test_cer={cer:.6f} char_accuracy={char_acc:.6f}")


if __name__ == "__main__":
    main()


# =============================================================================
# s4_on_s2 数据准备 TODO
# =============================================================================
# 本脚本只在 s2_on_s2（已有现成 *.original 对）上验证跑通。s4_on_s2 没有现成的
# *.original 文件，需要先补一个数据准备步骤，再用相同接口在此训练：
#
#   1. s4_on_s2 的输入是 s4 形态级 *.in（见 data/raw_s4_on_s2 或 outputs/ 下对应
#      目录），目标端仍是 s2 的形态串。
#   2. 用仓库根目录的 restore_to_original.py（restore 流程）把 *.in / *.out 的
#      reduced/编码形态还原为人类可读的 *.in.original / *.out.original：
#         .venv/bin/python restore_to_original.py \
#             --input  data/raw_s4_on_s2/{split}.in \
#             --output data/s4_on_s2/{split}.in.original
#      （out 端同理；具体参数以 restore_to_original.py 的 argparse 为准。）
#   3. 把三套 {train,val,test}.{in,out}.original 放到
#      data/<your_s4_subdir>/ 下，然后：
#         .venv/bin/python -u train_seq2seq.py --data_subdir <your_s4_subdir> ...
#   4. 注意：s4 端字符表可能与 s2 略有差异；因为词表只从“训练集”构建，
#      只要 s4 训练集覆盖了 test 端字符即可，否则未见字符会落到 UNK（已兜底）。
