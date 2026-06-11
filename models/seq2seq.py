"""
SyriacSeq2SeqModel — 字符级 encoder-decoder Transformer seq2seq baseline。

背景（为什么有这个文件）：
    本文件为论文返修重新实现一个 encoder-decoder seq2seq baseline。原始训练源码
    已丢失，只剩 .pth 产物，所以这里按原始 model_config*.json 记录的架构
    （emb_size=512, nhead=8, enc/dec layers=3, ffn=512, dropout=0.1）重写一份
    干净、可复现、可被 Optuna 调优的实现。所有架构维度都走构造函数参数，便于 HPO。

任务本质（char→char 翻译）：
    把辅音文本（src，例 "WBHLJN KLHJN ..."）翻译成 ETCBC 形态串
    （tgt，例 "W-B-HLJN KL/-HJN ..."）。逐字符 tokenize（含空格），属于论文所
    反对的“开放式生成”表述，这里作为 baseline 复现。

设计要点（都在下面对应处用注释解释“为什么”）：
    1. src / tgt 各自独立的 embedding（两端字符表不同：tgt 多了形态 marker）。
    2. 正弦位置编码（固定、不可学习）——Transformer 本身对位置不敏感，必须
       显式注入位置信息；正弦编码无参数、可外推到比训练更长的序列。
    3. forward 用 teacher forcing：解码器输入是“右移一位的 ground truth”，
       配合 causal mask 防止偷看未来，padding mask 屏蔽 PAD。
    4. generate() 做自回归 greedy 解码：从 SOS 起步，逐步贴上预测的下一个 token，
       直到 EOS 或 max_len。

代码风格遵循仓库约定：注释中文、标识符英文。本文件刻意自包含（只依赖
torch），不耦合 models/ 下既有组件库，以免改动现有文件或破坏其他模型。
"""

import math
from typing import Optional

import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# 特殊 token 的固定约定（与原始 model_config*.json 一致：PAD=0, SOS=1, EOS=2）
# UNK 原始词表里没有，但字符级任务在测试/推理时仍可能遇到训练集未见过的字符，
# 加一个 UNK=3 兜底，避免 KeyError 同时不影响已见字符的 id。
# -----------------------------------------------------------------------------
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
SPECIAL_TOKENS = ["PAD", "SOS", "EOS", "UNK"]


class PositionalEncoding(nn.Module):
    """正弦/余弦位置编码（Vaswani et al., 2017, 公式 PE(pos,2i)/PE(pos,2i+1)）。

    为什么用固定正弦编码而不是可学习的：
        - 无参数，不增加过拟合风险，HPO 时也不需要为它调任何东西；
        - 对位置具有“相对距离可由线性变换表示”的良好性质；
        - 可外推到比训练时更长的序列（推理时 max_len 可放大）。
    """

    def __init__(self, emb_size: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 预计算 [max_len, emb_size] 的位置编码表，注册为 buffer（随模型搬设备但不训练）
        pe = torch.zeros(max_len, emb_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        # div_term = 1 / 10000^(2i/d)，用 exp(log(...)) 的写法保证数值稳定
        div_term = torch.exp(
            torch.arange(0, emb_size, 2, dtype=torch.float) * (-math.log(10000.0) / emb_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维用 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维用 cos
        pe = pe.unsqueeze(0)  # [1, max_len, emb_size]，batch_first 下方便广播
        self.register_buffer("pe", pe)

    def forward(self, token_emb: torch.Tensor) -> torch.Tensor:
        # token_emb: [batch, seq_len, emb_size]
        seq_len = token_emb.size(1)
        token_emb = token_emb + self.pe[:, :seq_len, :]
        return self.dropout(token_emb)


class TokenEmbedding(nn.Module):
    """词嵌入层。按原论文/标准做法乘 sqrt(emb_size) 缩放。

    为什么乘 sqrt(d)：让 embedding 的尺度与正弦位置编码量级匹配，否则刚初始化时
    位置编码会“盖过”内容信息，影响早期训练稳定性。
    """

    def __init__(self, vocab_size: int, emb_size: int):
        super().__init__()
        # padding_idx=PAD_IDX：PAD 的 embedding 恒为 0 且不接收梯度
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=PAD_IDX)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class SyriacSeq2SeqModel(nn.Module):
    """字符级 encoder-decoder Transformer，用于辅音文本 → ETCBC 形态串翻译。

    所有架构维度都是构造函数参数，方便 Optuna 调优后重训。
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        emb_size: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        ffn_hid_dim: int = 512,
        dropout: float = 0.1,
        max_len: int = 128,
    ):
        super().__init__()
        # 记录超参，便于保存到 checkpoint / config，重建时无需再传一遍
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.emb_size = emb_size
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.ffn_hid_dim = ffn_hid_dim
        self.dropout = dropout
        self.max_len = max_len

        # 校验：nhead 必须整除 emb_size（多头注意力把维度均分到各头）
        assert emb_size % nhead == 0, (
            f"emb_size ({emb_size}) 必须能被 nhead ({nhead}) 整除"
        )

        # src / tgt 各自独立 embedding（两端字符表不同，不能共享）
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        # 位置编码 src/tgt 共用一个即可（与序列内容无关，只与位置有关）
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout, max_len=max_len)

        # 核心：nn.Transformer（含 encoder 与 decoder）。
        # batch_first=True：所有张量约定为 [batch, seq, feature]，更直观、少转置。
        # norm_first=True：Pre-LN（在子层前做 LayerNorm），训练更稳、更容易收敛，
        #   这是当下重训/HPO 的稳健默认（原始实现细节已不可考，取稳健选择）。
        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=ffn_hid_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )

        # 输出投影：decoder 隐状态 -> tgt 词表 logits
        self.generator = nn.Linear(emb_size, tgt_vocab_size)

        self._reset_parameters()

    def _reset_parameters(self):
        # Xavier 初始化（Transformer 常用），对多维权重做，bias / 1D 参数保持默认
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ---- mask 工具：为什么需要这些 mask ----
    @staticmethod
    def _generate_square_subsequent_mask(size: int, device: torch.device) -> torch.Tensor:
        """生成 causal（下三角）bool mask，形状 [size, size]。

        为什么需要：teacher forcing 时把整条 ground truth 一次喂进 decoder，但解码
        位置 t 只能看到 <= t 的 token，绝不能偷看未来（否则训练时“作弊”，推理时
        因为没有未来信息而崩掉）。

        为什么用 bool 而不是 float mask：新版 PyTorch 里 key_padding_mask 是 bool
        （True=屏蔽），若 attn_mask 用 float 会触发“mismatched mask type”弃用告警。
        统一用 bool（True=未来位置，禁止注意）最干净，nn.Transformer 内部会自动把
        它转成 -inf。
        """
        # triu(diagonal=1) 取严格上三角（未来位置）为 True=屏蔽
        return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)

    def _create_masks(self, src: torch.Tensor, tgt: torch.Tensor):
        """构造训练所需的全部 mask。

        - src/tgt padding mask：[batch, seq]，True 表示该位置是 PAD，需被忽略
          （为什么：不同长度句子 padding 到同长，PAD 不该参与注意力 / 不该贡献信息）。
        - tgt causal mask：见上，防止偷看未来。
        """
        device = src.device
        tgt_seq_len = tgt.size(1)

        tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len, device)
        # src 自注意力不需要 causal mask（encoder 可双向看整句），传 None
        src_mask = None

        src_padding_mask = (src == PAD_IDX)  # [batch, src_seq]
        tgt_padding_mask = (tgt == PAD_IDX)  # [batch, tgt_seq]
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def encode(self, src: torch.Tensor, src_padding_mask: Optional[torch.Tensor] = None):
        """单独跑 encoder，得到 memory（供自回归解码反复复用，避免重复编码）。"""
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask)
        return memory

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ):
        """单步/多步解码，返回 decoder 隐状态（未过 generator）。"""
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        out = self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return out

    def forward(self, src: torch.Tensor, tgt_input: torch.Tensor) -> torch.Tensor:
        """teacher forcing 前向。

        Args:
            src:       [batch, src_seq]，源端 id（含 SOS/EOS/PAD）
            tgt_input: [batch, tgt_seq]，解码器输入 = 右移一位的 ground truth
                       （即 tgt 去掉最后一个 token；训练循环负责右移，见 train 脚本）。
        Returns:
            logits: [batch, tgt_seq, tgt_vocab_size]
        """
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self._create_masks(src, tgt_input)

        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt_input))

        # memory_key_padding_mask 用 src 的 padding mask：decoder 做 cross-attention
        # 看 encoder memory 时，同样不能注意到源端的 PAD。
        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        return self.generator(outs)

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        max_len: Optional[int] = None,
        sos_idx: int = SOS_IDX,
        eos_idx: int = EOS_IDX,
    ) -> torch.Tensor:
        """自回归 greedy 解码（batch 化）。

        为什么这样做：推理时没有 ground truth，只能从 SOS 起步，每步取 logits 的
        argmax 作为下一个 token，再把它拼回去喂下一步，直到所有样本都产出 EOS 或
        达到 max_len。这与训练时的 teacher forcing 相对——训练喂真值，推理喂自己
        的上一步预测（exposure，存在 train/infer 分布差异，是 seq2seq 的固有特性）。

        Args:
            src: [batch, src_seq]
        Returns:
            [batch, gen_len] 的预测 id 序列（含开头 SOS；调用方负责截到 EOS）。
        """
        self.eval()
        device = src.device
        batch_size = src.size(0)
        if max_len is None:
            max_len = self.max_len

        src_padding_mask = (src == PAD_IDX)
        # 编码一次，后续每步复用 memory（关键优化：不重复编码源端）
        memory = self.encode(src, src_padding_mask=src_padding_mask)

        # 解码器输入初始化为单列 SOS
        ys = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
        # 记录哪些样本已经产出 EOS，后续不再更新（生成 PAD 占位即可）
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            tgt_seq_len = ys.size(1)
            tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len, device)
            out = self.decode(
                ys,
                memory,
                tgt_mask=tgt_mask,
                tgt_padding_mask=None,  # 解码过程中 ys 无 PAD（逐步生长）
                memory_key_padding_mask=src_padding_mask,
            )
            # 只取最后一个时间步的隐状态做下一个 token 的预测
            logits = self.generator(out[:, -1, :])  # [batch, tgt_vocab]
            next_token = logits.argmax(dim=-1)  # greedy

            # 已结束的样本强制填 PAD，避免 EOS 之后还冒出乱字符
            next_token = torch.where(finished, torch.full_like(next_token, PAD_IDX), next_token)
            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)

            finished = finished | (next_token == eos_idx)
            if bool(finished.all()):
                break

        return ys

    @torch.no_grad()
    def generate_constrained(
        self,
        src: torch.Tensor,
        src_to_tgt: torch.Tensor,   # [src_vocab] long: 源端 token id -> 同字符的 tgt token id;非骨架(SOS/EOS/PAD/UNK)= -1
        marker_mask: torch.Tensor,  # [tgt_vocab] bool: True = 自由形态标记 token(任何时候可出)
        max_len: Optional[int] = None,
        sos_idx: int = SOS_IDX,
        eos_idx: int = EOS_IDX,
    ) -> torch.Tensor:
        """约束(structure-aware)解码 —— 回应 R2-M2/M4 的 constrained-decoding encoder-decoder。

        约束:输出里"骨架字符"(辅音+词间空格,即源端字符)的子序列必须与源端逐一一致、
        顺序不变;模型只能在骨架字符之间自由插入形态标记(- / [ ] ~ < > = ! @ 等),
        既不能改写/增删辅音,也不能提前 EOS。这样 seq2seq 获得与离散化/MDLM 同等的结构
        合法性保证,使"约束 seq2seq vs 离散化模型"成为单变量比较(隔离离散化本身的贡献)。
        """
        self.eval()
        device = src.device
        B = src.size(0)
        if max_len is None:
            max_len = self.max_len
        # 每个样本的骨架 tgt-id 序列(把源端字符逐一映射到 tgt 词表,丢掉特殊/非骨架)
        s2t = src_to_tgt.tolist()
        skel = []
        for b in range(B):
            seq = [s2t[i] for i in src[b].tolist() if 0 <= i < len(s2t) and s2t[i] >= 0]
            skel.append(seq)
        src_padding_mask = (src == PAD_IDX)
        memory = self.encode(src, src_padding_mask=src_padding_mask)
        # 子序列约束(textual integrity):输入辅音必须作为输出的子序列、按序出现;
        # 但允许自由插入字母(mater lectionis 等修复字母)与形态标记。
        # 实现:基底允许任意非特殊 token;输入辅音未按序出完前禁止 EOS;
        #       连续 MAXRUN 步不消费下一个输入辅音 → 强制吐它(防打转/插入失控)。
        V = marker_mask.numel()
        free_base = torch.ones(V, dtype=torch.bool, device=device)
        for sp in (PAD_IDX, SOS_IDX, EOS_IDX, 3):   # 3 = UNK_IDX;特殊 token 不许自由产出
            if 0 <= sp < V:
                free_base[sp] = False
        ys = torch.full((B, 1), sos_idx, dtype=torch.long, device=device)
        ptr = [0] * B          # 已按序消费的输入辅音数
        run = [0] * B          # 自上次消费输入辅音以来的步数(标记/插入字母累计)
        MAXRUN = 8             # 超过则强制消费下一个输入辅音,杜绝插入失控
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        NEG = torch.finfo(torch.float32).min
        for _ in range(max_len - 1):
            tgt_mask = self._generate_square_subsequent_mask(ys.size(1), device)
            out = self.decode(ys, memory, tgt_mask=tgt_mask, tgt_padding_mask=None,
                              memory_key_padding_mask=src_padding_mask)
            logits = self.generator(out[:, -1, :])               # [B, V]
            allow = free_base.unsqueeze(0).expand(B, -1).clone()  # [B, V] bool
            fin = finished.tolist()
            for b in range(B):
                if fin[b]:
                    continue
                if ptr[b] < len(skel[b]):
                    if run[b] >= MAXRUN:               # 长时间不消费输入辅音 → 只许下一个输入辅音
                        allow[b] = False
                        allow[b, skel[b][ptr[b]]] = True
                    # EOS 在 free_base 里已为 False(输入未消费完不许结束)
                else:
                    allow[b, eos_idx] = True            # 输入辅音已全部按序出现 → 允许 EOS
            logits = logits.masked_fill(~allow, NEG)
            next_token = logits.argmax(dim=-1)
            nt = next_token.tolist()
            for b in range(B):
                if fin[b]:
                    continue
                if ptr[b] < len(skel[b]) and nt[b] == skel[b][ptr[b]]:
                    ptr[b] += 1; run[b] = 0            # 消费了一个输入辅音
                elif nt[b] != eos_idx:
                    run[b] += 1                         # 插入了字母/标记
            next_token = torch.where(finished, torch.full_like(next_token, PAD_IDX), next_token)
            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
            finished = finished | (next_token == eos_idx)
            if bool(finished.all()):
                break
        return ys
