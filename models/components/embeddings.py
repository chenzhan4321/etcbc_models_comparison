"""
嵌入层组件
"""

import torch
import torch.nn as nn


class SimpleEmbedding(nn.Module):
    """简单嵌入层"""

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * (self.embed_dim ** 0.5)