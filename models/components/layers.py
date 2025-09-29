"""
网络层组件
"""

import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    """分类头层"""

    def __init__(self, input_size: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(x))