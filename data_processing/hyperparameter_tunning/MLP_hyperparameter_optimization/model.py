from __future__ import annotations
from typing import List
import torch
from torch import nn

class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            last = h
        layers.append(nn.Linear(last, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
