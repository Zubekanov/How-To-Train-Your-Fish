"""Shared small MLP trunk (LayerNorm + ReLU). Kept shallow for CPU training."""
from __future__ import annotations

import torch.nn as nn


def make_mlp(in_dim: int, out_dim: int, hidden=(256, 256)) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)
