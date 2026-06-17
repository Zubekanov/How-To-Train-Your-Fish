"""Outcome estimators — both output a logit for P(p1 wins).

``PrivilegedCritic`` consumes the fully-observed god features (GOD_DIM) and also
serves as the asymmetric critic during PPO. ``PublicEstimator`` consumes only
mutual-knowledge features (PUB_DIM). Both are p1-oriented and seat-agnostic; the
acting seat's value is derived from ``P(p1 win)`` by the sign convention in
:mod:`fishrl.train.advantages`.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from fishrl.data.features import GOD_DIM, PUB_DIM
from fishrl.models.mlp import make_mlp


class _OutcomeHead(nn.Module):
    def __init__(self, in_dim: int, hidden):
        super().__init__()
        self.net = make_mlp(in_dim, 1, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)        # logit for P(p1 win)

    def p1_winprob(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


class PrivilegedCritic(_OutcomeHead):
    def __init__(self, hidden=(256, 256)):
        super().__init__(GOD_DIM, hidden)


class PublicEstimator(_OutcomeHead):
    def __init__(self, hidden=(256, 256)):
        super().__init__(PUB_DIM, hidden)
