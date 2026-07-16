"""Outcome estimators — both output a logit for P(p1 wins).

``PrivilegedCritic`` consumes the fully-observed god features (GOD_DIM) and is the
asymmetric PPO critic: **it is the only value head in the advantage loop** (see
``train/collector.collect_games(..., critic=...)``). ``PublicEstimator`` consumes
only mutual-knowledge features (PUB_DIM) and is **auxiliary/diagnostic** — trained
on the same terminal outcomes and reported by ``eval/metrics``, but never feeds a
policy gradient, so the two heads cannot silently disagree in the loop. (If the
public head were ever put in the loop it should be tied to the privileged one via
the consistency relation public = E[privileged | public info].)

Both heads are p1-oriented and seat-agnostic; the acting seat's value is derived
from P(p1 win) by the sign convention in :mod:`fishrl.train.advantages`.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from fishrl.data.features import GOD_DIM, GOD_SLOTS, PUB_DIM, PUB_SLOTS
from fishrl.models.entity_encoder import build_entity_encoder, layout_from_slots
from fishrl.models.mlp import make_mlp


class _OutcomeHead(nn.Module):
    def __init__(self, total_in: int, slots: dict, hidden, encoder: str, card_dim: int = 64):
        super().__init__()
        self.enc = build_entity_encoder(encoder, *layout_from_slots(slots), total_in, d=card_dim)
        in_dim = self.enc.enc_dim if self.enc is not None else total_in
        self.net = make_mlp(in_dim, 1, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.enc is not None:
            x = self.enc(x)
        return self.net(x).squeeze(-1)        # logit for P(p1 win)

    def p1_winprob(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


class PrivilegedCritic(_OutcomeHead):
    def __init__(self, hidden=(512, 512, 256), encoder: str = "flat", card_dim: int = 64):
        super().__init__(GOD_DIM, GOD_SLOTS, hidden, encoder, card_dim)


class PublicEstimator(_OutcomeHead):
    def __init__(self, hidden=(256, 256), encoder: str = "flat", card_dim: int = 64):
        super().__init__(PUB_DIM, PUB_SLOTS, hidden, encoder, card_dim)
