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

from fishrl.data.features import GOD_DIM, GOD_SLOTS, HANDS_DIM, HANDS_SLOTS, PUB_DIM, PUB_SLOTS
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


class PublicCritic(_OutcomeHead):
    """v3 PPO critic (critic_view="public"): the same outcome head over MUTUAL-KNOWLEDGE
    features. Measured basis (2026-08-17 audits): the god head's Brier edge over public
    has depreciated to ~0, so the asymmetric critic no longer buys lower-variance
    advantages — while its god encode + wider input cost real collection time.

    Carries a second 1-logit head over the same encoded features predicting the
    DECKOUT winner (parity credit — both audits' #1 lever: the critic moved only
    |ΔP|≈0.02 on a parity flip that near-decides deckout endgames, so GAE could not
    credit the draws that flip the clock). The head ALWAYS exists in this class so the
    loss weight (Config.critic_deckout_aux) is resume-tunable without state_dict
    surgery; at weight 0 it contributes no gradient."""

    DIM, SLOTS = PUB_DIM, PUB_SLOTS

    def __init__(self, hidden=(512, 512, 256), encoder: str = "entity", card_dim: int = 64):
        super().__init__(self.DIM, self.SLOTS, hidden, encoder, card_dim)
        in_dim = self.enc.enc_dim if self.enc is not None else self.DIM
        self.aux_head = nn.Linear(in_dim, 1)

    def forward_with_aux(self, x: torch.Tensor):
        if self.enc is not None:
            x = self.enc(x)
        return self.net(x).squeeze(-1), self.aux_head(x).squeeze(-1)


class HandsCritic(PublicCritic):
    """critic_view="hands" (2026-08-21): the PublicCritic head over the HANDS layout --
    both hands fully visible, no library rows. Same aux head, same contract."""
    DIM, SLOTS = HANDS_DIM, HANDS_SLOTS


def make_critic(view: str, hidden=(512, 512, 256), encoder: str = "flat",
                card_dim: int = 64):
    """The PPO critic for a Config.critic_view: "god" -> PrivilegedCritic (legacy),
    "public" -> PublicCritic, "hands" -> HandsCritic (both + deckout aux head)."""
    if view == "hands":
        return HandsCritic(hidden, encoder, card_dim)
    if view == "public":
        return PublicCritic(hidden, encoder, card_dim)
    return PrivilegedCritic(hidden, encoder, card_dim)
