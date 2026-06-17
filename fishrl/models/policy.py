"""Masked actor for the self-play policy.

Input is the actor observation: the fair perspective view concatenated with the
hand-guesser's belief (handled by the belief env). Output is logits over the flat
``Discrete(A.N)`` action space; only legal actions (per the env's action mask)
carry probability mass. The critic is the privileged estimator (asymmetric
actor-critic), so this module has no value head.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from fishrl.models.entity_encoder import build_entity_encoder, layout_from_slots
from fishrl.models.mlp import make_mlp
from fishrl.obs.encoder import OBS_DIM, SLOTS
from fishrl.obs import vocab as V
from fishrl.spaces import action_space as A

ACTOR_IN = OBS_DIM + V.N_NAMES          # perspective ⊕ hand-guess
_NEG_INF = -1e9


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Log-softmax over legal actions only (illegal entries -> log-prob -inf)."""
    neg = torch.full_like(logits, _NEG_INF)
    masked = torch.where(mask > 0, logits, neg)
    return Fn.log_softmax(masked, dim=-1)


class MaskedActor(nn.Module):
    def __init__(self, hidden=(256, 256), encoder: str = "flat"):
        super().__init__()
        self.enc = build_entity_encoder(encoder, *layout_from_slots(SLOTS), ACTOR_IN)
        in_dim = self.enc.enc_dim if self.enc is not None else ACTOR_IN
        self.net = make_mlp(in_dim, A.N, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.enc is not None:
            x = self.enc(x)
        return self.net(x)                  # raw logits [..., A.N]

    def log_probs(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return masked_log_softmax(self.forward(x), mask)

    @staticmethod
    def entropy(logp: torch.Tensor) -> torch.Tensor:
        """Entropy of a masked log-prob distribution (illegal terms contribute 0)."""
        p = logp.exp()
        return -(p * torch.where(torch.isinf(logp), torch.zeros_like(logp), logp)).sum(-1)
