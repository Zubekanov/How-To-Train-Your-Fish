"""Hand-guesser: predict the opponent's hand as per-name expected counts.

Input is the acting seat's fair perspective observation
(:func:`fishrl.obs.encoder.encode_observation`, which already contains the seat's
own hand and the public board) concatenated with the seat's PREVIOUS guess vector.
Output is a length-``N_NAMES`` non-negative expected-count vector (softplus).

Training regime is **supervised-only**: the guess is consumed by the actor as a
fixed feature (computed under ``no_grad`` in the belief env and baked into the
stored observation), so the ONLY gradient into this network is the Poisson NLL
against :func:`fishrl.data.features.opponent_hand_counts`. It is an honest
posterior over the opponent's hand, never trained end-to-end through the policy.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from fishrl.models.entity_encoder import build_entity_encoder, layout_from_slots
from fishrl.models.mlp import make_mlp
from fishrl.obs.encoder import OBS_DIM, SLOTS
from fishrl.obs import vocab as V

GUESS_DIM = V.N_NAMES
GUESSER_IN = OBS_DIM + GUESS_DIM


class HandGuesser(nn.Module):
    def __init__(self, hidden=(256, 256), encoder: str = "flat"):
        super().__init__()
        self.enc = build_entity_encoder(encoder, *layout_from_slots(SLOTS), GUESSER_IN)
        in_dim = self.enc.enc_dim if self.enc is not None else GUESSER_IN
        self.net = make_mlp(in_dim, GUESS_DIM, hidden)

    def forward(self, perspective: torch.Tensor, prev_guess: torch.Tensor) -> torch.Tensor:
        """Return expected counts (>=0), shape [..., N_NAMES]."""
        x = torch.cat([perspective, prev_guess], dim=-1)
        if self.enc is not None:
            x = self.enc(x)
        return Fn.softplus(self.net(x))
