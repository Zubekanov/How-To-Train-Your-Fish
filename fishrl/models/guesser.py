"""Hand-guesser: predict the opponent's hand as per-name expected counts.

Input is the acting seat's fair perspective observation
(:func:`fishrl.obs.encoder.encode_observation`, which already contains the seat's
own hand and the public board) concatenated with the seat's PREVIOUS guess vector.
Output is a length-``N_NAMES`` non-negative expected-count vector (softplus). It is
trained by Poisson NLL against :func:`fishrl.data.features.opponent_hand_counts`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from fishrl.models.mlp import make_mlp
from fishrl.obs.encoder import OBS_DIM
from fishrl.obs import vocab as V

GUESS_DIM = V.N_NAMES
GUESSER_IN = OBS_DIM + GUESS_DIM


class HandGuesser(nn.Module):
    def __init__(self, hidden=(256, 256)):
        super().__init__()
        self.net = make_mlp(GUESSER_IN, GUESS_DIM, hidden)

    def forward(self, perspective: torch.Tensor, prev_guess: torch.Tensor) -> torch.Tensor:
        """Return expected counts (>=0), shape [..., N_NAMES]."""
        x = torch.cat([perspective, prev_guess], dim=-1)
        return Fn.softplus(self.net(x))
