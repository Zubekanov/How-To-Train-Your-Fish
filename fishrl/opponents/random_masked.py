"""Uniform-random-over-legal-actions policy — the simplest baseline opponent."""
from __future__ import annotations

import numpy as np


class RandomMaskedPolicy:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, observation: dict) -> int:
        legal = np.flatnonzero(observation["action_mask"])
        if legal.size == 0:
            raise ValueError("no legal actions in mask")
        return int(self.rng.choice(legal))
