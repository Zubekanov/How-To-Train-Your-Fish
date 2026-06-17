"""Belief-augmented env: fold the hand-guesser's output into the actor's obs.

Wraps :class:`fishrl.env.aec_env.FishAEC`. Owns the (frozen, eval-mode) guesser
and a PER-SEAT carried previous guess (both seats interleave, so a shared belief
would leak one seat's guess into the other's input). ``observe`` runs the guesser
on the seat's fair perspective ⊕ its previous guess and returns the perspective
concatenated with the new guess; the action mask is passed through unchanged.
"""
from __future__ import annotations

import numpy as np
import torch

from fishrl.env.aec_env import FishAEC
from fishrl.models import device_of
from fishrl.obs import vocab as V


class BeliefAugmentedEnv:
    def __init__(self, guesser, belief: bool = True, **env_kwargs):
        self.env = FishAEC(**env_kwargs)
        self.guesser = guesser
        self.belief = belief          # when False, feed zeros for the belief channel
        self.last_guess: dict[str, np.ndarray] = {}

    # ── delegated state ──────────────────────────────────────────────────────
    @property
    def g(self):
        return self.env.g

    @property
    def agents(self):
        return self.env.agents

    @property
    def agent_selection(self):
        return self.env.agent_selection

    @property
    def terminations(self):
        return self.env.terminations

    @property
    def truncations(self):
        return self.env.truncations

    @property
    def rewards(self):
        return self.env.rewards

    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed, options=options)
        z = np.zeros(V.N_NAMES, dtype=np.float32)
        self.last_guess = {a: z.copy() for a in self.env.possible_agents}

    def agent_iter(self, max_iter: int = 2 ** 31):
        i = 0
        while self.env.agents and i < max_iter:
            yield self.env.agent_selection
            i += 1

    def step(self, action):
        self.env.step(action)

    def observe(self, agent: str) -> dict:
        base = self.env.observe(agent)
        persp = base["observation"]
        if not self.belief:                       # ablation: no belief, no guesser forward
            z = np.zeros(V.N_NAMES, dtype=np.float32)
            return {"observation": np.concatenate([persp, z]).astype(np.float32),
                    "action_mask": base["action_mask"]}
        prev = self.last_guess.get(agent, np.zeros(V.N_NAMES, dtype=np.float32))
        dev = device_of(self.guesser)
        with torch.no_grad():
            guess = self.guesser(
                torch.as_tensor(persp, dtype=torch.float32).unsqueeze(0).to(dev),
                torch.as_tensor(prev, dtype=torch.float32).unsqueeze(0).to(dev),
            ).squeeze(0).cpu().numpy().astype(np.float32)
        self.last_guess[agent] = guess
        return {"observation": np.concatenate([persp, guess]).astype(np.float32),
                "action_mask": base["action_mask"]}
