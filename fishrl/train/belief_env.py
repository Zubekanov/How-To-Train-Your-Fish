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
    def __init__(self, guesser, belief: bool = True, env=None, mode: str | None = None,
                 **env_kwargs):
        # `env` lets a caller inject a pre-built FishAEC subclass (e.g. a scenario env);
        # default constructs a plain FishAEC so existing callers are unchanged.
        # `mode` (Config.belief_mode) is the source of truth when given:
        #   "guesser"    -> the learned channel (legacy; requires `guesser`)
        #   "bookkeeper" -> the analytic hand-bookkeeper vector (stateless, no net)
        #   "none"       -> zeros
        # The legacy (guesser, belief) pair maps onto it for existing callers.
        self.env = env if env is not None else FishAEC(**env_kwargs)
        self.guesser = guesser
        if mode is None:
            mode = "guesser" if (belief and guesser is not None) else "none"
        assert mode in ("guesser", "bookkeeper", "none"), mode
        self.mode = mode
        self.belief = mode != "none"  # legacy readers
        self.last_guess: dict[str, np.ndarray] = {}
        # The input the guesser consumed at each seat's latest observe() — i.e. the
        # carried PREVIOUS guess. The collector buffers this as the training-time
        # guesser input so training conditions on exactly what inference fed.
        # (bookkeeper mode is stateless: both dicts stay zeros.)
        self.last_prev: dict[str, np.ndarray] = {}

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
    def decision_id(self) -> int:
        return self.env.decision_id

    @property
    def winner(self):
        # Scenario terminator result if one fired, else the engine's terminal winner
        # (identical to g.result's winner for normal self-play).
        return self.env.winner

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
        self.last_prev = {a: z.copy() for a in self.env.possible_agents}

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
        if self.mode == "bookkeeper":             # analytic vector; stateless, no net
            from fishrl.data.features import bookkeeper_counts
            bk = bookkeeper_counts(self.env.g, agent)
            self.last_prev[agent] = np.zeros(V.N_NAMES, dtype=np.float32)
            return {"observation": np.concatenate([persp, bk]).astype(np.float32),
                    "action_mask": base["action_mask"]}
        if not self.belief:                       # ablation: no belief, no guesser forward
            z = np.zeros(V.N_NAMES, dtype=np.float32)
            self.last_prev[agent] = z
            return {"observation": np.concatenate([persp, z]).astype(np.float32),
                    "action_mask": base["action_mask"]}
        prev = self.last_guess.get(agent, np.zeros(V.N_NAMES, dtype=np.float32))
        self.last_prev[agent] = prev
        dev = device_of(self.guesser)
        with torch.no_grad():
            guess = self.guesser(
                torch.as_tensor(persp, dtype=torch.float32).unsqueeze(0).to(dev),
                torch.as_tensor(prev, dtype=torch.float32).unsqueeze(0).to(dev),
            ).squeeze(0).cpu().numpy().astype(np.float32)
        self.last_guess[agent] = guess
        return {"observation": np.concatenate([persp, guess]).astype(np.float32),
                "action_mask": base["action_mask"]}
