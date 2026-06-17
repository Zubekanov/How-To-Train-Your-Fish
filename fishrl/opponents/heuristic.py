"""Evaluate a single policy against the engine's built-in heuristic AI.

This uses the engine's *sandbox* mode: p1 is the controlled (learning) seat, p2 is
the vendored heuristic AI (:mod:`fishrl.forgetful_fish.ai`), which the engine
drives itself. Only p1 ever receives a pending decision, so the same action /
mask / observation machinery as the AEC env applies — driven for one seat.

Use this for evaluation and curriculum, not training (the opponent is fixed).
"""
from __future__ import annotations

import numpy as np

from fishrl.forgetful_fish import engine as E
from fishrl.forgetful_fish.cards import load_decklist
from fishrl.env.apply import apply_atomic
from fishrl.env.driver import STOPS_MODES, is_terminal, terminal_winner
from fishrl.obs.encoder import encode_observation
from fishrl.spaces import action_space as A
from fishrl.spaces.compound import COMPOUND_TYPES, CompoundBuilder
from fishrl.spaces.masking import atomic_mask

SEAT = "p1"


class HeuristicMatch:
    """A single-seat (p1) driver versus the engine heuristic AI (p2)."""

    def __init__(self, stops_mode: str = "default", max_decisions: int = 4000):
        self.stops_mode = stops_mode
        self.max_decisions = max_decisions

    def reset(self, seed=None) -> dict:
        self.g = E.new_sandbox_game(load_decklist(), seed=seed, p1_name="p1",
                                    ai_profile="heuristic")
        self.g.players["p2"].name = "Heuristic AI"
        E.set_player_stops(self.g, SEAT, STOPS_MODES[self.stops_mode])
        self._builder = None
        self._decisions = 0
        self._sync()
        return self._obs()

    def _sync(self) -> None:
        """Ensure a compound builder exists iff p1's current pending needs one."""
        pend = self.g.pending
        if pend is not None and pend.player == SEAT and pend.type in COMPOUND_TYPES:
            if self._builder is None:
                self._builder = CompoundBuilder(self.g, SEAT, pend.type, pend.context or {})
        else:
            self._builder = None

    def _obs(self) -> dict:
        if self._builder is not None:
            mask = self._builder.mask()
        elif self.g.pending is not None and self.g.pending.player == SEAT:
            mask = atomic_mask(self.g, SEAT)
        else:
            mask = np.zeros(A.N, dtype=np.int8)
        prog = self._builder.progress() if self._builder is not None else 0.0
        return {"observation": encode_observation(self.g, SEAT, prog),
                "action_mask": mask.astype(np.int8)}

    def step(self, action: int):
        """Apply p1's action; the engine runs p2 forward. Returns (obs, reward,
        done, info). Reward is terminal ±1 for p1."""
        g = self.g
        pend = g.pending
        assert pend is not None and pend.player == SEAT, "not p1's decision"
        if self._builder is not None:
            if self._builder.feed(g, int(action)):
                self._builder = None
                self._decisions += 1
        else:
            if not apply_atomic(g, SEAT, int(action)):
                raise AssertionError(f"illegal action {A.decode(int(action))}")
            self._decisions += 1
        self._sync()
        done = is_terminal(g) or self._decisions >= self.max_decisions
        reward = 0.0
        if is_terminal(g):
            w = terminal_winner(g)
            reward = 0.0 if w is None else (1.0 if w == SEAT else -1.0)
        return self._obs(), reward, done, {"status": g.result.get("status")}


def evaluate(policy, n_games: int = 20, seed: int = 0, stops_mode: str = "default") -> dict:
    """Win/loss record of `policy` (a callable taking an observation -> action id)
    against the heuristic AI over `n_games`."""
    wins = losses = draws = 0
    for i in range(n_games):
        m = HeuristicMatch(stops_mode=stops_mode)
        obs = m.reset(seed=seed + i)
        done = False
        r = 0.0
        guard = 0
        while not done and guard < 20000:
            guard += 1
            obs, r, done, _ = m.step(policy(obs))
        if r > 0:
            wins += 1
        elif r < 0:
            losses += 1
        else:
            draws += 1
    return {"games": n_games, "wins": wins, "losses": losses, "draws": draws,
            "win_rate": wins / n_games}
