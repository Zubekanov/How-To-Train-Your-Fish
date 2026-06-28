"""Scenario base class.

A Scenario shapes the INITIAL-STATE DISTRIBUTION (which states the agent practices
from) and the TERMINATION/win-condition — never the reward type. Reward stays
terminal ±1 (the env applies it from `terminator`'s result). Each scenario:
  * `predicate(env)`   — when (during pool building) a state is a valid start-state.
  * `on_reset(env)`    — capture scenario-start references into `env.scn_ctx`.
  * `terminator(env)`  — return 'p1'/'p2'/'draw' to end early (terminal win/loss),
                         or None to keep playing (natural game-end still applies).
The pool is built lazily on first `sample` and cached on the instance.
"""
from __future__ import annotations

import copy


class Scenario:
    name = "base"
    # pool build budget; override per scenario as needed
    pool_n = 200
    pool_seed = 12345
    pool_max_games = 6000
    pool_max_decisions = 400

    def __init__(self):
        self._pool = None
        self._pool_games = 0

    # ── to override ──────────────────────────────────────────────────────────
    def predicate(self, env) -> bool:
        raise NotImplementedError

    def on_reset(self, env) -> None:
        pass

    def terminator(self, env):
        return None

    # ── pool management ──────────────────────────────────────────────────────
    def ensure_pool(self) -> int:
        if self._pool is None:
            from fishrl.train.scenarios.pool import build_snapshots
            self._pool, self._pool_games = build_snapshots(
                self.predicate, self.pool_n, seed=self.pool_seed,
                max_games=self.pool_max_games, max_decisions=self.pool_max_decisions)
            if not self._pool:
                raise RuntimeError(
                    f"scenario {self.name!r}: found 0 start-states in "
                    f"{self._pool_games} games — loosen the predicate or raise the budget")
        return len(self._pool)

    def sample(self, rng):
        """A fresh (deep-copied, independently mutable) start-state from the pool."""
        self.ensure_pool()
        return copy.deepcopy(self._pool[int(rng.integers(len(self._pool)))])
