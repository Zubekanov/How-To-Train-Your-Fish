"""Scenario base class.

A Scenario shapes the INITIAL-STATE DISTRIBUTION (which states the agent practices
from) and the TERMINATION/win-condition — never the reward type. Reward stays
terminal ±1 (the env applies it from `terminator`'s result). Each scenario:
  * `predicate(env)`    — when (during pool building) a state is a valid start-state.
  * `_manufacture(g, rng)` — (optional) rewrite the sampled snapshot in place into
                          the scenario's constructed start-state (state surgery).
  * `on_reset(env)`     — capture scenario-start references into `env.scn_ctx`.
  * `terminator(env)`   — return 'p1'/'p2'/'draw' to end early (terminal win/loss),
                          or None to keep playing (natural game-end still applies).

The pool is built lazily on first `sample` and cached on the instance. `sample` is
a template method: deep-copy a pool state, clear the per-turn scalars it carries,
`_manufacture` it, reseed the engine PRNG, then hand `engine_seat` (if set) to the
heuristic AI.

The current scenarios are COMPLETE manufactures: `_manufacture` empties every zone
(`pool_all_zones`) and rebuilds the board / hands / library from scratch, so the
sampled game contributes NO board texture — only a legal, self-consistent state
*skeleton* (turn/step/priority machinery at proven-reachable values). Hence the pool
is small (`pool_n`): it exists to borrow a handful of internally-consistent skeletons
without hand-authoring invariant-laden fields (turn_number ↔ held_step ↔ passed), not
to sample real play. Per-episode variety comes from `_manufacture`'s `rng` (the board)
and the engine-PRNG reseed (in-play draws), both seed-derived, so a fixed reset seed
stays reproducible. The one hazard a real skeleton carries is TRANSIENT per-turn state
the manufacture doesn't rebuild (a spent land drop, floating mana, an auto-yield); we
scrub those in `_reset_transient` so they can't leak real-game texture into the start.
"""
from __future__ import annotations

import copy
import random

from fishrl.forgetful_fish.state import _serialize_rng
from fishrl.train.scenarios.surgery import make_engine_heuristic


class Scenario:
    name = "base"
    # Pool build budget. These are COMPLETE manufactures, so the pool only supplies a
    # legal state skeleton (see module docstring) — a small pool covers the range of
    # turn/step skeletons; per-episode variety is the manufacture rng + PRNG reseed,
    # not pool diversity. Keep it > ~a dozen so turn_number (2..12) is well spread.
    pool_n = 24
    pool_seed = 12345
    pool_max_games = 6000
    pool_max_decisions = 400
    # Seat handed to the engine's heuristic AI after `_manufacture` (None: both
    # seats stay learner-controlled — the base self-play contract).
    engine_seat: str | None = None
    # Which vendored heuristic drives that seat. v1.2 (the current testbench line)
    # since 2026-07-10; earlier scenario games ran v1.0, so scenario_wr trends have
    # a step DOWN at the upgrade (stronger opponent), not a regression in the agent.
    engine_profile: str = "heuristic_1_2"

    def __init__(self):
        self._pool = None
        self._pool_games = 0

    # ── to override ──────────────────────────────────────────────────────────
    def predicate(self, env) -> bool:
        raise NotImplementedError

    def _manufacture(self, g, rng) -> None:
        """Rewrite the sampled snapshot in place. Base: no-op (snapshot as-is)."""

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
        """A fresh (deep-copied, independently mutable) start-state from the pool:
        transient per-turn scalars cleared, `_manufacture`d, engine PRNG reseeded,
        with `engine_seat` (if any) handed to the heuristic AI."""
        self.ensure_pool()
        g = copy.deepcopy(self._pool[int(rng.integers(len(self._pool)))])
        self._reset_transient(g)           # clear per-turn scalars the manufacture won't rebuild
        self._manufacture(g, rng)          # rebuild the board (consumes `rng` — unchanged behavior)
        self._reseed_engine_rng(g, rng)    # after the board, so a fixed seed still builds it byte-identically
        if self.engine_seat is not None:
            make_engine_heuristic(g, self.engine_seat, self.engine_profile)
        for pid in g.players:              # the leak this guards against must stay closed
            assert not g.players[pid].land_played_this_turn and not g.players[pid].mana_pool, \
                f"scenario {self.name!r}: transient base-game state leaked past the manufacture ({pid})"
        return g

    @staticmethod
    def _reset_transient(g) -> None:
        """Clear the per-turn bookkeeping a sampled skeleton carries but the manufacture
        never rebuilds — floating mana, the spent land drop, tap-undo, and any rest-of-turn
        auto-yield. Zone contents / knowledge / text are handled by `scrub` during the
        manufacture; this is the per-player scalar state that would otherwise leak real-game
        texture into the start — most visibly a land drop the agent never made, which the
        play-a-land mask reads straight off `land_played_this_turn`. (turn_number / step /
        held_step / passed are LEFT intact: they form the internally-consistent priority
        skeleton we deliberately borrow, and editing them by hand is what desyncs.)"""
        for p in g.players.values():
            p.mana_pool = {}
            p.land_played_this_turn = False
            p.tap_undo = []
            p.yield_mode = ""
            p.yield_here = {}

    @staticmethod
    def _reseed_engine_rng(g, rng) -> None:
        """Give the engine PRNG a fresh, seed-derived stream. A small skeleton pool means
        the same base state is drawn many times; without this, every draw of it would replay
        the identical in-engine shuffles/coin-flips during play. Called AFTER `_manufacture`
        so the manufacture's use of `rng` (hence the board) is byte-identical to before."""
        r = random.Random(int(rng.integers(1 << 63)))
        g.rng_state = _serialize_rng(r)
