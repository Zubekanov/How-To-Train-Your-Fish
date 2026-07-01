"""Snapshot-pool builder: roll out games and freeze GameStates at skill-moments.

A scenario's start-states come from REAL play (random self-play here), snapshotted
(``copy.deepcopy``) the instant a predicate holds. Because ``GameState`` is plain
data (its RNG is a serialized ``rng_state`` list rebuilt on demand), a deepcopy is a
complete, legal, reachable state — no hand-authored invariants to get wrong. The
builder (env-side ``_builder``) is NOT part of ``g``, so a snapshot of a compound
pending (e.g. declare_attackers) restores as a FRESH decision with nothing picked.
"""
from __future__ import annotations

import copy
import numpy as np

from fishrl.env.aec_env import FishAEC


def build_snapshots(predicate, n: int, seed: int = 0, max_games: int = 6000,
                    max_decisions: int = 400, per_game_cap: int = 3):
    """Random self-play until `n` states satisfy `predicate(env)` (or `max_games`
    exhausted). Returns (states, games_played). `per_game_cap` limits snapshots per
    game so the pool isn't dominated by one long correlated rollout, and at most ONE
    snapshot is taken per (game, turn) — consecutive qualifying decision points within
    a turn are near-identical states, so the cap is spread across the game instead.
    Deterministic: predicate/snapshotting never touches `rng`, so the same seed
    replays the same rollouts and freezes the same states."""
    rng = np.random.default_rng(seed)
    out: list = []
    gi = 0
    while len(out) < n and gi < max_games:
        env = FishAEC(max_decisions=max_decisions)
        env.reset(seed=seed * 1_000_003 + gi)
        gi += 1
        taken = 0
        taken_turns: set = set()
        guard = 0
        while env.agents and guard < max_decisions * 6 and len(out) < n:
            guard += 1
            s = env.agent_selection
            if env.terminations[s] or env.truncations[s]:
                env.step(None)
                continue
            if (taken < per_game_cap and env.g.turn_number not in taken_turns
                    and predicate(env)):
                out.append(copy.deepcopy(env.g))
                taken += 1
                taken_turns.add(env.g.turn_number)
            mask = env.observe(s)["action_mask"]
            legal = np.flatnonzero(mask)
            if legal.size == 0:
                env.step(None)
                continue
            env.step(int(rng.choice(legal)))
    return out, gi
