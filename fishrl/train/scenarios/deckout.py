"""Scenario #4 — deckout / card-advantage race.

Surgery forces the game to resolve through the deckout race rather than combat:
EXILE every creature from all zones (a truly creatureless deck, so no Dandân can be
redrawn and recast into a combat win) and HALVE the remaining shared library (a
principled size relative to the actual game state — not an arbitrary fixed count),
with p2 on the heuristic. With no creatures, the game can only end by decking, so
the terminator defers to the natural result (the env's max_decisions truncates the
rare library-refill loop, e.g. Day's Undoing, as a draw). Reward is the natural
terminal winner — the agent practises the late-game card/deckout race deliberately.

NOTE (honest limitation): the library is SHARED, so the deckout outcome is heavily
influenced by turn parity and the few mill/draw levers; treat this as "exposure to
deckout-race states" rather than a crisp single skill.
"""
from __future__ import annotations

from fishrl.train.scenarios.base import Scenario
from fishrl.train.scenarios.surgery import (
    make_engine_heuristic, remove_all_creatures, trim_library,
)


class DeckoutScenario(Scenario):
    name = "deckout"
    pool_seed = 505

    def predicate(self, env) -> bool:
        g = env.g
        p = g.pending
        return (p is not None and p.player == "p1" and p.type == "priority"
                and g.current_step in ("main1", "main2")
                and 4 <= g.turn_number <= 10
                and len(g.library) >= 4)         # enough that halving still leaves a race

    def sample(self, rng):
        g = super().sample(rng)
        remove_all_creatures(g)                  # truly creatureless -> no combat path
        trim_library(g, len(g.library) // 2)     # half-sized deck (relative, not arbitrary)
        make_engine_heuristic(g, "p2")
        return g

    def terminator(self, env):
        # Creatureless -> the only resolution is a deckout; defer to the natural
        # engine result (max_decisions truncates a pathological refill loop as a draw).
        return None
