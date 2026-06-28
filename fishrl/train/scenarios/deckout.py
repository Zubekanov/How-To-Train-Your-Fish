"""Scenario #4 — deckout / card-advantage race.

Surgery forces the game to resolve through the deckout race rather than combat:
clear both battlefields of creatures and trim the SHARED library to a handful of
cards, with p2 on the heuristic. The episode plays to a natural finish, which from
this state is almost always a deckout (drawing from an empty library loses). Reward
is the natural terminal winner — the agent practises the late-game card/deckout race
deliberately, instead of only stumbling into it.

NOTE (honest limitation): the library is SHARED, so the deckout outcome is heavily
influenced by turn parity and the few available mill/draw levers; this scenario is
better understood as "exposure to deckout-race states" than a crisp single skill.
A short horizon backstops the rare game that finds damage instead.
"""
from __future__ import annotations

from fishrl.train.scenarios.base import Scenario
from fishrl.train.scenarios.surgery import (
    clear_battlefield_creatures, make_engine_heuristic, trim_library,
)

LIBRARY = 8          # cards left in the shared library after trimming
HORIZON = 8          # backstop: turn-numbers before scoring if no natural deckout


class DeckoutScenario(Scenario):
    name = "deckout"
    pool_seed = 505

    def predicate(self, env) -> bool:
        g = env.g
        p = g.pending
        return (p is not None and p.player == "p1" and p.type == "priority"
                and g.current_step in ("main1", "main2")
                and 4 <= g.turn_number <= 10
                and len(g.library) >= LIBRARY + 4)

    def sample(self, rng):
        g = super().sample(rng)
        clear_battlefield_creatures(g)
        trim_library(g, LIBRARY)
        make_engine_heuristic(g, "p2")
        return g

    def on_reset(self, env) -> None:
        env.scn_ctx["turn0"] = env.g.turn_number

    def terminator(self, env):
        # Pure natural resolution (deckout): None defers to the engine winner. The
        # horizon only backstops a game that somehow avoids decking out.
        g = env.g
        if g.turn_number - env.scn_ctx["turn0"] >= HORIZON:
            return "draw"
        return None
