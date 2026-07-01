"""Scenario #4 — deckout / card-advantage race (manufactured from the full 80 deck).

Rather than fishing a low-library state out of self-play, we MANUFACTURE it: pool
every card across all zones, exile the creatures (so no combat path exists), then
deal each seat an equal random number of untapped lands (4–10, taken from the
deck) and a 7-card hand (any cards), keep 40 cards in the shared library, and put
the rest in the graveyard. p2 is the heuristic. With no creatures the game can only
resolve by decking, so the terminator defers to the natural result.
"""
from __future__ import annotations

from fishrl.forgetful_fish import engine as E
from fishrl.train.scenarios.base import Scenario
from fishrl.train.scenarios.surgery import (is_land, pool_all_zones, put_battlefield,
                                            put_exile, put_graveyard, put_hand,
                                            rebuild_library)

LIBRARY = 40         # cards kept in the shared library (half of the 80-card deck)
HAND = 7
LANDS_MIN, LANDS_MAX = 4, 10


class DeckoutScenario(Scenario):
    name = "deckout"
    pool_seed = 505
    engine_seat = "p2"

    def predicate(self, env) -> bool:
        # Any clean p1 main-phase priority (empty stack) — the manufacture rebuilds
        # every zone, so the snapshot only needs to be a safe decision point.
        g = env.g
        p = g.pending
        return (p is not None and p.player == "p1" and p.type == "priority"
                and g.current_step in ("main1", "main2")
                and not g.stack and 2 <= g.turn_number <= 12)

    def _manufacture(self, g, rng) -> None:
        n = int(rng.integers(LANDS_MIN, LANDS_MAX + 1))   # equal random lands per seat
        # 1) pool every instance, emptying every zone
        pool = pool_all_zones(g)
        # 2) exile creatures; split the rest into lands / nonlands
        lands, nonlands = [], []
        for iid in pool:
            o = g.objects.get(iid)
            if o is None:
                continue
            if E._is_creature(o):
                put_exile(g, iid)
            elif is_land(o):
                lands.append(iid)
            else:
                nonlands.append(iid)
        assert len(lands) >= 2 * n, "deckout: not enough lands to deal both seats"
        rng.shuffle(lands); rng.shuffle(nonlands)
        # 3) N untapped lands per seat
        for seat in ("p1", "p2"):
            for _ in range(n):
                put_battlefield(g, seat, lands.pop())
        # 4) 7-card hands (any cards), then 40-card library, rest to graveyard
        remaining = lands + nonlands
        rng.shuffle(remaining)
        assert len(remaining) >= 2 * HAND, "deckout: not enough cards for the hands"
        for seat in ("p1", "p2"):
            for _ in range(HAND):
                put_hand(g, seat, remaining.pop())
        rebuild_library(g, remaining[:LIBRARY])
        for iid in remaining[LIBRARY:]:
            put_graveyard(g, iid)
        for seat in ("p1", "p2"):                 # the invariants the docstring promises
            assert len(g.players[seat].hand) == HAND
            assert sum(1 for i in g.players[seat].battlefield
                       if is_land(g.objects[i])) == n

    def terminator(self, env):
        # Creatureless -> deckout is the only resolution, and it always arrives
        # (Day's Undoing exiles itself; nothing reintroduces the exiled creatures).
        return None
