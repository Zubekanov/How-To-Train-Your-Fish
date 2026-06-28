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
from fishrl.forgetful_fish.state import LibrarySlot
from fishrl.train.scenarios.base import Scenario
from fishrl.train.scenarios.surgery import make_engine_heuristic

LIBRARY = 40         # cards kept in the shared library (half of the 80-card deck)
HAND = 7


def _is_land(o) -> bool:
    return "Land" in (o.type_line or "")


class DeckoutScenario(Scenario):
    name = "deckout"
    pool_seed = 505

    def predicate(self, env) -> bool:
        # Any clean p1 main-phase priority (empty stack) — the manufacture rebuilds
        # every zone, so the snapshot only needs to be a safe decision point.
        g = env.g
        p = g.pending
        return (p is not None and p.player == "p1" and p.type == "priority"
                and g.current_step in ("main1", "main2")
                and not g.stack and 2 <= g.turn_number <= 12)

    def sample(self, rng):
        g = super().sample(rng)
        self._manufacture(g, rng)
        make_engine_heuristic(g, "p2")
        return g

    def _manufacture(self, g, rng) -> None:
        n = int(rng.integers(4, 11))          # equal random lands per seat, 4..10
        # 1) pool every instance, emptying every zone
        pool = []
        for pid in g.players:
            pool += g.players[pid].hand; g.players[pid].hand = []
            pool += g.players[pid].battlefield; g.players[pid].battlefield = []
        pool += [s.instance_id for s in g.library]; g.library = []
        pool += list(g.graveyard); g.graveyard = []
        pool += list(g.exile); g.exile = []
        pool += [s.source_instance_id for s in g.stack if s.source_instance_id]
        g.stack = []
        # 2) exile creatures; split the rest into lands / nonlands
        lands, nonlands = [], []
        for iid in pool:
            o = g.objects.get(iid)
            if o is None:
                continue
            if E._is_creature(o):
                g.exile.append(iid)
            elif _is_land(o):
                lands.append(iid)
            else:
                nonlands.append(iid)
        rng.shuffle(lands); rng.shuffle(nonlands)
        # 3) N untapped lands per seat
        for seat in ("p1", "p2"):
            for _ in range(n):
                if not lands:
                    break
                iid = lands.pop()
                o = g.objects[iid]
                o.tapped = False; o.controller = seat; o.entered_this_turn = False
                g.players[seat].battlefield.append(iid)
        # 4) 7-card hands (any cards), then 40-card library, rest to graveyard
        remaining = lands + nonlands
        rng.shuffle(remaining)
        for seat in ("p1", "p2"):
            for _ in range(HAND):
                if not remaining:
                    break
                iid = remaining.pop()
                g.objects[iid].controller = seat
                g.players[seat].hand.append(iid)
        for i, iid in enumerate(remaining):
            if i < LIBRARY:
                g.library.append(LibrarySlot(instance_id=iid, known_by={"p1": False, "p2": False}))
            else:
                g.graveyard.append(iid)

    def terminator(self, env):
        # Creatureless -> deckout is the only resolution, and it always arrives
        # (Day's Undoing exiles itself; nothing reintroduces the exiled creatures).
        return None
