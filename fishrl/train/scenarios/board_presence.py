"""Scenario #2 — condensed board / stack fight over the pivotal creature (manufactured).

Manufacture, don't fish from self-play: pool the whole deck, then give BOTH seats
4 life, exactly one creature on the battlefield, an equal random number of untapped
Islands (4–10), and a 7-card nonland grip — and remove every OTHER creature from the
deck. At 4 life a single 4-power Dandân is lethal, so the game is a tight fight over
that one creature (protect yours / remove theirs / counter the removal). There is no
"empty board = loss" rule (that would teach "creatures are sacred"): if both
creatures die, the deck is creatureless, so the game simply continues to a deckout.
p2 is the heuristic; the terminator defers to the natural result.
"""
from __future__ import annotations

from fishrl.forgetful_fish import engine as E
from fishrl.forgetful_fish.state import LibrarySlot
from fishrl.train.scenarios.base import Scenario
from fishrl.train.scenarios.surgery import make_engine_heuristic, pool_all_zones

LANDS_MIN, LANDS_MAX = 4, 10
HAND = 7
LIFE = 4


def _creatures(g, seat) -> int:
    return sum(1 for iid in g.players[seat].battlefield
               if iid in g.objects and E._is_creature(g.objects[iid]))


class BoardPresenceScenario(Scenario):
    name = "board_presence"
    pool_seed = 404

    def predicate(self, env) -> bool:
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
        n = int(rng.integers(LANDS_MIN, LANDS_MAX + 1))
        pool = pool_all_zones(g)
        creatures, islands, spells, other_lands = [], [], [], []
        for iid in pool:
            o = g.objects.get(iid)
            if o is None:
                continue
            if E._is_creature(o):
                creatures.append(iid)
            elif "Island" in (o.type_line or ""):
                islands.append(iid)
            elif "Land" in (o.type_line or ""):
                other_lands.append(iid)           # non-Island lands stay in the deck (library)
            else:
                spells.append(iid)
        rng.shuffle(creatures); rng.shuffle(islands); rng.shuffle(spells)
        rng.shuffle(other_lands)

        def put(seat, iid, tapped=False):
            o = g.objects[iid]
            o.tapped = tapped; o.controller = seat; o.entered_this_turn = False
            g.players[seat].battlefield.append(iid)

        for seat in ("p1", "p2"):                 # one creature each (ready to attack)
            if creatures:
                put(seat, creatures.pop())
        for iid in creatures:                     # every OTHER creature off the deck
            g.exile.append(iid)
        for seat in ("p1", "p2"):                 # N untapped Islands each (lethal is on)
            for _ in range(n):
                if not islands:
                    break
                put(seat, islands.pop())
        for seat in ("p1", "p2"):                 # 7-card nonland grip
            for _ in range(HAND):
                if not spells:
                    break
                iid = spells.pop()
                g.objects[iid].controller = seat
                g.players[seat].hand.append(iid)
            g.players[seat].life = LIFE
        remaining = islands + spells + other_lands   # creatureless library (non-Island lands kept)
        rng.shuffle(remaining)
        g.library = [LibrarySlot(instance_id=iid, known_by={"p1": False, "p2": False})
                     for iid in remaining]
        g.graveyard = []

    def terminator(self, env):
        # No empty-board rule: natural result (creature fight, or deckout if both die).
        return None
