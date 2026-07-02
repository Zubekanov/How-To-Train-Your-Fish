"""Scenario #2 — condensed board / stack fight over the pivotal creature (manufactured).

Manufacture, don't fish from self-play: pool the whole deck, then give BOTH seats
4 life, exactly one creature on the battlefield, an equal random number of untapped
lands (4–10; the FIRST land each seat gets is Island-TYPED so its Dandân is never
state-sacrificed for want of an Island, the rest are a random mix of land types —
every land taps for U; the unused lands stay in the library), and a 7-card nonland
grip — and remove every OTHER creature from the deck. At 4 life a single 4-power
Dandân is lethal, so the game is a tight fight over that one creature (protect
yours / remove theirs / counter the removal). There is no "empty board = loss" rule
(that would teach "creatures are sacred"): if both creatures die, the deck is
creatureless, so the game simply continues to a deckout. p2 is the heuristic; the
terminator defers to the natural result.
"""
from __future__ import annotations

from fishrl.forgetful_fish import engine as E
from fishrl.train.scenarios.base import Scenario
from fishrl.train.scenarios.surgery import (is_island, is_land, pool_all_zones,
                                            put_battlefield, put_exile, put_hand,
                                            rebuild_library)

LANDS_MIN, LANDS_MAX = 4, 10
HAND = 7
LIFE = 4


def _creatures(g, seat) -> int:
    return sum(1 for iid in g.players[seat].battlefield
               if iid in g.objects and E._is_creature(g.objects[iid]))


class BoardPresenceScenario(Scenario):
    name = "board_presence"
    pool_seed = 404
    engine_seat = "p2"

    def predicate(self, env) -> bool:
        g = env.g
        p = g.pending
        return (p is not None and p.player == "p1" and p.type == "priority"
                and g.active_player == "p1"  # held_step: p1 priority no longer implies p1's turn
                and g.current_step in ("main1", "main2")
                and not g.stack and 2 <= g.turn_number <= 12)

    def _manufacture(self, g, rng) -> None:
        n = int(rng.integers(LANDS_MIN, LANDS_MAX + 1))
        pool = pool_all_zones(g)
        creatures, islands, other_lands, spells = [], [], [], []
        for iid in pool:
            o = g.objects.get(iid)
            if o is None:
                continue
            if E._is_creature(o):
                creatures.append(iid)
            elif is_island(o):                    # Island-TYPED (type_line) — the guarantee pool
                islands.append(iid)
            elif is_land(o):                      # any other land -> the random mana-base pool
                other_lands.append(iid)
            else:
                spells.append(iid)
        assert len(creatures) >= 2, "board_presence: need a creature per seat"
        assert len(islands) >= 2, "board_presence: need an Island-typed land per seat"
        assert len(islands) + len(other_lands) >= 2 * n, "board_presence: not enough lands"
        assert len(spells) >= 2 * HAND, "board_presence: not enough nonland spells"
        rng.shuffle(creatures); rng.shuffle(islands)
        rng.shuffle(other_lands); rng.shuffle(spells)

        for seat in ("p1", "p2"):                 # one creature each (ready to attack)
            put_battlefield(g, seat, creatures.pop())
        for iid in creatures:                     # every OTHER creature off the deck
            put_exile(g, iid)
        for seat in ("p1", "p2"):                 # 1 Island-typed land each FIRST, so a
            put_battlefield(g, seat, islands.pop())  # seat's Dandân is never state-sacrificed
        lands = islands + other_lands             # the rest of the mana base is a random mix
        rng.shuffle(lands)
        for seat in ("p1", "p2"):                 # N random untapped lands each (lethal is on)
            for _ in range(n - 1):
                put_battlefield(g, seat, lands.pop())
        for seat in ("p1", "p2"):                 # 7-card nonland grip
            for _ in range(HAND):
                put_hand(g, seat, spells.pop())
            g.players[seat].life = LIFE
        remaining = lands + spells                # creatureless library (leftover lands kept)
        rng.shuffle(remaining)
        rebuild_library(g, remaining)
        for seat in ("p1", "p2"):                 # the invariants the docstring promises
            assert _creatures(g, seat) == 1, f"board_presence: {seat} must have exactly 1 creature"
            assert any(is_island(g.objects[i]) for i in g.players[seat].battlefield), \
                f"board_presence: {seat} controls no Island-typed land"
            assert len(g.players[seat].hand) == HAND

    def terminator(self, env):
        # No empty-board rule: natural result (creature fight, or deckout if both die).
        return None
