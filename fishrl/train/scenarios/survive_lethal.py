"""Scenario #5 — survive a lethal Dandân attack (manufactured; deny the swing).

The opponent is a minimal bot: 1–3 untapped Dandâns (a 4/1 each — "this creature
can't attack unless defending player controls an Island"), a random mana base of
4–10 lands including at least one Island (so the Dandâns aren't state-sacrificed),
NO cards in hand, driven by the engine heuristic. The agent starts on 4 life — so
a SINGLE Dandân is exactly lethal — with its own random 4–10 land mana base that
likewise includes an Island (the one the Dandâns need to be able to attack it) and
a RANDOM 4–7 card NONLAND grip. The scenario starts on the AGENT's turn and ends
when the bot's first turn passes back to the agent.

By default the attack is live: the agent controls an Island, so on its turn the
heuristic declares the Dandâns and swings for lethal. The agent wins only by
SURVIVING — remove or bounce the Dandâns, deploy a blocker, gain life past the
swing, or strip itself of Islands so the Dandâns can't attack at all. If the bot
finds no attack and its turn passes back to the agent (still alive), the agent is
credited the win; otherwise the lethal swing resolves to the natural p2 win.

Like ``known_threat_random`` this is a denial/recognition test with a random grip:
the measure is whether the agent answers (or recognises it can't answer) a known,
telegraphed lethal threat with whatever it happens to hold.
"""
from __future__ import annotations

from fishrl.forgetful_fish.state import LibrarySlot
from fishrl.train.scenarios.base import Scenario
from fishrl.train.scenarios.surgery import make_engine_heuristic, pool_all_zones

DANDAN = "Dandân"            # 4/1; can't attack unless the DEFENDER controls an Island
P1_LIFE = 4                  # a single 4-power Dandân is exactly lethal
N_DANDAN_MIN, N_DANDAN_MAX = 1, 3
HAND_MIN, HAND_MAX = 4, 7
LANDS_MIN, LANDS_MAX = 4, 10  # random mana base per side (every land taps for U)
# Both sides are dealt at least one Island: the agent so the Dandâns CAN attack it,
# the bot so its Dandâns aren't state-sacrificed for controlling no Island.


class SurviveLethalScenario(Scenario):
    name = "survive_lethal"
    pool_seed = 505
    CURATED_CARD = None          # a subclass names a card to guarantee in the agent's grip

    def predicate(self, env) -> bool:
        # Any clean p1 main-phase priority; the manufacture rebuilds both seats and
        # only needs the deck (pooled across zones) to contain a Dandân + an Island.
        g = env.g
        p = g.pending
        if not (p is not None and p.player == "p1" and p.type == "priority"
                and g.current_step in ("main1", "main2") and not g.stack):
            return False
        has_dandan = has_island = False
        for iid in g.objects:
            o = g.objects[iid]
            if o.name == DANDAN:
                has_dandan = True
            elif "Island" in (o.type_line or ""):
                has_island = True
            if has_dandan and has_island:
                return True
        return False

    def sample(self, rng):
        g = super().sample(rng)
        self._manufacture(g, rng)
        make_engine_heuristic(g, "p2")
        return g

    def _manufacture(self, g, rng) -> None:
        pool = pool_all_zones(g)
        islands, other_lands, dandans, nonland = [], [], [], []
        for iid in pool:
            o = g.objects.get(iid)
            if o is None:
                continue
            tl = o.type_line or ""
            if "Island" in tl:                     # the basic the Dandâns key off of
                islands.append(iid)
            elif "Land" in tl:                     # other lands -> library only (not the enabler)
                other_lands.append(iid)
            elif o.name == DANDAN:
                dandans.append(iid)
            else:
                nonland.append(iid)
        rng.shuffle(islands); rng.shuffle(other_lands)
        rng.shuffle(dandans); rng.shuffle(nonland)

        def put(seat, iid):                        # an untapped, ready (not sick) permanent
            o = g.objects[iid]
            o.tapped = False; o.controller = seat; o.entered_this_turn = False
            g.players[seat].battlefield.append(iid)

        # the bot: 1–3 ready Dandâns, empty hand
        n_dandan = int(rng.integers(N_DANDAN_MIN, N_DANDAN_MAX + 1))
        for _ in range(n_dandan):
            if dandans:
                put("p2", dandans.pop())
        g.players["p2"].hand = []

        # both sides: at least one Island (the agent so the Dandâns can attack, the bot
        # so they aren't sacrificed), then a random mana base on top (4–10 lands each,
        # any land type — every land taps for U; unused lands stay in the library)
        n_lands = {seat: int(rng.integers(LANDS_MIN, LANDS_MAX + 1)) for seat in ("p1", "p2")}
        for seat in ("p1", "p2"):
            if islands:
                put(seat, islands.pop())
                n_lands[seat] -= 1
        mana_base = islands + other_lands           # leftover lands, a random mix
        rng.shuffle(mana_base)
        for seat in ("p1", "p2"):
            for _ in range(max(0, n_lands[seat])):
                if mana_base:
                    put(seat, mana_base.pop())

        # the agent: 4 life and a 4–7 card NONLAND grip (leftover Dandâns count as
        # nonland cards too — a possible blocker). A curated subclass guarantees one
        # named answer (e.g. Vision Charm) in the grip; the rest of it stays random.
        g.players["p1"].life = P1_LIFE
        grip_pool = nonland + dandans
        rng.shuffle(grip_pool)
        n_hand = int(rng.integers(HAND_MIN, HAND_MAX + 1))
        hand = []
        if self.CURATED_CARD is not None:
            idx = next((i for i, iid in enumerate(grip_pool)
                        if g.objects[iid].name == self.CURATED_CARD), None)
            if idx is not None:
                hand.append(grip_pool.pop(idx))
        while len(hand) < n_hand and grip_pool:
            hand.append(grip_pool.pop())
        for iid in hand:
            g.objects[iid].controller = "p1"
        g.players["p1"].hand = hand

        # library: everything left (undealt lands + leftover grip); nothing stacked on top
        rest = mana_base + grip_pool
        rng.shuffle(rest)
        g.library = [LibrarySlot(instance_id=iid, known_by={"p1": False, "p2": False})
                     for iid in rest]
        g.graveyard = []

    def on_reset(self, env) -> None:
        env.scn_ctx["p2_seen"] = False

    def terminator(self, env):
        g = env.g
        c = env.scn_ctx
        if g.active_player == "p2":
            c["p2_seen"] = True
        # the bot's turn passed back to a still-living agent -> the swing was denied
        if c["p2_seen"] and g.active_player == "p1" and g.players["p1"].life > 0:
            return "p1"
        return None        # a lethal swing instead resolves to the natural p2 win


VISION = "Vision Charm"      # {U} instant; its land mode rewrites a basic land type


class SurviveLethalVisionScenario(SurviveLethalScenario):
    """survive_lethal with a guaranteed Vision Charm in the otherwise-random grip.

    Vision Charm's land mode ("each land of the first chosen type becomes the second
    chosen type until end of turn") rewrites the Island type off the board: with the
    Islands gone, the bot controls no Island, so its Dandâns are state-sacrificed (and
    even were they not, a defender with no Island can't be attacked by them). So a clean
    answer exists in hand — this variant measures whether the agent finds and casts it
    (in the right mode, on the right type) versus the random-grip base, where it usually
    can't. The {U} cost is trivially payable from the agent's Island mana base.
    """
    name = "survive_lethal_vision"
    pool_seed = 515
    CURATED_CARD = VISION

    def predicate(self, env) -> bool:
        # the random-grip frame, plus a Vision Charm somewhere in the deck to deal
        if not super().predicate(env):
            return False
        return any(env.g.objects[i].name == VISION for i in env.g.objects)
