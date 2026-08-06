"""Scenario #5 — survive a lethal Dandân attack (manufactured; deny the swing).

The opponent is a minimal bot: 1–3 untapped Dandâns (a 4/1 each — "this creature
can't attack unless defending player controls an Island"), a random mana base of
4–10 lands including at least one Island (so the Dandâns aren't state-sacrificed),
NO cards in hand, driven by the engine heuristic. The agent starts on 4 life — so
a SINGLE Dandân is exactly lethal — with its own random 4–10 land mana base that
likewise includes an Island (the one the Dandâns need to be able to attack it) and
a RANDOM 4–7 card NONLAND grip. The scenario starts on the AGENT's turn and ends
once the bot's first turn has FULLY passed (tracked by the engine's per-player-turn
counter, so a bot turn containing no agent decision at all — e.g. the bot just
draws a land — still ends the scenario on time, before the bot can dig into a fresh
Dandân on a later turn).

By default the attack is live: the agent controls an Island, so on its turn the
heuristic declares the Dandâns and swings for lethal. The agent wins only by
SURVIVING — remove or bounce the Dandâns, deploy a blocker, gain life past the
swing, or strip itself of Islands so the Dandâns can't attack at all. If the bot's
turn passes with the agent still alive, the agent is credited the win; otherwise
the lethal swing resolves to the natural p2 win.

Like ``known_threat_random`` this is a denial/recognition test with a random grip:
the measure is whether the agent answers (or recognises it can't answer) a known,
telegraphed lethal threat with whatever it happens to hold.
"""
from __future__ import annotations

from fishrl.train.scenarios.base import Scenario
from fishrl.train.scenarios.surgery import (is_island, is_land, pool_all_zones,
                                            put_battlefield, put_hand,
                                            rebuild_library)

DANDAN = "Dandân"            # 4/1; can't attack unless the DEFENDER controls an Island
P1_LIFE = 4                  # a single 4-power Dandân is exactly lethal
N_DANDAN_MIN, N_DANDAN_MAX = 1, 3
HAND_MIN, HAND_MAX = 4, 7
LANDS_MIN, LANDS_MAX = 4, 10  # random mana base per side (every land taps for U)
# Both sides are dealt at least one Island: the agent so the Dandâns CAN attack it,
# the bot so its Dandâns aren't state-sacrificed for controlling no Island.


class SurviveLethalScenario(Scenario):
    name = "survive_lethal"
    pool_seed = 606
    engine_seat = "p2"
    # Variant tuning knobs (subclasses override; defaults reproduce the base frame
    # byte-identically — the rng draw order is unchanged when the hooks are unset):
    n_dandan_min, n_dandan_max = N_DANDAN_MIN, N_DANDAN_MAX
    hand_min, hand_max = HAND_MIN, HAND_MAX
    CURATED_CARD = None          # a subclass names a card to guarantee in the agent's grip
    CURATED_CHOICES: tuple = ()  # ...or a set to draw ONE guaranteed card from per sample
    FILLER_EXCLUDE: tuple = ()   # names never dealt as random filler (library-only instead)

    def predicate(self, env) -> bool:
        # Any clean p1 main-phase priority; the manufacture rebuilds both seats and
        # only needs the deck (pooled across zones) to contain a Dandân + an Island.
        g = env.g
        p = g.pending
        if not (p is not None and p.player == "p1" and p.type == "priority"
                and g.active_player == "p1"  # held_step: p1 priority no longer implies p1's turn
                and g.current_step in ("main1", "main2") and not g.stack):
            return False
        has_dandan = has_island = False
        for iid in g.objects:
            o = g.objects[iid]
            if o.name == DANDAN:
                has_dandan = True
            elif is_island(o):
                has_island = True
            if has_dandan and has_island:
                return True
        return False

    def _manufacture(self, g, rng) -> None:
        pool = pool_all_zones(g)
        islands, other_lands, dandans, nonland = [], [], [], []
        for iid in pool:
            o = g.objects.get(iid)
            if o is None:
                continue
            if is_island(o):                       # the basic the Dandâns key off of
                islands.append(iid)
            elif is_land(o):                       # other lands -> library only (not the enabler)
                other_lands.append(iid)
            elif o.name == DANDAN:
                dandans.append(iid)
            else:
                nonland.append(iid)
        assert dandans, "survive_lethal: no Dandân in the deck"
        assert len(islands) >= 2, "survive_lethal: need an Island per seat"
        rng.shuffle(islands); rng.shuffle(other_lands)
        rng.shuffle(dandans); rng.shuffle(nonland)

        # the bot: ready Dandâns (1–3 in the base frame), empty hand
        n_dandan = min(int(rng.integers(self.n_dandan_min, self.n_dandan_max + 1)),
                       len(dandans))
        for _ in range(n_dandan):
            put_battlefield(g, "p2", dandans.pop())
        assert sum(1 for i in g.players["p2"].battlefield
                   if g.objects[i].name == DANDAN) == n_dandan >= self.n_dandan_min

        # both sides: at least one Island (the agent so the Dandâns can attack, the bot
        # so they aren't sacrificed), then a random mana base on top (4–10 lands each,
        # any land type — every land taps for U; unused lands stay in the library)
        n_lands = {seat: int(rng.integers(LANDS_MIN, LANDS_MAX + 1)) for seat in ("p1", "p2")}
        for seat in ("p1", "p2"):
            put_battlefield(g, seat, islands.pop())
            n_lands[seat] -= 1
        mana_base = islands + other_lands           # leftover lands, a random mix
        assert len(mana_base) >= n_lands["p1"] + n_lands["p2"], "survive_lethal: short on lands"
        rng.shuffle(mana_base)
        for seat in ("p1", "p2"):
            for _ in range(max(0, n_lands[seat])):
                put_battlefield(g, seat, mana_base.pop())
            assert any(is_island(g.objects[i]) for i in g.players[seat].battlefield)

        # the agent: 4 life and a random NONLAND grip (leftover Dandâns count as
        # nonland cards too — a possible blocker). A curated subclass guarantees one
        # named answer in the grip — a fixed CURATED_CARD (e.g. Vision Charm), or a
        # per-sample rng draw from CURATED_CHOICES; the rest of the grip stays
        # random, minus any FILLER_EXCLUDE names (those go to the library instead).
        g.players["p1"].life = P1_LIFE
        grip_pool = nonland + dandans
        rng.shuffle(grip_pool)
        n_hand = int(rng.integers(self.hand_min, self.hand_max + 1))
        curated = self.CURATED_CARD
        if self.CURATED_CHOICES:
            curated = self.CURATED_CHOICES[int(rng.integers(len(self.CURATED_CHOICES)))]
        hand, skipped = [], []
        if curated is not None:
            idx = next((i for i, iid in enumerate(grip_pool)
                        if g.objects[iid].name == curated), None)
            assert idx is not None, \
                f"survive_lethal: curated card {curated!r} not found in the deck"
            hand.append(grip_pool.pop(idx))
        while len(hand) < n_hand and grip_pool:
            iid = grip_pool.pop()
            (skipped if g.objects[iid].name in self.FILLER_EXCLUDE else hand).append(iid)
        for iid in hand:
            put_hand(g, "p1", iid)
        assert len(g.players["p1"].hand) >= self.hand_min, "survive_lethal: short grip"

        # library: everything left (undealt lands + leftover grip + excluded filler);
        # nothing stacked on top
        rest = mana_base + grip_pool + skipped
        rng.shuffle(rest)
        rebuild_library(g, rest)

    def on_reset(self, env) -> None:
        env.scn_ctx["turn0"] = env.g.turn_number   # the agent's turn at scenario start

    def terminator(self, env):
        g = env.g
        # `turn_number` increments once per PLAYER-turn (engine._begin_turn), so the
        # bot's turn is turn0+1 and it has FULLY passed once the count reaches
        # turn0+2 with the agent active again. Counting turns (rather than sampling
        # `active_player` at agent decision points) is robust to a bot turn that
        # contains no agent decision at all — with default stops the engine
        # fast-forwards straight through such a turn, and crediting the win late
        # would let the bot draw into a fresh Dandân and flip a deserved win.
        if (g.active_player == "p1" and g.turn_number >= env.scn_ctx["turn0"] + 2
                and g.players["p1"].life > 0):
            return "p1"        # the bot's turn passed back to a still-living agent
        return None            # a lethal swing instead resolves to the natural p2 win


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


# One-card answers to a lone Dandân, each exercising a decision surface the
# 2026-08-07 weakness probe measured at ~0 exemplar agreement (memory:
# weakness-probe-v13): Mind Bend / Crystal Spray answer through
# `choose_text_change` (rewrite "Island" out of the attack clause — the deck's
# removal-by-text line the agent never finds), Metamorphose through targeting
# (library-top bounce; its put-onto-battlefield rider is dead against this
# bot's EMPTY hand, so here it is clean removal).
ANSWERS = ("Metamorphose", "Mind Bend", "Crystal Spray")


class SurviveLethalSingleScenario(SurviveLethalScenario):
    """survive_lethal narrowed to a SINGLE Dandân and a single guaranteed answer.

    The bot has exactly ONE Dandân; the agent's grip is one rng-chosen card from
    ``ANSWERS`` plus 0–6 random nonland cards — with Vision Charm excluded from
    the random filler, so the land-mode dodge the `_vision` variant trains can't
    substitute for the targeted-answer line this one measures. Life stays at 4:
    the single Dandân is exactly lethal, and the win condition (survive the bot's
    turn) is unchanged. Against `survive_lethal` (can the agent answer with a
    random grip?) and `_vision` (does it find the known clean dodge?), this asks:
    does it find and correctly APPLY the deck's actual removal — the text-change
    choice or the bounce — when one is guaranteed to be in hand?"""
    name = "survive_lethal_single"
    pool_seed = 717
    n_dandan_min = n_dandan_max = 1
    hand_min, hand_max = 1, 7            # the answer + 0–6 random cards
    CURATED_CHOICES = ANSWERS
    FILLER_EXCLUDE = (VISION,)

    def predicate(self, env) -> bool:
        # the base frame, plus every candidate answer present to deal (the
        # per-sample rng may pick any of them)
        if not super().predicate(env):
            return False
        names = {env.g.objects[i].name for i in env.g.objects}
        return all(a in names for a in ANSWERS)
