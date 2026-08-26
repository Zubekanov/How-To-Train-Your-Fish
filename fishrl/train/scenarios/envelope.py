"""Envelope-calibrated state construction for the scenario curriculum.

The manufactured scenarios before 2026-08-21 rebuilt the board from hand-picked
constants (both players at 4 life, every other Dandân exiled, ten lands on turn 2,
...). Those states are legal but lie in corners no real game visits, and what a
policy learns in a corner does not have to transfer. This module replaces the
constants with a measured ENVELOPE: per turn-bucket distributions of life, lands,
Island-typed lands, hand size, fish, library and graveyard sizes, plus the
graveyard's card mix — all taken from 2,000 traced heuristic games (v1.3 mirror +
v1.2 vs v1.3, the 2026-08-21 Field Guide). A scenario is then just a handful of
OVERRIDES on top of one shared sampler, and "looks like a real game" is a property
of the sampler rather than of each scenario's author.

Construction rules the sampler enforces (each one a thing real games always do):
  * life is 20 - 4k (Dandân is the only damage source);
  * every seat that controls a fish controls >= 1 Island-typed land (else the fish
    would already have been state-sacrificed);
  * zone conservation: library = 80 - hands - battlefields - graveyard - exile.
    Nothing is deleted from the deck — the other Dandâns sit in the library;
  * the graveyard is drawn from the real cast mix (dead fish, Lapses, cantrips,
    removal), exile holds only resolved Day's Undoings;
  * lands on the active player's side are untapped at its main phase; the
    non-active side carries the taps its own turn left behind.

Reward is untouched: the scenario env still scores the natural game result.
"""
from __future__ import annotations

import numpy as np

from fishrl.forgetful_fish import engine as E
from fishrl.train.scenarios.surgery import (is_island, is_land, pool_all_zones,
                                            put_battlefield, put_exile, put_graveyard,
                                            put_hand, rebuild_library)

DANDAN = "Dandân"
UNDOING = "Day's Undoing"

# ── the envelope (p10..p90 bands / categorical probabilities per bucket) ──────
# Buckets are indexed by player-turn number; "lib12" is the library<=12 endgame
# regardless of turn. Values from the 2026-08-21 per-turn snapshot census.
ENVELOPE = {
    #            turn      life p(20,16,12,8,4)         lands   nonisl  hand    lib       gy
    # "1-5" measured 2026-08-24 from 600 traced v3-vs-1.3 games (turns 3-5, ~28k
    # decisions): life 20 in ~98%, hand p10-p90 5-7, library 61-64, fish 0/1.
    "1-5":    dict(turn=(3, 5),   life=(.97, .03, .00, .00, .00), lands=(2, 5),  hand=(5, 7), lib=(58, 65), gy=(0, 4)),
    "6-10":   dict(turn=(6, 10),  life=(.85, .10, .04, .01, .00), lands=(2, 4),  hand=(3, 6), lib=(55, 62), gy=(2, 8)),
    "11-16":  dict(turn=(11, 16), life=(.75, .12, .08, .04, .01), lands=(4, 7),  hand=(1, 6), lib=(41, 56), gy=(6, 18)),
    "17-24":  dict(turn=(17, 24), life=(.62, .15, .11, .08, .04), lands=(5, 10), hand=(1, 6), lib=(25, 47), gy=(11, 31)),
    "25+":    dict(turn=(25, 40), life=(.45, .18, .15, .12, .10), lands=(8, 14), hand=(0, 6), lib=(6, 36),  gy=(14, 44)),
    "lib12":  dict(turn=(28, 44), life=(.45, .18, .15, .12, .10), lands=(9, 15), hand=(0, 6), lib=(2, 12),  gy=(37, 48)),
}
LIFE_LEVELS = (20, 16, 12, 8, 4)
FISH_P = (0.71, 0.24, 0.05)                 # fish per player: 0 / 1 / 2
NONISLAND_P = (0.23, 0.31, 0.27, 0.14, 0.05)  # non-Island-typed lands per player: 0..4
# graveyard mix: expected cards per game (both seats) by name; lands enter via
# mills/cycles (small weights). Used as per-name sampling weight, divided by the
# number of copies so the weight is per physical card.
GY_MIX = {DANDAN: 6.3, "Memory Lapse": 6.8, "Accumulated Knowledge": 3.0, "Crystal Spray": 2.5,
          "Ponder": 2.5, "Brainstorm": 2.4, "Predict": 2.2, "Fact or Fiction": 1.9,
          "Mind Bend": 1.7, "Metamorphose": 1.5, "Vision Charm": 1.4, "Mystical Tutor": 1.3,
          UNDOING: 0.3, "Lonely Sandbar": 0.8, "The Surgical Bay": 0.3,
          "Island": 1.2, "Halimar Depths": 0.15, "Temple of Epiphany": 0.15,
          "Svyelunite Temple": 0.25, "Mystic Sanctuary": 0.15}
REMOVAL = ("Mind Bend", "Crystal Spray", "Metamorphose", "Vision Charm")
INSTANTS = ("Memory Lapse", "Crystal Spray", "Mind Bend", "Metamorphose", "Vision Charm",
            "Accumulated Knowledge", "Brainstorm", "Predict", "Mystical Tutor", "Fact or Fiction")
TOP_MANIP = ("Brainstorm", "Ponder", "Mystical Tutor", "Predict")
COST = {DANDAN: 2, "Mind Bend": 1, "Crystal Spray": 3, "Metamorphose": 2, "Vision Charm": 1,
        "Memory Lapse": 2, "Accumulated Knowledge": 2, "Brainstorm": 1, "Ponder": 1,
        "Predict": 2, "Mystical Tutor": 1, "Fact or Fiction": 4, UNDOING: 3}


def _cat(rng, probs):
    p = np.asarray(probs, dtype=float); p = p / p.sum()
    return int(rng.choice(len(p), p=p))


def _u(rng, lo, hi):
    return int(rng.integers(lo, hi + 1))


class Overrides:
    """The knobs a scenario sets on top of the envelope. Everything not set is
    sampled from the bucket. Fish counts are per seat; `fish_relation` re-samples
    until p1's count relates to p2's as asked ("ahead" / "behind" / "level")."""

    def __init__(self, *, p1_fish=None, p2_fish=None, total_fish_min=0, fish_relation=None,
                 p1_life_min=None, p1_life_max=None, lethal_on_p1=False,
                 p1_hand_require=(), p1_hand_require_one_of=(), p2_hand_require=(),
                 library=None, p1_untapped_min=0, p2_islands=None,
                 stack_spell=None, stack_target=None, top_known_to_p1=False):
        self.p1_fish, self.p2_fish = p1_fish, p2_fish
        self.total_fish_min, self.fish_relation = total_fish_min, fish_relation
        self.p1_life_min, self.p1_life_max, self.lethal_on_p1 = p1_life_min, p1_life_max, lethal_on_p1
        self.p1_hand_require = tuple(p1_hand_require)
        self.p1_hand_require_one_of = tuple(p1_hand_require_one_of)
        self.p2_hand_require = tuple(p2_hand_require)
        self.library = library
        self.p1_untapped_min = p1_untapped_min
        self.p2_islands = p2_islands
        self.stack_spell = stack_spell        # {"choices": {name: prob}} -> p2 spell put on the stack
        self.stack_target = stack_target      # None (fish-first legacy) | "fish" | "land"
        self.top_known_to_p1 = top_known_to_p1


def _pick(rng, v):
    """An override value may be an int, a (lo, hi) range or a {value: prob} dict."""
    if isinstance(v, dict):
        keys = list(v); return keys[_cat(rng, [v[k] for k in keys])]
    if isinstance(v, tuple):
        return _u(rng, v[0], v[1])
    return int(v)


def _pop_named(bag: list, g, name: str):
    for i, iid in enumerate(bag):
        if g.objects[iid].name == name:
            return bag.pop(i)
    return None


def _count_fish(g, seat) -> int:
    return sum(1 for i in g.players[seat].battlefield if i in g.objects and E._is_creature(g.objects[i]))


def _count_islands(g, seat) -> int:
    return sum(1 for i in g.players[seat].battlefield if i in g.objects and is_island(g.objects[i]))


def envelope_sample(g, rng, bucket: str, ov: Overrides | None = None) -> dict:
    """Rebuild `g` (a pooled skeleton) into an envelope-plausible start-state.
    Returns the sampled parameters (for tests / the plausibility audit)."""
    ov = ov or Overrides()
    env = ENVELOPE[bucket]
    active = g.active_player

    # ── 1. sample the per-seat parameters ─────────────────────────────────────
    for _ in range(200):
        fish = {s: (_pick(rng, getattr(ov, f"{s}_fish")) if getattr(ov, f"{s}_fish") is not None
                    else _cat(rng, FISH_P)) for s in ("p1", "p2")}
        if fish["p1"] + fish["p2"] < ov.total_fish_min:
            continue
        rel = ov.fish_relation
        if rel == "ahead" and not fish["p1"] > fish["p2"]: continue
        if rel == "behind" and not fish["p1"] < fish["p2"]: continue
        if rel == "level" and fish["p1"] != fish["p2"]: continue
        break
    life = {}
    for s in ("p1", "p2"):
        for _ in range(200):
            v = LIFE_LEVELS[_cat(rng, env["life"])]
            if s == "p1":
                if ov.p1_life_min is not None and v < ov.p1_life_min: continue
                if ov.p1_life_max is not None and v > ov.p1_life_max: continue
                if ov.lethal_on_p1 and v > 4 * fish["p2"]: continue
            break
        life[s] = v
    lands = {s: _u(rng, *env["lands"]) for s in ("p1", "p2")}
    nonisl = {}
    for s in ("p1", "p2"):
        n = min(_cat(rng, NONISLAND_P), lands[s] - 1)      # >= 1 Island-typed land always
        nonisl[s] = max(n, 0)
    if ov.p2_islands is not None:                          # e.g. exactly one Island (last-Island line)
        isl = _pick(rng, ov.p2_islands)
        lands["p2"] = max(lands["p2"], isl); nonisl["p2"] = lands["p2"] - isl
    if ov.p1_untapped_min:
        lands["p1"] = max(lands["p1"], ov.p1_untapped_min)
    # deck limits: 22 Island-typed (20 Island + 2 Sanctuary) and 12 other lands in
    # the whole deck; leave a couple of each for hands/library. Real late games sit
    # right at this ceiling, so clamp rather than reject.
    for _ in range(40):
        n_isl = {s: lands[s] - nonisl[s] for s in ("p1", "p2")}
        if sum(n_isl.values()) > 20:
            s = max(("p1", "p2"), key=lambda k: n_isl[k])
            if sum(nonisl.values()) < 10 and not (s == "p2" and ov.p2_islands is not None):
                nonisl[s] += 1                          # swap an Island for a utility land
            else:
                lands[s] -= 1                           # or just own one land fewer
        elif sum(nonisl.values()) > 10:
            s = max(("p1", "p2"), key=lambda k: nonisl[k]); nonisl[s] -= 1
        else:
            break
    hand = {s: _u(rng, *env["hand"]) for s in ("p1", "p2")}
    turn = _u(rng, *env["turn"])
    if (turn % 2 == 1) != (active == g.first_player):       # parity of turn_number ↔ active seat
        turn += 1

    # ── 2. empty every zone and sort the deck ─────────────────────────────────
    pool = pool_all_zones(g)
    creatures, islands, other_lands, spells = [], [], [], []
    for iid in pool:
        o = g.objects.get(iid)
        if o is None: continue
        (creatures if E._is_creature(o) else islands if is_island(o)
         else other_lands if is_land(o) else spells).append(iid)
    rng.shuffle(creatures); rng.shuffle(islands); rng.shuffle(other_lands); rng.shuffle(spells)

    # ── 3. battlefields ───────────────────────────────────────────────────────
    for s in ("p1", "p2"):
        n_isl = lands[s] - nonisl[s]
        assert len(islands) >= n_isl and len(other_lands) >= nonisl[s], "envelope: deck short of lands"
        for _ in range(n_isl): put_battlefield(g, s, islands.pop())
        for _ in range(nonisl[s]): put_battlefield(g, s, other_lands.pop())
        for _ in range(fish[s]): put_battlefield(g, s, creatures.pop())
    # tapped state: active seat untapped at its main phase; the other seat carries
    # 0..(lands-2) taps from its own turn; a stack spell taps p2 for its cost.
    for s in ("p1", "p2"):
        bf = [i for i in g.players[s].battlefield if is_land(g.objects[i])]
        if s == active and not (s == "p2" and ov.stack_spell):
            n_tap = _u(rng, 0, min(2, len(bf))) if s == "p2" else 0
        else:
            n_tap = _u(rng, 0, max(0, len(bf) - 2))
        if s == "p1":
            n_tap = min(n_tap, max(0, len(bf) - ov.p1_untapped_min))
        rng.shuffle(bf)
        for i in bf[:n_tap]: g.objects[i].tapped = True
    # summoning sickness: a fish on the active side may have been cast this turn
    for i in g.players[active].battlefield:
        if E._is_creature(g.objects[i]) and rng.random() < 0.3:
            g.objects[i].entered_this_turn = True

    # ── 4. hands: required cards first, then the natural (deck) mix ───────────
    rest = islands + other_lands + spells
    rng.shuffle(rest)
    req = {"p1": list(ov.p1_hand_require), "p2": list(ov.p2_hand_require)}
    if ov.p1_hand_require_one_of:
        req["p1"].append(str(rng.choice(list(ov.p1_hand_require_one_of))))
    stack_name = None
    if ov.stack_spell:
        ch = ov.stack_spell["choices"]
        stack_name = list(ch)[_cat(rng, list(ch.values()))]
    for s in ("p1", "p2"):
        for name in req[s]:
            iid = _pop_named(rest, g, name) or (_pop_named(creatures, g, name) if name == DANDAN else None)
            assert iid is not None, f"envelope: {name} not available for {s}'s hand"
            put_hand(g, s, iid)
        hand[s] = max(hand[s], len(g.players[s].hand))
    stack_iid = None
    if stack_name:
        stack_iid = _pop_named(rest, g, stack_name) or (_pop_named(creatures, g, stack_name) if stack_name == DANDAN else None)
        assert stack_iid is not None, f"envelope: {stack_name} not available for the stack"
    # fish in library: the remaining creatures join the deck pool
    rest += creatures
    rng.shuffle(rest)
    for s in ("p1", "p2"):
        while len(g.players[s].hand) < hand[s] and rest:
            put_hand(g, s, rest.pop())

    # ── 5. exile (resolved Undoings only), then library size, graveyard = rest ─
    if env["gy"][1] >= 14 and rng.random() < 0.25:
        iid = _pop_named(rest, g, UNDOING)
        if iid is not None: put_exile(g, iid)
    lib_n = _pick(rng, ov.library) if ov.library is not None else _u(rng, *env["lib"])
    lib_n = max(0, min(lib_n, len(rest)))
    gy_n = len(rest) - lib_n
    if ov.library is None:                                  # keep the graveyard inside its band
        lo, hi = env["gy"]
        if gy_n < lo: gy_n = min(lo, len(rest))
        if gy_n > hi: gy_n = hi
        lib_n = len(rest) - gy_n
    # graveyard drawn by the real cast mix (weight per physical card)
    copies = {}
    for iid in rest: copies[g.objects[iid].name] = copies.get(g.objects[iid].name, 0) + 1
    w = np.array([GY_MIX.get(g.objects[iid].name, 0.2) / copies[g.objects[iid].name] for iid in rest], dtype=float)
    gy_idx = set(rng.choice(len(rest), size=gy_n, replace=False, p=w / w.sum()).tolist()) if gy_n else set()
    gy = [iid for k, iid in enumerate(rest) if k in gy_idx]
    lib = [iid for k, iid in enumerate(rest) if k not in gy_idx]
    for iid in gy: put_graveyard(g, iid)
    rng.shuffle(lib)
    if ov.top_known_to_p1 and lib:
        rebuild_library(g, lib[1:], top=lib[0], top_known_by=("p1",))
    else:
        rebuild_library(g, lib)

    # ── 6. scalars: life, turn (held_step remapped), then the optional stack ───
    for s in ("p1", "p2"):
        g.players[s].life = life[s]
    old = g.turn_number
    g.turn_number = turn
    for p in g.players.values():
        if p.held_step and p.held_step[0] == old:
            p.held_step = [turn, p.held_step[1]]
    if stack_iid is not None:
        _put_spell_on_stack(g, "p2", stack_iid, rng, target=ov.stack_target)

    # ── invariants ────────────────────────────────────────────────────────────
    for s in ("p1", "p2"):
        assert _count_fish(g, s) == 0 or _count_islands(g, s) >= 1, "envelope: fish without an Island"
        assert life[s] in LIFE_LEVELS
    total = sum(len(g.players[s].hand) + len(g.players[s].battlefield) for s in g.players) \
        + len(g.library) + len(g.graveyard) + len(g.exile) + len(g.stack)
    assert total == len(g.objects), f"envelope: zone conservation broken ({total} != {len(g.objects)})"
    return {"bucket": bucket, "turn": turn, "fish": fish, "life": life, "lands": lands,
            "nonisland": nonisl, "hand": hand, "library": len(g.library), "graveyard": len(g.graveyard),
            "stack": stack_name}


def _put_spell_on_stack(g, caster: str, iid: str, rng, target: str | None = None) -> None:
    """p2 has cast `iid` and passed priority to p1 (the skeleton's priority state).
    Uses the engine's own play_card so the log, zone and knowledge bookkeeping are
    the real thing; targets are chosen legally for the board. `target` forces the
    class of the chosen target ("fish" / "land"; falls back to what exists) — the
    read_the_target scenario randomises it so the obs_ctx stack-target channel is
    the decisive input (2026-08-26)."""
    o = g.objects[iid]
    put_hand(g, caster, iid)
    if o.name == "Vision Charm":
        o.chosen["mode"] = "land"
    ok = E.play_card(g, caster, iid)
    assert ok, "envelope: play_card refused the stack spell"
    so = g.stack[-1]
    so.stack_id = f"scn{int(rng.integers(1 << 30)):08x}"
    if o.name in ("Mind Bend", "Crystal Spray", "Metamorphose"):
        opp = "p1" if caster == "p2" else "p2"
        perms = list(g.players[opp].battlefield)
        fish = [i for i in perms if E._is_creature(g.objects[i])]
        lands = [i for i in perms if is_land(g.objects[i])]
        pool = (lands if target == "land" and lands
                else fish if target == "fish" and fish
                else fish or perms)
        tgt = pool[int(rng.integers(len(pool)))]
        so.targets = [{"type": "object", "id": tgt}]
        g.log[-1] = g.log[-1].rstrip(".") + f" targeting {g.objects[tgt].name}."
    # pay for it: tap the caster's lands
    lands = [i for i in g.players[caster].battlefield if is_land(g.objects[i]) and not g.objects[i].tapped]
    for i in lands[:COST.get(o.name, 1)]:
        g.objects[i].tapped = True
    g.passed = {caster: True, ("p1" if caster == "p2" else "p2"): False}
    g.priority_player = "p1"
