"""The heuristic sandbox AI (p2): a card-aware policy that plays lands, casts
and responds with spells, attacks and blocks when profitable, and makes a
sensible choice for every resolution decision. The engine calls in through a
handful of hooks (take_priority, resolve_pending, choose_attackers,
choose_blocks, choose_trigger_target, choose_discards) whenever the pending
action belongs to an AI with ai_profile == "heuristic_1_3".

This is heuristic v1.3 — the current testbench MAINLINE (c0b2e05: adversarial-
agency Day's Undoing insurance, on top of the instant-speed-removal /
Day's-Undoing-gate / belief-audit line since v1.2; cumulative ~83% vs v1.0 in
the testbench arena). The belief-sampled evaluator folded into this file ships
with its master switch OFF (`_EVAL_GATES = False`) — that is what "v1.3
mainline" means; the evaluator-on variant is the testbench's side line
(v1.3-oracle, ai_oracle.py) and is NOT vendored. It lives behind its own
engine profile so it can serve as a SEPARATE PFSP pool opponent alongside
v1.1 (ai_v1_1.py) and v1.2 (ai_v1_2.py), both frozen at their releases; v1.0
(fishrl/forgetful_fish/ai.py) remains the default "heuristic" profile: the
eval anchor, the scenario bot, and the run's long-standing baseline. New
heuristic versions land as ai_v1_X.py + a profile, never by editing released
versions or replacing ai.py.

Fairness: the AI reads only public information, its own hand, and library
slots whose LibrarySlot.known_by marks them as seen by it — never the human's
hand or unknown library cards.

Control flow: the engine is re-entrant — an action taken here (play a land,
cast a spell) runs the game forward inside the call, eventually handing the
AI priority again, which re-enters take_priority for its next decision. Each
take_priority call therefore performs at most ONE action and returns; a
per-turn budget caps runaway loops.
"""
from __future__ import annotations

import copy as _copy
import math as _math
import pickle as _pickle
import random as _random
import sys as _sys

from fishrl.forgetful_fish import engine as E
from fishrl.forgetful_fish.state import (
    BASIC_TYPES, CYCLING, GameState, _is_land, enters_tapped, land_mana_color,
)

_MAX_ACTIONS_PER_TURN = 30   # belt-and-braces against policy bugs; not serialized


# ── Knowledge & evaluation helpers (no cheating) ───────────────────────────

def _known_top(g: GameState, player: str):
    """The top library card, but only if `player` has seen this slot."""
    if g.library and g.library[0].known_by.get(player):
        return g.objects.get(g.library[0].instance_id)
    return None


def _hand_cards(g: GameState, player: str) -> list:
    return [g.objects[iid] for iid in g.players[player].hand if iid in g.objects]


def _lands_in_play(g: GameState, player: str) -> int:
    return sum(1 for iid in g.players[player].battlefield
               if _is_land(g.objects[iid].type_line))


def _untapped_lands(g: GameState, player: str) -> list:
    return [iid for iid in g.players[player].battlefield
            if _is_land(g.objects[iid].type_line) and not g.objects[iid].tapped]


def _mana_view(g: GameState, player: str) -> dict:
    """Floating pool plus one symbol per untapped land (colour-aware, so a
    Vision-Charmed board taps for what it now actually produces)."""
    pool = dict(g.players[player].mana_pool)
    for iid in _untapped_lands(g, player):
        sym = land_mana_color(g.objects[iid])
        pool[sym] = pool.get(sym, 0) + 1
    return pool


def _affordable(g: GameState, player: str, cost: str) -> bool:
    colored, generic = E._parse_cost(cost)
    return E._can_afford(_mana_view(g, player), colored, generic)


def _sac_type(o) -> str | None:
    """The basic land type in a 'when you control no X, sacrifice' clause
    (Dandân: "Island"), read from the EFFECTIVE text so it tracks Mind Bend."""
    m = E._SAC_NO_TYPE_RE.search(o.oracle_text or "")
    if not m:
        return None
    typ = m.group(1).capitalize()
    return typ if typ in BASIC_TYPES else None


def _sac_creatures(g: GameState, player: str) -> list:
    """`player`'s creatures with a sacrifice clause (their Dandâns)."""
    return [iid for iid in g.players[player].battlefield
            if E._is_creature(g.objects[iid]) and _sac_type(g.objects[iid])]


def _creatures(g: GameState, player: str) -> list:
    return [iid for iid in g.players[player].battlefield
            if E._is_creature(g.objects[iid])]


# Card-value weights (hand / top-of-library desirability), centralised so they
# can be swept/tuned in one place; card_value reads only from here. Entries
# suffixed `_threat` apply when the card answers an opposing fish; the `land_*`
# tiers key off lands in play; the `AK_*` triple is Accumulated Knowledge's
# base + per-copy growth, capped. All values were hand-set, then arena-tuned.
_VALUE = {
    "land_lt4": 8.0, "land_lt6": 5.0, "land_ge6": 1.0, "land_utility_late": 0.5,
    "Dandân": 11.0,
    "Memory Lapse": 8.0,
    "Mind Bend": 9.0, "Mind Bend_threat": 11.0,
    "Fact or Fiction": 9.0,
    "Crystal Spray": 8.0, "Crystal Spray_threat": 10.0,
    "AK_base": 6.0, "AK_per": 4.0, "AK_cap": 11.0,
    "Predict": 6.0, "Predict_known": 10.0,
    "Metamorphose": 6.0,
    "Mystical Tutor": 5.0,
    "Brainstorm": 8.0,
    "Ponder": 7.0,
    "Vision Charm": 10.5,
    "Day's Undoing": 5.0,
    "default": 6.0,
}


def card_value(g: GameState, player: str, iid: str) -> float:
    """How much the AI wants this card in hand / on top of the library. Weights
    live in `_VALUE`; only the state-dependent branches (threat, land tiers, AK
    curve, deck-out) are decided here."""
    o = g.objects.get(iid)
    if not o:
        return 0.0
    name = o.name
    V = _VALUE
    if _is_land(o.type_line):
        lands = _lands_in_play(g, player)
        base = V["land_lt4"] if lands < 4 else (V["land_lt6"] if lands < 6 else V["land_ge6"])
        if name != "Island" and lands >= 4:
            base += V["land_utility_late"]                # utility lands edge out Islands late
        return base
    opp = E._OTHER[player]
    if name == "Mind Bend":
        return V["Mind Bend_threat"] if _sac_creatures(g, opp) else V["Mind Bend"]
    if name == "Crystal Spray":
        return V["Crystal Spray_threat"] if _sac_creatures(g, opp) else V["Crystal Spray"]
    if name == "Accumulated Knowledge":
        aks = sum(1 for cid in g.graveyard if g.objects[cid].name == "Accumulated Knowledge")
        base = min(V["AK_base"] + V["AK_per"] * aks, V["AK_cap"])
        # Library endgame: an AK in hand is mostly parity-suicide fuel (the
        # critic probe scores holding it at -3.5pp there); fade it so keeps/
        # putbacks stop hoarding the chain past the point it can fire.
        return min(base, 4.0) if _endgame(g) else base
    if name == "Predict":
        return V["Predict_known"] if _known_top(g, player) else V["Predict"]
    if name == "Day's Undoing":
        return 0.0 if _in_deckout_mode(g, player) else V["Day's Undoing"]  # a reset undoes the deck-out
    return V.get(name, V["default"])


# ── The priority driver ─────────────────────────────────────────────────────

def take_priority(g: GameState, player: str) -> None:
    """Called by the engine whenever the AI receives priority. Performs at most
    one action (the engine re-enters for the next decision) under a per-turn
    budget."""
    budget = getattr(g, "_ai_actions", None)
    if not budget or budget[0] != g.turn_number:
        budget = [g.turn_number, 0]
        g._ai_actions = budget                            # plain attr — not serialized
    budget[1] += 1
    if budget[1] > _MAX_ACTIONS_PER_TURN:                 # runaway safety: go passive
        E.pass_priority(g, player)
        return
    if _in_rollout:
        until = getattr(g, "_rollout_until", None)
        if until is not None and g.turn_number >= until:
            raise _RolloutHorizon()
    action = _choose_action(g, player)
    if _in_rollout:
        # Inside an evaluator rollout: plain heuristic, honouring a
        # hold-the-fish branch for the turn under evaluation.
        if (getattr(g, "_suppress_fish_turn", None) == g.turn_number
                and action[0] == "cast"
                and g.objects[action[1]].name == "Dandân"):
            action = _action_without_fish(g, player)
    elif _EVAL_GATES and g.turn_number >= _EVAL_MIN_TURN:
        action = _evaluator_gate(g, player, action)
    _execute(g, player, action)


def _execute(g: GameState, player: str, action: tuple) -> None:
    """Perform one chosen action; on any failure fall back to passing. After a
    successful action the engine has already continued the game re-entrantly."""
    kind = action[0]
    if kind == "land":
        if E.play(g, player, action[1]):
            return
    elif kind == "cycle":
        if E.cycle(g, player, action[1]):
            _finish_payment(g, player)
            return
    elif kind == "activate":
        _, iid, index = action
        if E.activate_ability(g, player, iid, index):
            _finish_payment(g, player)
            return
    elif kind == "cast":
        _, iid, mode, target = action
        if E.play(g, player, iid, hold=True, mode=mode):
            p = g.pending
            if p and p.type == "choose_targets" and p.player == player:
                if target is None or not E.complete_targets(g, player, [target]):
                    E.complete_targets(g, player, [], cancel=True)   # back out cleanly
                    E.pass_priority(g, player)
                    return
            _finish_payment(g, player)
            return
    if g.priority_player == player and g.pending and g.pending.type == "priority":
        E.pass_priority(g, player)


def _finish_payment(g: GameState, player: str) -> None:
    """Tap lands until a pending payment is covered (the cast/cycle completes
    inside the final tap). Affordability is pre-checked, so running out of
    lands means a policy bug — cancel rather than strand the payment."""
    guard = 0
    while (g.pending and g.pending.type == "pay" and g.pending.player == player
           and guard < 20):
        guard += 1
        land = _best_land_to_tap(g, player)
        if land is None or not E.tap(g, player, land):
            E.cancel_payment(g, player)                   # re-enters take_priority itself
            return


def _best_land_to_tap(g: GameState, player: str) -> str | None:
    """Which untapped land to tap for the pending payment. Tap by what a land
    actually PRODUCES, not its card name: a Mind-Bended Island is still named
    "Island" but now makes another colour, so tapping it for a {U} cost would
    only float dead mana. Prefer a land that pays a coloured pip we still need,
    then one that pays generic, and keep sacrifice outlets for last; never tap a
    land that can't pay anything toward what remains."""
    lands = _untapped_lands(g, player)
    pend = g.pending
    if not lands or not pend or pend.type != "pay":
        return None
    need = pend.context.get("need", {})
    generic = pend.context.get("generic", 0)

    def contribution(iid):
        sym = land_mana_color(g.objects[iid])
        if need.get(sym, 0) > 0:
            return 0                                      # covers a colour we still need
        if generic > 0:
            return 1                                      # covers generic
        return 2                                          # would only float — skip it

    def rank(iid):
        name = g.objects[iid].name
        if name == "Island":
            return 0
        if name in ("Svyelunite Temple", "The Surgical Bay"):
            return 2                                      # keep sacrifice outlets available
        return 1

    payable = [iid for iid in lands if contribution(iid) < 2]
    if not payable:
        return None
    return min(payable, key=lambda iid: (contribution(iid), rank(iid), iid))


# ── Choosing an action at priority ──────────────────────────────────────────

def _choose_action(g: GameState, player: str) -> tuple:
    if (g.active_player == player and g.current_step in E._MAIN_STEPS
            and not g.stack):
        if _in_deckout_mode(g, player):
            return _deckout_main_action(g, player)
        return _main_phase_action(g, player)
    return _response_action(g, player)


def _hand_by_name(g: GameState, player: str) -> dict:
    by = {}
    for iid in g.players[player].hand:
        o = g.objects.get(iid)
        if o:
            by.setdefault(o.name, []).append(iid)
    return by


# ── Toggleable strategic-dig logics (A/B sweep) ──────────────────────────────
# Each flag enables a stance-driven dig behaviour. ALL OFF (the default) is the
# committed baseline — no behaviour change. A tournament sets these per player.
_DIG_FLAGS = ("defend", "sweeper", "beatdown", "grind")
# Live-play defaults. A 320k-game A/B sweep found a small but consistent ~+0.4%
# edge for `sweeper` (dig for the Vision Charm board-wipe when being run over with
# no answer); `defend`/`beatdown`/`grind` stayed at noise, so they're off.
_DEFAULT_FLAGS = {"sweeper": True}
# Per-player overrides for A/B sweeps/tests; an explicit dict (even empty) for a
# player wins over _DEFAULT_FLAGS, so a sweep can isolate any toggle either way.
_PLAYER_FLAGS: dict = {}


def _flag(player: str, name: str) -> bool:
    flags = _PLAYER_FLAGS.get(player)
    if flags is None:
        flags = _DEFAULT_FLAGS                            # live play uses the defaults
    return bool(flags.get(name))


def _stance(g: GameState, player: str) -> str:
    """One label for the board state, driving what we dig/tutor/keep for. Only
    meaningful outside deck-out mode (which owns its own plan). Precedence:
    desperate > beatdown > defend > grind > develop."""
    if _in_deckout_mode(g, player):
        return "deckout"
    opp = E._OTHER[player]
    mine = len(_sac_creatures(g, player))
    theirs = len(_sac_creatures(g, opp))
    gap = theirs - mine                                   # +ve: behind on fish
    hand = _hand_by_name(g, player)
    have_answer = bool(hand.get("Crystal Spray") or hand.get("Mind Bend")
                       or hand.get("Vision Charm"))
    threatening = any(not E._attack_restricted(g, opp, a) for a in _sac_creatures(g, opp))
    if gap >= 2 and not have_answer:
        return "desperate"                                # being run over, no answer in hand
    if mine > theirs and _can_attack_opponent(g, player):
        return "beatdown"                                 # ahead on fish and able to close
    if 0 <= gap <= 1 or threatening:
        return "defend"                                   # even, or a fish is pointed at us
    if gap == 0 and g.players[player].hand and g.players[opp].hand:
        return "grind"                                    # stalled, both still loaded
    return "develop"


def _dig_bonus(g: GameState, player: str, iid: str) -> float:
    """Stance-driven bias added to card_value when CHOOSING what to keep/fetch
    (it never changes the in-hand value used elsewhere). Zero unless the matching
    flag is on, so all-off == baseline."""
    o = g.objects.get(iid)
    if not o:
        return 0.0
    name = o.name
    st = _stance(g, player)
    if st == "desperate" and _flag(player, "sweeper"):
        if name == "Vision Charm":
            return 10.0                                   # the sweeper we must find
        if name in ("Crystal Spray", "Mind Bend"):
            return 3.0                                    # single-target stopgap
        if name in ("Accumulated Knowledge", "Brainstorm", "Ponder",
                    "Predict", "Fact or Fiction"):
            return 1.0                                    # keep digging toward it
    elif st == "beatdown" and _flag(player, "beatdown"):
        if name == "Memory Lapse":
            return 3.0                                    # protect the clock
        if name == "Dandân":
            return 2.0
    elif st == "defend" and _flag(player, "defend"):
        if name in ("Crystal Spray", "Mind Bend"):
            return 4.0
        if name == "Vision Charm":
            return 2.0
        if name == "Metamorphose":
            return 1.0
    elif st == "grind" and _flag(player, "grind"):
        if name == "Memory Lapse":
            return 2.0
    return 0.0


def _stance_fetch(g: GameState, player: str, eligible: list):
    """Mystical Tutor target dictated by an active dig flag — the highest dig
    value among the eligible instants/sorceries — or None to fall back to the
    wishlist."""
    if not any(_flag(player, f) for f in _DIG_FLAGS):
        return None
    best = max(eligible, key=lambda iid: _dig_bonus(g, player, iid), default=None)
    if best is not None and _dig_bonus(g, player, best) > 0:
        return best
    return None


def _recover_top_action(g: GameState, player: str):
    """Grind: on our own turn a known valuable top (e.g. our Dandân just Memory-
    Lapsed there) goes to the OPPONENT's draw step next, so draw it back with an
    instant-speed draw instead of gifting it."""
    if not _flag(player, "grind") or g.active_player != player:
        return None
    if _stance(g, player) != "grind":
        return None
    top = _known_top(g, player)
    if top is None or _is_land(top.type_line) or card_value(g, player, top.instance_id) < 6.0:
        return None
    return _instant_top_draw(g, player, reserve=_counter_reserve(g, player))


def _main_phase_action(g: GameState, player: str) -> tuple:
    """Own main phase, empty stack: land drop, removal, threats, card advantage."""
    hand = _hand_by_name(g, player)
    opp = E._OTHER[player]

    land = _choose_land_drop(g, player, hand)
    if land is not None:
        return ("land", land)

    removal = _removal_action(g, player, hand)
    if removal is not None:
        return removal

    recover = _recover_top_action(g, player)              # grind: take our countered threat back
    if recover is not None:
        return recover

    # Threat: cast Dandân whenever affordable (it can't attack this turn anyway,
    # so either main phase is fine) — but ONLY if it survives: with no
    # Island-typed land of ours in play, the fish's own trigger sacrifices it
    # on resolution. Field log: the AI once donated three consecutive Dandâns
    # this way off a Halimar/Sandbar manabase and was beaten to death with no
    # blockers. The sac type is read from the card's EFFECTIVE text (a
    # Mind-Bent fish needs the rewritten land type instead).
    for iid in hand.get("Dandân", []):
        if _affordable(g, player, g.objects[iid].mana_cost):
            typ = _sac_type(g.objects[iid]) or "Island"
            if E.controls_basic_type(g, player, typ):
                return ("cast", iid, None, None)

    # Library endgame: steer the deck-out parity in EVERY game once the shared
    # library is small (emptying it outright wins on the spot). This is the
    # same steering deck-out mode uses; here it runs even while combat is live.
    if _endgame(g):
        lib = _deckout_library_action(g, player, hand)
        if lib is not None:
            return lib
        insurance = _deckout_insurance_action(g, player, hand)
        if insurance is not None:
            return insurance

    draw = _card_advantage_action(g, player, hand)
    if draw is not None:
        return draw

    # Day's Undoing: refill an empty hand when the opponent is far ahead on
    # cards (main2, so the whole turn was used first). Don't fire when we know
    # the top card — that means we've set the library up (scry/Brainstorm/etc.)
    # and the reshuffle would throw that away. (A swing-based gate,
    # min(7, theirs) - ours >= 3, measured exactly 50.0% — same trigger set in
    # practice, so the simpler hand-size gate stays.)
    # Pressure gate: the reset is symmetric — the player with the winning
    # board gets a fresh seven too, and resolution ENDS OUR TURN, so nothing
    # from our new hand deploys before their attack. Mirror logs: 9/200 games
    # the loser cast this within a round of dying, one even Lapse-protecting
    # it at 8 life into two fish, dying to the swing with two Dandâns in the
    # new hand — with all mana tapped by the war, the fresh seven could not
    # even respond. But facing lethal is not automatically a fold: with no
    # answer in the current hand and mana still OPEN after the cast, the new
    # seven can find an instant for their turn (Metamorphose bounces an
    # attacker; Vision Charm phases their Islands out and their fish sac
    # themselves), where passing just dies. So under a lethal swing (counting
    # our best blocks) the cast is allowed exactly as that Hail Mary: no
    # Metamorphose / Vision Charm already held (casting would shuffle the
    # answer away), and >= 2 mana spare after paying — every answer costs
    # <= 2. The far-behind-on-cards gates don't apply to a Hail Mary.
    if g.current_step == "main2" and hand.get("Day's Undoing"):
        # Rollout audit (2026-08-04, 933 determinized decline decisions):
        # the old refill gate (hand <= 2 AND their hand >= 4 AND surviving
        # the swing) declined casts that measured strongly positive -- cast
        # when 2+ cards BEHIND (-0.149 t=-8.0; 3+ behind -0.205) or facing
        # attackers (-0.064 t=-4.9), decline in quiet parity (+0.073
        # validates the pass). The deficit, not our absolute hand size, is
        # the load-bearing variable; the known-top guard stays (a reshuffle
        # throws our arrangement away). Arena 51.56% at 40k on top of the
        # instant-removal change; 100k confirm 51.56% [51.3, 51.9].
        # Boundary sweep (round 10): deficit >= 1 beats >= 2 (50.5% at 40k,
        # 50.41% [50.1, 50.7] at 100k) and >= 3 loses (49.3%) -- monotone
        # toward the looser gate, so any deficit at all justifies the refill.
        iid = hand["Day's Undoing"][0]
        if (_affordable(g, player, g.objects[iid].mana_cost)
                and len(g.players[player].hand) <= 4
                and _known_top(g, player) is None):
            deficit = (len(g.players[opp].hand)
                       - len(g.players[player].hand))
            attackers = [a for a in g.players[opp].battlefield
                         if E._is_creature(g.objects[a])
                         and not E._attack_restricted(g, opp, a)]
            if deficit >= 1 or attackers:
                return ("cast", iid, None, None)

    # NO flood valves: cycling a Sandbar or sacking the Bay "for value" when
    # flooded burns the deck-out war chest. These guaranteed one-card draws
    # are pure parity ammunition — the race, endgame-flip, and deck-out paths
    # spend them correctly. Swept the value-spend threshold 8 -> never:
    # monotone, with never-spend worth +4.8pp (54.8/54.9 on independent
    # seeds) over spending when flooded. Field logs showed a human banking
    # exactly these from the mid-game and unloading them at the death.
    return ("pass",)


def _choose_land_drop(g: GameState, player: str, hand: dict) -> str | None:
    """Which land (if any) to put down this turn: an untapped one when the turn
    has casting plans, otherwise the best tapped utility land."""
    if g.players[player].land_played_this_turn:
        return None
    land_iids = [iid for iid in g.players[player].hand
                 if _is_land(g.objects[iid].type_line)]
    if not land_iids:
        return None
    untapped = [iid for iid in land_iids if not enters_tapped(g, player, g.objects[iid])]
    tapped = [iid for iid in land_iids if iid not in untapped]
    # Hold Lonely Sandbar to cycle once the board is flooded.
    if _lands_in_play(g, player) >= 5:
        tapped = [iid for iid in tapped if g.objects[iid].name != "Lonely Sandbar"]
    has_spells = any(not _is_land(o.type_line) for o in _hand_cards(g, player))
    if untapped and has_spells:
        # Mana now: prefer a plain Island, keep special lands for later turns.
        return min(untapped, key=lambda iid: (g.objects[iid].name != "Island", iid))
    order = {"Halimar Depths": 0, "Temple of Epiphany": 1, "Mystic Sanctuary": 2,
             "Lonely Sandbar": 3, "Svyelunite Temple": 4, "The Surgical Bay": 5}
    candidates = tapped or untapped
    if not candidates:
        return None
    return min(candidates, key=lambda iid: (order.get(g.objects[iid].name, 6), iid))


# NB2: the THREAT-side twin of the rework below was also measured and dropped
# (2026-08-03, field report: long human games end ~5-vs-1 on card advantage):
# hold every Dandân past the first while the opponent has KNOWN removal in
# hand (Bend/Charm with 1+ mana open, Spray with 3+) — 49.9% at 40k. Each fish
# fed to removal is -1 (Spray's cantrip makes their answer free), but delaying
# our own clock costs exactly as much as the card saved. Fish ARE tempo; the
# card-economy lever, if it exists, is not in threat pacing.
#
# NB: an "economy-first" rework was measured and REJECTED wholesale (field
# hypothesis: the AI overvalues removal/counters over card-advantage engines).
# Every gate lost or washed at 40k: hold removal without board pressure 45.5%;
# don't Lapse removal aimed at our fish 45.9%; Vision-Charm counter gate part
# of the same 42.2% bundle; tutor-greed (FoF/AK over answers when unpressured)
# 50.0%; Mind Bend/Crystal Spray weights -2 measured 49.7%; strict AK seed
# gate 50.2/49.8 on independent seeds. Conclusion: fish are the format's
# CLOCK, so answering them IS card economy - a Lapse on their removal trades
# 1-for-1 with a real card while keeping our board; removal stops 4/turn. The
# combat-first frame is load-bearing, not a legacy bias.


def _removal_action(g: GameState, player: str, hand: dict) -> tuple | None:
    """Kill the opponent's Dandân: Crystal Spray (until end of turn + a card) >
    Mind Bend (permanent) > Vision Charm's land mode (kills ALL Dandâns — only
    when the trade is clearly profitable). Crystal Spray goes first: the fish dies
    to the sacrifice either way, so the until-end-of-turn change loses nothing,
    and its cantrip makes the kill card-neutral — Mind Bend's permanence is wasted
    on a creature that's leaving anyway."""
    opp = E._OTHER[player]
    targets = _sac_creatures(g, opp)
    mine = _sac_creatures(g, player)
    if targets:
        tgt = targets[0]
        # In the library endgame the cantrip flips: Crystal Spray's draw can
        # deck us (or hand over the last-card parity), so the drawless Mind
        # Bend leads, and the Spray also passes _safe_to_cast (its draw was
        # unaccounted — field log: the AI died to its own Spray at library 1
        # when a response cycle drained the last card mid-stack).
        order = ("Mind Bend", "Crystal Spray") if _endgame(g) else ("Crystal Spray", "Mind Bend")
        for name in order:
            for iid in hand.get(name, []):
                if not _affordable(g, player, g.objects[iid].mana_cost):
                    continue
                if name == "Crystal Spray" and not _safe_to_cast(g, player, name):
                    continue
                if (name == "Crystal Spray" and _endgame(g)
                        and not _parity_ok_after(g, player, 1, our_draw_next=False)):
                    continue
                return ("cast", iid, None, tgt)
    # Metamorphose: tempo removal when the permanent answers are missing or
    # unaffordable — bounce their fish to the SHARED top. It eats their next
    # draw step (they redraw their own fish) and a recast, and the bounced
    # fish is a KNOWN top worth 11 that _race_top_action can then steal with
    # a banked cycle/Bay. Field logs: the AI sat on Metamorphose from its
    # opening keep to death while being beaten down, and the human used the
    # same card as removal all game.
    # (Holding the LAST Metamorphose in the library endgame — only flip tool
    # left, no board pressure — was measured at exactly 50.0% (40k) and
    # dropped: the fish it declines to bounce compounds tempo at the same
    # rate the saved race tool pays. Twin result to the threat-pacing probe
    # by _hold-fish-for-known-removal's NB2 note above.)
    # Rider veto (sibling of the spare-fish Lapse veto): Metamorphose lets
    # the target's controller deploy a permanent from hand on resolution.
    # Bouncing their fish while we KNOW they hold another just swaps it at
    # instant speed — the bounce buys no tempo and we are down the card
    # (field log: bounced into a known spare, the human free-deployed off
    # the rider, and cleaning up the replacement cost a Sandbar and a Lapse
    # on top).
    if targets:
        for iid in hand.get("Metamorphose", []):
            if _affordable(g, player, g.objects[iid].mana_cost):
                if "Dandân" in _known_opp_hand_names(g, player):
                    break                                 # they replace it for free
                return ("cast", iid, None, targets[0])
    # Vision Charm hits both boards: cast only when the opponent loses more.
    if hand.get("Vision Charm") and targets:
        profitable = (not mine) or (len(targets) - len(mine) >= 2)
        if profitable:
            iid = hand["Vision Charm"][0]
            if _affordable(g, player, g.objects[iid].mana_cost):
                return ("cast", iid, "land", None)
    return None


def _card_advantage_action(g: GameState, player: str, hand: dict) -> tuple | None:
    """Own-main draw spells — only the ones that genuinely must be cast on our
    own turn. Everything castable at instant speed (Fact or Fiction, Accumulated
    Knowledge, Brainstorm, blind Predict) is deferred to the OPPONENT's end step
    (_end_step_draw_action): there our draw step is next, so Brainstorm put-backs
    come back to US instead of feeding their draw, a Memory Lapse on our spell
    only delays it into our own draw step, and after their end step nothing of
    theirs resolves before we untap. Casting the same spells on our own main does
    the opposite on every count. Two exceptions stay here:
      * Predict with a KNOWN top — on our own turn the opponent draws next, so
        the knowledge expires at their draw step; convert it now.
      * Ponder — a sorcery, it's now or never (the seat-aware reorder already
        steers its slots)."""
    reserve = _counter_reserve(g, player)                 # keep {1}{U} for the counter

    def castable(iid):
        o = g.objects[iid]
        if not _affordable(g, player, o.mana_cost):
            return False
        colored, generic = E._parse_cost(o.mana_cost)
        avail = sum(_mana_view(g, player).values())
        return avail - (sum(colored.values()) + generic) >= reserve

    options = []                                          # (priority, iid)
    if _known_top(g, player):                             # guaranteed Predict hit
        for iid in hand.get("Predict", []):
            options.append((0, iid))
    # Opportunistic PRIZE casts on our own turn (human doctrine, 2026-08-04:
    # "tapped out opponent is a good time"). These normally defer to the
    # opponent's end step — a Lapse on our own-turn cast tops the spell into
    # THEIR draw — but when the opponent is counter-dead the deferral only
    # buys them their untap step: at their end step they will have fresh
    # mana for the Lapse + cycle/Bay harvest (mirror mining: a Lapsed FoF is
    # harvested by the counterer 94:3). Strict addition, never a delay — the
    # end-step dump below still fires as always (a counter-BACKUP deferral
    # gate measured 41.5%, see the _end_step_draw_action notes).
    if _opp_counter_dead(g, player):
        for iid in hand.get("Fact or Fiction", []):
            options.append((0.5, iid))
        aks_gy = sum(1 for cid in g.graveyard
                     if g.objects[cid].name == "Accumulated Knowledge")
        if aks_gy >= 1 or len(hand.get("Accumulated Knowledge", [])) >= 2:
            for iid in hand.get("Accumulated Knowledge", []):
                options.append((0.7, iid))
    # Ponder stays ungated: in principle a Memory Lapse on our turn steals it
    # (the shared top + their draw next), but no sane opponent Lapses a 1-mana
    # cantrip (_counter_worthy never does), and gating it on their open mana
    # measured 47.9% — the lost early velocity outweighs the phantom threat.
    for iid in hand.get("Ponder", []):
        options.append((1, iid))
    # NB: Mystical Tutor is likewise NOT cast here — it puts the found card on
    # top of the SHARED library and the opponent draws next on our own turn;
    # _response_action casts it on their end step so our draw step takes it.
    for _, iid in sorted(options, key=lambda t: t[0]):
        name = g.objects[iid].name
        if _endgame(g):                                   # own main: their draw step is next
            drain = 3 if name == "Predict" else _library_drain(g, player, name)
            if drain is None or not _parity_ok_after(g, player, drain,
                                                     our_draw_next=False):
                continue
        if castable(iid) and _safe_to_cast(g, player, name):
            return ("cast", iid, None, None)
    return None


def _end_step_draw_action(g: GameState, player: str) -> tuple | None:
    """Draw-go: the best instant-speed draw spell at the OPPONENT's end step
    (the caller checks the timing). No mana reserve here — after their end step
    nothing of theirs resolves before we untap, so holding mana back buys
    nothing. No "strong position" gate on the big engines either. The naive
    argument (a Lapse here is self-punishing — the spell returns to the shared
    top and our draw step reclaims it) is NOT airtight: Lapse+Predict mills
    the countered spell away, never redrawn (humans play this line, and so do
    we now — see _instant_top_draw). But the punish needs Lapse + Predict + 4
    open mana to coincide with our cast, while a counter-backup gate delays
    EVERY engine until we hold spell+Lapse mana; measured 41.5% against an
    opponent without the punish and still only 44.9% against one WITH it —
    occasional losses are far cheaper than systematic delay.
    AK sequencing is a last-mover war over the shared graveyard: each
    AK we cast upgrades THEIR next one, so never seed an empty graveyard
    unless we hold the majority of the remaining chain (2+ copies) and are
    therefore the likely last mover."""
    hand = _hand_by_name(g, player)
    aks_gy = sum(1 for cid in g.graveyard
                 if g.objects[cid].name == "Accumulated Knowledge")
    # (Bait ordering — prize spells LAST into a known opponent Lapse — was
    # measured here at 49.8%: deferring the prize means it sometimes goes
    # uncast when the window's mana runs dry. The prizes stay first.)
    options = []                                          # (priority, iid)
    for iid in hand.get("Fact or Fiction", []):           # the true 3-for-1: EOTFOF
        options.append((0, iid))
    if aks_gy >= 1 or len(hand.get("Accumulated Knowledge", [])) >= 2:
        for iid in hand.get("Accumulated Knowledge", []):
            options.append((1, iid))
    # (Holding these cheap instants back as reactive "ammo" — dumping only
    # under discard pressure — measured 48.2%: their reactive uses are too
    # rare to pay for the per-turn value forgone.)
    for iid in hand.get("Brainstorm", []):                # put-backs return to us here
        options.append((2, iid))
    # Blind Predict stays in the dump: it names Island (20/80 deck, ~25% for
    # the bonus draw), so it is a card-neutral-plus cantrip AND a denial mill.
    # Holding it for a known-top window instead measured 49.4% - the option
    # value never repays the per-turn value forgone.
    # Predict is an OPPORTUNISTIC card (field doctrine): the cast wants a
    # sculpted top, and _decide_name_card only converts knowledge that
    # happens to exist at resolution. Hold it out of the blind dump when we
    # HOLD the tools to sculpt that top (Brainstorm/Ponder in hand — with
    # Brainstorm the combo chains inside this very window: putbacks make a
    # known-dud top and Predict names it — or a Halimar / Sanctuary land
    # drop coming); with no arranger held there is nothing to wait for and
    # the dump is right (unconditional holding measured 49.4%; two
    # known-top prize guards measured 49.9%).
    arrangers = (hand.get("Brainstorm") or hand.get("Ponder")
                 or hand.get("Halimar Depths") or hand.get("Mystic Sanctuary"))
    if _known_top(g, player) is not None or not arrangers:
        for iid in hand.get("Predict", []):               # names the known top, else Island
            options.append((3, iid))
    for _, iid in sorted(options, key=lambda t: t[0]):
        name = g.objects[iid].name
        if _endgame(g):
            # Library endgame: this is where field games are decided. A value
            # cast that flips the last-card parity to us is suicide-by-AK (we
            # lost every long human game exactly this way). At their end step
            # OUR draw step is next. Blind Predict's drain is unpredictable —
            # skip it entirely here.
            drain = _library_drain(g, player, name)
            if drain is None or not _parity_ok_after(g, player, drain,
                                                     our_draw_next=True):
                continue
        if (_affordable(g, player, g.objects[iid].mana_cost)
                and _safe_to_cast(g, player, name)):
            return ("cast", iid, None, None)
    return None


# ── Deck-out mode (winning by emptying the shared library) ──────────────────
#
# library / graveyard / exile are ONE shared, ordered pool, and a player loses
# the moment they must draw from an empty library — at their OWN draw step OR
# from a spell/ability (so the AI must never over-draw itself: see
# _safe_to_cast). During the AI's
# main phase the opponent draws next, so the opponent's draw steps see library
# sizes L, L-2, L-4 ... and the AI's see L-1, L-3 ... — the opponent decks out
# iff len(library) is EVEN when the AI ends its turn. So when combat can't close
# the game (an opponent who never plays an Island leaves every Dandân unable to
# attack), the AI commits to the deck-out: it stops spending its deck-
# manipulation cards for value and hoards them, then once the shared deck is
# small it steers the parity — picking a play that leaves the library at an
# even, smaller size (emptying it outright is an immediate win) and never
# growing it. Re-entrancy re-reads L after every action, so multi-card draws
# self-correct.

_DECKOUT_LIBRARY = 20   # actively steer/mill the shared deck once it is this small


def _can_attack_opponent(g: GameState, player: str) -> bool:
    """Whether combat can still close the game: the opponent controls an Island
    (so the AI's Dandâns — current or future — may attack it) or the AI already
    controls a creature not structurally barred from attacking."""
    if E.controls_basic_type(g, E._OTHER[player], "Island"):
        return True
    return any(E._is_creature(g.objects[iid]) and not E._attack_restricted(g, player, iid)
               for iid in g.players[player].battlefield)


def _in_deckout_mode(g: GameState, player: str) -> bool:
    """Combat can no longer win the game (the opponent never develops an Island,
    so the AI's Dandâns can't attack). The deck-out is now the only path, so the
    AI conserves its deck-manipulation cards from the start and steers the
    library parity once it shrinks below _DECKOUT_LIBRARY. Early on, before
    either side has lands down, this also reads as "hopeless" for a turn or two
    against a normal Islands opponent — harmless, since it just defers a draw
    spell until they reveal their Islands and combat opens back up."""
    return not _can_attack_opponent(g, player)


def _library_drain(g: GameState, player: str, name: str) -> int | None:
    """How many cards casting/cycling this card nets out of the shared library
    right now (positive shrinks it). None when unpredictable (a blind Predict);
    a non-positive result wouldn't help (or would grow the library)."""
    n = len(g.library)
    if name in ("Ponder", "Lonely Sandbar"):              # cantrip / cycle: draw one
        return min(1, n)
    if name == "Brainstorm":                              # draw 3, put 2 back on top
        return min(3, n) - 2
    if name == "Fact or Fiction":                         # reveal 5, all leave the library
        return min(5, n)
    if name == "Accumulated Knowledge":
        aks = sum(1 for cid in g.graveyard if g.objects[cid].name == "Accumulated Knowledge")
        return min(1 + aks, n)
    if name == "Predict":                                 # mill 1 + draw 2 on a guaranteed hit
        return min(3, n) if _known_top(g, player) else None
    if name == "Vision Charm":                            # mill mode: top four
        return min(4, n)
    return None


def _safe_to_cast(g: GameState, player: str, name: str) -> bool:
    """Drawing from an empty library now LOSES the game (for spell draws too, not
    just the draw step), so whether casting `name` is safe for US — it won't make
    us draw past the last card. Milling/revealing doesn't draw, so only the draw
    portion can deck us; conserve the spell when the shared deck is too thin.
    In the library endgame, when the opponent VISIBLY can pull a card at
    instant speed (ready Surgical Bay, or a KNOWN cycle/Brainstorm/AK with
    the mana — _opp_theft_ready), every draw cast needs a one-card margin
    against that response drain: their pull takes a card mid-stack and OUR
    resolution kills us — the stack-kill window of _stack_kill_response,
    pointed back at us. Crystal Spray learned this from a field death; the
    rest of the suite inherits it on the visible evidence only (a blanket
    any-open-mana margin measured 49.7%: paralysis costs more than the
    deaths it prevents)."""
    L = len(g.library)
    opp_open = len(_untapped_lands(g, E._OTHER[player])) > 0
    margin = 1 if (_endgame(g) and _opp_theft_ready(g, player)) else 0
    if name == "Brainstorm":
        return L >= 3 + margin                            # draws three (before putting two back)
    if name == "Crystal Spray":
        # The cantrip draw is easy to forget: field log has the AI dying to
        # its own Spray at library 1 after the opponent's response cycle
        # drained the last card mid-stack. Its margin applies at ANY library
        # size (the original rule, kept strictest).
        return L >= (2 if opp_open else 1)
    if name == "Accumulated Knowledge":
        aks = sum(1 for cid in g.graveyard if g.objects[cid].name == "Accumulated Knowledge")
        return L >= 1 + aks + margin
    if name == "Predict":                                 # mills one, then draws up to two
        return L >= 3 + margin
    if name in ("Ponder", "Lonely Sandbar"):              # draws one
        return L >= 1 + margin
    return True                                           # Fact or Fiction / Vision Charm: no draw


# ── The library endgame (all games, not just no-Island stalls) ──────────────
#
# Every long game of Forgetful Fish ends in the deck-out race: the shared
# library empties and someone's draw step hits nothing. Field logs of human
# wins showed the AI losing this race in EVERY long game — it kept casting
# "value" draw spells (AK chains, FoF) with a 10-card library, drawing itself
# to death, because parity awareness only existed in _in_deckout_mode (combat
# hopeless). Once the library is small, the race dominates card value in all
# games, so draw decisions below this size are parity-gated.

_ENDGAME_LIBRARY = 5    # swept 4..26: unimodal peak at 5 (the race only truly
#                         dominates value in the last few cards - any earlier and
#                         the opponent still holds enough flips to retake the count)


def _endgame(g: GameState) -> bool:
    return 0 < len(g.library) <= _ENDGAME_LIBRARY


def _parity_ok_after(g: GameState, player: str, drain: int, *,
                     our_draw_next: bool) -> bool:
    """Deck-out accounting: after a play of ours removes `drain` cards, does
    the OPPONENT still take the losing draw from the empty library? Natural
    draw steps alternate, so whoever takes the next natural draw takes draws
    1, 3, 5, ... and dies on draw L+1 iff L is even."""
    L = len(g.library) - drain
    if L < 0:
        return False
    return (L % 2 == 1) if our_draw_next else (L % 2 == 0)


def _deckout_library_action(g: GameState, player: str, hand: dict) -> tuple | None:
    """The deck-manipulation play that best advances the deck-out: leave the
    shared library at an EVEN size (the opponent draws the last card) and as
    small as possible — emptying it is an immediate win. Skip anything that would
    flip the parity against us or grow the library."""
    L = len(g.library)
    if L == 0 or L >= _DECKOUT_LIBRARY:                   # empty (opp decks next), or still
        return None                                       # plenty: hoard our tools, just wait
    opp = E._OTHER[player]
    reserve = 2 if hand.get("Memory Lapse") and _sac_creatures(g, opp) else 0

    def affordable(cost):
        if not _affordable(g, player, cost):
            return False
        colored, generic = E._parse_cost(cost)
        avail = sum(_mana_view(g, player).values())
        return avail - (sum(colored.values()) + generic) >= reserve

    # Prefer drawing engines (they refill our hand with more steering tools) over
    # the pure-mill Vision Charm; an outright win (new_L == 0) beats everything.
    rank = {"Fact or Fiction": 0, "Accumulated Knowledge": 1, "Predict": 2,
            "Brainstorm": 3, "Ponder": 4, "Vision Charm": 5, "Lonely Sandbar": 6}
    best = None                                           # (sort_key, action)
    for name, iids in hand.items():
        drain = _library_drain(g, player, name)
        if drain is None or drain <= 0:                   # unpredictable or no shrink
            continue
        if not _safe_to_cast(g, player, name):            # never deck OURSELVES out
            continue
        new_L = L - drain
        if new_L % 2 != 0:                                # would hand the deck-out to us
            continue
        cost = "{U}" if name == "Lonely Sandbar" else g.objects[iids[0]].mana_cost
        if not affordable(cost):
            continue
        action = (("cycle", iids[0]) if name == "Lonely Sandbar"
                  else ("cast", iids[0], "mill" if name == "Vision Charm" else None, None))
        key = (new_L != 0, rank.get(name, 9), new_L)
        if best is None or key < best[0]:
            best = (key, action)
    return best[1] if best else None


def _deckout_insurance_action(g: GameState, player: str, hand: dict) -> tuple | None:
    """The deck-out race is LOST and we hold the reset: cast Day's Undoing
    rather than certainly deck (mirror logs: a third of all decking losses died
    holding it, several holding two). Fires only at our own main when the count
    is against us with the end near (L odd and <= 3: the opponent takes the
    last card and our next draw after that is from the empty library), every
    cheaper out is gone — no parity-fixing drain (the caller tried
    _deckout_library_action first), no Sandbar/Metamorphose in hand and no
    Surgical Bay down for the losing-parity flip windows — and combat cannot
    kill them before we deck. A Memory Lapse response doesn't beat the cast:
    the Lapse-back GROWS the library by one, which flips the count our way."""
    L = len(g.library)
    if L not in (1, 3, 5, 7):                             # odd, near the end; L == 0 at our
        # Recast window (field game 2026-08-06): our first insurance cast
        # was Memory-Lapsed onto the library — even L, our Day's Undoing on
        # the known top, another copy still in hand. The docstring's "the
        # Lapse-back flips the count our way" is wrong twice over there:
        # the natural count now hands THEM our reset card, and one banked
        # cycle of theirs re-flips the parity at will (the observed loss:
        # opponent cycled Lonely Sandbar and we decked holding the second
        # copy). They also just spent a counter — push the copy through.
        top = _known_top(g, player)
        _recast = (L in (2, 4) and top is not None
                   and top.name == "Day's Undoing"
                   and len(hand.get("Day's Undoing", [])) >= 1)
        _fear = (L <= 8 and L % 2 == 0
                 and len(g.players[E._OTHER[player]].hand) >= 4
                 and len(hand.get("Day's Undoing", [])) >= 2)
        if not (_recast or _fear):
            return None                                   # main means THEY deck next — never reset
    undoings = hand.get("Day's Undoing", [])
    if not undoings or not _affordable(g, player, g.objects[undoings[0]].mana_cost):
        return None
    if (hand.get("Lonely Sandbar") or hand.get("Metamorphose"))             and len(undoings) < 2:
        return None                                       # the flip windows can still save us
    if any(g.objects[iid].name == "The Surgical Bay"
           for iid in g.players[player].battlefield):
        return None
    opp = E._OTHER[player]
    attacks_left = (L + 1) // 2                           # full turns we still get
    power = sum(g.objects[c].power or 0 for c in g.players[player].battlefield
                if E._is_creature(g.objects[c]) and not E._attack_restricted(g, player, c))
    if power * attacks_left >= g.players[opp].life:
        return None                                       # the board race is faster than the reset
    return ("cast", undoings[0], None, None)


def _deckout_main_action(g: GameState, player: str) -> tuple:
    """Own main phase while combat is hopeless: develop mana, stay alive against
    the opponent's fish, steer the shared library toward an even, empty deck so
    the opponent draws the last card, and empty Dandâns out of hand (they can't
    attack, so casting them only frees room to hoard deck-manipulation). Never
    spends a draw spell for value — those are the parity tools — and casts
    Day's Undoing only as last-resort insurance when the race is LOST
    (_deckout_insurance_action); while the race is live a reset would
    reshuffle every zone back and undo it."""
    hand = _hand_by_name(g, player)
    land = _choose_land_drop(g, player, hand)
    if land is not None:
        return ("land", land)
    removal = _removal_action(g, player, hand)            # survival comes before the deck-out
    if removal is not None:
        return removal
    lib = _deckout_library_action(g, player, hand)        # steer parity (once the deck is small)
    if lib is not None:
        return lib
    insurance = _deckout_insurance_action(g, player, hand)  # race lost, no tools: reset it
    if insurance is not None:
        return insurance
    for iid in hand.get("Dandân", []):                    # park dead fish to free hand space
        if _affordable(g, player, g.objects[iid].mana_cost):
            # Same survives-check as the main-phase cast (3042bbf): a parked
            # fish is a free BLOCKER only if it lives — with no land of its
            # sac type under our control (their Vision Charm just rewrote our
            # Islands, or our only Island got Mind-Bent) the "parking" is a
            # straight donation to the shared graveyard. Field log 2026-08-03
            # turn 19: cast into its own Charm's Island->Plains change and
            # sacrificed on resolution (the utility lands still paid the
            # blue, so affordability alone doesn't guard this).
            typ = _sac_type(g.objects[iid]) or "Island"
            if not E.controls_basic_type(g, player, typ):
                break
            return ("cast", iid, None, None)
    return ("pass",)


_TOP_RACE_MIN_VALUE = 7.0   # only race for a real prize, not a mediocre top
_KEEP_MIN = 3.0             # scry/reorder bar: a card worth drawing (vs a dud)


def _counter_reserve(g: GameState, player: str) -> int:
    """{1}{U} held back for Memory Lapse on OWN-turn value spending — mana
    tapped on our turn stays tapped through the opponent's entire next turn."""
    return 2 if (_hand_by_name(g, player).get("Memory Lapse")
                 and g.turn_number >= 3) else 0


def _instant_top_draw(g, player, reserve=0):
    """The cheapest affordable INSTANT-speed draw that pulls the current top card
    into our hand — expendable tools first (cycle a Lonely Sandbar, sac The
    Surgical Bay) before spending a card-draw spell. Returns an action tuple, or
    None if we can't draw at instant speed right now.
    `reserve`: mana that must stay untapped AFTER the draw. Own-turn callers pass
    2 while holding Memory Lapse — the counter-mana hold of _card_advantage_action,
    which this path silently bypassed (field log 2026-08-04: Metamorphose put the
    fish on a known top, the recover path fired the Lapse+Predict punish down to
    one mana, and the recast fish walked through the now-dead Lapse for lethal)."""
    hand = _hand_by_name(g, player)
    avail = sum(_mana_view(g, player).values())
    for iid in hand.get("Lonely Sandbar", []):            # cycling {U}: discard, draw one
        if _affordable(g, player, "{U}") and avail - 1 >= reserve:
            return ("cycle", iid)
    if g.library:                                         # sac The Surgical Bay: draw one
        bays = [iid for iid in g.players[player].battlefield
                if g.objects[iid].name == "The Surgical Bay" and not g.objects[iid].tapped]
        others = [iid for iid in _untapped_lands(g, player)
                  if g.objects[iid].name != "The Surgical Bay"]
        if bays and len(others) >= 2 + reserve:
            return ("activate", bays[0], 1)
    def clears_reserve(iid):
        colored, generic = E._parse_cost(g.objects[iid].mana_cost)
        return avail - (sum(colored.values()) + generic) >= reserve
    for name in ("Accumulated Knowledge", "Brainstorm"):  # instants that draw off the top
        for iid in hand.get(name, []):
            if (_affordable(g, player, g.objects[iid].mana_cost)
                    and _safe_to_cast(g, player, name) and clears_reserve(iid)):
                return ("cast", iid, None, None)
    # (Two measured non-improvements here: preferring Predict over drawing a
    # sub-premium known top was exactly 50.0% - the situation is too rare to
    # matter; Vision Charm's mill-4 as an extra top-denial measured 47.4% -
    # the charm is worth more as removal/deck-out ammo than as denial.)
    # Last resort: Predict the KNOWN top away. It doesn't take the card — it
    # names it, MILLS it, and draws 2 off the guaranteed hit — so it's denial
    # plus value rather than a steal (the Lapse+Predict punish: counter their
    # spell, then mill it so it is never redrawn). Worse than drawing a premium
    # card we want, hence tried last; far better than gifting the top back.
    if _known_top(g, player) is not None:
        for iid in hand.get("Predict", []):
            if (_affordable(g, player, g.objects[iid].mana_cost)
                    and _safe_to_cast(g, player, "Predict") and clears_reserve(iid)):
                return ("cast", iid, None, None)
    return None


def _opp_drawing_off_top(g, player):
    """The opponent has an instant-speed draw on the stack right now — a draw
    spell, OR a cycling / Surgical Bay draw ability (both resolve as a 'draw_1'
    activated ability). It will pull the top card when it resolves, so we can
    respond and take the top first (our response resolves first — LIFO)."""
    opp = E._OTHER[player]
    for so in g.stack:
        if so.controller != opp:
            continue
        if so.kind == "spell":
            o = g.objects.get(so.source_instance_id)
            # Predict belongs here too (rollout audit 2026-08-04): their
            # Predict MILLS our known top -- the Lapse-punish pointed at us --
            # and responding with an instant draw steals it first (-0.079
            # win-prob to pass, t=-6.7, n=218; Predict alone -0.085 t=-6.8).
            # Their FoF measured neutral (n=20) and stays out.
            if o and o.name in ("Brainstorm", "Accumulated Knowledge",
                                "Crystal Spray", "Predict"):
                return True
        elif so.kind == "activated" and (so.chosen or {}).get("effect") == "draw_1":
            return True                                   # opponent's cycling / Surgical Bay draw
    return False


def _race_top_action(g, player):
    """A KNOWN, valuable card on top of the SHARED library is a contested
    resource. If the opponent would draw it before we naturally would — their
    draw step is imminent (their upkeep) or they're drawing right now (a draw
    spell on the stack) — snatch it first with an instant-speed draw/cycle. The
    top is known to both players after a tutor / Mystic Sanctuary / Memory Lapse,
    so this is a real race we can win by responding."""
    if _in_deckout_mode(g, player):
        return None                                       # racing burns the parity we're steering
    # Already racing: if the TOPMOST stack object is ours, our response resolves
    # first and takes the prize - casting a second draw on top of it would just
    # steal the card from our own first spell and burn a card for nothing (this
    # used to double-cast at the opponent's upkeep, and re-fire off their draw
    # spell sitting UNDER our response). If the opponent responds over us, the
    # topmost object is theirs again and the race legitimately re-opens.
    if g.stack and g.stack[-1].controller == player:
        return None
    top = _known_top(g, player)
    if (top is None or _is_land(top.type_line)            # not worth an instant draw for a land
            or card_value(g, player, top.instance_id) < _TOP_RACE_MIN_VALUE):
        return None
    opp = E._OTHER[player]
    # The top is only "ours to lose" while our own draw step is still ahead
    # this turn. On our OWN turn after the draw step (main1 onward) the next
    # natural draw is the OPPONENT's — a known good top there (a spell we just
    # Memory Lapsed, their end-step tutor target) is theirs unless we take it
    # at instant speed now.
    imminent = ((g.active_player == opp
                 and (g.current_step == "upkeep" or _opp_drawing_off_top(g, player)))
                or (g.active_player == player
                    and g.current_step not in ("untap", "upkeep", "draw")))
    if not imminent:
        return None                                       # our own draw step still gets it
    # Library endgame: a race draw is also a parity flip — don't spend it when
    # the flip hands the opponent the last card (every race window here has
    # their natural draw next; approximated as a one-card drain).
    if _endgame(g) and not _parity_ok_after(g, player, 1, our_draw_next=False):
        return None
    # OWN-TURN initiation into a READY theft tool loses the race by
    # construction: top-races are won by the LAST responder (LIFO), so with
    # their Bay/known instant draw + mana visibly up, our racer just triggers
    # the steal and burns the spell (rollout audit: -0.022 t=-4.6 n=803;
    # their-turn races stay correct at +0.062 — their draw step takes the
    # card anyway, so forcing the exchange is right there). Arena 50.50% at
    # 40k, confirmed 50.61% [50.3, 50.9] at 100k.
    if (g.active_player == player and _opp_theft_ready(g, player)
            and not _opp_drawing_off_top(g, player)):
        return None
    # Their-turn races untap before any exposure; own-turn races tap down
    # through their whole next turn — hold the counter mana there.
    reserve = _counter_reserve(g, player) if g.active_player == player else 0
    return _instant_top_draw(g, player, reserve=reserve)


def _natural_next_drawer(g: GameState, player: str) -> str:
    """Who takes the next NATURAL draw step off the shared library (no spell
    adjustment - see _next_drawer for the spell-aware version): before/at the
    active player's draw step it is the active player, afterwards the other."""
    pre_draw = g.current_step in ("", "untap", "upkeep")
    active = g.active_player if g.active_player in ("p1", "p2") else player
    if pre_draw:
        return active
    return E._OTHER[active]


def _survival_refill_action(g: GameState, player: str):
    """The shared library is EMPTY and our natural draw is next: that draw loses
    on the spot. Metamorphose is the one instant that can refill the library
    (its target - an OPPONENT permanent - goes on top, +1 card), turning our
    lethal draw into a real one and handing the empty-library problem back.
    Prefer denying a creature; any permanent works. (Seen in the field: the AI
    died with Metamorphose in hand rather than refill.)"""
    if len(g.library) != 0 or g.stack:
        return None
    if _natural_next_drawer(g, player) != player:
        return None                                       # the opponent takes the lethal draw
    hand = _hand_by_name(g, player)
    metas = hand.get("Metamorphose", [])
    if not metas or not _affordable(g, player, g.objects[metas[0]].mana_cost):
        return None
    targets = list(g.players[E._OTHER[player]].battlefield)
    if not targets:
        return None
    best = min(targets, key=lambda iid: (0 if E._is_creature(g.objects[iid]) else 1, iid))
    return ("cast", metas[0], None, best)


def _losing_parity_flip(g: GameState, player: str):
    """Library endgame, losing parity: the current count has US taking the
    losing draw from the empty library. Flip it with an expendable GUARANTEED
    one-card draw (cycle a Lonely Sandbar / sac The Surgical Bay) — the move
    the deck-out race is actually won with. No removal flip left: Metamorphose
    flips the other way (+1 — the targeted opponent permanent goes on top),
    which fixes the parity just as well and denies them a permanent doing it.
    Callers pick the windows; parity direction comes from whoever's natural
    draw is next, so the same check serves every window."""
    if not _endgame(g):
        return None
    ours_next = _natural_next_drawer(g, player) == player
    if _parity_ok_after(g, player, 0, our_draw_next=ours_next):
        return None
    hand = _hand_by_name(g, player)
    for iid in hand.get("Lonely Sandbar", []):
        if _affordable(g, player, "{U}"):
            return ("cycle", iid)
    bays = [iid for iid in g.players[player].battlefield
            if g.objects[iid].name == "The Surgical Bay"
            and not g.objects[iid].tapped]
    others = [iid for iid in _untapped_lands(g, player)
              if g.objects[iid].name != "The Surgical Bay"]
    if bays and len(others) >= 2:
        return ("activate", bays[0], 1)
    metas = hand.get("Metamorphose", [])
    opp_perms = list(g.players[E._OTHER[player]].battlefield)
    if (metas and opp_perms
            and _affordable(g, player, g.objects[metas[0]].mana_cost)):
        best = min(opp_perms, key=lambda iid: (
            0 if E._is_creature(g.objects[iid]) else 1, iid))
        return ("cast", metas[0], None, best)
    return None


# Pre-draw mill and mandatory draw count of a resolving object, for the
# stack-kill window: drawing from an empty library loses on the spot, so an
# object that MUST draw R cards kills its controller if the library holds
# fewer than R when it resolves (after its own m-card mill).
_DRAW_ON_RESOLVE = {
    "Brainstorm": (0, 3),
    "Ponder": (0, 1),
    "Crystal Spray": (0, 1),
    "Predict": (1, 2),
    # Accumulated Knowledge computed live (1 + graveyard count)
}


def _stack_kill_response(g: GameState, player: str) -> tuple | None:
    """Field critique (2026-08-03 game 4): near-empty library, ANY pending
    draw on the stack is a kill window — respond with a drain that leaves
    the library below what their resolution must draw, and the resolution
    itself kills them (LIFO: our drain resolves first). This is the
    offensive twin of the drain-counter defense in _counter_worthy, and the
    reason a small library demands urgency: the parity count only decides
    the game if nobody forces the issue on the stack first. One action per
    priority; if the first drain doesn't finish the job we get priority
    again before their object resolves."""
    L = len(g.library)
    if not (0 < L <= _ENDGAME_LIBRARY + 3) or not g.stack:
        return None
    opp = E._OTHER[player]
    their = None                                          # topmost object of theirs that must draw
    for so in reversed(g.stack):
        if so.controller != opp:
            continue
        if so.kind == "activated" and (so.chosen or {}).get("effect") == "draw_1":
            their = (0, 1)
            break
        if so.kind == "spell":
            o = g.objects.get(so.source_instance_id)
            if o and o.name == "Accumulated Knowledge":
                their = (0, 1 + sum(1 for c in g.graveyard
                                    if g.objects[c].name == "Accumulated Knowledge"))
                break
            if o and o.name in _DRAW_ON_RESOLVE:
                their = _DRAW_ON_RESOLVE[o.name]
                break
    if their is None:
        return None
    their_mill, their_need = their
    need_drain = L - their_mill - (their_need - 1)        # our drain >= this kills them
    if need_drain <= 0:
        return None                                       # already dead on resolution: let it happen
    hand = _hand_by_name(g, player)
    aks_gy = sum(1 for c in g.graveyard if g.objects[c].name == "Accumulated Knowledge")
    # (drain we cause before their resolution, self draw-requirement, action)
    tools = []
    for iid in hand.get("Lonely Sandbar", []):
        tools.append((min(1, L), 1, "{U}", ("cycle", iid)))
    for iid in g.players[player].battlefield:
        o = g.objects[iid]
        if o.name == "The Surgical Bay" and not o.tapped:
            others = [x for x in _untapped_lands(g, player) if x != iid]
            if len(others) >= 2:
                tools.append((min(1, L), 1, None, ("activate", iid, 1)))
    for iid in hand.get("Vision Charm", []):
        tools.append((min(4, L), 0, "{U}", ("cast", iid, "mill", None)))
    for iid in hand.get("Predict", []):
        tools.append((3, 3, g.objects[iid].mana_cost, ("cast", iid, None, None)))
    for iid in hand.get("Accumulated Knowledge", []):
        tools.append((1 + aks_gy, 1 + aks_gy, g.objects[iid].mana_cost,
                      ("cast", iid, None, None)))
    for iid in hand.get("Brainstorm", []):
        tools.append((1, 3, g.objects[iid].mana_cost, ("cast", iid, None, None)))
    for iid in hand.get("Fact or Fiction", []):
        tools.append((min(5, L), 0, g.objects[iid].mana_cost, ("cast", iid, None, None)))
    for drain, self_req, cost, action in tools:           # list order = spend preference
        if drain < need_drain or L < self_req:
            continue
        if cost is not None and not _affordable(g, player, cost):
            continue
        return action
    return None


def _response_action(g: GameState, player: str) -> tuple:
    """Anything outside my own quiet main phase: counter the opponent's spell
    with Memory Lapse when it threatens us, race a known card off the top, else
    pass."""
    opp = E._OTHER[player]
    survive = _survival_refill_action(g, player)          # empty library, our draw next
    if survive is not None:
        return survive
    kill = _stack_kill_response(g, player)                # their pending draw = kill window
    if kill is not None:
        return kill
    top = g.stack[-1] if g.stack else None
    if (top is not None and top.kind == "spell" and top.controller == opp
            and _counter_worthy(g, player, top)):
        lapses = _hand_by_name(g, player).get("Memory Lapse", [])
        if lapses and _affordable(g, player, g.objects[lapses[0]].mana_cost):
            return ("cast", lapses[0], None, top.source_instance_id)
    # Race the opponent for a known, valuable top card (their upkeep, or in
    # response to their draw spell) before they draw it.
    race = _race_top_action(g, player)
    if race is not None:
        return race
    # Instant-speed removal on THEIR turn (rollout discovery 2026-08-04):
    # the whole removal suite is instants, but the removal logic only ran at
    # our own main. Auditing pass-vs-cast-now on their turn measured -0.042
    # (t=-7.0, n=799) across Spray -0.035 / Charm -0.041 / Metamorphose
    # -0.052 alike: the fish dies before it attacks again, on mana that was
    # otherwise idle through their whole turn. Arena 54.92% at 40k,
    # confirmed 54.95% [54.6, 55.3] at 100k.
    if g.active_player == opp and not g.stack:
        removal = _removal_action(g, player, _hand_by_name(g, player))
        if removal is not None:
            return removal
    # Losing-parity flip at the LAST card: when the library is down to one and
    # the count is against us, the opponent's next draw step takes it and our
    # end-step window never comes (mirror log: died decked with an untapped
    # Surgical Bay and twelve lands while the last card sat on top through our
    # own end step and their upkeep — a Lapse-back response war had flipped
    # the parity mid-turn). Take it in any quiet window that still precedes
    # the decisive draw: their upkeep, or our own turn once our draw step is
    # past. At L >= 2 the end-step site below stays the flip of choice — it
    # acts LAST, so the opponent can't re-flip before our draw; firing early
    # there measured a wash. Unlike the end-step site this window also runs
    # in deck-out mode: the flip IS the race-winning spend that mode is
    # hoarding its tools for.
    if (len(g.library) == 1 and not g.stack
            and ((g.active_player == opp and g.current_step == "upkeep")
                 or (g.active_player == player
                     and g.current_step not in ("", "untap", "upkeep", "draw")))):
        flip = _losing_parity_flip(g, player)
        if flip is not None:
            return flip
    # The OPPONENT's end step with an empty stack: the draw-go window. Fire the
    # instant-speed draw engines deferred from our own main phase (see
    # _card_advantage_action), then Mystical Tutor — it puts the found card on
    # top of the SHARED library and our draw step is next, so WE draw it (not
    # the opponent, as we would casting it on our own turn). Deck-out mode
    # hoards instead: spending draw spells for value burns the parity tools.
    if g.active_player == opp and g.current_step == "end" and not g.stack:
        if not _in_deckout_mode(g, player):
            draw = _end_step_draw_action(g, player)
            if draw is not None:
                return draw
            # Losing parity with our draw step next: flip it (see
            # _losing_parity_flip; the parity-gated window above already
            # tried the draw spells whose drain flips).
            flip = _losing_parity_flip(g, player)
            if flip is not None:
                return flip
        tutors = _hand_by_name(g, player).get("Mystical Tutor", [])
        if tutors and _affordable(g, player, g.objects[tutors[0]].mana_cost):
            # Theft guard: the tutored card sits on the SHARED top until our
            # draw step, and an opponent instant draw takes it first (field
            # log: a tutored Mind Bend stolen by a response Brainstorm; the
            # mirror's own-turn race does the same). Hold the tutor when we
            # KNOW they hold an instant draw with mana for it, or their
            # Surgical Bay is ready.
            known = _known_opp_hand_names(g, player)
            opp_open = len(_untapped_lands(g, opp))
            bays = [iid for iid in g.players[opp].battlefield
                    if g.objects[iid].name == "The Surgical Bay"
                    and not g.objects[iid].tapped]
            theft = ((known & {"Brainstorm", "Lonely Sandbar"} and opp_open >= 1)
                     or ("Accumulated Knowledge" in known and opp_open >= 2)
                     or (bays and opp_open >= 3))
            if not theft:
                return ("cast", tutors[0], None, None)
    # NB: our OWN upkeep is deliberately NOT a draw window. Floyd's "cast in
    # your upkeep" rule protects a specific must-resolve spell from Memory
    # Lapse, but as a general window it taps the most expensive mana of the
    # game — everything spent at upkeep crowds out our own main phase, unlike
    # the opponent's end step where we untap immediately after. Measured:
    # 45.4% vs not doing it.
    return ("pass",)


def _known_opp_hand_names(g: GameState, player: str) -> set:
    """Names of opponent-hand cards we have legitimately SEEN (their known_by
    marks us): Lapse-backs they redrew, tutor reveals, library slots we
    arranged. Measured: >= 1 known card at half of all decision points."""
    opp = E._OTHER[player]
    return {g.objects[iid].name for iid in g.players[opp].hand
            if iid in g.objects and player in (g.objects[iid].known_by or [])}


def _total_copies(g: GameState, name: str) -> int:
    """Copies of `name` in THIS game, counted from the object registry (every
    real card lives in g.objects regardless of zone). Never hard-code deck
    composition: this pool is nonstandard — a shared 80 with 8 Memory Lapses,
    10 Dandâns, 20 Islands — and a '4-of' assumption already shipped one bug
    (the counter belief zeroing out once four Lapses hit the graveyard)."""
    return sum(1 for o in g.objects.values() if o.name == name)


def _opp_counter_belief(g: GameState, player: str) -> float:
    """P(the opponent's UNKNOWN cards hold a Memory Lapse), validated against
    the RL hand-guesser (it=485k) over 2,936 mirror states with true hidden
    hands. Retention-weighted hypergeometric: players HOLD counters, so each
    unknown hand slot counts DOUBLE a library slot (w=2, fit empirically —
    Brier 0.204 vs 0.244 for the full neural guesser and 0.229 for the
    uniform census). Known copies are certainty, not belief."""
    if "Memory Lapse" in _known_opp_hand_names(g, player):
        return 1.0
    opp = E._OTHER[player]
    seen = sum(1 for iid in g.players[player].hand
               if g.objects[iid].name == "Memory Lapse")
    seen += sum(1 for cid in g.graveyard
                if g.objects[cid].name == "Memory Lapse")
    seen += sum(1 for cid in getattr(g, "exile", [])
                if g.objects[cid].name == "Memory Lapse")
    seen += sum(1 for so in g.stack
                if (o := g.objects.get(so.source_instance_id)) is not None
                and o.name == "Memory Lapse")
    unseen = _total_copies(g, "Memory Lapse") - seen
    h_unknown = sum(1 for iid in g.players[opp].hand
                    if iid in g.objects
                    and player not in (g.objects[iid].known_by or []))
    if unseen <= 0 or h_unknown <= 0:
        return 0.0
    p_slot = 2.0 * h_unknown / (2.0 * h_unknown + len(g.library))
    return 1.0 - (1.0 - p_slot) ** unseen


def _opp_counter_dead(g: GameState, player: str) -> bool:
    """The opponent cannot Lapse right now — or almost surely holds none:
    fewer than two untapped lands (hard fact), or the retention-weighted
    belief below 15% (probe: that band holds a Lapse 1.8% of the time)."""
    if len(_untapped_lands(g, E._OTHER[player])) < 2:
        return True
    return _opp_counter_belief(g, player) < 0.15


def _opp_theft_ready(g: GameState, player: str) -> bool:
    """Whether the opponent can pull the shared top card into their hand (or
    drain one off the library) at instant speed right now, on visible or
    legitimately-known evidence only: a KNOWN instant draw with the mana for
    it, or their ready Surgical Bay (public information)."""
    opp = E._OTHER[player]
    known = _known_opp_hand_names(g, player)
    opp_open = len(_untapped_lands(g, opp))
    bays = [iid for iid in g.players[opp].battlefield
            if g.objects[iid].name == "The Surgical Bay"
            and not g.objects[iid].tapped]
    return (bool(known & {"Brainstorm", "Lonely Sandbar"} and opp_open >= 1)
            or bool("Accumulated Knowledge" in known and opp_open >= 2)
            or bool(bays and opp_open >= 3))


def _counter_worthy(g: GameState, player: str, so) -> bool:
    """Whether the opponent's spell on the stack is worth a Memory Lapse."""
    inst = g.objects.get(so.source_instance_id)
    if not inst:
        return False
    name = inst.name
    # Library-endgame defense (before every other rule, deck-out mode
    # included): a drain spell can be a KILL SHOT, not value. Two shapes,
    # both from one field game (2026-08-03 game 3, turn 17): the AI flipped
    # the race with its Surgical Bay, the human responded with Vision Charm
    # mill-four, the library hit zero and the AI's own pending Bay draw
    # killed it — while it held the Memory Lapse that stops all of it (a
    # Lapse even puts the Charm on top for that pending draw to steal).
    #   * fatal_now: their drain empties the library while a draw of OURS
    #     sits under it on the stack — we lose on resolution, at any L.
    #   * flips: in the endgame their drain turns a count we are winning
    #     into one we lose; the Lapse-back (library +1) must itself leave
    #     the count survivable, else the Lapse is wasted doom-delay.
    L = len(g.library)
    if L > 0:
        if name == "Vision Charm":
            their_drain = (min(4, L) if (inst.chosen or {}).get("mode") == "mill"
                           else None)
        else:
            their_drain = _library_drain(g, player, name)
        if their_drain is not None and their_drain > 0:
            ours_next = _natural_next_drawer(g, player) == player
            own_pending_draw = any(
                so.controller == player
                and ((so.kind == "activated"
                      and (so.chosen or {}).get("effect") == "draw_1")
                     or (so.kind == "spell"
                         and (o := g.objects.get(so.source_instance_id)) is not None
                         and o.name in ("Brainstorm", "Accumulated Knowledge",
                                        "Crystal Spray", "Ponder", "Predict",
                                        "Fact or Fiction")))
                for so in g.stack[:-1])
            fatal_now = their_drain >= L and own_pending_draw
            flips = (_endgame(g)
                     and _parity_ok_after(g, player, 0, our_draw_next=ours_next)
                     and not _parity_ok_after(g, player, their_drain,
                                              our_draw_next=ours_next))
            lapse_ok = _parity_ok_after(g, player, -1, our_draw_next=ours_next)
            if fatal_now or (flips and lapse_ok):
                return True
    if _in_deckout_mode(g, player):
        # Racing to deck the opponent out: only Day's Undoing matters — it
        # reshuffles every zone back into the library and resets the race.
        # Their drawing/milling only empties the shared deck faster (good for
        # us), a lone fish is handled by blocks/removal, and Memory Lapse would
        # itself grow the library by putting the countered spell back on top.
        return name == "Day's Undoing"
    if name == "Dandân":
        # Rollout audit (2026-08-04): 2,853 determinized decline decisions,
        # mean -0.032 win-prob, t=-9.7 -- EVERY decline reason of the old
        # branch measured negative (hold-removal -0.034 t=-8.9, "comfortable"
        # -0.029 t=-3.8, known-spare -0.024 n.s.). Lapse-first denies the
        # fish AND makes their next draw a known dead redraw, while removal
        # stays banked for the next threat; the old doctrine priced the Lapse
        # as "only a delay" and ignored the top-lock and the recast tax.
        # Arena: 52.06% at 40k, confirmed 52.04% [51.73, 52.35] at 100k.
        return True
    # Anything aimed at our stuff (Mind Bend / Crystal Spray / Metamorphose on
    # our permanents, or a counter on our own spell). No grab-plan gate here:
    # even a redrawn removal spell bought the permanent a turn.
    own = set(g.players[player].battlefield)
    own.update(s.source_instance_id for s in g.stack if s.controller == player)
    if any(t.get("id") in own for t in (so.targets or [])):
        # Counter-protection by REDRAW, not by war (human doctrine, 2026-08-04):
        # their Memory Lapse on our instant DRAW spell during THEIR turn is a
        # delay, not a loss — the spell tops the shared library and OUR draw
        # step is next, so we redraw it for free while they are down a premium
        # counter. Escalating buys one turn of tempo at the price of counter
        # parity (mirror mining: 29/200 games did exactly that). Fight only
        # when the counterer can actually steal the topped spell (theft tool
        # visible) or in the library endgame, where resolve-vs-top-back is a
        # parity question the drain branch above owns.
        if (inst.name == "Memory Lapse" and not _endgame(g)
                and _natural_next_drawer(g, player) == player
                and not _opp_theft_ready(g, player)):
            tid = (so.targets or [{}])[0].get("id")
            tso = next((s for s in g.stack if s.source_instance_id == tid), None)
            tobj = g.objects.get(tid)
            if (tso is not None and tso.controller == player and tobj is not None
                    and tobj.name in ("Brainstorm", "Accumulated Knowledge",
                                      "Fact or Fiction", "Predict",
                                      "Mystical Tutor")):
                return False
        # War-depth cap on FISH defense (field log: a four-Lapse war
        # protecting one 2-mana Dandân, which died to a Mind Bend a turn
        # later anyway; the human doctrine — fish are removal checks, the
        # premium counters belong to the card war). One Lapse per fish:
        # if the counter chain under their spell roots in a Dandân of ours
        # and one of our Lapses is already fighting for it, stop escalating.
        # (Wholesale never-defend-the-fish measured 45.9% in the rejected
        # economy rework — the cap only stops the SECOND Lapse onward.)
        if any(s.controller == player
               and (o := g.objects.get(s.source_instance_id)) is not None
               and o.name == "Memory Lapse" for s in g.stack):
            root, cur, hops = None, so, 0
            while cur is not None and hops < 12:
                hops += 1
                tids = [t.get("id") for t in (cur.targets or [])]
                if not tids:
                    break
                below = next((s for s in g.stack
                              if s.source_instance_id == tids[0]), None)
                if below is None:
                    root = g.objects.get(tids[0])
                    break
                ro = g.objects.get(below.source_instance_id)
                if ro is not None and ro.name != "Memory Lapse" \
                        and ro.name not in ("Crystal Spray", "Mind Bend",
                                            "Metamorphose", "Vision Charm"):
                    root = ro
                    break
                cur = below
            if (root is not None and root.name == "Dandân"
                    and root.controller == player):
                return False
        return True
    # Value spells: a Lapse puts the spell on top of the SHARED library, so on
    # OUR turn the opponent's draw step simply reclaims it — countering without
    # a plan to draw it ourselves at instant speed (cycle / Bay / Brainstorm /
    # AK, after paying the Lapse) only converts their spell into a one-turn
    # delay at the price of our premium counter. On THEIR turn the counter is
    # a steal (our draw is next) and needs no plan.
    if (g.active_player == player
            and name in ("Fact or Fiction", "Day's Undoing", "Accumulated Knowledge")
            and not _can_grab_after_lapse(g, player)):
        return False
    if name in ("Fact or Fiction", "Day's Undoing"):
        return True
    if name == "Accumulated Knowledge":
        aks = sum(1 for cid in g.graveyard if g.objects[cid].name == "Accumulated Knowledge")
        return aks >= 2
    if name == "Vision Charm" and (inst.chosen or {}).get("mode") == "land":
        return bool(_sac_creatures(g, player))            # it would kill our fish
    return False


def _can_grab_after_lapse(g: GameState, player: str) -> bool:
    """After paying Memory Lapse ({1}{U}), can we still take — or Predict-mill —
    the returned top card at instant speed before the opponent's draw step?
    Mirrors the tools in _instant_top_draw with a two-mana margin for the Lapse
    itself. Predict is the strongest follow-up: the Lapsed spell is a KNOWN
    top, so Predict names it, mills it (never redrawn), and draws 2."""
    avail = sum(_mana_view(g, player).values())
    hand = _hand_by_name(g, player)
    if avail >= 3 and (hand.get("Lonely Sandbar")
                       or (hand.get("Brainstorm") and _safe_to_cast(g, player, "Brainstorm"))):
        return True
    if avail >= 4:
        if (hand.get("Accumulated Knowledge")
                and _safe_to_cast(g, player, "Accumulated Knowledge")):
            return True
        if hand.get("Predict") and _safe_to_cast(g, player, "Predict"):
            return True                                   # Lapse + Predict: deny and draw 2
        bays = [iid for iid in g.players[player].battlefield
                if g.objects[iid].name == "The Surgical Bay" and not g.objects[iid].tapped]
        others = [iid for iid in _untapped_lands(g, player)
                  if g.objects[iid].name != "The Surgical Bay"]
        if bays and len(others) >= 4:                     # 2 for the Lapse + 2 for the Bay
            return True
    return False


# ── Combat ──────────────────────────────────────────────────────────────────

def choose_attackers(g: GameState, player: str, eligible: list) -> list:
    """Attack with everything — unless OUTNUMBERED. With symmetric 4/1 fish a
    block is a costless 1-for-1 trade, so attacking presents the defender a
    losing choice (trade, or take 4): ablation from the old conservative rules
    ran 50.0% -> 52.2% widening toward always-attack. But attacking taps the
    team: with fewer fish than theirs, racing is arithmetic suicide (we deal
    4a while taking 4b unblocked, a < b) — field log: a lone fish kept
    attacking into a two-fish board and its owner died on turn 6 having never
    blocked. Behind on board, the team stays home to trade as blockers."""
    opp = E._OTHER[player]
    if len(eligible) >= len(_creatures(g, opp)):
        return eligible
    return []


def choose_blocks(g: GameState, player: str, eligible: list) -> dict:
    """Every block here is a one-for-one trade (4/1 vs 4/1). Block everything
    when possible: like attacking, a block is a 1-for-1 board trade that costs
    no hand cards, and declining it just donates 4 damage. Measured: gating
    blocks on life/board conditions only ever lost winrate."""
    attackers = list(g.combat.attackers.keys())
    if not attackers:
        return {}
    blocks = {}
    pool = list(eligible)
    for aid in attackers:                                 # one blocker per attacker
        if not pool:
            break
        blocks[aid] = [pool.pop(0)]
    return blocks


# ── Resolution decisions ────────────────────────────────────────────────────

def resolve_pending(g: GameState) -> None:
    """Resolve every pending decision that belongs to the heuristic AI (one
    decision often leads straight into the next, e.g. mulligan -> bottom)."""
    guard = 0
    while guard < 25:
        guard += 1
        p = g.pending
        if not p or p.type == "priority":                 # priority is take_priority's job
            return
        pl = g.players.get(p.player)
        if not pl or not pl.is_ai or getattr(pl, "ai_profile", "") != "heuristic_1_3":
            return
        handler = _PENDING_HANDLERS.get(p.type)
        if handler is None:
            return
        marker = id(p)
        handler(g, p.player, p.context)
        if g.pending is not None and id(g.pending) == marker:
            return                                        # no progress — bail, don't spin


def _decide_play_order(g, player, ctx):
    E.choose_play_order(g, player, "first")


_CANTRIPS = ("Brainstorm", "Ponder")   # the 1-mana diggers that find land #2 off one land


def _keepable(lands: int, cantrips: int, kept: int) -> bool:
    """Whether to keep a fresh seven that will be bottomed to `kept` cards.
    Low curve, so 2-4 lands is ideal; 1 land needs cantrips to find #2 (fewer are
    demanded as we mull lower); 5+ lands is a flood; never dig below five."""
    if kept <= 4:                                         # mull floor: a shaky 4 beats a 3
        return lands >= 1
    if lands == 0 or lands >= 5:                          # no mana / flooded
        return False
    if lands >= 2:                                        # 2..4 lands
        return True
    return cantrips >= (2 if kept >= 6 else 1)            # 1 land: lean on the diggers


def _decide_mulligan(g, player, ctx):
    cards = _hand_cards(g, player)
    lands = sum(1 for o in cards if _is_land(o.type_line))
    cantrips = sum(1 for o in cards if o.name in _CANTRIPS)
    kept = E._OPENING_HAND - g.players[player].mulligans
    keep = _keepable(lands, cantrips, kept)
    E.mulligan_decision(g, player, "keep" if keep else "mulligan")


def _decide_bottom(g, player, ctx):
    """Bottom to ~3 lands, and when land-light protect the diggers (a 1-land keep
    that ships its Brainstorm/Ponder has thrown away its escape hatch)."""
    n = ctx.get("count", 0)
    hand = list(g.players[player].hand)
    lands = [iid for iid in hand if _is_land(g.objects[iid].type_line)]
    nonlands = [iid for iid in hand if iid not in lands]
    excess_lands = lands[3:]                              # keep up to three lands
    land_light = (len(lands) - len(excess_lands)) <= 1    # keeping a single land

    def ship_rank(iid):                                   # lower = bottomed sooner
        v = card_value(g, player, iid)
        if g.objects[iid].name in _CANTRIPS and land_light:
            v += 6.0                                      # the digger is the escape hatch — keep it
        return v

    bottomable = excess_lands + sorted(nonlands, key=ship_rank)
    E.bottom_cards(g, player, bottomable[:n])


def _draw_desirability(g, player, iid) -> float:
    """How much the AI wants to DRAW this card next (vs its abstract value):
    lands rate by how short on mana we are. The Surgical Bay and Lonely
    Sandbar stop being 'lands' once mana is met: each is a banked guaranteed
    draw — deck-out war-chest ammunition — so they never rate as gift-safe
    duds for an opponent slot, and our own scry keeps them instead of
    bottoming them (field log: a Halimar arrangement served the human a Bay
    on his next draw as if it were a spare Island; he banked it and later
    Lapse-stole a Fact or Fiction with it)."""
    o = g.objects.get(iid)
    if not o:
        return 0.0
    if _is_land(o.type_line):
        if _lands_in_play(g, player) < 4:
            return 7.0
        return 5.0 if o.name in ("The Surgical Bay", "Lonely Sandbar") else 1.5
    return card_value(g, player, iid)


def _decide_scry(g, player, ctx):
    """Scry on the SHARED library. Keep the cards we want where WE draw them and
    bottom the rest — but the next draw is usually the opponent's, so for a slot
    they draw we instead keep a dud on top (waste their draw) and bottom anything
    that would help them (deny it; we weren't getting it anyway). If we hold a
    castable Predict and would keep nothing of our own, leave one card we draw on
    top: a known top is a guaranteed Predict hit (name it, mill it, draw two)."""
    cards = [c["instance_id"] for c in ctx.get("cards", [])]
    if not cards:
        E.complete_scry(g, player, [], [])
        return
    opp = E._OTHER[player]
    seats = _draw_assignment(g, player, len(cards), 0)
    order = _fill_slots(g, player, cards, seats)
    tops, bottoms = [], []
    for iid, seat in zip(order, seats):
        if seat == player:
            keep = _draw_desirability(g, player, iid) >= _KEEP_MIN   # a card we want to draw
        else:
            keep = _draw_desirability(g, opp, iid) < _KEEP_MIN       # a dud we're glad to feed them
        (tops if keep else bottoms).append(iid)
    if not tops and bottoms and seats[0] == player:
        hand = _hand_by_name(g, player)
        if any(_affordable(g, player, g.objects[i].mana_cost) and _safe_to_cast(g, player, "Predict")
               for i in hand.get("Predict", [])):
            tops.append(bottoms.pop(0))                   # set a known top for Predict
    E.complete_scry(g, player, tops, bottoms)


def _decide_reorder(g, player, ctx):
    """Ponder / Halimar Depths. Reorder the looked-at cards across the slots their
    future drawers will take (Ponder draws the first one itself; the rest, like
    Halimar's three, usually go to the opponent first off the shared library) so
    our best cards land where WE draw them and the duds where THEY do. Still
    shuffle to dig when nothing here is worth keeping for us."""
    ids = [c["instance_id"] for c in ctx.get("cards", [])]
    seats = _draw_assignment(g, player, len(ids), ctx.get("draw_after", 0))
    order = _fill_slots(g, player, ids, seats)
    if ctx.get("allow_shuffle") and all(
            _draw_desirability(g, player, iid) < _KEEP_MIN for iid in ids):
        E.complete_reorder(g, player, order, shuffle=True)
    else:
        E.complete_reorder(g, player, order)


def _decide_putback(g, player, ctx):
    """Brainstorm: keep our best in hand and put the least-wanted cards back on
    top. On our own turn the opponent draws first off the shared library, so steer
    the worst of those toward their draw and keep the better ones for our own
    slots (we re-draw them — Predict fuel); cast at the opponent's end step the
    next draw is ours, so we keep both."""
    n = ctx.get("slots", 0)
    hand = list(g.players[player].hand)
    seats = _draw_assignment(g, player, n, 0)             # putback has no immediate draw
    opp = E._OTHER[player]
    all_gift = n > 0 and all(s == opp for s in seats)

    def keep_value(iid):
        v = card_value(g, player, iid) + _dig_bonus(g, player, iid)
        if all_gift:
            # Every slot feeds the opponent (their pending stack draws, or
            # their natural draws own-turn): a shipped card is lost AND arms
            # them, so price the gift side too — an excess Island beats a
            # live Predict out the door.
            v += _draw_desirability(g, opp, iid)
        return v

    chosen = sorted(hand, key=keep_value)[:n]             # ship the least wanted (dig-aware)
    order = _fill_slots(g, player, chosen, seats)
    E.complete_putback(g, player, order)


def _next_drawer(g, player):
    """Who draws the next card off the SHARED library. On our own turn the
    opponent's draw step precedes our next one, so they draw next — unless we
    hold a castable PONDER, the one draw spell we actually cast on our own
    main (it takes the top before their draw). Brainstorm and Accumulated
    Knowledge do NOT flip this: draw-go policy defers them to the OPPONENT'S
    end step, which is AFTER their draw step has taken the current top (field
    logs: with a castable AK in hand, Halimar arrangements seated slot 1 as
    'ours', parked the best keep there — a Dandân, once — and handed it to
    the opponent's very next draw). On the opponent's turn (e.g. an
    instant-speed tutor at their end step) our draw step is next."""
    opp = E._OTHER[player]
    if g.active_player != player:
        return player
    hand = _hand_by_name(g, player)
    for iid in hand.get("Ponder", []):
        if _affordable(g, player, g.objects[iid].mana_cost) and _safe_to_cast(g, player, "Ponder"):
            return player
    return opp


def _opp_pending_stack_draws(g, player):
    """Top-of-library slots that OPPONENT objects already on the stack will
    consume when they resolve — their draw spells and draw_1 activations
    sitting under our resolving one. Field log 2026-08-04: the AI Brainstormed
    over a human Brainstorm to race a Lapsed AK back (correctly), then seated
    its putbacks as its own future draws — the pending Brainstorm below ate
    both, gifting a live Predict. Mills count too: a slot the opponent's
    Predict mills is just as lost to us as one they draw."""
    opp = E._OTHER[player]
    total = 0
    for so in g.stack:
        if so.controller != opp:
            continue
        if so.kind == "activated" and (so.chosen or {}).get("effect") == "draw_1":
            total += 1
        elif so.kind == "spell":
            o = g.objects.get(so.source_instance_id)
            if o is None:
                continue
            if o.name in _DRAW_ON_RESOLVE:
                mill, draw = _DRAW_ON_RESOLVE[o.name]
                total += mill + draw
            elif o.name == "Accumulated Knowledge":
                total += 1 + sum(1 for cid in g.graveyard
                                 if g.objects[cid].name == "Accumulated Knowledge")
            elif o.name == "Fact or Fiction":
                total += 5
    return total


def _draw_assignment(g, player, n, draw_after=0):
    """Which seat draws each of the top n library slots, in order. The first
    `draw_after` slots are the resolving effect's own immediate draws (always the
    controller — e.g. Ponder's draw-one); next come any slots consumed by
    OPPONENT draw objects pending on the stack (they resolve before anyone's
    draw step); the rest fall to future draw steps off the SHARED library,
    alternating from whoever draws next (_next_drawer). On our own turn that
    next draw is usually the opponent's, so leaving a card on top most often
    hands it to them."""
    opp = E._OTHER[player]
    seats = [player] * min(draw_after, n)
    pend = _opp_pending_stack_draws(g, player)
    while len(seats) < n and pend > 0:
        seats.append(opp)
        pend -= 1
    nxt = _next_drawer(g, player)
    while len(seats) < n:
        seats.append(nxt)
        nxt = opp if nxt == player else player
    return seats


# NB: known-opponent-hand awareness was measured here and came out neutral.
# The engine legitimately exposes cards we have seen enter their hand
# (known_by survives the draw - Lapse-backs, tutor reveals, slots we
# arranged; at half of all decision points we know >= 1 of their cards, most
# often Memory Lapse). Adjusting gift values by known holdings (redundant
# copy -1.5 / known-AK +3, and an AK-only variant) both measured 49.9-50.0%
# at 40k: the mine+theirs ranking already avoids the bad gifts. The
# known-hand channel remains available for future consumers (Lapse-aware
# sequencing, Day's-aware hoarding) - expect mirror-neutral, human-facing
# value only.


def _fill_slots(g, player, ids, seats):
    """Order `ids` into the slots described by `seats` (one seat per slot, same
    length). Because the library is shared, a card in an opponent-drawn slot is a
    gift: keep the cards worth most to US in our own slots, and steer the cards
    least useful to the OPPONENT into theirs. Keep-priority is value-to-us plus
    value-to-them (mine+theirs): high means 'we want it and don't want them to
    have it' -> keep; low means a dead card -> safe to hand over. Within our slots
    our best is drawn soonest; within theirs their worst is drawn soonest."""
    opp = E._OTHER[player]
    mine = {i: _draw_desirability(g, player, i) for i in ids}
    theirs = {i: _draw_desirability(g, opp, i) for i in ids}
    n_me = sum(1 for s in seats if s == player)
    ranked = sorted(ids, key=lambda i: mine[i] + theirs[i], reverse=True)
    my_cards = ranked[:n_me]
    opp_cards = ranked[n_me:]
    my_cards.sort(key=lambda i: mine[i], reverse=True)    # our best, drawn soonest
    opp_cards.sort(key=lambda i: theirs[i])               # their worst, drawn soonest
    order, mi, oi = [], 0, 0
    for s in seats:
        if s == player:
            order.append(my_cards[mi]); mi += 1
        else:
            order.append(opp_cards[oi]); oi += 1
    return order


def _decide_search(g, player, ctx):
    """Mystical Tutor puts the found card on TOP of the SHARED library, so the
    NEXT draw gets it — and on our own turn that draw is usually the OPPONENT's.
    Only commit to a card when we draw next; otherwise fail to find (just
    shuffle) rather than hand the opponent a free instant/sorcery."""
    eligible = ctx.get("eligible", [])
    if not eligible or _next_drawer(g, player) != player:
        E.complete_library_search(g, player, None)        # shuffle, find nothing
        return
    pick = _stance_fetch(g, player, eligible)             # dig flags override the wishlist
    if pick is not None:
        E.complete_library_search(g, player, pick)
        return
    by_name = {}
    for iid in eligible:
        by_name.setdefault(g.objects[iid].name, []).append(iid)
    hand = _hand_by_name(g, player)
    hand_names = set(hand)
    opp = E._OTHER[player]
    # Fetch doctrine (field review 2026-08-03: tutoring Memory Lapse on an
    # empty board is the floor of the card): answer live pressure first
    # (Vision Charm beats a multi-fish board, the cantrip Spray / permanent
    # Bend a single fish); the RESET when we're losing the card war (same
    # hand-gap gate as the Day's Undoing cast itself); extend an AK chain we
    # already dominate; otherwise the engines. Memory Lapse is the fallback,
    # not the default — reactive cards are fetched by dig flags/stance when
    # a specific fight is coming, not stockpiled a card down.
    wishlist = []
    opp_fish = _sac_creatures(g, opp)
    if opp_fish:
        if len(opp_fish) >= 2 and "Vision Charm" not in hand_names:
            wishlist.append("Vision Charm")
        for nm in ("Crystal Spray", "Mind Bend"):
            if nm not in hand_names:
                wishlist.append(nm)
    if (len(g.players[player].hand) <= 2 and len(g.players[opp].hand) >= 4
            and "Day's Undoing" not in hand_names):
        wishlist.append("Day's Undoing")
    if len(hand.get("Accumulated Knowledge", [])) >= 2:
        wishlist.append("Accumulated Knowledge")
    wishlist += ["Fact or Fiction", "Accumulated Knowledge", "Memory Lapse"]
    for name in wishlist:
        if by_name.get(name):
            E.complete_library_search(g, player, by_name[name][0])
            return
    best = max(eligible, key=lambda iid: card_value(g, player, iid), default=None)
    E.complete_library_search(g, player, best)


def _decide_fof_split(g, player, ctx):
    """Splitting the opponent's Fact or Fiction. The caster keeps ONE pile and
    bins the other, so we make the two piles as close in value as we can — then
    whichever they keep, they gain the least. Value is judged from the CASTER's
    seat (they do the choosing). Naive value-balanced partition: deal each card,
    richest first, onto the lighter pile — a stopgap until the ML splitter lands.
    (Replaces the old isolate-the-bomb 1/4 split, which usually just handed them
    the fat pile.)"""
    revealed = list(ctx.get("revealed", []))
    if not revealed:
        E.complete_fof_split(g, player, [], [])
        return
    caster = ctx.get("caster", E._OTHER[player])
    pile1, pile2 = [], []
    v1 = v2 = 0.0
    for iid in sorted(revealed, key=lambda i: card_value(g, caster, i), reverse=True):
        val = card_value(g, caster, iid)
        if v1 <= v2:
            pile1.append(iid)
            v1 += val
        else:
            pile2.append(iid)
            v2 += val
    E.complete_fof_split(g, player, pile1, pile2)


def _decide_fof_choose(g, player, ctx):
    v1 = sum(card_value(g, player, iid) for iid in ctx.get("pile1_ids", []))
    v2 = sum(card_value(g, player, iid) for iid in ctx.get("pile2_ids", []))
    E.complete_fof_choose(g, player, 1 if v1 >= v2 else 2)


def _decide_name_card(g, player, ctx):
    """Predict: name the known top card (guaranteed hit), else Island (the most
    common card in the deck)."""
    top = _known_top(g, player)
    name = top.name if top is not None else "Island"
    if name not in ctx.get("names", []):
        name = "Island"
    E.complete_name_card(g, player, name)


def _decide_text_change(g, player, ctx):
    """Turn the type a sacrifice clause needs into one its controller has none
    of — Island into Swamp kills a Dandân."""
    frm, controller = "Island", None
    for tid in ctx.get("change_targets", []):
        o = g.objects.get(tid)
        if o and _sac_type(o):
            frm, controller = _sac_type(o), o.controller
            break
    to = next((t for t in BASIC_TYPES
               if t != frm and (controller is None
                                or not E.controls_basic_type(g, controller, t))), "Swamp")
    E.complete_text_change(g, player, frm, to)


def _decide_put_from_hand(g, player, ctx):
    """Metamorphose's consolation: free tempo — put the best eligible permanent
    onto the battlefield (a creature over an untapped land over the rest)."""
    eligible = ctx.get("eligible", [])

    def rank(iid):
        o = g.objects[iid]
        if E._is_creature(o):
            return 0
        if _is_land(o.type_line) and not enters_tapped(g, player, o):
            return 1
        return 2

    best = min(eligible, key=lambda iid: (rank(iid), iid), default=None)
    E.complete_put_from_hand(g, player, best)


def _decide_discard(g, player, ctx):
    E.discard_to_hand_size(g, player, choose_discards(g, player, ctx.get("count", 0)))


def _decide_graveyard(g, player, ctx):
    """Resolution "may" for a targeted recursion (Mystic Sanctuary): it puts the
    chosen card on top of the SHARED library, so on our own turn the OPPONENT
    draws it next. Take the best card back only when WE draw next; otherwise
    decline the "may" rather than gift the opponent a free instant/sorcery (the
    same shared-deck logic as Mystical Tutor — see _decide_search)."""
    eligible = ctx.get("eligible", [])
    if not eligible:
        E.complete_graveyard_choice(g, player, None)
        return
    if _next_drawer(g, player) != player and ctx.get("may", True):
        E.complete_graveyard_choice(g, player, None)      # decline rather than feed the opponent
        return
    pick = max(eligible, key=lambda iid: card_value(g, player, iid), default=None)
    E.complete_graveyard_choice(g, player, pick)


_PENDING_HANDLERS = {
    "choose_play_order": _decide_play_order,
    "mulligan": _decide_mulligan,
    "bottom": _decide_bottom,
    "scry": _decide_scry,
    "reorder": _decide_reorder,
    "putback": _decide_putback,
    "search_library": _decide_search,
    "fof_split": _decide_fof_split,
    "fof_choose": _decide_fof_choose,
    "name_card": _decide_name_card,
    "choose_text_change": _decide_text_change,
    "put_from_hand": _decide_put_from_hand,
    "discard": _decide_discard,
    "choose_graveyard": _decide_graveyard,
}


# ── Other engine-facing hooks ───────────────────────────────────────────────

def choose_trigger_target(g: GameState, t, legal: list) -> str | None:
    """A trigger that targets (Mystic Sanctuary): pick the most valuable card."""
    return max(legal, key=lambda iid: card_value(g, t.controller, iid), default=None)


def _deckout_keep_value(g: GameState, player: str, iid: str) -> float:
    """How much to KEEP a card when the deck-out is the only win (higher = keep).
    The deck-manipulation cards ARE the win condition, so they outrank
    everything; a Dandân, removal or Day's Undoing is dead weight against an
    Islandless opponent; excess lands go once a mana base is down."""
    o = g.objects.get(iid)
    if not o:
        return -1.0
    name = o.name
    if name == "Predict" or _library_drain(g, player, name) not in (None, 0):
        return 12.0                                       # cantrips / mill / Fact or Fiction
    if _is_land(o.type_line):
        return 8.0 if _lands_in_play(g, player) < 4 else 2.0
    if name == "Memory Lapse":
        return 5.0                                        # one answer to a Day's Undoing reset
    return 1.0                                            # Dandân, Mind Bend, Crystal Spray, ...


def choose_discards(g: GameState, player: str, excess: int) -> list:
    """Cleanup: discard the least valuable cards. When the deck-out is the only
    win, keep the deck-manipulation tools and pitch the dead combat cards;
    otherwise fall back to general card value (excess lands rate lowest)."""
    hand = list(g.players[player].hand)
    key = (_deckout_keep_value if _in_deckout_mode(g, player) else card_value)
    return sorted(hand, key=lambda iid: key(g, player, iid))[:excess]


# ── Play-time evaluator (v1.3: belief-sampled Lapse / fish-pace gate) ───────
# Round-10 finding: the Memory Lapse cast/decline and cast-fish-vs-hold
# decisions hold ~+4pp that no feature rule captures — oracle substitution
# measured 53.8-54.9% against the pure heuristic while every distilled rule
# LOST (bundle 43.2%, singles 46.5-48.6%): the oracle agrees with the rules
# 82% of the time and its deviations are not feature-separable. The edge
# does NOT survive below ~10 belief samples (S=4 halves it), but the
# rollout horizon amortizes almost entirely into the value function: a TRUE
# two-turn horizon (my action resolves + their answering turn) measures
# 54.8% vs the pre-fold heuristic, identical to full-game rollouts (54.6%),
# while horizon 1 costs ~2pp (52.8%). (The prototype's "horizon 1 = horizon
# 40" finding was an artifact: the re-entrant engine ignored the pump-loop
# turn cap, so every setting silently rolled ~18 turns — see
# _RolloutHorizon.) So these two decision classes are decided by
# evaluation: resample the unseen zones S times, resolve TWO turns under
# each sampled world for every candidate (common random numbers), score
# with the fitted value model, and deviate from the heuristic's choice only
# when the challenger also wins a 2S fresh-seed head-to-head
# (winner's-curse guard). Measured: ~60ms mean / 126ms p90 per gated
# decision, ~9 per game; every other decision stays a microsecond rule.
# Master switch, OFF in mainline v1.3: the evaluator's +4-5pp costs a
# ~60x benchmark slowdown (40s -> ~45min per 40k arena), which is the
# wrong trade for the rapid-prototyping mainline. The oracle line lives on
# as `ai_oracle.py` (this file exec'd with the switch on) — v1.3-oracle, a
# side evolution benchmarked on its own cadence.
_EVAL_GATES = False
_EVAL_S = 10                # sampled worlds per candidate; confirm on 2S
_EVAL_MIN_TURN = 4
_EVAL_MAX_PER_GAME = 12
_EVAL_HORIZON_TURNS = 2     # TRUE rollout horizon (see _RolloutHorizon)
_in_rollout = False
_EVAL_LOG_HOOK = None       # datagen taps gated decisions here; inert if None


class _RolloutHorizon(Exception):
    """Unwinds a rollout clone once its horizon turn is reached. The engine
    is re-entrant — one take_priority cascades the game forward internally,
    so a pump-loop turn check alone never truncates anything (profiled: the
    '1-turn' playouts were silently running ~18 turns, 87% of evaluator
    cost). Raising from the rollout decision path is the only cut that
    works; the half-advanced clone is then scored as-is and discarded."""

# Logistic value model fitted on 577k turn-boundary snapshots from 20k
# mirror games (holdout Brier 0.199 vs 0.227 for the life/hand heuristic;
# reliability near-diagonal in every decile). It is applied to DETERMINIZED
# worlds, where the sampled zones are legitimately visible — the hidden
# information is handled by averaging over worlds, not by peeking.
_VALUE_MODEL = {
    "features": ["life_diff", "hand_diff", "fish_diff", "lands_diff",
                 "lapse_diff", "draw_diff", "removal_diff", "flip_diff",
                 "days_diff", "lib", "turn", "active", "parity",
                 "parity_end", "endgame", "deckout_gap"],
    "mu": [0.03347541324462002, -4.3317046123990714e-05, 0.005174221159510691,
           0.0005111411442630904, -0.0035455002252486397, 0.0010331115500571786,
           0.00969652077485532, 0.002700817825830821, -0.0031079980593963336,
           43.831765256263644, 17.15805956960183, 0.500140780399903,
           0.49901020549606684, 0.03163010707973802, 0.06362624319922376,
           1.1316080153862147],
    "sd": [6.43109902410327, 2.244893865134053, 0.889574715306431,
           1.9957883330372, 1.2046600932805966, 1.3661428751940359,
           1.3719670923338447, 0.6916343024174401, 0.7569755956196632,
           18.019472645102024, 11.572062002542202, 0.4999999811813636,
           0.4999990213058695, 0.17501326737066913, 0.24408593745537477,
           3.5576894048725496],
    "w": [0.21281565092725677, 0.4935295292771423, 0.7875495692542026,
          0.15319870459089563, 0.1988436119825829, 0.1641356560456295,
          0.3844891492811589, 0.22548257686711926, 0.025416630774613855,
          0.02965511910279908, 0.027900825041739755, 0.2093043647648059,
          0.013332941979475784, 0.04335203570511088, -0.03137061690304697,
          -0.005536017694998151],
    "b": 0.014640200567328822,
}
_VF_DRAW = ("Brainstorm", "Ponder", "Predict", "Accumulated Knowledge",
            "Fact or Fiction", "Mystical Tutor")
_VF_REMOVAL = ("Crystal Spray", "Mind Bend", "Vision Charm", "Metamorphose")


def _value_features(g: GameState, pov: str) -> dict:
    opp = E._OTHER[pov]

    def hand_count(pid, names):
        return sum(1 for i in g.players[pid].hand
                   if i in g.objects and g.objects[i].name in names)
    L = len(g.library)
    nxt = _natural_next_drawer(g, pov)
    parity = 1.0 if ((L % 2 == 1) == (nxt == pov)) else 0.0
    endg = 1.0 if 0 < L <= 12 else 0.0
    return {
        "life_diff": g.players[pov].life - g.players[opp].life,
        "hand_diff": len(g.players[pov].hand) - len(g.players[opp].hand),
        "fish_diff": (len(_sac_creatures(g, pov))
                      - len(_sac_creatures(g, opp))),
        "lands_diff": _lands_in_play(g, pov) - _lands_in_play(g, opp),
        "lapse_diff": (hand_count(pov, ("Memory Lapse",))
                       - hand_count(opp, ("Memory Lapse",))),
        "draw_diff": hand_count(pov, _VF_DRAW) - hand_count(opp, _VF_DRAW),
        "removal_diff": (hand_count(pov, _VF_REMOVAL)
                         - hand_count(opp, _VF_REMOVAL)),
        "flip_diff": (int(_instant_top_draw(g, pov) is not None)
                      - int(_instant_top_draw(g, opp) is not None)),
        "days_diff": (hand_count(pov, ("Day's Undoing",))
                      - hand_count(opp, ("Day's Undoing",))),
        "lib": L, "turn": g.turn_number,
        "active": 1.0 if g.active_player == pov else 0.0,
        "parity": parity, "parity_end": parity * endg, "endgame": endg,
        "deckout_gap": max(0, 20 - L),
    }


def _position_score(g: GameState, player: str) -> float:
    st = g.result.get("status")
    if st == f"{player}_wins":
        return 1.0
    if st == f"{E._OTHER[player]}_wins":
        return 0.0
    f = _value_features(g, player)
    m = _VALUE_MODEL
    z = m["b"]
    for k, mu, sd, w in zip(m["features"], m["mu"], m["sd"], m["w"]):
        z += w * ((f[k] - mu) / sd)
    return 1.0 / (1.0 + _math.exp(-z))


def _determinize(g: GameState, player: str, rng) -> None:
    """Resample what `player` cannot see: unknown opponent-hand cards and
    library slots unknown to BOTH players are pooled and redealt (interactive
    cards at retention weight 2 for the hand slots). Slots the opponent has
    seen keep their cards so their knowledge stays consistent."""
    opp = E._OTHER[player]
    hand_pos = [k for k, iid in enumerate(g.players[opp].hand)
                if player not in (g.objects[iid].known_by or [])]
    lib_pos = [k for k, slot in enumerate(g.library)
               if not slot.known_by.get(player) and not slot.known_by.get(opp)]
    pool = ([g.players[opp].hand[k] for k in hand_pos]
            + [g.library[k].instance_id for k in lib_pos])
    if len(pool) < 2:
        return
    hand_fill = []
    avail = list(pool)
    for _ in hand_pos:
        weights = [2.0 if g.objects[i].name in (
                       "Memory Lapse", "Crystal Spray", "Mind Bend",
                       "Vision Charm", "Metamorphose") else 1.0
                   for i in avail]
        pick = rng.choices(range(len(avail)), weights=weights)[0]
        hand_fill.append(avail.pop(pick))
    rng.shuffle(avail)
    for k, iid in zip(hand_pos, hand_fill):
        g.players[opp].hand[k] = iid
        g.objects[iid].known_by = []
    for k, iid in zip(lib_pos, avail):
        g.library[k].instance_id = iid
        g.objects[iid].known_by = []


def _rollout_play(g: GameState, until_turn: int, max_pumps: int = 20000):
    """Pump a cloned game with this module playing both seats (plain
    heuristic — `_in_rollout` is set by the caller) until the turn cap."""
    stall = 0
    for _ in range(max_pumps):
        if g.result.get("status") != "ongoing" or g.turn_number >= until_turn:
            break
        p = g.pending
        if p is None:
            break
        marker = (id(p), len(g.log), len(g.stack))
        if p.type == "priority":
            take_priority(g, p.player)
        else:
            E._resolve_ai_pending(g)
        now = g.pending
        if (g.result.get("status") == "ongoing" and now is p
                and (id(now), len(g.log), len(g.stack)) == marker):
            stall += 1
            if stall >= 3:
                if now.type == "priority":
                    E.pass_priority(g, now.player)
                    stall = 0
                else:
                    break
        else:
            stall = 0


def _eval_scores(g: GameState, player: str, cands: list, n: int,
                 did: int) -> list:
    """Mean position score per candidate over n sampled worlds, common
    random numbers. A candidate is an action tuple or a hold-the-fish dict."""
    global _in_rollout
    scores = [0.0] * len(cands)
    old_mod, old_flag = E._ai_mod, _in_rollout
    E._ai_mod = lambda *a, **k: _SELF
    _in_rollout = True
    try:
        # Pickle transport: ~4x cheaper than deepcopy on this state shape,
        # and each sampled world is serialized ONCE then loaded per branch.
        blob_g = _pickle.dumps(g, _pickle.HIGHEST_PROTOCOL)
        for s in range(n):
            rng = _random.Random(did * 1000 + s)
            det = _pickle.loads(blob_g)
            _determinize(det, player, rng)
            until = det.turn_number + _EVAL_HORIZON_TURNS
            blob_det = _pickle.dumps(det, _pickle.HIGHEST_PROTOCOL)
            for ci, branch in enumerate(cands):
                b = _pickle.loads(blob_det)
                b._rollout_until = until
                try:
                    if isinstance(branch, dict):
                        b._suppress_fish_turn = branch["suppress_fish"]
                    else:
                        _execute(b, player, branch)
                    _rollout_play(b, until)
                except _RolloutHorizon:
                    pass                             # scored as-is below
                scores[ci] += _position_score(b, player)
    finally:
        E._ai_mod = old_mod
        _in_rollout = old_flag
    return [x / n for x in scores]


def _action_without_fish(g: GameState, player: str) -> tuple:
    pl = g.players[player]
    fish = [i for i in pl.hand if g.objects[i].name == "Dandân"]
    for i in fish:
        pl.hand.remove(i)
    try:
        return _choose_action(g, player)
    finally:
        pl.hand.extend(fish)


def _gate_candidates(g: GameState, player: str, action: tuple):
    """The two evaluator-gated decision classes. cands[0] is always the
    heuristic's own choice."""
    opp = E._OTHER[player]
    kind = action[0]
    top = g.stack[-1] if g.stack else None
    cast_name = g.objects[action[1]].name if kind == "cast" else None
    if cast_name == "Memory Lapse" and top is not None:
        return [action, ("pass",)]
    if (top is not None and top.kind == "spell" and top.controller == opp
            and cast_name != "Memory Lapse"):
        lapses = _hand_by_name(g, player).get("Memory Lapse", [])
        if lapses and _affordable(g, player, g.objects[lapses[0]].mana_cost):
            return [action, ("cast", lapses[0], None, top.source_instance_id)]
        return None
    if cast_name == "Dandân" and g.active_player == player and not g.stack:
        return [action, {"suppress_fish": g.turn_number}]
    return None


def _evaluator_gate(g: GameState, player: str, action: tuple) -> tuple:
    st = getattr(g, "_evp", None)
    if st is None:
        st = {"n": 0, "done": set(), "ctr": 0, "sup": None}
        g._evp = st                                   # plain attr — not serialized
    if (st["sup"] == g.turn_number and action[0] == "cast"
            and g.objects[action[1]].name == "Dandân"):
        return _action_without_fish(g, player)        # hold already decided
    key = (g.turn_number, g.current_step)
    if st["n"] >= _EVAL_MAX_PER_GAME or key in st["done"]:
        return action
    cands = _gate_candidates(g, player, action)
    if cands is None:
        return action
    st["done"].add(key)
    st["n"] += 1
    st["ctr"] += 1
    did = (g.turn_number * 100003 + len(g.log) * 101 + st["ctr"]) & 0x7FFFFFFF
    screen = _eval_scores(g, player, cands, _EVAL_S, did)
    best = max(range(len(cands)), key=lambda i: screen[i])
    hold = None
    if best != 0:
        st["ctr"] += 1
        did2 = (did + 500 + st["ctr"]) & 0x7FFFFFFF
        hold = _eval_scores(g, player, [cands[0], cands[best]],
                            2 * _EVAL_S, did2)
        if hold[1] <= hold[0]:
            best = 0
    if _EVAL_LOG_HOOK is not None:
        try:
            _EVAL_LOG_HOOK(g, player, cands, screen, hold, best)
        except Exception:                                 # noqa: BLE001 - telemetry only
            pass
    pick = cands[best]
    if isinstance(pick, dict):
        st["sup"] = g.turn_number
        return _action_without_fish(g, player)
    return pick


_SELF = _sys.modules[__name__]
