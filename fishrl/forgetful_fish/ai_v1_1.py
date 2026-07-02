"""The heuristic sandbox AI (p2): a card-aware policy that plays lands, casts
and responds with spells, attacks and blocks when profitable, and makes a
sensible choice for every resolution decision. The engine calls in through a
handful of hooks (take_priority, resolve_pending, choose_attackers,
choose_blocks, choose_trigger_target, choose_discards) whenever the pending
action belongs to an AI with ai_profile == "heuristic_1_1".

This is heuristic v1.1 — the stronger testbench line (seat-aware shared-deck
manipulation, draw-go discipline, full combat aggression). It lives behind its
own engine profile so it can serve as a SEPARATE PFSP pool opponent; the
previous heuristic (v1.0, fishrl/forgetful_fish/ai.py) remains the default
"heuristic" profile: the eval anchor, the scenario bot, and the pool member
the run has been training against.

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
    "Memory Lapse": 10.0,
    "Mind Bend": 9.0, "Mind Bend_threat": 11.0,
    "Fact or Fiction": 9.0,
    "Crystal Spray": 8.0, "Crystal Spray_threat": 10.0,
    "AK_base": 6.0, "AK_per": 4.0, "AK_cap": 11.0,
    "Predict": 6.0, "Predict_known": 10.0,
    "Metamorphose": 6.0,
    "Mystical Tutor": 8.0,
    "Brainstorm": 8.0,
    "Ponder": 7.0,
    "Vision Charm": 7.0,
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
        return min(V["AK_base"] + V["AK_per"] * aks, V["AK_cap"])
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
    _execute(g, player, _choose_action(g, player))


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
    return _instant_top_draw(g, player)


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
    # so either main phase is fine).
    for iid in hand.get("Dandân", []):
        if _affordable(g, player, g.objects[iid].mana_cost):
            return ("cast", iid, None, None)

    draw = _card_advantage_action(g, player, hand)
    if draw is not None:
        return draw

    # Day's Undoing: refill an empty hand when the opponent is far ahead on
    # cards (main2, so the whole turn was used first). Don't fire when we know
    # the top card — that means we've set the library up (scry/Brainstorm/etc.)
    # and the reshuffle would throw that away. (A swing-based gate,
    # min(7, theirs) - ours >= 3, measured exactly 50.0% — same trigger set in
    # practice, so the simpler hand-size gate stays.)
    if g.current_step == "main2" and hand.get("Day's Undoing"):
        iid = hand["Day's Undoing"][0]
        if (_affordable(g, player, g.objects[iid].mana_cost)
                and len(g.players[player].hand) <= 2
                and len(g.players[opp].hand) >= 4
                and _known_top(g, player) is None):
            return ("cast", iid, None, None)

    # Flood valve: sac The Surgical Bay to draw a card once we're flooded, as
    # long as the draw is safe (a non-empty library) and two OTHER untapped lands
    # can pay {1}{U} (it taps itself as part of the cost).
    if _lands_in_play(g, player) >= 6 and len(g.library) >= 1:
        bays = [iid for iid in g.players[player].battlefield
                if g.objects[iid].name == "The Surgical Bay" and not g.objects[iid].tapped]
        others = [iid for iid in _untapped_lands(g, player)
                  if g.objects[iid].name != "The Surgical Bay"]
        if bays and len(others) >= 2:
            return ("activate", bays[0], 1)               # {1}{U}, T, Sacrifice: Draw a card

    # Flood valve: cycle Lonely Sandbar once the board has plenty of lands.
    if _lands_in_play(g, player) >= 5:
        for iid in g.players[player].hand:
            o = g.objects.get(iid)
            if o and o.name in CYCLING and _affordable(g, player, "{1}"):
                return ("cycle", iid)

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
        for name in ("Crystal Spray", "Mind Bend"):
            for iid in hand.get(name, []):
                if _affordable(g, player, g.objects[iid].mana_cost):
                    return ("cast", iid, None, tgt)
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
    reserve = 0
    if hand.get("Memory Lapse") and g.turn_number >= 3:
        reserve = 2                                       # keep {1}{U} for the counter

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
        if castable(iid) and _safe_to_cast(g, player, g.objects[iid].name):
            return ("cast", iid, None, None)
    return None


def _end_step_draw_action(g: GameState, player: str) -> tuple | None:
    """Draw-go: the best instant-speed draw spell at the OPPONENT's end step
    (the caller checks the timing). No mana reserve here — after their end step
    nothing of theirs resolves before we untap, so holding mana back buys
    nothing. No "strong position" gate on the big engines either: at THIS
    window a Memory Lapse on our spell is self-punishing (the spell returns to
    the shared top and our draw step reclaims it, while their premium counter
    hits the graveyard), so there is no position weak enough to justify
    holding a resolvable Fact or Fiction — a counter-backup gate measured
    41.5%. AK sequencing is a last-mover war over the shared graveyard: each
    AK we cast upgrades THEIR next one, so never seed an empty graveyard
    unless we hold the majority of the remaining chain (2+ copies) and are
    therefore the likely last mover."""
    hand = _hand_by_name(g, player)
    aks_gy = sum(1 for cid in g.graveyard
                 if g.objects[cid].name == "Accumulated Knowledge")
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
    for iid in hand.get("Predict", []):                   # blind: card-neutral + a denial mill
        options.append((3, iid))
    for _, iid in sorted(options, key=lambda t: t[0]):
        name = g.objects[iid].name
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
    portion can deck us; conserve the spell when the shared deck is too thin."""
    L = len(g.library)
    if name == "Brainstorm":
        return L >= 3                                     # draws three (before putting two back)
    if name == "Accumulated Knowledge":
        aks = sum(1 for cid in g.graveyard if g.objects[cid].name == "Accumulated Knowledge")
        return L >= 1 + aks
    if name == "Predict":                                 # mills one, then draws up to two
        return L >= 3
    if name in ("Ponder", "Lonely Sandbar"):              # draws one
        return L >= 1
    return True                                           # Fact or Fiction / Vision Charm: no draw


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


def _deckout_main_action(g: GameState, player: str) -> tuple:
    """Own main phase while combat is hopeless: develop mana, stay alive against
    the opponent's fish, steer the shared library toward an even, empty deck so
    the opponent draws the last card, and empty Dandâns out of hand (they can't
    attack, so casting them only frees room to hoard deck-manipulation). Never
    spends a draw spell for value and never casts Day's Undoing — the former are
    the parity tools, the latter would reshuffle every zone back and reset the
    race."""
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
    for iid in hand.get("Dandân", []):                    # park dead fish to free hand space
        if _affordable(g, player, g.objects[iid].mana_cost):
            return ("cast", iid, None, None)
    return ("pass",)


_TOP_RACE_MIN_VALUE = 7.0   # only race for a real prize, not a mediocre top
_KEEP_MIN = 3.0             # scry/reorder bar: a card worth drawing (vs a dud)


def _instant_top_draw(g, player):
    """The cheapest affordable INSTANT-speed draw that pulls the current top card
    into our hand — expendable tools first (cycle a Lonely Sandbar, sac The
    Surgical Bay) before spending a card-draw spell. Returns an action tuple, or
    None if we can't draw at instant speed right now."""
    hand = _hand_by_name(g, player)
    for iid in hand.get("Lonely Sandbar", []):            # cycling {U}: discard, draw one
        if _affordable(g, player, "{U}"):
            return ("cycle", iid)
    if g.library:                                         # sac The Surgical Bay: draw one
        bays = [iid for iid in g.players[player].battlefield
                if g.objects[iid].name == "The Surgical Bay" and not g.objects[iid].tapped]
        others = [iid for iid in _untapped_lands(g, player)
                  if g.objects[iid].name != "The Surgical Bay"]
        if bays and len(others) >= 2:
            return ("activate", bays[0], 1)
    for name in ("Accumulated Knowledge", "Brainstorm"):  # instants that draw off the top
        for iid in hand.get(name, []):
            if _affordable(g, player, g.objects[iid].mana_cost) and _safe_to_cast(g, player, name):
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
            if o and o.name in ("Brainstorm", "Accumulated Knowledge", "Crystal Spray"):
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
    return _instant_top_draw(g, player)


def _response_action(g: GameState, player: str) -> tuple:
    """Anything outside my own quiet main phase: counter the opponent's spell
    with Memory Lapse when it threatens us, race a known card off the top, else
    pass."""
    opp = E._OTHER[player]
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
        tutors = _hand_by_name(g, player).get("Mystical Tutor", [])
        if tutors and _affordable(g, player, g.objects[tutors[0]].mana_cost):
            return ("cast", tutors[0], None, None)
    # NB: our OWN upkeep is deliberately NOT a draw window. Floyd's "cast in
    # your upkeep" rule protects a specific must-resolve spell from Memory
    # Lapse, but as a general window it taps the most expensive mana of the
    # game — everything spent at upkeep crowds out our own main phase, unlike
    # the opponent's end step where we untap immediately after. Measured:
    # 45.4% vs not doing it.
    return ("pass",)


def _counter_worthy(g: GameState, player: str, so) -> bool:
    """Whether the opponent's spell on the stack is worth a Memory Lapse."""
    inst = g.objects.get(so.source_instance_id)
    if not inst:
        return False
    name = inst.name
    if _in_deckout_mode(g, player):
        # Racing to deck the opponent out: only Day's Undoing matters — it
        # reshuffles every zone back into the library and resets the race.
        # Their drawing/milling only empties the shared deck faster (good for
        # us), a lone fish is handled by blocks/removal, and Memory Lapse would
        # itself grow the library by putting the countered spell back on top.
        return name == "Day's Undoing"
    if name == "Dandân":
        # Memory Lapse only DELAYS a fish — it goes back on top of the shared
        # library to be redrawn — so it's a poor use of a premium counter unless
        # the fish actually matters. Don't spend it when we hold a permanent
        # answer (Crystal Spray / Mind Bend), or when we're comfortable: healthy
        # life, no attacking board, and our own creatures to block or race with.
        # Otherwise it's our only answer — counter it.
        hand = _hand_by_name(g, player)
        if hand.get("Crystal Spray") or hand.get("Mind Bend"):
            return False                                  # remove it permanently instead
        opp = E._OTHER[player]
        attackers = [a for a in g.players[opp].battlefield
                     if E._is_creature(g.objects[a]) and not E._attack_restricted(g, opp, a)]
        return (g.players[player].life <= 12 or bool(attackers)
                or not _sac_creatures(g, player))
    # Anything aimed at our stuff (Mind Bend / Crystal Spray / Metamorphose on
    # our permanents, or a counter on our own spell). No grab-plan gate here:
    # even a redrawn removal spell bought the permanent a turn.
    own = set(g.players[player].battlefield)
    own.update(s.source_instance_id for s in g.stack if s.controller == player)
    if any(t.get("id") in own for t in (so.targets or [])):
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
    """After paying Memory Lapse ({1}{U}), can we still take the returned top
    card at instant speed before the opponent's draw step? Mirrors the tools in
    _instant_top_draw with a two-mana margin for the Lapse itself."""
    avail = sum(_mana_view(g, player).values())
    hand = _hand_by_name(g, player)
    if avail >= 3 and (hand.get("Lonely Sandbar")
                       or (hand.get("Brainstorm") and _safe_to_cast(g, player, "Brainstorm"))):
        return True
    if avail >= 4:
        if (hand.get("Accumulated Knowledge")
                and _safe_to_cast(g, player, "Accumulated Knowledge")):
            return True
        bays = [iid for iid in g.players[player].battlefield
                if g.objects[iid].name == "The Surgical Bay" and not g.objects[iid].tapped]
        others = [iid for iid in _untapped_lands(g, player)
                  if g.objects[iid].name != "The Surgical Bay"]
        if bays and len(others) >= 4:                     # 2 for the Lapse + 2 for the Bay
            return True
    return False


# ── Combat ──────────────────────────────────────────────────────────────────

def choose_attackers(g: GameState, player: str, eligible: list) -> list:
    """Attack with everything, always. With symmetric 4/1 fish every block is a
    1-for-1 board trade that costs neither side a hand card, so attacking only
    ever presents the defender a losing choice: trade (fine for us) or take 4
    (life pressure that devalues their hand in the attrition war). Measured by
    ablation from the old conservative rules (hold back unless outnumbering or
    lethal): every widening of the attack condition won more — the full
    gradient runs 50.0% -> 52.2% from always-hold-heuristics to always-attack."""
    return eligible


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
        if not pl or not pl.is_ai or getattr(pl, "ai_profile", "") != "heuristic_1_1":
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
    lands rate by how short on mana we are."""
    o = g.objects.get(iid)
    if not o:
        return 0.0
    if _is_land(o.type_line):
        return 7.0 if _lands_in_play(g, player) < 4 else 1.5
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

    def keep_value(iid):
        return card_value(g, player, iid) + _dig_bonus(g, player, iid)

    chosen = sorted(hand, key=keep_value)[:n]             # ship the least wanted (dig-aware)
    seats = _draw_assignment(g, player, len(chosen), 0)   # putback has no immediate draw
    order = _fill_slots(g, player, chosen, seats)
    E.complete_putback(g, player, order)


def _next_drawer(g, player):
    """Who draws the next card off the SHARED library. On our own turn the
    opponent's draw step precedes our next one, so they draw next — unless we
    still hold a castable draw spell that takes the top card ourselves first
    (Brainstorm/Ponder/Accumulated Knowledge all draw off the top). On the
    opponent's turn (e.g. an instant-speed tutor at their end step) our draw
    step is next, so we draw next."""
    opp = E._OTHER[player]
    if g.active_player != player:
        return player
    hand = _hand_by_name(g, player)
    for name in ("Brainstorm", "Ponder", "Accumulated Knowledge"):
        for iid in hand.get(name, []):
            if _affordable(g, player, g.objects[iid].mana_cost) and _safe_to_cast(g, player, name):
                return player
    return opp


def _draw_assignment(g, player, n, draw_after=0):
    """Which seat draws each of the top n library slots, in order. The first
    `draw_after` slots are the resolving effect's own immediate draws (always the
    controller — e.g. Ponder's draw-one); the rest fall to future draw steps off
    the SHARED library, alternating from whoever draws next (_next_drawer). On our
    own turn that next draw is usually the opponent's, so leaving a card on top
    most often hands it to them."""
    opp = E._OTHER[player]
    seats = [player] * min(draw_after, n)
    nxt = _next_drawer(g, player)
    while len(seats) < n:
        seats.append(nxt)
        nxt = opp if nxt == player else player
    return seats


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
    hand_names = {g.objects[i].name for i in g.players[player].hand if i in g.objects}
    opp = E._OTHER[player]
    wishlist = []
    if _sac_creatures(g, opp) and "Mind Bend" not in hand_names:
        wishlist.append("Mind Bend")
    if "Memory Lapse" not in hand_names:
        wishlist.append("Memory Lapse")
    wishlist += ["Fact or Fiction", "Accumulated Knowledge"]
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
