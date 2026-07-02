"""The heuristic sandbox AI (p2): a card-aware policy that plays lands, casts
and responds with spells, attacks and blocks when profitable, and makes a
sensible choice for every resolution decision. The engine calls in through a
handful of hooks (take_priority, resolve_pending, choose_attackers,
choose_blocks, choose_trigger_target, choose_discards) whenever the pending
action belongs to an AI with ai_profile == "heuristic".

This is heuristic v1.0 — the line the RL run has been training against: the
default "heuristic" profile, the vs-heuristic eval anchor, and the scenario
bot. The stronger testbench heuristic (v1.1, fishrl/forgetful_fish/ai_v1_1.py)
lives behind the separate "heuristic_1_1" profile as an additional PFSP pool
opponent only, so the training baseline stays comparable.

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


def card_value(g: GameState, player: str, iid: str) -> float:
    """How much the AI wants this card in hand / on top of the library."""
    o = g.objects.get(iid)
    if not o:
        return 0.0
    name = o.name
    if _is_land(o.type_line):
        lands = _lands_in_play(g, player)
        base = 6.0 if lands < 4 else (3.0 if lands < 6 else 1.0)
        if name != "Island" and lands >= 4:
            base += 0.5                                   # utility lands edge out Islands late
        return base
    opp = E._OTHER[player]
    if name == "Dandân":
        return 9.0
    if name == "Memory Lapse":
        return 8.0
    if name == "Mind Bend":
        return 9.0 if _sac_creatures(g, opp) else 7.0
    if name == "Fact or Fiction":
        return 7.0
    if name == "Crystal Spray":
        return 8.0 if _sac_creatures(g, opp) else 6.0
    if name == "Accumulated Knowledge":
        aks = sum(1 for cid in g.graveyard if g.objects[cid].name == "Accumulated Knowledge")
        return min(4.0 + 2.0 * aks, 9.0)
    if name == "Predict":
        return 8.0 if _known_top(g, player) else 4.0
    if name == "Metamorphose":
        return 5.0
    if name == "Mystical Tutor":
        return 5.0
    if name == "Brainstorm":
        return 5.0
    if name == "Ponder":
        return 4.0
    if name == "Vision Charm":
        return 4.0
    if name == "Day's Undoing":
        return 0.0 if _in_deckout_mode(g, player) else 2.0  # a reset would undo the deck-out
    return 3.0


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

    # Threat: cast Dandân whenever affordable (it can't attack this turn anyway,
    # so either main phase is fine).
    for iid in hand.get("Dandân", []):
        if _affordable(g, player, g.objects[iid].mana_cost):
            return ("cast", iid, None, None)

    draw = _card_advantage_action(g, player, hand)
    if draw is not None:
        return draw

    # Day's Undoing: refill an empty hand when the opponent is far ahead on
    # cards (main2, so the whole turn was used first).
    if g.current_step == "main2" and hand.get("Day's Undoing"):
        iid = hand["Day's Undoing"][0]
        if (_affordable(g, player, g.objects[iid].mana_cost)
                and len(g.players[player].hand) <= 2
                and len(g.players[opp].hand) >= 4):
            return ("cast", iid, None, None)

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
    """Kill the opponent's Dandân: Mind Bend (permanent) > Crystal Spray (until
    end of turn + a card) > Vision Charm's land mode (kills ALL Dandâns — only
    when the trade is clearly profitable)."""
    opp = E._OTHER[player]
    targets = _sac_creatures(g, opp)
    mine = _sac_creatures(g, player)
    if targets:
        tgt = targets[0]
        for name in ("Mind Bend", "Crystal Spray"):
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
    """The best affordable draw spell — kept on a leash when Memory Lapse mana
    should stay open."""
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

    aks = sum(1 for cid in g.graveyard if g.objects[cid].name == "Accumulated Knowledge")
    options = []                                          # (priority, iid, target)
    for iid in hand.get("Fact or Fiction", []):
        options.append((0, iid))
    if aks >= 1:
        for iid in hand.get("Accumulated Knowledge", []):
            options.append((1, iid))
    if _known_top(g, player):                             # guaranteed Predict hit
        for iid in hand.get("Predict", []):
            options.append((2, iid))
    for iid in hand.get("Accumulated Knowledge", []):
        options.append((3, iid))
    for iid in hand.get("Brainstorm", []):
        options.append((4, iid))
    for iid in hand.get("Ponder", []):
        options.append((5, iid))
    for iid in hand.get("Predict", []):
        options.append((6, iid))
    for iid in hand.get("Mystical Tutor", []):
        options.append((7, iid))
    for _, iid in sorted(options, key=lambda t: t[0]):
        if castable(iid):
            return ("cast", iid, None, None)
    return None


# ── Deck-out mode (winning by emptying the shared library) ──────────────────
#
# library / graveyard / exile are ONE shared, ordered pool, and a player loses
# only when they must draw at their OWN draw step with an empty library
# (spell-induced draws from an empty library merely whiff). During the AI's
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


def _response_action(g: GameState, player: str) -> tuple:
    """Anything outside my own quiet main phase: counter the opponent's spell
    with Memory Lapse when it threatens us, otherwise pass."""
    opp = E._OTHER[player]
    top = g.stack[-1] if g.stack else None
    if (top is not None and top.kind == "spell" and top.controller == opp
            and _counter_worthy(g, player, top)):
        lapses = _hand_by_name(g, player).get("Memory Lapse", [])
        if lapses and _affordable(g, player, g.objects[lapses[0]].mana_cost):
            return ("cast", lapses[0], None, top.source_instance_id)
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
        return True
    # Anything aimed at our stuff (Mind Bend / Crystal Spray / Metamorphose on
    # our permanents, or a counter on our own spell).
    own = set(g.players[player].battlefield)
    own.update(s.source_instance_id for s in g.stack if s.controller == player)
    if any(t.get("id") in own for t in (so.targets or [])):
        return True
    if name in ("Fact or Fiction", "Day's Undoing"):
        return True
    if name == "Accumulated Knowledge":
        aks = sum(1 for cid in g.graveyard if g.objects[cid].name == "Accumulated Knowledge")
        return aks >= 2
    if name == "Vision Charm" and (inst.chosen or {}).get("mode") == "land":
        return bool(_sac_creatures(g, player))            # it would kill our fish
    return False


# ── Combat ──────────────────────────────────────────────────────────────────

def choose_attackers(g: GameState, player: str, eligible: list) -> list:
    """Attack with everything when it's profitable: free damage, outnumbering
    the blockers, or a lethal race — otherwise hold back."""
    opp = E._OTHER[player]
    blockers = E._eligible_blockers(g, opp)
    a, b = len(eligible), len(blockers)
    if b == 0:
        return eligible
    power = sum(g.objects[iid].power for iid in eligible)
    excess = power - 4 * b                                # damage that gets through blocks
    if excess >= g.players[opp].life:
        return eligible                                   # lethal even through blocks
    if g.players[player].life <= 8 and b >= a:
        return []                                         # too far behind to trade into blocks
    if a > b:
        return eligible                                   # outnumber their blockers
    return []


def choose_blocks(g: GameState, player: str, eligible: list) -> dict:
    """Every block here is a one-for-one trade (4/1 vs 4/1). Block everything
    when unblocked damage would be lethal; trade when level or behind on life."""
    attackers = list(g.combat.attackers.keys())
    if not attackers:
        return {}
    incoming = sum(g.objects[a].power for a in attackers if a in g.objects)
    life = g.players[player].life
    opp = E._OTHER[player]
    must_block = incoming >= life
    want_trades = life <= 12 or len(_creatures(g, player)) >= len(_creatures(g, opp))
    if not (must_block or want_trades):
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
        if not pl or not pl.is_ai or getattr(pl, "ai_profile", "") != "heuristic":
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


def _decide_mulligan(g, player, ctx):
    lands = sum(1 for o in _hand_cards(g, player) if _is_land(o.type_line))
    keep = 2 <= lands <= 5 or g.players[player].mulligans >= 2
    E.mulligan_decision(g, player, "keep" if keep else "mulligan")


def _decide_bottom(g, player, ctx):
    n = ctx.get("count", 0)
    hand = list(g.players[player].hand)
    lands = [iid for iid in hand if _is_land(g.objects[iid].type_line)]
    extra_lands = lands[3:]                               # keep three lands
    rest = sorted((iid for iid in hand if iid not in extra_lands),
                  key=lambda iid: card_value(g, player, iid))
    E.bottom_cards(g, player, (extra_lands + rest)[:n])


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
    tops, bottoms = [], []
    for c in ctx.get("cards", []):
        iid = c["instance_id"]
        if _draw_desirability(g, player, iid) >= 3.0:
            tops.append(iid)
        else:
            bottoms.append(iid)
    E.complete_scry(g, player, tops, bottoms)


def _decide_reorder(g, player, ctx):
    ids = [c["instance_id"] for c in ctx.get("cards", [])]
    order = sorted(ids, key=lambda iid: -_draw_desirability(g, player, iid))
    if ctx.get("allow_shuffle") and all(
            _draw_desirability(g, player, iid) < 3.0 for iid in ids):
        E.complete_reorder(g, player, order, shuffle=True)
    else:
        E.complete_reorder(g, player, order)


def _decide_putback(g, player, ctx):
    """Brainstorm: put back the two least wanted cards, the better one on top
    (drawn again first). They become known top slots — Predict fuel."""
    n = ctx.get("slots", 0)
    hand = list(g.players[player].hand)
    chosen = sorted(hand, key=lambda iid: card_value(g, player, iid))[:n]
    chosen.sort(key=lambda iid: -card_value(g, player, iid))
    E.complete_putback(g, player, chosen)


def _decide_search(g, player, ctx):
    """Mystical Tutor: removal if it has a job, else the counter, else draw."""
    eligible = ctx.get("eligible", [])
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
    """Splitting the opponent's Fact or Fiction: isolate the best card so they
    must choose between it and everything else."""
    revealed = list(ctx.get("revealed", []))
    if not revealed:
        E.complete_fof_split(g, player, [], [])
        return
    best = max(revealed, key=lambda iid: card_value(g, player, iid))
    rest = [iid for iid in revealed if iid != best]
    E.complete_fof_split(g, player, [best], rest)


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
    """Resolution "may" for a targeted recursion (Mystic Sanctuary): the AI chose
    the target when the ability went on the stack, so put the best eligible card
    on top of the library."""
    eligible = ctx.get("eligible", [])
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
