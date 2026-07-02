"""Forgetful Fish game flow: the opening (roll for first player, choose play
order, deal opening hands) and the sandbox AI opponent, which simply passes its
turns and priority.

The sandbox game is admin-only: p1 is the human, p2 is the AI. The game starts
on no phase; both players roll d20 until someone rolls higher, the winner is
prompted to play first or second, then 7 cards are dealt to each player (the
player who will take the first turn is dealt first).
"""
from __future__ import annotations

import re
import uuid

from fishrl.forgetful_fish.state import (
    GameState, PendingDecision, StackObject, LibrarySlot, Combat, new_game, draw_card, draw_cards,
    play_card, resolve_top, tap_land, untap_last, shuffle_library, _is_land, _is_permanent,
    _mark_seen, _public_object, enters_tapped, mark_library_known, forget_rearranged,
    PERMANENT_ABILITIES,
    CYCLING, _reverse_source, rng_from_state, _serialize_rng,
    BASIC_TYPES, add_text_change, revert_text_changes, expire_eot_text_changes,
    controls_basic_type, land_mana_color, _drop_pool, MODAL_SPELLS, modal_mode_text,
)

_OTHER = {"p1": "p2", "p2": "p1"}
_OPENING_HAND = 7


# Heuristic AI versions, keyed by ai_profile. "heuristic" is v1.0 (the RL run's
# long-standing opponent/eval anchor); "heuristic_1_1" is the stronger testbench
# line, exposed as a SEPARATE profile so it can join the PFSP pool without moving
# the training baseline.
_HEURISTIC_PROFILES = ("heuristic", "heuristic_1_1")


def _ai_mod(g: "GameState" = None, pid: str = None):
    """The heuristic AI module for `pid`'s profile, imported lazily: the ai modules
    import this engine at module top, so the engine side of the cycle must resolve
    at call time. Without arguments (or for the default profile) this is v1.0."""
    if g is not None and pid is not None:
        p = g.players.get(pid)
        if p is not None and getattr(p, "ai_profile", "heuristic") == "heuristic_1_1":
            from fishrl.forgetful_fish import ai_v1_1
            return ai_v1_1
    from fishrl.forgetful_fish import ai
    return ai


def _heuristic(g: "GameState", pid: str) -> bool:
    """Whether `pid` is an AI playing a heuristic profile (any version, vs the
    passive auto-passing opponent)."""
    p = g.players.get(pid)
    return bool(p and p.is_ai and getattr(p, "ai_profile", "heuristic") in _HEURISTIC_PROFILES)


def _draw_or_lose(g: GameState, player: str, n: int = 1) -> bool:
    """Draw `n` cards as a game action. Attempting to draw from an empty library
    loses the game (CR 104.3c / 120.3) — for ANY draw, the natural draw step AND
    spell/ability draws (Brainstorm, Ponder, Accumulated Knowledge, Predict, …),
    not just the once-per-turn draw. Returns True if the player lost, so the
    caller can stop processing the rest of the effect."""
    drawn = draw_cards(g, player, n)
    if len(drawn) < n:
        _lose(g, player, "drawing from an empty library")
        return True
    return False


# Turn structure (CR 500–514): the ordered steps of a turn. Untap and cleanup
# normally grant no priority; the main steps are where lands/sorceries are legal.
_STEPS = (
    "untap", "upkeep", "draw", "main1", "begin_combat", "declare_attackers",
    "declare_blockers", "combat_damage", "main2", "end", "cleanup",
)
_MAIN_STEPS = ("main1", "main2")


def new_sandbox_game(decklist, *, seed=None, game_id=None, p1_name="You",
                     ai_profile="heuristic") -> GameState:
    """Create a sandbox game (p2 = AI), roll for first player, and let the AI
    auto-resolve its choice if it won the roll. `p1_name` is set before the roll
    so even the opening log lines read with the player's name. `ai_profile`
    selects the opponent: "heuristic" (default — actually plays) or "passive"
    (the original auto-passing opponent)."""
    g = new_game(decklist, p1_name=p1_name, p2_name="Sandbox AI", seed=seed, game_id=game_id)
    g.players["p2"].is_ai = True
    g.players["p2"].ai_profile = (ai_profile if ai_profile in _HEURISTIC_PROFILES + ("passive",)
                                  else "heuristic")
    g.current_step = ""        # pregame: no phase yet
    g.active_player = ""       # undecided until the roll is resolved
    _roll_to_start(g)
    _resolve_ai_pending(g)     # if the AI won the roll it chooses immediately
    return g


def new_multiplayer_game(decklist, *, p1_name, p2_name, seed=None, game_id=None) -> GameState:
    """Create a two-human game: roll for the first player and wait. Neither player
    is an AI, so the roll winner (who may be p2) makes the play-order choice and
    the whole opening is driven by each player's own actions — no auto-resolve."""
    g = new_game(decklist, p1_name=p1_name, p2_name=p2_name, seed=seed, game_id=game_id)
    g.current_step = ""        # pregame: no phase yet
    g.active_player = ""       # undecided until the roll is resolved
    _roll_to_start(g)          # sets pending=choose_play_order for the roll winner
    return g


def _roll_to_start(g: GameState) -> None:
    rng = rng_from_state(g.rng_state)
    winner = "p1"
    for _ in range(50):
        r1, r2 = rng.randint(1, 20), rng.randint(1, 20)
        g.log.append(f"{g.players['p1'].name} rolled {r1}; {g.players['p2'].name} rolled {r2}.")
        if r1 != r2:
            winner = "p1" if r1 > r2 else "p2"
            break
        g.log.append("Tie, rolling again.")
    g.rng_state = _serialize_rng(rng)
    g.log.append(f"{g.players[winner].name} wins the roll.")
    g.pending = PendingDecision(type="choose_play_order", player=winner)


def _resolve_ai_pending(g: GameState) -> None:
    """Auto-resolve a pending decision that belongs to the AI."""
    pending = g.pending
    if not pending:
        return
    player = g.players.get(pending.player)
    if not player or not player.is_ai:
        return
    if _heuristic(g, pending.player):                    # real choices live in ai.py
        _ai_mod(g, pending.player).resolve_pending(g)
        return
    if pending.type == "choose_play_order":
        choose_play_order(g, pending.player, "first")   # the AI always plays first
    elif pending.type == "put_from_hand":
        complete_put_from_hand(g, pending.player, None)  # the passive AI puts nothing
    elif pending.type == "fof_split":
        cards = pending.context["revealed"]              # the passive AI splits evenly
        half = (len(cards) + 1) // 2
        complete_fof_split(g, pending.player, cards[:half], cards[half:])


def choose_play_order(g: GameState, player: str, choice: str) -> bool:
    """The roll winner chooses to play "first" or "second"; deal opening hands
    and start the mulligan phase."""
    if not g.pending or g.pending.type != "choose_play_order" or g.pending.player != player:
        return False
    first = player if choice == "first" else _OTHER[player]
    g.first_player = first
    g.pending = None
    g.log.append(f"{g.players[player].name} chooses to play {choice}.")
    _deal_opening_hands(g, first)
    _start_mulligans(g)
    return True


# ── Mulligans (London): keep or mulligan in turn, then bottom one card per
#    mulligan taken. The player taking the first turn decides first; decisions
#    alternate until both have kept. ───────────────────────────────────────

def _start_mulligans(g: GameState) -> None:
    for p in g.players.values():
        p.kept = False
    _set_mulligan_turn(g, g.first_player)


def _set_mulligan_turn(g: GameState, player: str) -> None:
    """Prompt `player` to keep/mulligan — unless they have already mulliganed
    down to nothing (>= the opening hand size), in which case they auto-keep
    zero cards with no prompt or bottoming choice."""
    if g.players[player].mulligans >= _OPENING_HAND:
        _keep_hand(g, player)
        return
    g.pending = PendingDecision(type="mulligan", player=player,
                                context={"mulligans": g.players[player].mulligans})
    _resolve_ai_mulligan(g)


def mulligan_decision(g: GameState, player: str, action: str) -> bool:
    if not g.pending or g.pending.type != "mulligan" or g.pending.player != player:
        return False
    if action == "mulligan":
        g.players[player].mulligans += 1
        _mulligan_hand(g, player)
        kept_to = _OPENING_HAND - g.players[player].mulligans
        g.log.append(f"{g.players[player].name} mulligans (will keep {max(kept_to, 0)}).")
        _next_mulligan_decider(g, player)
    else:
        _keep_hand(g, player)
    return True


def _mulligan_hand(g: GameState, player: str) -> None:
    """Shuffle the hand back, reshuffle the library (clearing position
    knowledge), and draw a fresh seven."""
    p = g.players[player]
    for iid in p.hand:
        g.library.append(LibrarySlot(instance_id=iid))
    p.hand = []
    shuffle_library(g)                                # reshuffles and clears all knowledge
    for _ in range(_OPENING_HAND):
        if g.library:
            p.hand.append(g.library.pop(0).instance_id)


def _keep_hand(g: GameState, player: str) -> None:
    p = g.players[player]
    bottom = min(p.mulligans, _OPENING_HAND)
    if bottom == 0:
        p.kept = True
        g.log.append(f"{p.name} keeps.")
        _next_mulligan_decider(g, player)
    elif bottom >= len(p.hand):
        # Mulliganed down to nothing: the whole hand goes to the bottom.
        _bottom_instances(g, player, list(p.hand))
        p.kept = True
        g.log.append(f"{p.name} keeps 0 cards.")
        _next_mulligan_decider(g, player)
    else:
        g.pending = PendingDecision(type="bottom", player=player, context={"count": bottom})


def bottom_cards(g: GameState, player: str, instance_ids: list) -> bool:
    """Resolve a keep: put the chosen cards on the bottom in the chosen order."""
    if not g.pending or g.pending.type != "bottom" or g.pending.player != player:
        return False
    need = g.pending.context.get("count", 0)
    hand = g.players[player].hand
    chosen, seen = [], set()
    for iid in instance_ids:
        if iid in hand and iid not in seen:
            chosen.append(iid)
            seen.add(iid)
    if len(chosen) != need:
        return False
    _bottom_instances(g, player, chosen)
    g.players[player].kept = True
    g.log.append(f"{g.players[player].name} keeps {len(hand)} cards (bottomed {need}).")
    _next_mulligan_decider(g, player)
    return True


def _bottom_instances(g: GameState, player: str, instance_ids: list) -> None:
    p = g.players[player]
    for iid in instance_ids:
        if iid in p.hand:
            p.hand.remove(iid)
            slot = LibrarySlot(instance_id=iid)
            slot.known_by[player] = True          # revealed to them at the bottom
            g.library.append(slot)


def _next_mulligan_decider(g: GameState, last: str) -> None:
    other = _OTHER[last]
    if not g.players[other].kept:
        nxt = other
    elif not g.players[last].kept:
        nxt = last
    else:
        nxt = None
    if nxt is None:
        _finish_mulligans(g)
    else:
        _set_mulligan_turn(g, nxt)


def _resolve_ai_mulligan(g: GameState) -> None:
    p = g.pending
    if p and p.type == "mulligan" and g.players.get(p.player) and g.players[p.player].is_ai:
        if _heuristic(g, p.player):               # may mulligan, then bottom cards
            _ai_mod(g, p.player).resolve_pending(g)
            return
        mulligan_decision(g, p.player, "keep")    # the passive AI always keeps


def _finish_mulligans(g: GameState) -> None:
    g.pending = None
    g.turn_number = 0
    _begin_turn(g, g.first_player)


def _deal_opening_hands(g: GameState, first: str) -> None:
    # Simultaneous deal: the player taking the first turn is dealt first.
    for pid in (first, _OTHER[first]):
        for _ in range(_OPENING_HAND):
            if g.library:
                g.players[pid].hand.append(g.library.pop(0).instance_id)


# ── Turn structure, priority, and the stack (CR 117, 405, 500–514) ────────

def _begin_turn(g: GameState, player: str) -> None:
    g.active_player = player
    g.turn_number += 1
    g.turns_since_stop += 1
    g.players[player].land_played_this_turn = False
    for pl in g.players.values():                    # "pass rest of turn" ends with the turn
        pl.yield_mode = ""
    # Each player numbers their own turns: 1/p1, 1/p2, 2/p1, 2/p2, … (the global
    # turn counter still alternates, so the player turn is half it, rounded up).
    g.log.append(f"Turn {(g.turn_number + 1) // 2}: {g.players[player].name}.")
    _enter_step(g, "untap")


def _enter_step(g: GameState, step: str) -> None:
    g.current_step = step
    g.passed = {"p1": False, "p2": False}
    p = g.players[g.active_player]

    if step == "untap":                              # CR 502 — no priority
        p.tap_undo = []
        for iid in p.battlefield:
            g.objects[iid].tapped = False
            g.objects[iid].entered_this_turn = False
        _advance_step(g)
        return

    if step == "draw":                               # CR 504
        first_turn = g.turn_number == 1 and g.active_player == g.first_player
        if not first_turn and _draw_or_lose(g, g.active_player):
            return

    if step == "begin_combat":                       # CR 507 — fresh combat each turn
        g.combat = Combat()

    if step == "declare_attackers":                  # CR 508 — turn-based, then priority
        _declare_attackers_step(g)
        return

    if step == "declare_blockers":                   # CR 509
        _declare_blockers_step(g)
        return

    if step == "combat_damage":                      # CR 510
        _combat_damage_step(g)
        return

    if step == "cleanup":                            # CR 514 — normally no priority
        _cleanup(g)
        return

    _give_priority(g, g.active_player)                # active player gets priority first


# ── Combat (CR 507–510): declarations are turn-based actions, not priority ──

def _is_creature(o) -> bool:
    return "creature" in (o.type_line or "").lower()


_CANT_ATTACK_RE = re.compile(r"can'?t attack unless defending player controls an? (\w+)", re.I)


def _attack_restricted(g: GameState, player: str, iid: str) -> bool:
    """Whether a 'can't attack unless defending player controls a [type]' clause
    (Dandân) currently bars this creature, reading the effective type words."""
    o = g.objects.get(iid)
    if not o:
        return True
    m = _CANT_ATTACK_RE.search(o.oracle_text or "")
    return bool(m) and not controls_basic_type(g, _OTHER[player], m.group(1))


def _eligible_attackers(g: GameState, player: str) -> list:
    # a creature can attack if it's untapped, didn't enter this turn, and isn't
    # barred by a "can't attack unless..." clause
    return [iid for iid in g.players[player].battlefield
            if _is_creature(g.objects[iid])
            and not g.objects[iid].tapped and not g.objects[iid].entered_this_turn
            and not _attack_restricted(g, player, iid)]


def _eligible_blockers(g: GameState, player: str) -> list:
    # a tapped creature can't block; summoning sickness doesn't stop blocking
    return [iid for iid in g.players[player].battlefield
            if _is_creature(g.objects[iid]) and not g.objects[iid].tapped]


def _set_attackers(g: GameState, player: str, ids: list) -> None:
    defender = _OTHER[player]
    for iid in ids:
        g.objects[iid].tapped = True                 # attacking taps (CR 508.1f)
        g.combat.attackers[iid] = {"target": defender, "blockers": []}
    if ids:
        g.log.append(f"{g.players[player].name} attacks with "
                     + ", ".join(g.objects[i].name for i in ids) + ".")


def _declare_attackers_step(g: GameState) -> None:
    active = g.active_player
    eligible = _eligible_attackers(g, active)
    if not eligible:
        _give_priority(g, active)
        return
    if g.players[active].is_ai:                       # heuristic: profitable attacks only;
        chosen = (_ai_mod(g, active).choose_attackers(g, active, eligible)
                  if _heuristic(g, active) else eligible)   # passive: everything it can
        _set_attackers(g, active, chosen)
        _give_priority(g, active)
        return
    # The human is always stopped to declare attackers (overrides yields/stops).
    g.pending = PendingDecision(type="declare_attackers", player=active,
                                context={"eligible": eligible})


def declare_attackers(g: GameState, player: str, attacker_ids: list) -> bool:
    if not g.pending or g.pending.type != "declare_attackers" or g.pending.player != player:
        return False
    eligible = set(g.pending.context.get("eligible", []))
    chosen = [iid for iid in (attacker_ids or []) if iid in eligible]
    g.pending = None
    if chosen:
        _set_attackers(g, player, chosen)
    g.passed = {"p1": False, "p2": False}
    _give_priority(g, player)                          # active player gets priority (CR 508.5)
    return True


def _declare_blockers_step(g: GameState) -> None:
    if not g.combat.attackers:                         # CR 508.8 — no attackers, skip
        _advance_step(g)
        return
    defender = _OTHER[g.active_player]
    eligible = _eligible_blockers(g, defender)
    yielded = g.players[defender].yield_mode == "unconditional" or bool(g.players[defender].yield_here)
    if eligible and _heuristic(g, defender):           # the heuristic AI blocks when profitable
        _apply_blocks(g, defender, eligible, _ai_mod(g, defender).choose_blocks(g, defender, eligible))
        return
    if not eligible or g.players[defender].is_ai or yielded:
        _give_priority(g, g.active_player)            # nothing to block / passive AI / yielded past it
        return
    # The defender is prompted to block unless they have yielded (6 / phase bar).
    g.pending = PendingDecision(type="declare_blockers", player=defender,
                                context={"attackers": list(g.combat.attackers.keys()),
                                         "eligible": eligible})


def _apply_blocks(g: GameState, player: str, eligible: list, blocks: dict) -> None:
    """Record the defender's block assignments (one blocker blocks one attacker)
    and hand priority back to the active player (CR 509.6)."""
    eligible = set(eligible)
    used = set()
    for aid, bids in (blocks or {}).items():
        if aid not in g.combat.attackers:
            continue
        for bid in (bids or []):
            if bid in eligible and bid not in used:    # a creature can block only one attacker
                g.combat.attackers[aid]["blockers"].append(bid)
                used.add(bid)
    blocked = sum(1 for a in g.combat.attackers.values() if a["blockers"])
    if blocked:
        g.log.append(f"{g.players[player].name} blocks {blocked} attacker"
                     + ("s" if blocked != 1 else "") + ".")
    g.passed = {"p1": False, "p2": False}
    _give_priority(g, g.active_player)                 # active player gets priority (CR 509.6)


def declare_blockers(g: GameState, player: str, blocks: dict) -> bool:
    if not g.pending or g.pending.type != "declare_blockers" or g.pending.player != player:
        return False
    eligible = g.pending.context.get("eligible", [])
    g.pending = None
    _apply_blocks(g, player, eligible, blocks)
    return True


def _combat_damage_step(g: GameState) -> None:
    if not g.combat.attackers:
        _advance_step(g)
        return
    _resolve_combat_damage(g)
    _give_priority(g, g.active_player)                 # CR 510.4


def _resolve_combat_damage(g: GameState) -> None:
    # All combat damage is dealt simultaneously (CR 510.1-510.2).
    for aid, info in g.combat.attackers.items():
        atk = g.objects.get(aid)
        if not atk:
            continue
        blockers = [b for b in info.get("blockers", []) if b in g.objects]
        if blockers:
            for i, bid in enumerate(blockers):         # 1 damage to each blocker up to power
                if i < atk.power:
                    g.objects[bid].damage_marked += 1
            atk.damage_marked += sum(g.objects[b].power for b in blockers)  # blockers hit back
        else:
            target = info.get("target")                # unblocked -> hit the player
            g.players[target].life -= atk.power
            g.log.append(f"{atk.name} deals {atk.power} damage to {g.players[target].name}.")
    _check_sba(g)                                      # lethal damage -> creatures die
    g.combat = Combat()                                # combat over -> clear the markers


def _advance_step(g: GameState) -> None:
    # A step/phase is ending: empty all mana pools (CR 500.4).
    for pl in g.players.values():
        pl.mana_pool = {}
    idx = _STEPS.index(g.current_step)
    if idx + 1 < len(_STEPS):
        _enter_step(g, _STEPS[idx + 1])
    else:
        _begin_turn(g, _OTHER[g.active_player])       # after cleanup -> next turn


_MAX_AUTO_TURNS = 6     # safety: never auto-pass more than this many turns in a row


def _opponent_object_on_stack(g: GameState, player: str) -> bool:
    opp = _OTHER[player]
    return any(so.controller == opp for so in g.stack)


def _context(g: GameState, player: str) -> str:
    """Whose turn it is, from `player`'s perspective."""
    return "mine" if g.active_player == player else "theirs"


def _has_stop(g: GameState, player: str) -> bool:
    return g.current_step in g.players[player].stops.get(_context(g, player), [])


def _human_should_stop(g: GameState, player: str) -> bool:
    """Decide whether the human takes priority here, or auto-passes. By default
    you auto-pass except at a stop or when there is something to respond to;
    a yield-until-here passes (ignoring stops) until its target, then stops; the
    unconditional yield (key 6) ignores stops to the end of the turn."""
    yh = g.players[player].yield_here
    if yh:
        if _context(g, player) == yh.get("context") and g.current_step == yh.get("phase"):
            g.players[player].yield_here = {}        # reached -> clear and stop
            return True
        return False                                 # not yet -> pass (ignoring stops)
    if g.players[player].yield_mode == "unconditional":
        return False
    if g.turns_since_stop >= _MAX_AUTO_TURNS:        # runaway safety net
        return True
    # Once interactively held in this step (e.g. pulled in to respond to an
    # opponent's spell in a phase with no stop), the step behaves as if it had a
    # stop: the player passes out of it manually instead of being auto-passed
    # the moment the stack clears. Explicit yields above still skip through.
    if g.players[player].held_step == [g.turn_number, g.current_step]:
        return True
    return _opponent_object_on_stack(g, player) or _has_stop(g, player)


def _give_priority(g: GameState, player: str) -> None:
    _check_sba(g)                                     # SBAs checked before priority (CR 117.5)
    if g.result.get("status") != "ongoing":
        return
    _check_state_triggers(g)                          # state triggers queue here (CR 603.8)
    if _flush_triggers(g):                            # queued triggers hit the stack now (CR 603.3);
        return                                        # paused for the holding area — don't grant priority
    g.priority_player = player
    g.pending = PendingDecision(type="priority", player=player)
    if g.players[player].is_ai:
        if _heuristic(g, player):                     # the heuristic AI may act here
            _ai_mod(g, player).take_priority(g, player)
        else:                                         # the passive AI always passes
            pass_priority(g, player)
        return
    if _human_should_stop(g, player):
        g.turns_since_stop = 0
        # Mark the step as interactively held so every later priority grant in
        # it also stops (see _human_should_stop) — self-expires on step change.
        g.players[player].held_step = [g.turn_number, g.current_step]
        return
    pass_priority(g, player)                           # auto-pass


def pass_priority(g: GameState, player: str) -> bool:
    if g.priority_player != player:
        return False
    g.passed[player] = True
    other = _OTHER[player]
    if g.passed.get(other):                           # all players passed in succession
        if g.stack:
            _resolve_top(g)                           # CR 608 / 405.5 — may pause for a decision
        else:
            g.priority_player = None
            _advance_step(g)                          # empty stack -> step/phase ends
    else:
        _give_priority(g, other)
    return True


# ── Spell effects + resolution (CR 608) ───────────────────────────────────
#
# A card's effect is a function effect(g, controller, instance_id) registered by
# card name. It either completes immediately (returns falsy) or sets a pending
# decision and returns True to *pause* resolution — the spell is finished later
# when that decision is resolved. New cards register an effect and reuse the
# library primitives / decision helpers below.

_EFFECTS = {}


def register_effect(name):
    def deco(fn):
        _EFFECTS[name] = fn
        return fn
    return deco


# Spells that need a target chosen as they're cast (CR 601.2c). `legal(g, caster)`
# returns the instance ids that may be chosen; `count` is how many are required.
# The chosen targets are stored on the spell's StackObject and read by its effect.
_TARGETS = {}   # card name -> {count, legal, prompt}


def register_targets(name, *, count, legal, prompt):
    _TARGETS[name] = {"count": count, "legal": legal, "prompt": prompt}


def _all_targets_illegal(g: GameState, obj) -> bool:
    """CR 608.2b: are every one of a spell's chosen targets now illegal? Legality
    is re-checked against the card's current legal-target set at resolution."""
    name = g.objects[obj.source_instance_id].name if obj.source_instance_id in g.objects else None
    spec = _TARGETS.get(name)
    if not spec or not obj.targets:
        return False
    legal = set(spec["legal"](g, obj.controller))
    return all(t.get("id") not in legal for t in obj.targets)


# ── Triggered & activated abilities (use the stack; CR 113 / 603.3) ────────
#
# An ability that uses the stack — a triggered ability, or a non-mana activated
# ability — is a StackObject with kind "triggered"/"activated" whose
# source_instance_id points at the permanent it came from, so the UI can render
# it as a square ability "card": the source's name + art, the ability type, and
# the effect text. Mana abilities and special actions (playing a land) never
# come here — they don't use the stack. Triggers fire when their event happens
# but are only PUT on the stack the next time a player would receive priority
# (CR 603.3), which is why you can't respond to the trigger condition itself.

_TRIGGERS = {}   # card name -> [ {event, type, text, effect} ]


def register_trigger(name, event, text, *, type_label="Triggered ability", condition=None, target=None):
    """`condition(g, source_iid) -> bool` gates whether the trigger fires (e.g.
    Mystic Sanctuary's "when this land enters untapped"). `target` declares that
    the ability targets — its target is chosen as the ability is put on the stack
    (CR 603.3d), e.g. {"zone":"graveyard","types":(...),"may":True}."""
    def deco(fn):
        _TRIGGERS.setdefault(name, []).append(
            {"event": event, "type": type_label, "text": text, "effect": fn,
             "condition": condition, "target": target})
        return fn
    return deco


def queue_etb_triggers(g: GameState, iid: str) -> None:
    """Queue the 'enters the battlefield' triggers of the permanent `iid`. They
    go on the stack later, in _flush_triggers."""
    inst = g.objects.get(iid)
    if not inst:
        return
    for idx, spec in enumerate(_TRIGGERS.get(inst.name, [])):
        if spec["event"] != "etb":
            continue
        if spec.get("condition") and not spec["condition"](g, iid):
            continue                                      # intervening-if did not hold
        g.pending_triggers.append(StackObject(
            stack_id=uuid.uuid4().hex, kind="triggered",
            source_instance_id=iid, controller=inst.controller,
            description=spec["text"], chosen={"ability": idx},
        ))
        g.log.append(f"{inst.name}'s ability triggers.")


def _trigger_target_spec(g: GameState, t: StackObject):
    """The `target` spec of a queued trigger (or None) — read from its source's
    registered trigger."""
    src = g.objects.get(t.source_instance_id)
    if not src:
        return None
    specs = _TRIGGERS.get(src.name, [])
    idx = (t.chosen or {}).get("ability", 0)
    spec = specs[idx] if 0 <= idx < len(specs) else None
    return spec.get("target") if spec else None


def _trigger_legal_targets(g: GameState, tspec: dict) -> list:
    """The instance ids a trigger with `tspec` may target right now."""
    if tspec and tspec.get("zone") == "graveyard":
        return [iid for iid in g.graveyard
                if any(ty in (g.objects[iid].type_line or "").lower() for ty in tspec["types"])]
    return []


def _trigger_needs_choice(g: GameState, t: StackObject) -> bool:
    """Whether putting `t` on the stack needs a target choice from its controller."""
    tspec = _trigger_target_spec(g, t)
    return bool(tspec) and bool(_trigger_legal_targets(g, tspec))


def _place_on_stack(g: GameState, t: StackObject) -> None:
    if t in g.pending_triggers:
        g.pending_triggers.remove(t)
    g.stack.append(t)
    g.passed = {"p1": False, "p2": False}


def _auto_place_trigger(g: GameState, t: StackObject) -> None:
    """Put an AI's trigger on the stack, auto-choosing a target if it needs one."""
    tspec = _trigger_target_spec(g, t)
    if tspec:
        legal = _trigger_legal_targets(g, tspec)
        if legal:
            chosen = (_ai_mod(g, t.controller).choose_trigger_target(g, t, legal)
                      if _heuristic(g, t.controller) else legal[0])
            if chosen is not None:
                t.targets = [{"type": "object", "id": chosen}]
    _place_on_stack(g, t)


def _flush_triggers(g: GameState) -> bool:
    """Put queued triggers on the stack in APNAP order (CR 603.3b): the active
    player's first (so they resolve last). A player with more than one trigger, or
    a trigger that needs a target, gets the MTGO-style holding area (CR 603.3d) —
    they click each trigger to target it / put it on the stack. Returns True if it
    paused for such a decision (priority must not be granted yet)."""
    return _place_triggers(g)


def _place_triggers(g: GameState) -> bool:
    """Advance trigger placement; returns True if paused for a holding-area
    decision, False once all queued triggers are on the stack."""
    for who in (g.active_player, _OTHER[g.active_player]):
        mine = [t for t in g.pending_triggers if t.controller == who]
        if not mine:
            continue
        if g.players[who].is_ai:                           # AI: auto-order + auto-target
            for t in list(mine):
                _auto_place_trigger(g, t)
            return _place_triggers(g)
        if len(mine) == 1 and not _trigger_needs_choice(g, mine[0]):
            _place_on_stack(g, mine[0])                    # nothing to choose -> straight on
            return _place_triggers(g)
        _start_order_triggers(g, who, mine)                # human orders / targets them
        return True
    return False                                           # everything placed


def _start_order_triggers(g: GameState, player: str, triggers: list) -> None:
    g.pending = PendingDecision(type="order_triggers", player=player, context={
        "triggers": [{
            "stack_id": t.stack_id,
            "source": t.source_instance_id,
            "name": g.objects[t.source_instance_id].name if t.source_instance_id in g.objects else "",
            "card_key": g.objects[t.source_instance_id].card_key if t.source_instance_id in g.objects else "",
            "text": t.description,
            "needs_target": _trigger_needs_choice(g, t),
        } for t in triggers],
        "prompt": "Put your triggered abilities on the stack (click each in the order you choose).",
    })


def _continue_placement(g: GameState) -> None:
    """Resume placement after a trigger was put on the stack (and any target
    chosen); grant priority once they're all placed."""
    if _place_triggers(g):
        _resolve_ai_pending(g)
        return
    _give_priority(g, g.active_player)


def place_trigger(g: GameState, player: str, stack_id) -> bool:
    """The controller clicks a trigger in the holding area to put it on the stack;
    if it needs a target, that's chosen first (CR 603.3d)."""
    if not g.pending or g.pending.type != "order_triggers" or g.pending.player != player:
        return False
    t = next((x for x in g.pending_triggers if x.stack_id == stack_id and x.controller == player), None)
    if not t:
        return False
    tspec = _trigger_target_spec(g, t)
    if tspec and tspec.get("zone") == "graveyard" and _trigger_legal_targets(g, tspec):
        g.pending = None
        # Choosing the target is mandatory once a legal target exists (CR 601.2c /
        # 603.3d) — the ability's "may" is exercised at resolution, not here.
        return start_graveyard_choice(g, player, t.source_instance_id, types=tspec["types"],
                                      may=False, then="place_trigger", trig_stack_id=stack_id)
    g.pending = None
    _place_on_stack(g, t)
    _continue_placement(g)
    return True


def _resolve_ability(g: GameState, obj: StackObject) -> None:
    """Resolve a triggered/activated ability: run its effect (which may pause for
    a decision); the ability then simply ceases to exist (no graveyard)."""
    src = g.objects.get(obj.source_instance_id)
    if (obj.chosen or {}).get("state_trigger") == "sacrifice":
        # No intervening 'if' (CR 603.4): sacrifice regardless of the current count.
        pl = g.players.get(obj.controller)
        if src and pl and obj.source_instance_id in pl.battlefield:
            _remove_from_battlefield(g, pl, obj.source_instance_id, src, f"{src.name} is sacrificed.")
        _after_resolve(g)
        return
    if obj.kind == "activated":
        effect = _ACTIVATED_EFFECTS.get((obj.chosen or {}).get("effect"))
        if effect and effect(g, obj.controller, obj.source_instance_id):
            _resolve_ai_pending(g)                        # a paused decision may belong to the AI
            return
        _after_resolve(g)
        return
    specs = _TRIGGERS.get(src.name, []) if src else []
    idx = (obj.chosen or {}).get("ability", 0)
    spec = specs[idx] if 0 <= idx < len(specs) else None
    if spec and spec.get("effect") and spec["effect"](g, obj.controller, obj.source_instance_id, obj):
        _resolve_ai_pending(g)                            # e.g. the AI's Halimar Depths reorder
        return                                            # paused for a decision — finish later
    _after_resolve(g)


# -- activated abilities (CR 602): pay the cost, then add mana immediately (a
#    mana ability) or put the non-mana ability on the stack -------------------

_ACTIVATED_EFFECTS = {}   # effect key -> fn(g, controller, source_iid) -> paused?


def _act_draw_1(g: GameState, controller: str, source: str) -> bool:
    _draw_or_lose(g, controller)                          # empty library -> the activator loses
    return False


_ACTIVATED_EFFECTS["draw_1"] = _act_draw_1


def _pay_source(g: GameState, player: str, iid: str, ab: dict) -> None:
    """Pay the part of an ability's cost that comes from the source itself: tap
    it, and sacrifice it if required."""
    obj = g.objects[iid]
    if ab["tap"]:
        obj.tapped = True
    if ab["sac"] and iid in g.players[player].battlefield:
        g.players[player].battlefield.remove(iid)
        g.graveyard.append(iid)
        _mark_seen(g, iid)


def activate_ability(g: GameState, player: str, iid: str, index: int) -> bool:
    """Activate ability `index` of permanent `iid`. A mana cost is paid from the
    pool or by entering payment mode (tap lands) like casting a spell. Mana
    abilities may also be activated *during* a payment (CR 605.3a) — they pay the
    cost being asked for instead of floating."""
    pend = g.pending
    in_pay = bool(pend and pend.type == "pay" and pend.player == player)
    if not in_pay and (g.priority_player != player or (pend and pend.type != "priority")):
        return False
    p = g.players[player]
    if iid not in p.battlefield:
        return False
    abils = PERMANENT_ABILITIES.get(g.objects[iid].name, [])
    if not (0 <= index < len(abils)):
        return False
    ab = abils[index]
    if ab["tap"] and g.objects[iid].tapped:
        return False
    if in_pay:                                            # a mana ability pays the pending cost
        if not ab["adds"] or iid == pend.context.get("source"):
            return False                                  # only mana abilities; not the cost's own source
        ctx, pool = pend.context, p.mana_pool
        _pay_source(g, player, iid, ab)
        ctx["ops"].append({"op": "source", "kind": "sac" if ab["sac"] else "tap",
                           "iid": iid, "sym": "U", "n": ab["adds"]})
        for _ in range(ab["adds"]):                       # the ability's mana ({U} here) pays the cost
            pool["U"] = pool.get("U", 0) + 1
            paid = _alloc_one(ctx, pool, "U")             # ...or floats if it can't
            if paid is not None:
                ctx["ops"].append({"op": "alloc", "sym": "U", "paid": paid})
        if _payment_done(ctx):
            _complete_payment(g, player, ctx)
        return True
    colored, generic = _parse_cost(ab["cost"])        # coloured pips preserved ({1}{U} != {2})
    ctx = _new_payment({"kind": "activate", "source": iid, "index": index,
                        "name": g.objects[iid].name}, p.mana_pool, colored, generic)
    if _payment_done(ctx):
        _resolve_activation(g, player, iid, index)
        return True
    g.pending = PendingDecision(type="pay", player=player, context=ctx)
    return True


def _complete_payment(g: GameState, player: str, ctx: dict) -> None:
    """A cost has been fully paid (allocated mana is already spent): cast/activate.
    Any unspent mana stays floating in the pool."""
    g.pending = None
    if ctx.get("kind") == "activate":
        _resolve_activation(g, player, ctx["source"], ctx["index"])
    elif ctx.get("kind") == "cycle":
        _resolve_cycle(g, player, ctx["source"])
    else:
        _finish_cast(g, player, ctx["instance_id"], ctx.get("hold", False), ctx.get("targets"))


def _resolve_activation(g: GameState, player: str, iid: str, index: int) -> None:
    """Cost paid: tap/sacrifice the source, then add mana (off the stack) or put
    the non-mana ability on the stack."""
    obj = g.objects[iid]
    ab = PERMANENT_ABILITIES[obj.name][index]
    _pay_source(g, player, iid, ab)
    if ab["adds"]:                                        # mana ability: immediate, no stack
        g.players[player].mana_pool["U"] = g.players[player].mana_pool.get("U", 0) + ab["adds"]
        g.players[player].tap_undo.append(              # undoable with z, like tapping a land
            {"kind": "sac" if ab["sac"] else "tap", "iid": iid, "mana": ab["adds"]})
        return                                            # keep priority (silent, like tapping a land)
    g.stack.append(StackObject(
        stack_id=uuid.uuid4().hex, kind="activated", source_instance_id=iid, controller=player,
        description=ab["text"], chosen={"ability": index, "effect": ab["effect"]}))
    g.log.append(f"{g.players[player].name} activates {obj.name}.")
    g.passed = {"p1": False, "p2": False}
    _give_priority(g, player)
    if g.priority_player == player:                       # default: pass after activating
        pass_priority(g, player)


def cycle(g: GameState, player: str, iid: str) -> bool:
    """Activate a card's cycling ability from hand (CR 702.29): pay its mana cost
    (from the pool or by entering payment mode), discard it, and put 'draw a card'
    on the stack. Usable any time the player has priority (instant speed)."""
    pend = g.pending
    if pend and pend.type == "pay":                      # cycling isn't a mana ability mid-payment
        return False
    if g.priority_player != player or (pend and pend.type != "priority"):
        return False
    if iid not in g.players[player].hand or g.objects[iid].name not in CYCLING:
        return False
    cost = CYCLING[g.objects[iid].name]
    pool = g.players[player].mana_pool
    # Cycling cost is blue ({U}), not generic — a Swamped land can't pay it.
    ctx = _new_payment({"kind": "cycle", "source": iid, "name": g.objects[iid].name},
                       pool, {"U": cost}, 0)
    if _payment_done(ctx):
        _resolve_cycle(g, player, iid)
        return True
    g.pending = PendingDecision(type="pay", player=player, context=ctx)
    return True


def _resolve_cycle(g: GameState, player: str, iid: str) -> None:
    """Cost paid: discard the card (the rest of the cost), then put 'draw a card'
    on the stack."""
    obj = g.objects[iid]
    if iid in g.players[player].hand:                    # discard is part of the cost
        g.players[player].hand.remove(iid)
        g.graveyard.append(iid)
        _mark_seen(g, iid)
    g.stack.append(StackObject(
        stack_id=uuid.uuid4().hex, kind="activated", source_instance_id=iid, controller=player,
        description="Draw a card.", chosen={"effect": "draw_1"}))
    g.log.append(f"{g.players[player].name} cycles {obj.name}.")
    g.passed = {"p1": False, "p2": False}
    _give_priority(g, player)                             # the ability waits on the stack like any other


# -- library primitives -----------------------------------------------------

def look_top(g: GameState, n: int) -> list:
    """The instance ids of the top n library cards (in order)."""
    return [slot.instance_id for slot in g.library[:n]]


def reorder_top(g: GameState, order: list) -> bool:
    """Reorder the top len(order) library slots to match `order` (instance ids)."""
    top = g.library[:len(order)]
    by_id = {s.instance_id: s for s in top}
    if set(order) != set(by_id):
        return False
    g.library[:len(order)] = [by_id[iid] for iid in order]
    return True


# -- resolution -------------------------------------------------------------

def _resolve_top(g: GameState) -> None:
    """Resolve the top stack object: a permanent enters the battlefield; a spell
    runs its effect (which may pause for a decision) then heads to the graveyard."""
    obj = g.stack.pop()
    if obj.kind in ("triggered", "activated"):        # an ability resolves and ceases to exist
        _resolve_ability(g, obj)
        return
    iid = obj.source_instance_id
    inst = g.objects.get(iid) if iid else None
    if not inst:
        _after_resolve(g)
        return
    if _is_permanent(inst.type_line):
        inst.entered_this_turn = True
        inst.tapped = enters_tapped(g, obj.controller, inst)
        g.players[obj.controller].battlefield.append(iid)
        _mark_seen(g, iid)
        queue_etb_triggers(g, iid)                    # ETB triggers (creatures, etc.)
        _after_resolve(g)
        return
    # CR 608.2b: a spell whose targets are all now illegal is removed from the
    # stack without resolving — none of its other text happens (e.g. its draw).
    if _all_targets_illegal(g, obj):
        g.graveyard.append(iid)
        _mark_seen(g, iid)
        g.log.append(f"{inst.name} is removed from the stack (no targets).")
        _after_resolve(g)
        return
    effect = _EFFECTS.get(inst.name)
    if effect and effect(g, obj.controller, iid, obj):
        _resolve_ai_pending(g)                         # a paused decision may belong to the AI
        return                                         # paused for a decision — finish later
    _spell_to_graveyard(g, iid)


def _spell_to_graveyard(g: GameState, iid: str) -> None:
    g.graveyard.append(iid)
    _mark_seen(g, iid)
    _after_resolve(g)


def _after_resolve(g: GameState) -> None:
    g.passed = {"p1": False, "p2": False}
    _give_priority(g, g.active_player)                  # active player gets priority again


# -- the reorder-top decision (boutique deck UI) ----------------------------

def _finish_resolution(g: GameState, ctx: dict) -> None:
    """Finish a paused resolution: a spell heads to the graveyard, an ability
    simply ends. Either way the active player gets priority again."""
    resolving = ctx.get("resolving")
    if ctx.get("resolving_kind") == "ability":
        _after_resolve(g)
    elif resolving in g.objects:
        _spell_to_graveyard(g, resolving)
    else:
        _after_resolve(g)


def start_reorder(g: GameState, player: str, source_iid: str, n: int, *, draw_after: int = 0,
                  allow_shuffle: bool = False, resolving_kind: str = "spell") -> bool:
    """Pause resolution while `player` reorders the top n library cards. The
    paused source is a spell (-> graveyard when done) or an ability (just ends)."""
    top = look_top(g, n)
    for iid in top:
        mark_library_known(g, iid, [player])           # the looker now knows these cards
    g.pending = PendingDecision(type="reorder", player=player, context={
        "cards": [_public_object(g.objects[i]) for i in top],
        "draw_after": draw_after, "allow_shuffle": allow_shuffle, "resolving": source_iid,
        "resolving_kind": resolving_kind,
        "resolving_name": g.objects[source_iid].name, "resolving_by": player,
    })
    g.log.append(f"{g.players[player].name} looks at the top {n}.")
    return True


def complete_reorder(g: GameState, player: str, order: list, *, shuffle: bool = False) -> bool:
    if not g.pending or g.pending.type != "reorder" or g.pending.player != player:
        return False
    ctx = g.pending.context
    name = g.players[player].name
    if shuffle and ctx.get("allow_shuffle"):
        shuffle_library(g)                             # clears everyone's knowledge
        g.log.append(f"{name} shuffles their library.")
    else:
        if not reorder_top(g, order):
            return False
        forget_rearranged(g, g.library[:len(order)], player)   # private order
        g.log.append(f"{name} puts the cards back on top.")
    g.pending = None
    if _draw_or_lose(g, player, ctx.get("draw_after", 0)):
        return True                                        # drew from an empty library -> lost
    _finish_resolution(g, ctx)
    return True


# -- the scry decision (look at the top n; keep some on top, put the rest on
#    the bottom) — same window as reorder, with a "bottom of library" drop area.

def start_scry(g: GameState, player: str, source_iid, n: int,
               *, resolving_kind: str = "spell") -> bool:
    """Pause resolution while `player` scries n (CR 701.x). `source_iid` may be
    None (e.g. a debug scry with no spell/ability behind it)."""
    top = look_top(g, n)
    for iid in top:
        mark_library_known(g, iid, [player])           # the scryer sees all n
    name = g.objects[source_iid].name if source_iid in g.objects else "Scry"
    g.pending = PendingDecision(type="scry", player=player, context={
        "cards": [_public_object(g.objects[i]) for i in top], "count": len(top),
        "resolving": source_iid, "resolving_kind": resolving_kind,
        "resolving_name": name, "resolving_by": player,
    })
    return True                                        # logged as one line when the scry completes


def debug_scry(g: GameState, player: str, n: int) -> bool:
    """Admin/debug: start a scry of `n` with no source (for testing the UI)."""
    if not isinstance(n, int) or n <= 0:
        return False
    if not (g.pending and g.pending.type == "priority" and g.pending.player == player):
        return False
    return start_scry(g, player, None, n, resolving_kind="ability")


def complete_scry(g: GameState, player: str, top_order: list, bottom_order: list) -> bool:
    if not g.pending or g.pending.type != "scry" or g.pending.player != player:
        return False
    ctx = g.pending.context
    n = ctx.get("count", 0)
    scried = {c["instance_id"] for c in ctx["cards"]}
    if set(top_order) & set(bottom_order) or set(top_order) | set(bottom_order) != scried:
        return False                                   # must partition exactly the scried cards
    by_id = {s.instance_id: s for s in g.library[:n]}
    rest = g.library[n:]
    g.library = [by_id[i] for i in top_order] + rest + [by_id[i] for i in bottom_order]
    forget_rearranged(g, by_id.values(), player)       # the scryer's keep/bottom is private
    g.pending = None
    parts = ([f"{len(top_order)} to the top"] if top_order else []) \
        + ([f"{len(bottom_order)} to the bottom"] if bottom_order else [])
    g.log.append(f"{g.players[player].name} scries {' and '.join(parts)}.")
    _finish_resolution(g, ctx)
    return True


# -- the choose-a-card-in-the-graveyard decision (e.g. Mystic Sanctuary) -----

def start_graveyard_choice(g: GameState, player: str, source_iid: str, *, types,
                           may: bool = True, resolving_kind: str = "spell",
                           then: str = "resolve", trig_stack_id=None,
                           restrict_to=None, prompt: str | None = None) -> bool:
    """Pause to choose a graveyard card whose type matches one of `types`. No
    eligible card (and a "may") means nothing happens. `then="place_trigger"`
    makes the choice the trigger's target as it goes on the stack (CR 603.3d),
    rather than the resolving effect. `restrict_to` limits eligibility to a single
    id (the resolution "may" for an ability that already chose its target)."""
    eligible = [iid for iid in g.graveyard
                if any(t in (g.objects[iid].type_line or "").lower() for t in types)]
    if restrict_to is not None:
        eligible = [iid for iid in eligible if iid == restrict_to]
    if not eligible:
        return False
    label = " or ".join(types)
    g.pending = PendingDecision(type="choose_graveyard", player=player, context={
        "eligible": eligible, "may": may, "resolving": source_iid, "resolving_kind": resolving_kind,
        "resolving_name": g.objects[source_iid].name, "resolving_by": player,
        "then": then, "trig_stack_id": trig_stack_id,
        "prompt": prompt or f"Choose a target {label} card in the graveyard.",
    })
    return True


def complete_graveyard_choice(g: GameState, player: str, card_iid) -> bool:
    if not g.pending or g.pending.type != "choose_graveyard" or g.pending.player != player:
        return False
    ctx = g.pending.context
    if card_iid is not None and card_iid not in ctx["eligible"]:
        return False
    if card_iid is None and not ctx.get("may", True):
        return False                                  # declining a mandatory target choice
    if ctx.get("then") == "place_trigger":            # targeting a trigger as it's put on the stack
        t = next((x for x in g.pending_triggers if x.stack_id == ctx.get("trig_stack_id")), None)
        if t is not None:
            if card_iid is not None:
                t.targets = [{"type": "object", "id": card_iid}]
                g.log.append(f"{g.players[player].name}'s ability targets {g.objects[card_iid].name}.")
            _place_on_stack(g, t)
        g.pending = None
        _continue_placement(g)
        return True
    if card_iid is not None:                          # resolution-time choice (legacy path)
        g.graveyard.remove(card_iid)
        g.library.insert(0, LibrarySlot(instance_id=card_iid, known_by={"p1": True, "p2": True}))
        g.log.append(f"{g.players[player].name} puts {g.objects[card_iid].name} on top of the library.")
    g.pending = None
    _finish_resolution(g, ctx)
    return True


# -- searching the library for a card and putting it on top (e.g. Mystical
#    Tutor) — the graveyard choice run on the library, then a shuffle.

def start_library_search(g: GameState, player: str, source_iid: str, *, types,
                         resolving_kind: str = "spell") -> bool:
    """Pause while `player` looks through the whole library and may put a card
    matching `types` on top. Searching always opens — even with no match the
    player can look at the deck and choose to find nothing (then just shuffle)."""
    eligible = [slot.instance_id for slot in g.library
                if any(t in (g.objects[slot.instance_id].type_line or "").lower() for t in types)]
    for slot in g.library:
        mark_library_known(g, slot.instance_id, [player])   # the searcher sees the whole deck
    # The library/graveyard/exile are one shared, fixed pool of cards, so seeing
    # the whole library lets the searcher deduce the opponent's hand by elimination
    # (everything not in the library or a public zone). Record that knowledge.
    opp = _OTHER[player]
    for iid in g.players[opp].hand:
        o = g.objects.get(iid)
        if o and player not in (o.known_by or []):
            o.known_by.append(player)
    label = " or ".join(types)
    g.pending = PendingDecision(type="search_library", player=player, context={
        "eligible": eligible,
        "cards": [_public_object(g.objects[slot.instance_id]) for slot in g.library],   # whole library
        "resolving": source_iid, "resolving_kind": resolving_kind,
        "resolving_name": g.objects[source_iid].name, "resolving_by": player,
        "prompt": f"Search your library for an {label} card, then shuffle and put it on top.",
    })
    g.log.append(f"{g.players[player].name} searches the library.")
    return True


def complete_library_search(g: GameState, player: str, card_iid) -> bool:
    if not g.pending or g.pending.type != "search_library" or g.pending.player != player:
        return False
    ctx = g.pending.context
    chosen = None
    if card_iid is not None:
        if card_iid not in ctx["eligible"]:
            return False
        chosen = card_iid
        g.library = [s for s in g.library if s.instance_id != chosen]   # take the found card out
    shuffle_library(g)                                 # "shuffle and put that card on top"
    if chosen is not None:                             # "reveal it" -> known to everyone
        g.library.insert(0, LibrarySlot(instance_id=chosen, known_by={"p1": True, "p2": True}))
        g.log.append(f"{g.players[player].name} reveals {g.objects[chosen].name} "
                     "and puts it on top of the library.")
    else:
        g.log.append(f"{g.players[player].name} shuffles (found nothing).")
    g.pending = None
    _finish_resolution(g, ctx)
    return True


# -- targeting a spell as it's cast (CR 601.2c) -----------------------------

def complete_targets(g: GameState, player: str, target_ids, *, cancel: bool = False) -> bool:
    """Lock in the chosen targets for a spell being cast, then pay for / cast it.
    `cancel` aborts the cast and hands priority back (the card stays in hand)."""
    if not g.pending or g.pending.type != "choose_targets" or g.pending.player != player:
        return False
    ctx = g.pending.context
    if cancel:
        g.pending = PendingDecision(type="priority", player=player)
        return True
    legal = set(ctx.get("legal", []))
    chosen = [t for t in (target_ids or []) if t in legal]
    if len(set(chosen)) != ctx.get("count", 1):
        return False                                  # must choose exactly the required number
    g.pending = None
    targets = [{"type": "object", "id": iid} for iid in chosen]
    return _begin_cast(g, player, ctx["instance_id"], ctx.get("hold", False), targets)


# -- the "put a card from your hand onto the battlefield" decision. Optional
#    ("may"), and it can belong to the OPPONENT of the spell's caster (e.g.
#    Metamorphose), so it is the first decision that doesn't go to the caster.

_PUT_TYPES = ("artifact", "creature", "enchantment", "land")


def start_put_from_hand(g: GameState, player: str, source_iid, *, resolving_kind: str = "spell") -> bool:
    """Pause while `player` may put an artifact/creature/enchantment/land card
    from their hand onto the battlefield. No eligible card -> nothing to do."""
    hand = g.players[player].hand
    eligible = [iid for iid in hand
                if any(t in (g.objects[iid].type_line or "").lower() for t in _PUT_TYPES)]
    if not eligible:
        return False
    by = g.objects[source_iid].controller if source_iid in g.objects else player
    g.pending = PendingDecision(type="put_from_hand", player=player, context={
        "eligible": eligible, "resolving": source_iid, "resolving_kind": resolving_kind,
        "resolving_name": g.objects[source_iid].name if source_iid in g.objects else "Metamorphose",
        "resolving_by": by,
        "prompt": "Put an artifact, creature, enchantment, or land card from your hand onto the battlefield.",
    })
    g.log.append(f"{g.players[player].name} may put a card onto the battlefield.")
    return True


def complete_put_from_hand(g: GameState, player: str, card_iid) -> bool:
    if not g.pending or g.pending.type != "put_from_hand" or g.pending.player != player:
        return False
    ctx = g.pending.context
    if card_iid is not None:                          # chose a card -> put it onto the battlefield
        if card_iid not in ctx["eligible"] or card_iid not in g.players[player].hand:
            return False
        g.players[player].hand.remove(card_iid)
        perm = g.objects[card_iid]
        perm.controller = player
        perm.entered_this_turn = True
        perm.tapped = enters_tapped(g, player, perm)
        g.players[player].battlefield.append(card_iid)
        _mark_seen(g, card_iid)
        queue_etb_triggers(g, card_iid)
        g.log.append(f"{g.players[player].name} puts {perm.name} onto the battlefield.")
    else:                                             # declined ("may")
        g.log.append(f"{g.players[player].name} puts nothing onto the battlefield.")
    g.pending = None
    _finish_resolution(g, ctx)
    return True


# -- the put-cards-back decision (choose n cards from hand to put on top of the
#    library, in any order) — the scry window run in reverse: the hand is the
#    pool and the "top of library" zone has n dotted slots (e.g. Brainstorm).

def start_putback(g: GameState, player: str, source_iid, n: int,
                  *, resolving_kind: str = "spell") -> bool:
    """Pause while `player` chooses n cards from hand to put on top of the
    library (leftmost slot = topmost). Only that player sees the hand pool —
    current_view hides a decision's context from the opponent."""
    hand = g.players[player].hand
    n = min(n, len(hand))
    if n <= 0:
        return False
    g.pending = PendingDecision(type="putback", player=player, context={
        "cards": [_public_object(g.objects[i]) for i in hand], "slots": n,
        "resolving": source_iid, "resolving_kind": resolving_kind,
        "resolving_name": g.objects[source_iid].name if source_iid in g.objects else "Put back",
        "resolving_by": player,
    })
    return True


def complete_putback(g: GameState, player: str, order: list) -> bool:
    if not g.pending or g.pending.type != "putback" or g.pending.player != player:
        return False
    ctx = g.pending.context
    n = ctx.get("slots", 0)
    hand = g.players[player].hand
    if len(order) != n or len(set(order)) != n or any(iid not in hand for iid in order):
        return False                                   # must name exactly n cards in hand
    for iid in order:
        hand.remove(iid)
    for iid in reversed(order):                        # insert so order[0] ends up on top
        card = g.objects[iid]
        # known to the controller; and to anyone who already knew this exact card
        knows = {pid: (pid == player) or (pid in (card.known_by or [])) for pid in g.players}
        g.library.insert(0, LibrarySlot(instance_id=iid, known_by=knows))
    g.pending = None
    g.log.append(f"{g.players[player].name} puts {n} cards on top of the library.")
    _finish_resolution(g, ctx)
    return True


# -- Fact or Fiction: reveal the top five, an opponent splits them into two
#    piles, then the caster keeps one pile (hand) and bins the other (graveyard).
#    Two decisions: the split belongs to the opponent (the passive AI auto-splits
#    evenly), the choice belongs to the caster.

def start_fof_split(g: GameState, caster: str, source_iid: str) -> bool:
    """Reveal the top five (set aside, known to everyone) and hand the opponent a
    split decision. No cards to reveal means the spell does nothing."""
    revealed = []
    for _ in range(5):
        if not g.library:
            break
        revealed.append(g.library.pop(0).instance_id)
    if not revealed:
        return False
    for cid in revealed:
        g.objects[cid].known_by = ["p1", "p2"]            # revealed to both players
    opp = _OTHER[caster]
    g.pending = PendingDecision(type="fof_split", player=opp, context={
        "cards": [_public_object(g.objects[i]) for i in revealed], "revealed": revealed,
        "caster": caster, "resolving": source_iid, "resolving_kind": "spell",
        "resolving_name": g.objects[source_iid].name, "resolving_by": caster,
        "pile1": list(revealed), "pile2": [],            # live arrangement (everything starts in pile 1)
    })
    g.log.append(f"{g.players[caster].name} reveals the top {len(revealed)} cards.")
    return True


def update_fof_split(g: GameState, player: str, pile2) -> bool:
    """Record the splitter's in-progress arrangement (cards listed in `pile2`; the
    rest stay in pile 1) without finalizing, so the caster can watch the split
    take shape over the SSE stream. Only the decider may, and only revealed cards
    count."""
    if not g.pending or g.pending.type != "fof_split" or g.pending.player != player:
        return False
    ctx = g.pending.context
    p2 = {iid for iid in (pile2 or []) if iid in ctx.get("revealed", [])}
    ctx["pile2"] = [iid for iid in ctx.get("revealed", []) if iid in p2]
    ctx["pile1"] = [iid for iid in ctx.get("revealed", []) if iid not in p2]
    return True


def complete_fof_split(g: GameState, player: str, pile1, pile2) -> bool:
    """The opponent's split: pile1 and pile2 must partition the revealed cards.
    Hands the caster the pile choice."""
    if not g.pending or g.pending.type != "fof_split" or g.pending.player != player:
        return False
    ctx = g.pending.context
    p1, p2 = list(pile1 or []), list(pile2 or [])
    if set(p1) & set(p2) or set(p1) | set(p2) != set(ctx["revealed"]):
        return False                                      # must partition exactly the revealed cards
    caster = ctx["caster"]
    g.pending = PendingDecision(type="fof_choose", player=caster, context={
        "pile1": [_public_object(g.objects[i]) for i in p1],
        "pile2": [_public_object(g.objects[i]) for i in p2],
        "pile1_ids": p1, "pile2_ids": p2,
        "resolving": ctx["resolving"], "resolving_kind": ctx["resolving_kind"],
        "resolving_name": ctx["resolving_name"], "resolving_by": caster,
    })
    g.log.append(f"{g.players[player].name} splits the cards into piles of {len(p1)} and {len(p2)}.")
    _resolve_ai_pending(g)                                # the pile choice may belong to the AI caster
    return True


def complete_fof_choose(g: GameState, player: str, pile: int) -> bool:
    """The caster keeps the chosen pile (to hand) and bins the other (graveyard)."""
    if not g.pending or g.pending.type != "fof_choose" or g.pending.player != player:
        return False
    ctx = g.pending.context
    if pile not in (1, 2):
        return False
    keep = ctx["pile1_ids"] if pile == 1 else ctx["pile2_ids"]
    binned = ctx["pile2_ids"] if pile == 1 else ctx["pile1_ids"]
    for cid in keep:
        g.players[player].hand.append(cid)                # stays known to both (it was revealed)
    for cid in binned:
        g.graveyard.append(cid); _mark_seen(g, cid)
    g.log.append(f"{g.players[player].name} puts {len(keep)} cards into hand and {len(binned)} into the graveyard.")
    g.pending = None
    _finish_resolution(g, ctx)
    return True


# -- "name a card" (chosen during resolution): the player picks a card name,
#    then a registered follow-up runs (e.g. Predict). The UI offers a filtered,
#    scrollable list of the card names in the deck.

_AFTER_NAME = {}   # key -> fn(g, player, name, ctx) run once a name is chosen


def register_after_name(key):
    def deco(fn):
        _AFTER_NAME[key] = fn
        return fn
    return deco


def start_name_card(g: GameState, player: str, source_iid: str, *, then: str,
                    resolving_kind: str = "spell") -> bool:
    """Pause while `player` names a card. `then` selects the follow-up to run with
    the chosen name."""
    names = sorted({o.name for o in g.objects.values() if o.name})
    g.pending = PendingDecision(type="name_card", player=player, context={
        "names": names, "then": then,
        "resolving": source_iid, "resolving_kind": resolving_kind,
        "resolving_name": g.objects[source_iid].name if source_iid in g.objects else "Name a card",
        "resolving_by": player, "prompt": "Name a card.",
    })
    return True


def complete_name_card(g: GameState, player: str, name) -> bool:
    if not g.pending or g.pending.type != "name_card" or g.pending.player != player:
        return False
    ctx = g.pending.context
    if name not in ctx["names"]:                          # must be a card name from the list
        return False
    g.pending = None
    g.log.append(f"{g.players[player].name} names {name}.")
    fn = _AFTER_NAME.get(ctx["then"])
    if fn:
        fn(g, player, name, ctx)
    else:
        _finish_resolution(g, ctx)
    return True


def _mill(g: GameState, player: str):
    """Put the top card of the library into the (shared) graveyard; return it."""
    if not g.library:
        return None
    iid = g.library.pop(0).instance_id
    g.graveyard.append(iid)
    _mark_seen(g, iid)
    g.log.append(f"{g.players[player].name} mills {g.objects[iid].name}.")
    return iid


# ---------------------------------------------------------------------------
# Text-changing effects: Crystal Spray, Mind Bend, Vision Charm. A two-list
# "turn X to Y" decision rewrites one basic land type word with another on the
# chosen object(s). Vision Charm is modal (mill four / change a land type).
# ---------------------------------------------------------------------------

_AFTER_TEXT_CHANGE = {}   # key -> fn(g, player, frm, to, ctx)


def register_after_text_change(key):
    def deco(fn):
        _AFTER_TEXT_CHANGE[key] = fn
        return fn
    return deco


def start_text_change(g: GameState, player: str, source_iid: str, *, then: str,
                      change_targets, resolving_kind: str = "spell") -> bool:
    """Pause while `player` picks 'turn X to Y' from two basic-land-type lists."""
    g.pending = PendingDecision(type="choose_text_change", player=player, context={
        "from_types": list(BASIC_TYPES), "to_types": list(BASIC_TYPES), "then": then,
        "change_targets": list(change_targets),
        "resolving": source_iid, "resolving_kind": resolving_kind,
        "resolving_name": g.objects[source_iid].name if source_iid in g.objects else "",
        "resolving_by": player, "prompt": "Replace one basic land type with another.",
    })
    return True


def complete_text_change(g: GameState, player: str, frm, to) -> bool:
    if not g.pending or g.pending.type != "choose_text_change" or g.pending.player != player:
        return False
    if frm not in BASIC_TYPES or to not in BASIC_TYPES:
        return False
    ctx = g.pending.context
    g.pending = None
    g.log.append(f"{g.players[player].name} turns {frm} into {to}.")
    fn = _AFTER_TEXT_CHANGE.get(ctx["then"])
    if fn:
        fn(g, player, frm, to, ctx)
    else:
        _finish_resolution(g, ctx)
    return True


# Crystal Spray {2}{U}: change the text of target spell or permanent until end of
# turn, then draw a card. (On a spell that resolves into a permanent the change
# carries over, keeping its end-of-turn duration — see _sweep_text_changes.)
register_targets("Crystal Spray", count=1,
                 legal=lambda g, caster: (
                     [s.source_instance_id for s in g.stack if s.kind == "spell"]
                     + [iid for pl in g.players.values() for iid in pl.battlefield]),
                 prompt="Change the text of target spell or permanent.")


@register_effect("Crystal Spray")
def _effect_crystal_spray(g: GameState, controller: str, iid: str, obj=None) -> bool:
    targets = (obj.targets if obj else None) or []
    tgt = [t["id"] for t in targets if t.get("id") in g.objects]
    return start_text_change(g, controller, iid, then="crystal_spray", change_targets=tgt)


@register_after_text_change("crystal_spray")
def _after_crystal_spray(g: GameState, player: str, frm: str, to: str, ctx: dict) -> None:
    for tgt in ctx.get("change_targets", []):
        o = g.objects.get(tgt)
        if o:
            add_text_change(o, frm, to, eot=True, turn=g.turn_number)
    if _draw_or_lose(g, player):                          # empty library -> the caster loses
        return
    _finish_resolution(g, ctx)


# Mind Bend {U}: change the text of target permanent indefinitely.
register_targets("Mind Bend", count=1,
                 legal=lambda g, caster: [iid for pl in g.players.values() for iid in pl.battlefield],
                 prompt="Change the text of target permanent.")


@register_effect("Mind Bend")
def _effect_mind_bend(g: GameState, controller: str, iid: str, obj=None) -> bool:
    targets = (obj.targets if obj else None) or []
    tgt = [t["id"] for t in targets if t.get("id") in g.objects]
    return start_text_change(g, controller, iid, then="mind_bend", change_targets=tgt)


@register_after_text_change("mind_bend")
def _after_mind_bend(g: GameState, player: str, frm: str, to: str, ctx: dict) -> None:
    for tgt in ctx.get("change_targets", []):
        o = g.objects.get(tgt)
        if o:
            add_text_change(o, frm, to, eot=False, turn=g.turn_number)
    _finish_resolution(g, ctx)


# Vision Charm {U}: modal (mode chosen at cast time) — mill four (shared library),
# or change each land of one type into another type until end of turn.
@register_effect("Vision Charm")
def _effect_vision_charm(g: GameState, controller: str, iid: str, obj=None) -> bool:
    mode = g.objects[iid].chosen.get("mode", "mill")
    if mode == "land":
        return start_text_change(g, controller, iid, then="vision_land", change_targets=[])
    for _ in range(4):                                    # mill four off the shared library
        if not _mill(g, controller):
            break
    return False


@register_after_text_change("vision_land")
def _after_vision_land(g: GameState, player: str, frm: str, to: str, ctx: dict) -> None:
    for pl in g.players.values():                         # each land of the first type
        for tid in list(pl.battlefield):
            o = g.objects.get(tid)
            if o and _is_land(o.type_line):
                add_text_change(o, frm, to, eot=True, turn=g.turn_number)
    _finish_resolution(g, ctx)


# -- worked examples ---------------------------------------------------------
# Ponder (a spell): look at the top 3, reorder or shuffle, draw 1.
@register_effect("Ponder")
def _effect_ponder(g: GameState, controller: str, iid: str, obj=None) -> bool:
    return start_reorder(g, controller, iid, 3, draw_after=1, allow_shuffle=True)


# Brainstorm (a spell): draw three cards, then put two cards from your hand on
# top of your library in any order. The scry window run in reverse — the hand
# is the pool, the "top of library" zone has two slots.
def _forget_hand(g: GameState, owner: str) -> None:
    """Wipe other players' knowledge of `owner`'s hand. Used after a private
    draw-and-put-back (Brainstorm): an opponent can no longer track which cards
    the owner holds, nor which of the new draws were kept versus put back."""
    for iid in g.players[owner].hand:
        o = g.objects.get(iid)
        if o:
            o.known_by = [pid for pid in (o.known_by or []) if pid == owner]


@register_effect("Brainstorm")
def _effect_brainstorm(g: GameState, controller: str, iid: str, obj=None) -> bool:
    if _draw_or_lose(g, controller, 3):                # fewer than three left -> lost
        return True
    _forget_hand(g, controller)                        # the new hand is hidden again
    return start_putback(g, controller, iid, 2)


# Accumulated Knowledge (a spell): draw a card, then one more for each
# Accumulated Knowledge already in the graveyard. The graveyard is shared, so
# that single count covers "all graveyards"; the resolving copy is still on the
# stack (not yet in the graveyard), so it isn't counted.
@register_effect("Accumulated Knowledge")
def _effect_accumulated_knowledge(g: GameState, controller: str, iid: str, obj=None) -> bool:
    extra = sum(1 for cid in g.graveyard if g.objects[cid].name == "Accumulated Knowledge")
    if _draw_or_lose(g, controller, 1 + extra):        # not enough left to draw -> lost
        return True
    return False


@register_effect("Fact or Fiction")
def _effect_fact_or_fiction(g: GameState, controller: str, iid: str, obj=None) -> bool:
    return start_fof_split(g, controller, iid)


# Predict (a spell): name a card, then mill one. If the milled card has that
# name, draw two cards; otherwise draw a card. The milled card stays milled.
@register_effect("Predict")
def _effect_predict(g: GameState, controller: str, iid: str, obj=None) -> bool:
    return start_name_card(g, controller, iid, then="predict")


@register_after_name("predict")
def _after_predict(g: GameState, player: str, name: str, ctx: dict) -> None:
    milled = _mill(g, player)
    hit = milled is not None and g.objects[milled].name == name
    if _draw_or_lose(g, player, 2 if hit else 1):          # hit -> draw two, otherwise one
        return                                             # drew from an empty library -> lost
    _finish_resolution(g, ctx)


# Metamorphose (an instant): target a permanent an opponent controls; put it on
# top of the (shared) library, then THAT opponent may put an artifact/creature/
# enchantment/land card from their hand onto the battlefield. The "may" decision
# belongs to the opponent — the first decision that isn't the caster's.
register_targets("Metamorphose", count=1,
                 legal=lambda g, caster: list(g.players[_OTHER[caster]].battlefield),
                 prompt="Choose a permanent an opponent controls.")


@register_effect("Metamorphose")
def _effect_metamorphose(g: GameState, controller: str, iid: str, obj=None) -> bool:
    targets = (obj.targets if obj else None) or []
    tgt = targets[0]["id"] if targets else None
    perm = g.objects.get(tgt)
    owner = perm.controller if perm else None
    if not perm or owner not in g.players or tgt not in g.players[owner].battlefield:
        return False                                  # target gone -> spell does nothing
    g.players[owner].battlefield.remove(tgt)
    perm.tapped = False
    perm.entered_this_turn = False
    # onto the top of the shared library, known to everyone (it was on the battlefield)
    g.library.insert(0, LibrarySlot(instance_id=tgt, known_by={"p1": True, "p2": True}))
    g.log.append(f"{g.players[controller].name} puts {perm.name} on top of the library.")
    return start_put_from_hand(g, owner, iid)         # the opponent's "may" decision


# Memory Lapse (an instant): counter target spell on the stack — but put the
# countered spell on top of its owner's (shared) library instead of the
# graveyard. The first card that targets a spell on the stack rather than a
# permanent; the targeting UI highlights stack cards the same way.
register_targets("Memory Lapse", count=1,
                 legal=lambda g, caster: [s.source_instance_id for s in g.stack if s.kind == "spell"],
                 prompt="Counter target spell on the stack.")


@register_effect("Memory Lapse")
def _effect_memory_lapse(g: GameState, controller: str, iid: str, obj=None) -> bool:
    targets = (obj.targets if obj else None) or []
    tgt = targets[0]["id"] if targets else None
    so = next((s for s in g.stack if s.kind == "spell" and s.source_instance_id == tgt), None)
    if not so:
        return False                                  # target spell gone -> Memory Lapse fizzles
    g.stack.remove(so)                                # counter it
    # onto the top of the shared library instead of the graveyard, known to all
    g.library.insert(0, LibrarySlot(instance_id=tgt, known_by={"p1": True, "p2": True}))
    g.log.append(f"{g.players[controller].name} counters {g.objects[tgt].name} "
                 "and puts it on top of the library.")
    return False                                      # Memory Lapse itself -> graveyard


# Mystical Tutor (an instant): search the library for an instant or sorcery,
# reveal it, then shuffle and put it on top.
@register_effect("Mystical Tutor")
def _effect_mystical_tutor(g: GameState, controller: str, iid: str, obj=None) -> bool:
    return start_library_search(g, controller, iid, types=("instant", "sorcery"))


# Day's Undoing (a sorcery): each player shuffles hand + graveyard into the
# (shared) library and draws seven; then, on the caster's turn, end the turn
# (CR 720) — so the caster can't use the fresh hand until their next turn.
@register_effect("Day's Undoing")
def _effect_days_undoing(g: GameState, controller: str, iid: str, obj=None) -> bool:
    for pl in g.players.values():                     # hands + the shared graveyard...
        for cid in pl.hand:
            g.library.append(LibrarySlot(instance_id=cid))
        pl.hand = []
    for cid in g.graveyard:
        g.library.append(LibrarySlot(instance_id=cid))
    g.graveyard = []
    shuffle_library(g)                                # ...shuffled into the library (knowledge cleared)
    for pl in g.players.values():                     # then each player draws seven
        for _ in range(7):
            if not g.library:
                break
            slot = g.library.pop(0)
            g.objects[slot.instance_id].known_by = []   # freshly shuffled -> known to nobody
            pl.hand.append(slot.instance_id)
    g.log.append("Each player shuffles hand and graveyard into the library and draws seven.")
    if g.active_player == controller:                 # "If it's your turn, end the turn."
        _end_the_turn(g, iid)
        return True                                   # exiled by end-the-turn, not put in the graveyard
    return False


# Halimar Depths (a triggered ability): on entering, look at the top 3 and
# put them back in any order. Reuses the reorder window; finishes as an ability.
@register_trigger("Halimar Depths", "etb",
                  "Look at the top three cards of your library, then put them back in any order.")
def _trig_halimar_depths(g: GameState, controller: str, source_iid: str, obj=None) -> bool:
    return start_reorder(g, controller, source_iid, 3, resolving_kind="ability")


# Temple of Epiphany (a triggered ability): on entering, scry 1.
@register_trigger("Temple of Epiphany", "etb", "Scry 1.")
def _trig_temple_of_epiphany(g: GameState, controller: str, source_iid: str, obj=None) -> bool:
    return start_scry(g, controller, source_iid, 1, resolving_kind="ability")


# Mystic Sanctuary (a triggered ability): only when it enters UNTAPPED, you may
# put target instant/sorcery from the graveyard on top of your library. The target
# is chosen (mandatorily, if a legal one exists) as the ability is put on the stack
# (CR 603.3d); the "may" — whether to actually move it — is exercised at resolution.
@register_trigger("Mystic Sanctuary", "etb",
                  "You may put target instant or sorcery card from your graveyard on top of your library.",
                  condition=lambda g, iid: not g.objects[iid].tapped,
                  target={"zone": "graveyard", "types": ("instant", "sorcery"), "may": True})
def _trig_mystic_sanctuary(g: GameState, controller: str, source_iid: str, obj=None) -> bool:
    targets = (obj.targets if obj else None) or []
    tgt = targets[0]["id"] if targets else None
    if tgt and tgt in g.graveyard:                        # target still legal -> exercise the "may"
        name = g.objects[tgt].name
        return start_graveyard_choice(
            g, controller, source_iid, types=("instant", "sorcery"), may=True,
            then="resolve", resolving_kind="ability", restrict_to=tgt,
            prompt=f"You may put {name} on top of your library.")
    return False                                          # no target (illegal/none) -> nothing


_MANA_TOKEN = re.compile(r"\{([^}]+)\}")
_COLORS = ("W", "U", "B", "R", "G", "C")


def _parse_cost(cost: str):
    """Parse a mana cost like "{2}{U}" into (colored {sym: n}, generic int)."""
    colored, generic = {}, 0
    for tok in _MANA_TOKEN.findall(cost or ""):
        t = tok.strip().upper()
        if t.isdigit():
            generic += int(t)
        elif t in _COLORS:
            colored[t] = colored.get(t, 0) + 1
        # {X} / hybrid / Phyrexian don't appear in this format
    return colored, generic


def _can_afford(pool: dict, colored: dict, generic: int) -> bool:
    """Whether `pool` (with `extra` untapped lands of any colour) covers a cost."""
    avail = dict(pool)
    for sym, n in colored.items():
        if avail.get(sym, 0) < n:
            return False
        avail[sym] -= n
    return sum(avail.values()) >= generic


# -- colour-aware payment: a cost is (need = coloured pips, generic count); mana
#    is spent one pip at a time, a colour paying its own pip first then generic.

def _alloc_one(ctx: dict, pool: dict, sym: str):
    """Spend one `sym` mana from the pool toward the cost. Returns the slot paid
    (the colour for a matching pip, or 'generic'), or None if it can't pay."""
    if pool.get(sym, 0) <= 0:
        return None
    need = ctx["need"]
    if need.get(sym, 0) > 0:                       # pays its own coloured pip first
        need[sym] -= 1
        if need[sym] <= 0:
            del need[sym]
        paid = sym
    elif ctx["generic"] > 0:                       # otherwise generic
        ctx["generic"] -= 1
        paid = "generic"
    else:
        return None
    _drop_pool(pool, sym, 1)
    return paid


def _auto_allocate(ctx: dict, pool: dict) -> None:
    """Greedily spend pool mana: matching coloured pips first, then generic."""
    for sym in list(ctx["need"]):
        while ctx["need"].get(sym, 0) > 0 and pool.get(sym, 0) > 0:
            paid = _alloc_one(ctx, pool, sym)
            ctx["ops"].append({"op": "alloc", "sym": sym, "paid": paid})
    for sym in list(pool):
        while ctx["generic"] > 0 and pool.get(sym, 0) > 0:
            paid = _alloc_one(ctx, pool, sym)
            ctx["ops"].append({"op": "alloc", "sym": sym, "paid": paid})


def _payment_done(ctx: dict) -> bool:
    return not ctx["need"] and ctx["generic"] <= 0


def _exactly_covers(pool: dict, need: dict, generic: int) -> bool:
    """Whether the floating pool is *exactly* the remaining cost — the same total,
    with enough of each required colour (any surplus colours go to generic). Only
    then is the pool auto-spent; otherwise the player clicks mana to allocate it."""
    if sum(pool.values()) != sum(need.values()) + generic:
        return False
    return all(pool.get(sym, 0) >= n for sym, n in need.items())


def _maybe_autopay(ctx: dict, pool: dict) -> None:
    """Spend the whole pool on the cost, but only when it covers it exactly."""
    if _exactly_covers(pool, ctx["need"], ctx["generic"]):
        _auto_allocate(ctx, pool)


def _new_payment(ctx_extra: dict, pool: dict, colored: dict, generic: int) -> dict:
    """Build a pay context, snapshot the pool for cancel, and auto-spend it only
    if it exactly covers the cost."""
    ctx = {"need": dict(colored), "generic": int(generic), "ops": [],
           "pool_before": dict(pool), **ctx_extra}
    _maybe_autopay(ctx, pool)
    return ctx


def _target_names(g: GameState, targets) -> list:
    """Display names for chosen targets (a permanent/spell by its card name, a
    player by their name)."""
    names = []
    for t in targets or []:
        tid = t.get("id")
        if t.get("type") == "player" and tid in g.players:
            names.append(g.players[tid].name)
        elif tid in g.objects:
            names.append(g.objects[tid].name)
    return names


def _finish_cast(g: GameState, player: str, instance_id: str, hold: bool, targets=None) -> None:
    """Put a paid-for spell on the stack. By default the caster passes priority
    after casting; holding Ctrl (`hold`) keeps priority so several spells can be
    stacked before passing."""
    play_card(g, player, instance_id)                 # spell -> stack ("X casts Spell.")
    inst = g.objects[instance_id]
    mode_text = modal_mode_text(inst.name, inst.chosen.get("mode", ""))
    if mode_text and g.log:                           # modal spell: name the chosen mode in the log
        g.log[-1] = g.log[-1].rstrip(".") + " (" + mode_text.rstrip(".") + ")."
    if targets:
        g.stack[-1].targets = list(targets)           # carry the chosen targets on the spell
        names = _target_names(g, targets)             # ...and name them in the cast log
        if names and g.log:
            g.log[-1] = g.log[-1].rstrip(".") + " targeting " + ", ".join(names) + "."
    g.passed = {"p1": False, "p2": False}
    _give_priority(g, player)                          # caster gets priority (CR 117.3c)
    if not hold and g.priority_player == player:
        pass_priority(g, player)                       # default: pass after casting


def play(g: GameState, player: str, instance_id: str, hold: bool = False, mode=None) -> bool:
    """Play a card from hand. Always goes through priority/timing: the player
    must hold priority (so nothing can be played during the opening/mulligan
    phase), lands and sorcery-speed spells only during their own main phase
    with an empty stack, instants any time. `hold` keeps priority after casting
    a spell (Ctrl), otherwise the caster passes priority by default. `mode` is the
    chosen mode of a modal spell (CR 700.2), chosen as it is cast."""
    if instance_id not in g.players[player].hand:
        return False
    if g.priority_player != player:
        return False
    if g.pending and g.pending.type != "priority":    # e.g. mid-payment
        return False
    obj = g.objects[instance_id]
    modes = MODAL_SPELLS.get(obj.name)
    if modes:                                          # lock in the chosen mode before casting
        keys = {m["key"] for m in modes}
        obj.chosen["mode"] = mode if mode in keys else modes[0]["key"]
    tl = (obj.type_line or "").lower()
    is_land = "land" in tl
    sorcery_speed = is_land or "instant" not in tl    # only instants are any-time
    if sorcery_speed and (g.active_player != player or g.current_step not in _MAIN_STEPS or g.stack):
        return False
    if is_land:
        if g.players[player].land_played_this_turn:
            return False
        play_card(g, player, instance_id)             # land -> battlefield (special action)
        queue_etb_triggers(g, instance_id)            # ETB triggers (the land enters)
        g.players[player].land_played_this_turn = True
        g.passed = {"p1": False, "p2": False}
        _give_priority(g, player)                      # retains priority; flushes the trigger
        return True
    # A spell that needs a target chooses it first (CR 601.2c), before paying.
    spec = _TARGETS.get(obj.name)
    if spec:
        legal = spec["legal"](g, player)
        if len(legal) < spec["count"]:
            return False                              # not enough legal targets to cast
        g.pending = PendingDecision(type="choose_targets", player=player, context={
            "instance_id": instance_id, "name": obj.name, "hold": hold,
            "count": spec["count"], "legal": legal, "chosen": [], "prompt": spec["prompt"],
        })
        return True
    return _begin_cast(g, player, instance_id, hold)


def _begin_cast(g: GameState, player: str, instance_id: str, hold: bool, targets=None) -> bool:
    """Pay a spell's cost and cast it (targets already chosen). Enters payment
    mode if the floating pool can't cover the cost; the targets ride along so the
    cast can finish once the cost is paid."""
    obj = g.objects[instance_id]
    colored, generic = _parse_cost(obj.mana_cost)
    pool = g.players[player].mana_pool
    ctx = _new_payment({"instance_id": instance_id, "name": obj.name,
                        "hold": hold, "targets": targets or []}, pool, colored, generic)
    if _payment_done(ctx):                            # floating pool already covered it
        _finish_cast(g, player, instance_id, hold, targets)
        return True
    g.pending = PendingDecision(type="pay", player=player, context=ctx)
    return True


def _trigger_if_priority(g: GameState, player: str) -> None:
    if g.priority_player == player and g.pending and g.pending.type == "priority":
        pass_priority(g, player)


def set_yield(g: GameState, player: str, mode: str) -> bool:
    """Arm / cancel a "pass rest of turn" yield for the human. Setting a mode
    while you hold priority starts passing immediately. Cancelling (mode "")
    also clears any yield-until-here."""
    if g.players[player].is_ai or mode not in ("", "conditional", "unconditional"):
        return False
    g.players[player].yield_mode = mode
    if mode == "":
        g.players[player].yield_here = {}            # key 5 cancels yield-until-here too
        return True
    _trigger_if_priority(g, player)
    return True


def set_yield_here(g: GameState, player: str, context: str, phase: str) -> bool:
    """Yield (ignoring stops) until `phase` on the chosen turn (`context`:
    "mine"/"theirs"), then stop. Only one at a time."""
    if g.players[player].is_ai or context not in ("mine", "theirs") or phase not in _STEPS:
        return False
    g.players[player].yield_here = {"context": context, "phase": phase}
    _trigger_if_priority(g, player)
    return True


def set_stop(g: GameState, player: str, context: str, phase: str) -> bool:
    """Toggle a manual-priority stop at `phase` on the chosen turn (`context`)."""
    if g.players[player].is_ai or context not in ("mine", "theirs") or phase not in _STEPS:
        return False
    stops = g.players[player].stops.setdefault(context, [])
    if phase in stops:
        stops.remove(phase)
    else:
        stops.append(phase)
    return True


def set_player_stops(g: GameState, player: str, stops: dict) -> bool:
    """Replace a player's manual-priority stops wholesale — used to seed a game
    from a saved preference (profile for members, browser storage for guests).
    Sanitises to known steps in canonical order; ignores AI players."""
    if g.players[player].is_ai or not isinstance(stops, dict):
        return False
    clean = {}
    for context in ("mine", "theirs"):
        wanted = stops.get(context) or []
        clean[context] = [step for step in _STEPS if step in wanted]   # valid steps, deduped, ordered
    g.players[player].stops = clean
    return True


def tap(g: GameState, player: str, instance_id: str) -> bool:
    """Tap one of `player`'s lands. While paying for a spell the mana goes
    straight into the payment; otherwise it floats in the mana pool."""
    pend = g.pending
    if pend and pend.type == "pay" and pend.player == player:
        obj = g.objects.get(instance_id)
        p = g.players[player]
        if (not obj or instance_id not in p.battlefield or not _is_land(obj.type_line)
                or obj.tapped or instance_id == pend.context.get("source")):
            return False                              # the source taps as part of its own cost
        obj.tapped = True
        ctx, pool = pend.context, p.mana_pool
        sym = land_mana_color(obj)
        pool[sym] = pool.get(sym, 0) + 1
        paid = _alloc_one(ctx, pool, sym)             # try to pay into the cost (matching pip, else generic)
        if paid is not None:
            ctx["ops"].append({"op": "tap", "iid": instance_id, "sym": sym, "paid": paid})
        else:                                         # ...float only if it can't pay anything
            ctx["ops"].append({"op": "float", "iid": instance_id, "sym": sym})
        if _payment_done(ctx):                        # fully paid -> cast / activate
            _complete_payment(g, player, ctx)
        return True
    # Otherwise it's a mana ability adding to the pool, which (like every other
    # action) is legal only while this player actually holds priority — CR
    # 605.3a. Without priority and outside a payment, do nothing.
    if not (pend and pend.type == "priority" and pend.player == player):
        return False
    return tap_land(g, player, instance_id)


def allocate_mana(g: GameState, player: str, sym: str) -> bool:
    """Click a floating mana during a payment to spend it on the cost (a matching
    coloured pip first, otherwise generic)."""
    pend = g.pending
    if not pend or pend.type != "pay" or pend.player != player:
        return False
    sym = (sym or "").strip().upper()
    ctx, pool = pend.context, g.players[player].mana_pool
    paid = _alloc_one(ctx, pool, sym)
    if paid is None:
        return False                                  # that mana can't pay anything left
    ctx["ops"].append({"op": "alloc", "sym": sym, "paid": paid})
    if _payment_done(ctx):
        _complete_payment(g, player, ctx)
    return True


def _undo_payment_op(g: GameState, player: str, ctx: dict, op: dict) -> None:
    """Reverse a single payment op (for z): a floated land tap / mana ability, or
    an allocation (a clicked mana spent on the cost)."""
    pool = g.players[player].mana_pool
    if op["op"] == "tap":                             # a land tapped and spent on the cost
        _reverse_source(g, player, {"kind": "tap", "iid": op["iid"]})
        if op["paid"] == "generic":
            ctx["generic"] += 1
        elif op["paid"]:
            ctx["need"][op["paid"]] = ctx["need"].get(op["paid"], 0) + 1
    elif op["op"] == "float":                         # a land tapped, mana floated (couldn't pay)
        _reverse_source(g, player, {"kind": "tap", "iid": op["iid"]})
        _drop_pool(pool, op["sym"], 1)
    elif op["op"] == "source":                        # a mana ability (tap/sac) floated mana
        _reverse_source(g, player, {"kind": op.get("kind", "tap"), "iid": op["iid"]})
        _drop_pool(pool, op.get("sym", "U"), op.get("n", 1))
    else:                                             # alloc: return the mana, restore the slot
        pool[op["sym"]] = pool.get(op["sym"], 0) + 1
        if op["paid"] == "generic":
            ctx["generic"] += 1
        elif op["paid"]:
            ctx["need"][op["paid"]] = ctx["need"].get(op["paid"], 0) + 1


def untap(g: GameState, player: str) -> str | None:
    """z: undo the last mana tap. While paying, untap the last land tapped for
    the payment (and owe that mana again); otherwise undo a floating-mana tap."""
    pend = g.pending
    if pend and pend.type == "pay" and pend.player == player and pend.context.get("ops"):
        ctx = pend.context
        op = ctx["ops"].pop()
        _undo_payment_op(g, player, ctx, op)
        return op.get("iid") if op["op"] == "tap" else None
    return untap_last(g, player)


_CHAT_MAX = 60


def send_chat(g: GameState, player: str, text: str) -> bool:
    """Append a table-talk message from `player` (collapsed whitespace, capped
    length and transcript). Chat is public — both players and spectators see it."""
    text = " ".join((text or "").split())[:300]
    if not text or player not in g.players:
        return False
    g.chat.append({"by": player, "text": text})
    if len(g.chat) > _CHAT_MAX:
        del g.chat[:-_CHAT_MAX]
    return True


def cancel_payment(g: GameState, player: str) -> bool:
    """Abort a spell payment: restore the pool mana spent and untap the lands
    tapped for it; the spell stays in hand."""
    pend = g.pending
    if not pend or pend.type != "pay" or pend.player != player:
        return False
    p = g.players[player]
    ctx = pend.context
    p.mana_pool.clear()                               # restore the pool to before the payment...
    p.mana_pool.update(ctx.get("pool_before", {}))
    for op in ctx.get("ops", []):                     # ...and untap / un-sacrifice every source tapped
        if op["op"] in ("tap", "float", "source"):
            _reverse_source(g, player, {"kind": op.get("kind", "tap"), "iid": op["iid"]})
    g.pending = None
    _give_priority(g, player)
    return True


def end_turn(g: GameState, player: str) -> bool:
    """Convenience: pass priority repeatedly through the rest of your turn."""
    if g.players[player].is_ai:
        return False
    start = g.turn_number
    guard = 0
    while (g.active_player == player and g.priority_player == player and not g.stack
           and g.turn_number == start and g.result.get("status") == "ongoing" and guard < 30):
        pass_priority(g, player)
        guard += 1
    return True


def _cleanup(g: GameState) -> None:
    p = g.players[g.active_player]
    for o in g.objects.values():                      # "until end of turn" text changes wear off
        expire_eot_text_changes(o)
    for pl in g.players.values():                     # CR 514.2 — remove all marked damage
        for iid in pl.battlefield:
            g.objects[iid].damage_marked = 0
    excess = len(p.hand) - p.max_hand_size
    if excess > 0:
        if p.is_ai:
            if _heuristic(g, g.active_player):        # heuristic: discard the worst cards
                ids = _ai_mod(g, g.active_player).choose_discards(g, g.active_player, excess)
            else:                                     # passive: from the end of the hand
                ids = list(p.hand[-excess:])
            for iid in ids:
                if iid in p.hand:
                    p.hand.remove(iid)
                    g.graveyard.append(iid)
            g.log.append(f"{p.name} discards {excess}.")
        else:
            g.pending = PendingDecision(type="discard", player=g.active_player,
                                        context={"count": excess})
            return                                    # wait for the human to discard
    _advance_step(g)                                  # cleanup is last -> next turn


def _end_the_turn(g: GameState, source_iid: str) -> None:
    """CR 720 "end the turn": pending triggers cease to exist, the whole stack
    (including the spell doing this) is exiled — not put in the graveyard —
    combat is cleared, and the game jumps straight to the cleanup step. The rest
    of the turn is skipped: no end step, no further priority, and any
    end-of-turn / delayed triggers never happen."""
    g.pending_triggers = []                           # pending triggers vanish (never hit the stack)
    for so in list(g.stack):                          # exile everything on the stack...
        sid = so.source_instance_id
        if sid and sid in g.objects and sid not in g.exile:
            g.exile.append(sid); _mark_seen(g, sid)
    g.stack = []
    if source_iid in g.objects and source_iid not in g.exile:
        g.exile.append(source_iid); _mark_seen(g, source_iid)   # ...including this card itself
    g.log.append(f"{g.objects[source_iid].name} ends the turn and exiles the stack.")
    g.combat = Combat()                               # everything removed from combat
    g.current_step = "cleanup"                        # skip straight to cleanup (CR 720.1)
    g.passed = {"p1": False, "p2": False}
    _cleanup(g)                                       # discard to hand size, damage wears off, then next turn


def discard_to_hand_size(g: GameState, player: str, instance_ids: list) -> bool:
    if not g.pending or g.pending.type != "discard" or g.pending.player != player:
        return False
    need = g.pending.context.get("count", 0)
    hand = g.players[player].hand
    chosen, seen = [], set()
    for iid in instance_ids:
        if iid in hand and iid not in seen:
            chosen.append(iid)
            seen.add(iid)
    if len(chosen) != need:
        return False
    for iid in chosen:
        hand.remove(iid)
        g.graveyard.append(iid)
    g.pending = None
    g.log.append(f"{g.players[player].name} discards {need}.")
    _advance_step(g)
    return True


_SAC_NO_TYPE_RE = re.compile(r"when you control no (\w+?)s?\s*,?\s*sacrifice", re.I)


def _remove_from_battlefield(g: GameState, pl, iid: str, o, reason: str) -> None:
    pl.battlefield.remove(iid)
    g.graveyard.append(iid)
    o.damage_marked = 0
    o.tapped = False
    g.combat.attackers.pop(iid, None)
    for info in g.combat.attackers.values():
        if iid in info.get("blockers", []):
            info["blockers"].remove(iid)
    g.log.append(reason)


def _sweep_text_changes(g: GameState) -> None:
    """Text changes only apply on the stack/battlefield; revert any object that
    has left those zones (graveyard, exile, hand, library)."""
    in_play = {s.source_instance_id for s in g.stack}
    for pl in g.players.values():
        in_play.update(pl.battlefield)
    for o in g.objects.values():
        if (o.text_changes or o.text_orig) and o.instance_id not in in_play:
            revert_text_changes(o)


def _check_sba(g: GameState) -> None:
    # Creatures with lethal marked damage are destroyed (CR 704.5g).
    for pl in g.players.values():
        for iid in list(pl.battlefield):
            o = g.objects.get(iid)
            if o and _is_creature(o) and o.toughness > 0 and o.damage_marked >= o.toughness:
                _remove_from_battlefield(g, pl, iid, o, f"{o.name} dies.")
    _sweep_text_changes(g)
    for pid, p in g.players.items():
        if p.has_lost:
            continue
        if p.life <= 0:
            _lose(g, pid, f"life total {p.life}")


def _state_trigger_pending(g: GameState, iid: str) -> bool:
    """Whether a state trigger from `iid` is already pending or on the stack — it
    mustn't re-trigger until that instance leaves the stack (CR 603.8)."""
    return any((t.chosen or {}).get("state_trigger") and t.source_instance_id == iid
               for t in (list(g.stack) + list(g.pending_triggers)))


def _check_state_triggers(g: GameState) -> None:
    """Queue state-triggered abilities whose condition currently holds (CR 603.8):
    Dandân's "When you control no [type], sacrifice this creature." This is a real
    trigger (uses the stack), NOT a state-based action — it's put on the stack at
    the next priority, can be responded to, and resolves regardless of the count at
    resolution (no intervening 'if'). It won't pile up while an instance is pending,
    and re-triggers if that instance leaves the stack with the condition still true.
    Reads the effective (possibly text-changed) type words."""
    for pid, pl in g.players.items():
        for iid in list(pl.battlefield):
            o = g.objects.get(iid)
            if not o:
                continue
            m = _SAC_NO_TYPE_RE.search(o.oracle_text or "")
            if not m or controls_basic_type(g, pid, m.group(1)):
                continue                                  # no clause, or the condition is false
            if _state_trigger_pending(g, iid):
                continue                                  # an instance is already pending/on the stack
            g.pending_triggers.append(StackObject(
                stack_id=uuid.uuid4().hex, kind="triggered",
                source_instance_id=iid, controller=pid,
                description=f"When you control no {m.group(1).capitalize()}s, sacrifice this creature.",
                chosen={"state_trigger": "sacrifice"},
            ))
            g.log.append(f"{o.name}'s ability triggers.")


def _lose(g: GameState, pid: str, reason: str) -> None:
    p = g.players[pid]
    if p.has_lost:
        return
    p.has_lost = True
    winner = _OTHER[pid]
    g.result = {"status": f"{winner}_wins", "winner": winner, "reason": reason}
    g.pending = None
    g.priority_player = None
    g.log.append(f"{p.name} loses: {reason}.")
