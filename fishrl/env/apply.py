"""Decode a flat action id into exactly one engine call (atomic decisions).

Compound decisions are handled by the env-side builder (:mod:`fishrl.spaces.compound`);
this module covers priority/pay and the single/atomic pending decisions. Every
function here assumes the action was legal under the mask, so the engine call
should succeed; it returns the engine's bool for defensive checking.
"""
from __future__ import annotations

from fishrl.forgetful_fish import engine as E
from fishrl.forgetful_fish.state import MODAL_SPELLS
from fishrl.spaces import action_space as A


def apply_atomic(g, viewer: str, action: int) -> bool:
    """Apply one atomic action for `viewer` against the current pending decision."""
    pend = g.pending
    ctx = pend.context or {}
    t = pend.type
    name, i = A.decode(action)
    hand = g.players[viewer].hand
    bf = g.players[viewer].battlefield

    if t == "priority":
        if name == "PASS":
            return E.pass_priority(g, viewer)
        if name == "END_TURN":
            return E.end_turn(g, viewer)
        if name == "PLAY_HAND":
            return E.play(g, viewer, hand[i])
        if name == "PLAY_HAND_ALT":
            iid = hand[i]
            modes = MODAL_SPELLS.get(g.objects[iid].name) or []
            alt = modes[1]["key"] if len(modes) > 1 else None
            return E.play(g, viewer, iid, mode=alt)
        if name == "CYCLE_HAND":
            return E.cycle(g, viewer, hand[i])
        if name == "ACTIVATE":
            return E.activate_ability(g, viewer, bf[i // A.ABIL_SLOTS], i % A.ABIL_SLOTS)

    elif t == "pay":
        if name == "CANCEL_PAY":
            return E.cancel_payment(g, viewer)
        if name == "TAP_LAND":
            return E.tap(g, viewer, bf[i])
        if name == "ALLOC_MANA":
            return E.allocate_mana(g, viewer, A.COLORS[i])
        if name == "ACTIVATE":
            return E.activate_ability(g, viewer, bf[i // A.ABIL_SLOTS], i % A.ABIL_SLOTS)

    elif t == "choose_play_order":
        return E.choose_play_order(g, viewer, "first" if i == 0 else "second")
    elif t == "mulligan":
        return E.mulligan_decision(g, viewer, "keep" if i == 0 else "mulligan")
    elif t == "fof_choose":
        return E.complete_fof_choose(g, viewer, 1 if i == 0 else 2)
    elif t == "choose_text_change":
        frm, to = A.text_change_pair(i)
        return E.complete_text_change(g, viewer, frm, to)
    elif t == "choose_targets":
        if name == "TARGET_CANCEL":
            return E.complete_targets(g, viewer, [], cancel=True)
        return E.complete_targets(g, viewer, [ctx["legal"][i]])
    elif t == "choose_graveyard":
        pick = None if name == "PICK_NONE" else ctx["eligible"][i]
        return E.complete_graveyard_choice(g, viewer, pick)
    elif t == "search_library":
        pick = None if name == "PICK_NONE" else ctx["eligible"][i]
        return E.complete_library_search(g, viewer, pick)
    elif t == "put_from_hand":
        pick = None if name == "PICK_NONE" else ctx["eligible"][i]
        return E.complete_put_from_hand(g, viewer, pick)
    elif t == "name_card":
        return E.complete_name_card(g, viewer, ctx["names"][i])
    elif t == "order_triggers":
        return E.place_trigger(g, viewer, ctx["triggers"][i]["stack_id"])

    raise ValueError(f"unhandled atomic action {name} for pending {t}")
