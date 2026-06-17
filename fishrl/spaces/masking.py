"""Legality masks for atomic (non-compound) decisions.

Every unmasked action is guaranteed to be accepted by the engine when applied —
the mask is the contract that lets the policy emit a single discrete id without
the env ever issuing an illegal engine call. Compound decisions are masked by the
builder (see :mod:`fishrl.spaces.compound`); this module covers everything else.

Priority/pay legality is read from :func:`state.current_view` (which already
refines play/cycle/ability timing for the viewer); decision legality is read from
``g.pending.context`` (whose id lists are legal by construction).
"""
from __future__ import annotations

import numpy as np

from fishrl.forgetful_fish import engine as E
from fishrl.forgetful_fish.state import (
    PERMANENT_ABILITIES, MODAL_SPELLS, _is_land, current_view, land_mana_color,
)
from fishrl.spaces import action_space as A


def atomic_mask(g, viewer: str) -> np.ndarray:
    """Length-N 0/1 legality mask for the current atomic pending decision."""
    m = np.zeros(A.N, dtype=np.int8)
    pend = g.pending
    if pend is None:
        return m
    t = pend.type
    ctx = pend.context or {}

    if t == "priority":
        _priority_mask(g, viewer, m)
    elif t == "pay":
        _pay_mask(g, viewer, ctx, m)
    elif t == "choose_play_order":
        m[A.aid("PLAY_ORDER", 0)] = 1
        m[A.aid("PLAY_ORDER", 1)] = 1
    elif t == "mulligan":
        m[A.aid("MULLIGAN", 0)] = 1
        m[A.aid("MULLIGAN", 1)] = 1
    elif t == "fof_choose":
        m[A.aid("FOF_CHOOSE", 0)] = 1
        m[A.aid("FOF_CHOOSE", 1)] = 1
    elif t == "choose_text_change":
        for li in range(len(A.BASICS) * len(A.BASICS)):
            frm, to = A.text_change_pair(li)
            if frm != to:
                m[A.aid("TEXT_CHANGE", li)] = 1
    elif t == "choose_targets":
        legal = ctx.get("legal", [])
        for k in range(min(len(legal), A.PICK_K)):
            m[A.aid("PICK_SINGLE", k)] = 1
        m[A.aid("TARGET_CANCEL")] = 1
    elif t == "choose_graveyard":
        _single_pick_mask(ctx.get("eligible", []), m, allow_none=ctx.get("may", True))
    elif t == "search_library":
        _single_pick_mask(ctx.get("eligible", []), m, allow_none=True)
    elif t == "put_from_hand":
        _single_pick_mask(ctx.get("eligible", []), m, allow_none=True)
    elif t == "name_card":
        _single_pick_mask(ctx.get("names", []), m, allow_none=False)
    elif t == "order_triggers":
        _single_pick_mask(ctx.get("triggers", []), m, allow_none=False)
    return m


def _single_pick_mask(items, m, *, allow_none: bool) -> None:
    for k in range(min(len(items), A.PICK_K)):
        m[A.aid("PICK_SINGLE", k)] = 1
    if allow_none:
        m[A.aid("PICK_NONE")] = 1


def _affordable(g, viewer: str, cost: str) -> bool:
    """Whether `viewer` can pay `cost` from floating mana + untapped lands (the
    same coarse estimate the heuristic AI uses; ignores sacrifice-for-mana)."""
    colored, generic = E._parse_cost(cost)
    pool = dict(g.players[viewer].mana_pool)
    for iid in g.players[viewer].battlefield:
        o = g.objects[iid]
        if _is_land(o.type_line) and not o.tapped:
            sym = land_mana_color(o)
            pool[sym] = pool.get(sym, 0) + 1
    return E._can_afford(pool, colored, generic)


def _priority_mask(g, viewer: str, m) -> None:
    m[A.aid("PASS")] = 1
    if g.active_player == viewer and not g.stack:
        m[A.aid("END_TURN")] = 1
    view = current_view(g, viewer)
    hand_view = view["players"][viewer]["hand"]
    hand_ids = g.players[viewer].hand
    for i, card in enumerate(hand_view[:A.HAND]):
        if card.get("can_play"):
            o = g.objects[hand_ids[i]]
            playable = True
            if not _is_land(o.type_line):               # spells: need mana + legal targets
                if not _affordable(g, viewer, o.mana_cost):
                    playable = False
                spec = E._TARGETS.get(o.name)
                if spec is not None and len(spec["legal"](g, viewer)) < spec["count"]:
                    playable = False
            if playable:
                m[A.aid("PLAY_HAND", i)] = 1
                if card.get("modes"):                   # modal spell: alternate mode
                    m[A.aid("PLAY_HAND_ALT", i)] = 1
        if card.get("can_cycle"):
            m[A.aid("CYCLE_HAND", i)] = 1
    bf = view["players"][viewer]["battlefield"]
    for i, card in enumerate(bf[:A.BF]):
        for ab in card.get("abilities", []):
            idx = ab["index"]
            if ab.get("available") and idx < A.ABIL_SLOTS:
                m[A.aid("ACTIVATE", i * A.ABIL_SLOTS + idx)] = 1


def _pay_mask(g, viewer: str, ctx: dict, m) -> None:
    m[A.aid("CANCEL_PAY")] = 1
    need = ctx.get("need", {})
    generic = ctx.get("generic", 0)
    source = ctx.get("source")
    bf = g.players[viewer].battlefield

    def can_pay(sym: str) -> bool:
        return need.get(sym, 0) > 0 or generic > 0

    for i, iid in enumerate(bf[:A.BF]):
        o = g.objects.get(iid)
        if not o:
            continue
        # tap a land that pays a still-needed pip (skip the cost's own source)
        if _is_land(o.type_line) and not o.tapped and iid != source:
            from fishrl.forgetful_fish.state import land_mana_color
            if can_pay(land_mana_color(o)):
                m[A.aid("TAP_LAND", i)] = 1
        # activate a mana ability (Svyelunite / Surgical Bay add {U}) into the cost
        for idx, ab in enumerate(PERMANENT_ABILITIES.get(o.name, [])):
            if (ab["adds"] and idx < A.ABIL_SLOTS and iid != source
                    and not (ab["tap"] and o.tapped) and can_pay("U")):
                m[A.aid("ACTIVATE", i * A.ABIL_SLOTS + idx)] = 1
    # spend floating mana that can still pay something
    pool = g.players[viewer].mana_pool
    for ci, c in enumerate(A.COLORS):
        if pool.get(c, 0) > 0 and can_pay(c):
            m[A.aid("ALLOC_MANA", ci)] = 1
