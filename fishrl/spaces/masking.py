"""Legality masks for atomic (non-compound) decisions.

Every unmasked action is guaranteed to be accepted by the engine when applied —
the mask is the contract that lets the policy emit a single discrete id without
the env ever issuing an illegal engine call. Compound decisions are masked by the
builder (see :mod:`fishrl.spaces.compound`); this module covers everything else.

Two human-only "undo" affordances are deliberately withheld from the agent, so it
cannot stall indefinitely by oscillating in and out of a commitment (each such
oscillation is a counted decision with zero game progress):

  * ``CANCEL_PAY`` — aborting a half-made payment. Instead, casts/activations are
    only offered when the player can actually pay (see ``_affordable``), and the
    pay mask never offers a tap that would strand the payment, so once committed
    the cost is always completable.
  * ``TARGET_CANCEL`` — aborting target selection (targets are chosen before
    paying, so cancelling would hand the card back for free). ``choose_targets``
    is always mandatory-count, so the agent simply picks.

Priority/pay legality is read from :func:`state.current_view` (which already
refines play/cycle/ability timing for the viewer); decision legality is read from
``g.pending.context`` (whose id lists are legal by construction).
"""
from __future__ import annotations

import numpy as np

from fishrl.forgetful_fish import engine as E
from fishrl.forgetful_fish.state import (
    CYCLING, PERMANENT_ABILITIES, MODAL_SPELLS, _is_land, current_view, land_mana_color,
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
        # NB: no TARGET_CANCEL — targets are mandatory and chosen before paying, so
        # cancelling would be a free, reversible abort of the cast.
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


# ── mana availability ─────────────────────────────────────────────────────────
# Each untapped permanent exposes one or more mana OPTIONS; every option taps the
# source, so only one per source can be used. We model affordability off the MAX
# option per source (e.g. Svyelunite Temple can make {U}{U} by sacrificing), and we
# exclude the cost's own source (an ability that taps/sacrifices itself can't help
# pay for itself — e.g. The Surgical Bay's draw needs two OTHER lands).

def _mana_options(g, viewer: str, exclude_iid):
    """List of (iid, ('tap_land'|'activate', idx_or_None), sym, amount) for every
    untapped source's mana options. Ability mana is always {U} in this game."""
    opts = []
    for iid in g.players[viewer].battlefield:
        if iid == exclude_iid:
            continue
        o = g.objects.get(iid)
        if o is None or o.tapped:
            continue
        if _is_land(o.type_line):
            opts.append((iid, ("tap_land", None), land_mana_color(o), 1))
        for idx, ab in enumerate(PERMANENT_ABILITIES.get(o.name, [])):
            if ab["adds"] and not (ab["tap"] and o.tapped):   # an available mana ability
                opts.append((iid, ("activate", idx), "U", ab["adds"]))
    return opts


def _source_max(opts: list) -> dict:
    """iid -> (sym, amount) of that source's largest mana option."""
    best: dict = {}
    for iid, _action, sym, amt in opts:
        if amt > best.get(iid, (None, 0))[1]:
            best[iid] = (sym, amt)
    return best


def _potential_pool(g, viewer: str, exclude_iid) -> dict:
    """Floating mana + each untapped source's MAX producible mana."""
    pool = dict(g.players[viewer].mana_pool)
    for sym, amt in _source_max(_mana_options(g, viewer, exclude_iid)).values():
        pool[sym] = pool.get(sym, 0) + amt
    return pool


def _affordable(g, viewer: str, colored: dict, generic: int, *, exclude_iid=None) -> bool:
    """Whether `viewer` can pay (colored pips + generic) from floating mana and the
    max mana of every untapped source (except `exclude_iid`)."""
    return E._can_afford(_potential_pool(g, viewer, exclude_iid), colored, generic)


def _priority_mask(g, viewer: str, m) -> None:
    m[A.aid("PASS")] = 1
    if g.active_player == viewer and not g.stack:
        m[A.aid("END_TURN")] = 1
    view = current_view(g, viewer)
    hand_view = view["players"][viewer]["hand"]
    hand_ids = g.players[viewer].hand
    for i, card in enumerate(hand_view[:A.HAND]):
        o = g.objects[hand_ids[i]]
        if card.get("can_play"):
            playable = True
            if not _is_land(o.type_line):               # spells: need mana + legal targets
                colored, generic = E._parse_cost(o.mana_cost)
                if not _affordable(g, viewer, colored, generic):
                    playable = False
                spec = E._TARGETS.get(o.name)
                if spec is not None and len(spec["legal"](g, viewer)) < spec["count"]:
                    playable = False
            if playable:
                m[A.aid("PLAY_HAND", i)] = 1
                if card.get("modes"):                   # modal spell: alternate mode
                    m[A.aid("PLAY_HAND_ALT", i)] = 1
        # cycling costs {U} (always blue, even if the card's land type was changed);
        # gate it so the agent can't commit to a cost it can't pay (there is no cancel).
        if card.get("can_cycle") and _affordable(g, viewer, {"U": CYCLING.get(o.name, 0)}, 0):
            m[A.aid("CYCLE_HAND", i)] = 1
    bf_ids = g.players[viewer].battlefield
    bf = view["players"][viewer]["battlefield"]
    for i, card in enumerate(bf[:A.BF]):
        iid = bf_ids[i]
        name = g.objects[iid].name
        for ab in card.get("abilities", []):
            idx = ab["index"]
            if not (ab.get("available") and idx < A.ABIL_SLOTS):
                continue
            spec = PERMANENT_ABILITIES[name][idx]
            # mana abilities cost no mana (they ARE mana); a non-mana ability must be
            # payable (colour-aware) from OTHER sources, since this one taps/sacs itself.
            if spec["adds"] or _affordable(g, viewer, *E._parse_cost(spec["cost"]), exclude_iid=iid):
                m[A.aid("ACTIVATE", i * A.ABIL_SLOTS + idx)] = 1


def _pay_mask(g, viewer: str, ctx: dict, m) -> None:
    """Offer only payment actions that keep the cost completable — never a tap that
    would strand the payment (there is no CANCEL_PAY to bail out)."""
    need = ctx.get("need", {})
    generic = ctx.get("generic", 0)
    source = ctx.get("source")
    bf_ids = g.players[viewer].battlefield
    idx_of = {iid: i for i, iid in enumerate(bf_ids)}

    opts = _mana_options(g, viewer, source)
    src_max = _source_max(opts)
    floating = dict(g.players[viewer].mana_pool)

    def can_pay(sym: str) -> bool:
        return need.get(sym, 0) > 0 or generic > 0

    def others_cover(skip_iid) -> bool:
        """Can the remaining cost be paid WITHOUT this source (others at max)?"""
        pool = dict(floating)
        for iid, (sym, amt) in src_max.items():
            if iid == skip_iid:
                continue
            pool[sym] = pool.get(sym, 0) + amt
        return E._can_afford(pool, need, generic)

    for iid, action, sym, amt in opts:
        if not can_pay(sym):                            # this mana can't pay anything left
            continue
        if not others_cover(iid) and amt < src_max[iid][1]:
            continue                                    # source is required -> only its MAX option
        i = idx_of.get(iid)
        if i is None or i >= A.BF:
            continue
        if action[0] == "tap_land":
            m[A.aid("TAP_LAND", i)] = 1
        elif action[1] < A.ABIL_SLOTS:
            m[A.aid("ACTIVATE", i * A.ABIL_SLOTS + action[1])] = 1

    # spend floating mana that can still pay something (always safe — it only
    # reduces the cost and consumes no source)
    pool = g.players[viewer].mana_pool
    for ci, c in enumerate(A.COLORS):
        if pool.get(c, 0) > 0 and can_pay(c):
            m[A.aid("ALLOC_MANA", ci)] = 1
