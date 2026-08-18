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

import re

import numpy as np

from fishrl.forgetful_fish import engine as E
from fishrl.forgetful_fish.state import (
    CYCLING, PERMANENT_ABILITIES, MODAL_SPELLS, _available_mana, _can_play_now,
    _is_land, current_view, land_mana_color,
)
from fishrl.spaces import action_space as A

# ── memoized text-pure engine helpers ────────────────────────────────────────────
# `land_mana_color` (regex over type_line/oracle_text) and `_is_land` (str.lower +
# substring) are pure functions of a card's EFFECTIVE text -- which text-change
# effects rewrite, so the cache keys on the strings themselves, never the instance.
# `_mana_options` runs them per untapped permanent per affordability probe, which
# made module-level `re.search` a measurable slice of collection. fishrl-side only:
# the vendored engine keeps calling its own copies.
_LAND_COLOR_CACHE: dict = {}
_IS_LAND_CACHE: dict = {}


def _land_color(o) -> str:
    key = (o.type_line, o.oracle_text)
    c = _LAND_COLOR_CACHE.get(key)
    if c is None:
        c = land_mana_color(o)
        _LAND_COLOR_CACHE[key] = c
    return c


def _is_land_tl(tl) -> bool:
    v = _IS_LAND_CACHE.get(tl)
    if v is None:
        v = _is_land(tl)
        _IS_LAND_CACHE[tl] = v
    return v

# ── choose_text_change action-space treatment (Config.text_change_mode) ──────────
# "full"   -- the legacy 25-way from->to block (minus the no-op diagonal).
# "guided" -- {EFFECT, NO-OP}: the type actually written on the targeted card ->
#             a type absent from the target's controller's permanents, plus one
#             provably-inert pair (decline). Collapses a 5x5 space the policy had
#             to learn from +-1 terminal reward into the 2-way choice that is all
#             the game semantics actually offer.
# "auto"   -- {EFFECT} alone: a forced decision the collector's single-legal-action
#             fast path plays with no policy forward (auto-resolve).
# Process-global, set once from Config at trainer/worker/panel init (the
# features.set_public_encoding pattern). Guided masking only ever SHRINKS the legal
# set; on any state it cannot analyse it falls back to the full mask.
_TEXT_CHANGE_MODE = "full"
_BASIC_WORD = {b: re.compile(rf"\b{b}\b") for b in A.BASICS}


def set_text_change_mode(mode: str) -> None:
    global _TEXT_CHANGE_MODE
    assert mode in ("full", "guided", "auto"), mode
    _TEXT_CHANGE_MODE = mode


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
        _text_change_mask(g, ctx, m)
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


def _full_text_change_mask(ctx: dict, m) -> None:
    """The legacy mask: every from!=to pair the engine will accept."""
    from_types = ctx.get("from_types") or list(A.BASICS)
    to_types = ctx.get("to_types") or list(A.BASICS)
    for li in range(len(A.BASICS) * len(A.BASICS)):
        frm, to = A.text_change_pair(li)
        if frm != to and frm in from_types and to in to_types:
            m[A.aid("TEXT_CHANGE", li)] = 1


def _guided_text_change(g, ctx: dict):
    """(effect_id, noop_id) for the guided/auto modes, or (None, None) to fall back.

    EFFECT = (frm, to) where `frm` is the basic-type word actually written on the
    change targets (the pool's cards only ever carry one) and `to` is the first
    canonical type ABSENT from the targets' controller's permanents — the pair that
    severs the "control an Island" linkage (Dandân's sacrifice clause, islandwalk).
    NO-OP = a from-type present in the engine's from_types but written on none of the
    targets: the rewrite provably matches nothing. Both ids are ordinary
    TEXT_CHANGE[5*frm+to] actions the engine accepts, so the mask contract holds."""
    targets = [g.objects[iid] for iid in ctx.get("change_targets", []) if iid in g.objects]
    if not targets:
        return None, None
    from_types = ctx.get("from_types") or list(A.BASICS)
    to_types = ctx.get("to_types") or list(A.BASICS)

    def written_on(o, b: str) -> bool:
        pat = _BASIC_WORD[b]
        return bool(pat.search(o.type_line or "") or pat.search(o.oracle_text or ""))

    # frm: the basic type written on the most targets (ties break in canonical order)
    frm = best_n = None
    for b in A.BASICS:
        if b not in from_types:
            continue
        n = sum(1 for o in targets if written_on(o, b))
        if n > 0 and (best_n is None or n > best_n):
            frm, best_n = b, n
    if frm is None:
        return None, None                       # nothing the rewrite could touch

    controller = targets[0].controller or ""
    if controller not in g.players:
        return None, None
    ctrl_bf = [g.objects[iid] for iid in g.players[controller].battlefield
               if iid in g.objects]
    to = next((b for b in A.BASICS
               if b != frm and b in to_types
               and not any(_BASIC_WORD[b].search(o.type_line or "") for o in ctrl_bf)),
              None)
    if to is None:
        return None, None                       # controller covers all types (not this pool)
    effect = A.aid("TEXT_CHANGE", A.BASICS.index(frm) * len(A.BASICS) + A.BASICS.index(to))

    noop_frm = next((b for b in A.BASICS
                     if b in from_types and not any(written_on(o, b) for o in targets)),
                    None)
    noop = None
    if noop_frm is not None:
        noop_to = next((b for b in A.BASICS if b != noop_frm and b in to_types), None)
        if noop_to is not None:
            noop = A.aid("TEXT_CHANGE",
                         A.BASICS.index(noop_frm) * len(A.BASICS) + A.BASICS.index(noop_to))
    return effect, noop


def _text_change_mask(g, ctx: dict, m) -> None:
    if _TEXT_CHANGE_MODE != "full":
        effect, noop = _guided_text_change(g, ctx)
        if effect is not None:
            m[effect] = 1
            if _TEXT_CHANGE_MODE == "guided" and noop is not None:
                m[noop] = 1
            return                              # guided/auto: the reduced set
    _full_text_change_mask(ctx, m)              # legacy, and the guided fallback


# ── mana availability ─────────────────────────────────────────────────────────
# Each untapped permanent exposes one or more mana OPTIONS; every option taps the
# source, so only one per source can be used. We model affordability off the MAX
# option per source (e.g. Svyelunite Temple can make {U}{U} by sacrificing), and we
# exclude the cost's own source (an ability that taps/sacrifices itself can't help
# pay for itself — e.g. The Surgical Bay's draw needs two OTHER lands).

def _mana_options(g, viewer: str, exclude_iid):
    """List of (iid, ('tap_land'|'activate', idx_or_None), sym, amount) for every
    untapped source's mana options. Ability mana is always {U} in this game.

    Only ADDRESSABLE sources are returned: a permanent past the action space's
    battlefield-slot cap (``A.BF``) or an ability past ``A.ABIL_SLOTS`` has no
    TAP_LAND/ACTIVATE id, so the pay mask could never emit it. Counting such a
    source toward affordability would let a cast be offered at priority and then
    strand with an empty pay mask (the {U}-spell strand). Affordability
    (``_potential_pool``) and the pay mask both route through here, so applying the
    cap once keeps them consistent by construction."""
    opts = []
    for i, iid in enumerate(g.players[viewer].battlefield):
        if i >= A.BF:                         # beyond the addressable slot cap -> the
            break                             # pay mask can't tap it; don't count it
        if iid == exclude_iid:
            continue
        o = g.objects.get(iid)
        if o is None or o.tapped:
            continue
        if _is_land_tl(o.type_line):
            opts.append((iid, ("tap_land", None), _land_color(o), 1))
        for idx, ab in enumerate(PERMANENT_ABILITIES.get(o.name, [])):
            if idx >= A.ABIL_SLOTS:           # ability id the action space can't reach
                continue
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
    """Object-native priority mask: re-derives the view's can_play / can_cycle /
    modes / ability-availability annotations straight from engine objects and the
    same engine helpers `current_view` uses (`_can_play_now`, `_available_mana`),
    skipping the full UI-view build (which was ~8% of collection wall). MUST stay
    identical to `_priority_mask_ref` -- guarded by test_priority_mask_equivalence."""
    m[A.aid("PASS")] = 1
    if g.active_player == viewer and not g.stack:
        m[A.aid("END_TURN")] = 1
    obj = g.objects
    pend = g.pending
    # current_view's timing refinement (CR 605.3a): mana abilities at your priority
    # OR during your payment; everything else only at your priority.
    your_priority = pend is not None and pend.type == "priority" and pend.player == viewer
    in_your_pay = pend is not None and pend.type == "pay" and pend.player == viewer
    for i, iid in enumerate(g.players[viewer].hand[:A.HAND]):
        o = obj[iid]
        if _can_play_now(g, viewer, o):
            playable = True
            if not _is_land_tl(o.type_line):            # spells: need mana + legal targets
                colored, generic = E._parse_cost(o.mana_cost)
                if not _affordable(g, viewer, colored, generic):
                    playable = False
                spec = E._TARGETS.get(o.name)
                if spec is not None and len(spec["legal"](g, viewer)) < spec["count"]:
                    playable = False
            if playable:
                m[A.aid("PLAY_HAND", i)] = 1
                if o.name in MODAL_SPELLS:              # modal spell: alternate mode
                    m[A.aid("PLAY_HAND_ALT", i)] = 1
        # cycling costs {U} (always blue, even if the card's land type was changed);
        # gate it so the agent can't commit to a cost it can't pay (there is no cancel).
        cyc = CYCLING.get(o.name)
        if (cyc is not None and your_priority and _available_mana(g, viewer) >= cyc
                and _affordable(g, viewer, {"U": cyc}, 0)):
            m[A.aid("CYCLE_HAND", i)] = 1
    for i, iid in enumerate(g.players[viewer].battlefield[:A.BF]):
        o = obj[iid]
        specs = PERMANENT_ABILITIES.get(o.name)
        if not specs:
            continue
        for idx, spec in enumerate(specs):
            if idx >= A.ABIL_SLOTS:
                continue
            avail = not (spec["tap"] and o.tapped)      # _ability_view's base availability
            if avail:
                avail = (your_priority or in_your_pay) if spec["adds"] else your_priority
            if not avail:
                continue
            # mana abilities cost no mana (they ARE mana); a non-mana ability must be
            # payable (colour-aware) from OTHER sources, since this one taps/sacs itself.
            if spec["adds"] or _affordable(g, viewer, *E._parse_cost(spec["cost"]), exclude_iid=iid):
                m[A.aid("ACTIVATE", i * A.ABIL_SLOTS + idx)] = 1


def _priority_mask_ref(g, viewer: str, m) -> None:
    """REFERENCE priority mask (the original current_view path) -- the behavioural
    contract `_priority_mask` must stay identical to. Kept as the equivalence oracle
    (test_priority_mask_equivalence), like the *_ref encoders."""
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
    would strand the payment (there is no CANCEL_PAY to bail out).

    The rule is per-OPTION completability: an option is offered iff floating mana +
    THIS option's mana + every OTHER untapped source at its MAX covers the remaining
    cost. That preserves the anti-stall guarantee — taking an offered option leaves a
    remainder that is still affordable from floating + remaining sources at max, and
    in any affordable state each remaining source's MAX option passes this very test
    (this-at-max + others-at-max IS the affordability pool), so the mask can never go
    empty mid-payment. But unlike the old "required source -> only its MAX option"
    rule, it does not over-force: with Svyelunite Temple + Island paying {U}{U}, the
    Temple is required, yet its tap-for-one still completes (1 + the Island's 1), so
    the agent may tap instead of being forced into an unnecessary sacrifice."""
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

    def completable(this_iid, this_sym: str, this_amt: int) -> bool:
        """Can the remaining cost be paid from floating + THIS option's mana + every
        OTHER source at max? (This option taps its source, so the source's other —
        possibly larger — options are forgone.)"""
        pool = dict(floating)
        pool[this_sym] = pool.get(this_sym, 0) + this_amt
        for iid, (sym, amt) in src_max.items():
            if iid == this_iid:
                continue
            pool[sym] = pool.get(sym, 0) + amt
        return E._can_afford(pool, need, generic)

    for iid, action, sym, amt in opts:
        if not can_pay(sym):                            # this mana can't pay anything left
            continue
        if not completable(iid, sym, amt):
            continue                                    # this option would strand the payment
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
