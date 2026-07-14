"""Observation encoder: project a per-player view into a fixed-size float vector.

The source of truth is :func:`state.current_view`, which already enforces hidden-
information legality — the opponent's unseen hand cards and unknown library slots
arrive as ``{"known": False}`` and are encoded as a single "hidden" row carrying
no identity. Public, non-identity scalars (life, turn, mulligans, library count)
are read from the game state directly; those are common knowledge.

Layout is fixed and flat (MLP-friendly). Observation rows are zone-slot aligned
with the action space so a card's feature row and its action slot share an index.
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

from fishrl.forgetful_fish.state import current_view, text_variant, BASIC_TYPES
from fishrl.obs import vocab as V

# Basic land types, in canonical order, with whole-word matchers. Forgetful Fish
# rewrites the basic-type word in a card's text (Mind Bend / Crystal Spray /
# Vision Charm change "Island" into another type, permanently or until end of
# turn). The engine stores the EFFECTIVE (already-rewritten) type_line/oracle_text,
# so we read the current type off those strings.
_BASIC_RE = [re.compile(r"\b" + t + r"s?\b", re.I) for t in BASIC_TYPES]


@lru_cache(maxsize=8192)
def _basic_bits(text: str) -> tuple:
    """Which canonical basic-land types appear as whole words in `text` (5-tuple of
    0/1). This is a PURE function of the string, so it's memoized: the same handful
    of card texts recur across every zone, every encoder (perspective/god/public),
    every decision -- and `re.Pattern.search` was ~30% of collection self-time. Card
    texts are few, so the cache is ~100% hit after warmup."""
    t = text or ""
    return tuple(1.0 if pat.search(t) else 0.0 for pat in _BASIC_RE)


def _basic_multihot(text: str, out: np.ndarray, off: int) -> None:
    for i, b in enumerate(_basic_bits(text)):
        if b:
            out[off + i] = 1.0


# ── per-card feature vector ───────────────────────────────────────────────────
# name one-hot (N_NAMES) + unknown bit; generic type flags(4); effective basic
# type in the type line (5); basic types referenced in the oracle text (5, e.g. a
# Dandân's "Island" clause after Mind Bend); text-altered flag(1); stats(4);
# perm flags(3); known bit(1).
CARD_F = V.N_NAMES + 1 + 4 + 5 + 5 + 1 + 4 + 3 + 1

# ── fixed zone slot counts (pad / truncate) ───────────────────────────────────
SLOTS = {
    "own_hand": 12, "opp_hand": 12, "own_bf": 34, "opp_bf": 34,
    "graveyard": 32, "exile": 8, "stack": 6, "library": 8,
}
_ZONE_ROWS = sum(SLOTS.values())

# ── global / decision scalar block ────────────────────────────────────────────
_PER_PLAYER = 12          # life, hand_count, lands, untapped, pool_total, pool×5, mulligans, has_lost
_GAME = 5                 # turn, active_is_self, priority_is_self, library_count, stack_depth
_COMBAT = 4               # am_attacking, am_defending, n_attackers, n_blockers
_PAY = 4                  # need_U, need_total, generic, in_pay
_BUILDER = 1              # compound-builder progress (injected by the env)
_CLOCK = 3                # deckout clock: parity, next_drawer_is_self, self_decks_first
_GLOBALS = (_PER_PLAYER * 2 + _GAME + V.N_STEPS + _COMBAT + V.N_PENDING + _PAY
            + _BUILDER + _CLOCK)

OBS_DIM = _ZONE_ROWS * CARD_F + _GLOBALS

# ── deckout clock ─────────────────────────────────────────────────────────────
# The library is SHARED and both seats draw one per turn, so the seat forced to draw from an
# empty library is decided by (library_count mod 2) XOR (whose draw is next) -- and every
# EXTRA card drawn flips it, which makes Brainstorm/Predict/Fact-or-Fiction deckout-tempo
# tools, not just card advantage. It decides ~81% of the deckout/board_presence games.
#
# The count is already encoded -- but only as `len(library)/80.0`, one smooth float. Probing
# the 592h checkpoint (fishrl/eval/probe_deckout_clock.py): an MLP handed a DIRECT supervised
# label cannot recover parity from that float (0.508 vs a 0.514 base rate), and the trained
# actor had not learned it either (trunk 0.525). Same information, and the ENCODING is the
# whole difference between 0.508 and 1.000. Parity from a smooth scalar is the textbook thing
# ReLU nets fail at, so this is a RE-ENCODING of information already present, not new
# information -- nothing here is hidden from the viewer.
_PRE_DRAW = ("untap", "upkeep", "draw", "")


def deckout_clock(g, viewer: str) -> tuple:
    """(library_parity, next_drawer_is_viewer, viewer_decks_first), each 0/1.

    Zeroed in the PREGAME, where `active_player` is "" because the play-order roll has not
    resolved. The clock is undefined there (nobody has drawn), and any viewer-dependent
    fallback would make the two seats DISAGREE about who draws next: this must be a function
    of the GAME, only re-oriented per seat."""
    active = g.active_player
    if active not in ("p1", "p2"):
        return 0.0, 0.0, 0.0
    n = len(g.library)
    opp = "p2" if active == "p1" else "p1"
    nxt = opp if g.current_step not in _PRE_DRAW else active   # the draw step precedes main1
    loser = nxt if n % 2 == 0 else ("p2" if nxt == "p1" else "p1")
    return float(n % 2), float(nxt == viewer), float(loser == viewer)


def _encode_card_into(card: dict | None, viewer: str, v: np.ndarray) -> None:
    """Write a card's feature row into ``v`` (a CARD_F-wide slice, assumed zeroed).

    In-place to avoid a per-card temporary allocation + row-copy: this runs ~150x per
    decision (3 encoders x ~50 occupied slots) and `_encode_card` was the single
    hottest function in collection. The name one-hot stays a single sparse write
    (`v[idx]=1.0`) -- it is NOT densely constructed, so precomputing it would be a
    20-wide copy in place of one assignment (slower)."""
    if not card or card.get("known") is False or "name" not in card:
        v[V.N_NAMES] = 1.0          # "unknown/hidden" bit
        return
    name = card.get("name", "")
    idx = V.NAME_INDEX.get(name)
    if idx is not None:
        v[idx] = 1.0
    else:
        v[V.N_NAMES] = 1.0
    o = V.N_NAMES + 1
    tl = card.get("type_line") or ""
    tll = tl.lower()
    v[o + 0] = float("land" in tll)
    v[o + 1] = float("creature" in tll)
    v[o + 2] = float("instant" in tll)
    v[o + 3] = float("sorcery" in tll)
    o += 4
    _basic_multihot(tl, v, o)                  # effective basic land type (5)
    o += 5
    _basic_multihot(card.get("oracle_text") or "", v, o)   # basic types named in text (5)
    o += 5
    v[o] = 1.0 if card.get("text_variant") else 0.0        # text-altered (Island lineage moved)
    o += 1
    v[o + 0] = (card.get("power") or 0) / 10.0
    v[o + 1] = (card.get("toughness") or 0) / 10.0
    v[o + 2] = (card.get("damage_marked") or 0) / 10.0
    v[o + 3] = sum((card.get("counters") or {}).values()) / 10.0
    o += 4
    v[o + 0] = float(bool(card.get("tapped")))
    v[o + 1] = float(bool(card.get("entered_this_turn")))
    v[o + 2] = float(card.get("controller") == viewer)
    o += 3
    v[o] = 1.0                       # known bit


def _encode_card(card: dict | None, viewer: str) -> np.ndarray:
    """Allocating wrapper kept for callers that want a standalone row."""
    v = np.zeros(CARD_F, dtype=np.float32)
    _encode_card_into(card, viewer, v)
    return v


def _zone(cards: list, n: int, viewer: str) -> np.ndarray:
    rows = np.zeros((n, CARD_F), dtype=np.float32)
    for i, card in enumerate(cards[:n]):
        _encode_card_into(card, viewer, rows[i])   # write into the row; no temp + copy
    return rows.reshape(-1)


def encode_observation_ref(g, viewer: str, builder_progress: float = 0.0) -> np.ndarray:
    """REFERENCE observation encoder: the dict-based path through the engine's UI-oriented
    `current_view`. This is the behavioural CONTRACT -- `encode_observation` (the fast
    object-native path actually used in collection) must stay bit-identical to it, guarded by
    tests/test_encoder_equivalence.py. Kept as the source of truth and the equivalence oracle.

    `builder_progress` is the fraction of a compound decision already specified (0 when none)."""
    view = current_view(g, viewer)
    opp = "p2" if viewer == "p1" else "p1"
    me_v, op_v = view["players"][viewer], view["players"][opp]

    parts = [
        _zone(me_v.get("hand", []), SLOTS["own_hand"], viewer),
        _zone(op_v.get("hand", []), SLOTS["opp_hand"], viewer),
        _zone(me_v.get("battlefield", []), SLOTS["own_bf"], viewer),
        _zone(op_v.get("battlefield", []), SLOTS["opp_bf"], viewer),
        _zone(view.get("graveyard", []), SLOTS["graveyard"], viewer),
        _zone(view.get("exile", []), SLOTS["exile"], viewer),
        _zone([s.get("card") for s in view.get("stack", [])], SLOTS["stack"], viewer),
        _zone([s for s in view.get("library", []) if s.get("known")],
              SLOTS["library"], viewer),
    ]

    g_vec = np.zeros(_GLOBALS, dtype=np.float32)
    k = 0
    for pid in (viewer, opp):
        pv = view["players"][pid]
        bf = pv.get("battlefield", [])
        lands = [c for c in bf if "land" in (c.get("type_line") or "").lower()]
        pool = pv.get("mana_pool", {})
        g_vec[k + 0] = pv.get("life", 0) / 20.0
        g_vec[k + 1] = (pv.get("hand_count", len(pv.get("hand", [])))) / 12.0
        g_vec[k + 2] = len(lands) / 20.0
        g_vec[k + 3] = sum(1 for c in lands if not c.get("tapped")) / 20.0
        g_vec[k + 4] = sum(pool.values()) / 10.0
        for ci, c in enumerate(("W", "U", "B", "R", "G")):
            g_vec[k + 5 + ci] = pool.get(c, 0) / 10.0
        g_vec[k + 10] = g.players[pid].mulligans / 7.0
        g_vec[k + 11] = float(bool(pv.get("has_lost")))
        k += _PER_PLAYER
    g_vec[k + 0] = g.turn_number / 40.0
    g_vec[k + 1] = float(g.active_player == viewer)
    g_vec[k + 2] = float(g.priority_player == viewer)
    g_vec[k + 3] = len(g.library) / 80.0
    g_vec[k + 4] = len(g.stack) / 6.0
    k += _GAME
    g_vec[k + V.STEP_INDEX.get(g.current_step, 0)] = 1.0
    k += V.N_STEPS
    # combat
    atk = g.combat.attackers or {}
    am_attacking = any(a in g.players[viewer].battlefield for a in atk)
    am_defending = any(info.get("target") == viewer for info in atk.values())
    g_vec[k + 0] = float(am_attacking)
    g_vec[k + 1] = float(am_defending)
    g_vec[k + 2] = len(atk) / 20.0
    g_vec[k + 3] = sum(len(info.get("blockers", [])) for info in atk.values()) / 20.0
    k += _COMBAT
    # pending one-hot + pay sub-state
    pend = g.pending
    if pend is not None:
        pi = V.PENDING_INDEX.get(pend.type)
        if pi is not None:
            g_vec[k + pi] = 1.0
    k += V.N_PENDING
    if pend is not None and pend.type == "pay":
        ctx = pend.context or {}
        need = ctx.get("need", {})
        g_vec[k + 0] = need.get("U", 0) / 4.0
        g_vec[k + 1] = sum(need.values()) / 6.0
        g_vec[k + 2] = ctx.get("generic", 0) / 6.0
        g_vec[k + 3] = 1.0
    k += _PAY
    g_vec[k] = float(builder_progress)
    k += _BUILDER
    g_vec[k], g_vec[k + 1], g_vec[k + 2] = deckout_clock(g, viewer)   # viewer-oriented
    k += _CLOCK

    parts.append(g_vec)
    return np.concatenate(parts).astype(np.float32)


# ── fast object-native path ───────────────────────────────────────────────────
# `encode_observation_ref` builds the policy input by reading the UI view (`current_view`),
# which constructs a rich per-object dict (ability legality, can_play, cycle, modes, full
# asdict of pending/combat) -- the net reads NONE of that, and re-parses the ~12 fields it
# does want back out through ~17 dict.get + a str.lower PER card. Profiling: that round-trip
# (_public_object + _ability_view + the dict reads) was the dominant share of collection.
# The functions below skip the dict entirely: they read CardInstance attributes directly,
# re-deriving the viewer's visibility filter fishrl-side (engine objects used READ-ONLY, so no
# vendored edit). They MUST stay bit-identical to the *_ref path -- see test_encoder_equivalence.


# A card row's first _STATIC_END floats (name one-hot/unknown, type flags, both
# basic-type multihots, text-altered bit) plus the trailing known bit are a PURE
# FUNCTION of (name, type_line, oracle_text, text_variant) -- that is the cache
# key, and "template" does NOT mean immutable over the game. This format's
# text/type-changing effects (Mind Bend / Crystal Spray / Vision Charm) work by
# REWRITING those very attributes: the next encode reads the changed strings,
# computes a different key, and takes a different template; an until-end-of-turn
# change reverting restores the old key. Guarded by the text-change cases in
# test_encoder_equivalence. Distinct keys number a few dozen, while
# `_encode_obj_into` runs ~100x per decision, so the identity-derived part is
# built once per key and row-copied; only the 7 truly per-call floats (stats,
# tapped/entered/controller) are written each time. text_variant is IN the key
# (not derived from the strings): a text-change chain can exist on a card whose
# printed text never contained the changed word, leaving the strings untouched
# while the variant bit fires.
_STATIC_END = V.N_NAMES + 1 + 4 + 5 + 5 + 1
_TEMPLATES: dict = {}


def _static_template(name: str, tl: str, ot: str, tv) -> np.ndarray:
    key = (name, tl, ot, tv)
    t = _TEMPLATES.get(key)
    if t is None:
        t = np.zeros(CARD_F, dtype=np.float32)
        idx = V.NAME_INDEX.get(name)
        if idx is not None:
            t[idx] = 1.0
        else:
            t[V.N_NAMES] = 1.0
        off = V.N_NAMES + 1
        tll = tl.lower()
        t[off + 0] = float("land" in tll)
        t[off + 1] = float("creature" in tll)
        t[off + 2] = float("instant" in tll)
        t[off + 3] = float("sorcery" in tll)
        _basic_multihot(tl, t, off + 4)                    # effective basic land type (5)
        _basic_multihot(ot, t, off + 9)                    # basic types named in text (5)
        t[off + 14] = 1.0 if tv else 0.0                   # text-altered (Island lineage moved)
        t[CARD_F - 1] = 1.0                                # known bit
        _TEMPLATES[key] = t
    return t


def _encode_obj_into(o, viewer: str, v: np.ndarray) -> None:
    """Object-native twin of `_encode_card_into`: write a CardInstance's CARD_F-wide row from
    its attributes. `o is None` encodes the hidden/unknown slot (opp's unseen hand, empty stack
    source). Field-for-field identical to `_encode_card_into` (the equivalence oracle), via a
    cached static template + the 7 dynamic floats."""
    if o is None:
        v[V.N_NAMES] = 1.0          # "unknown/hidden" bit
        return
    np.copyto(v, _static_template(o.name, o.type_line or "", o.oracle_text or "",
                                  text_variant(o)))
    off = _STATIC_END
    v[off + 0] = (o.power or 0) / 10.0
    v[off + 1] = (o.toughness or 0) / 10.0
    v[off + 2] = (o.damage_marked or 0) / 10.0
    v[off + 3] = sum((o.counters or {}).values()) / 10.0
    v[off + 4] = float(bool(o.tapped))
    v[off + 5] = float(bool(o.entered_this_turn))
    v[off + 6] = float(o.controller == viewer)


def _zone_obj(objs: list, n: int, viewer: str) -> np.ndarray:
    rows = np.zeros((n, CARD_F), dtype=np.float32)
    for i, o in enumerate(objs[:n]):
        _encode_obj_into(o, viewer, rows[i])
    return rows.reshape(-1)


def encode_observation(g, viewer: str, builder_progress: float = 0.0) -> np.ndarray:
    """Fast path used in collection: bit-identical to `encode_observation_ref` (guarded by
    tests/test_encoder_equivalence.py) but reads engine objects directly instead of the UI view.

    Visibility filter (mirrors `current_view`): own hand + both battlefields + graveyard + exile
    are fully visible; the opponent's hand is per-card known-or-hidden; the library shows only the
    slots this viewer knows. A hidden card / missing stack source is encoded as the unknown slot."""
    opp = "p2" if viewer == "p1" else "p1"
    obj = g.objects

    parts = [
        _zone_obj([obj[iid] for iid in g.players[viewer].hand], SLOTS["own_hand"], viewer),
        _zone_obj([obj[iid] if viewer in (obj[iid].known_by or []) else None
                   for iid in g.players[opp].hand], SLOTS["opp_hand"], viewer),
        _zone_obj([obj[iid] for iid in g.players[viewer].battlefield], SLOTS["own_bf"], viewer),
        _zone_obj([obj[iid] for iid in g.players[opp].battlefield], SLOTS["opp_bf"], viewer),
        _zone_obj([obj[iid] for iid in g.graveyard], SLOTS["graveyard"], viewer),
        _zone_obj([obj[iid] for iid in g.exile], SLOTS["exile"], viewer),
        _zone_obj([obj.get(s.source_instance_id) for s in g.stack], SLOTS["stack"], viewer),
        _zone_obj([obj[s.instance_id] for s in g.library if s.known_by.get(viewer)],
                  SLOTS["library"], viewer),
    ]

    g_vec = np.zeros(_GLOBALS, dtype=np.float32)
    k = 0
    for pid in (viewer, opp):
        p = g.players[pid]
        lands = [obj[iid] for iid in p.battlefield
                 if "land" in (obj[iid].type_line or "").lower()]
        pool = p.mana_pool
        g_vec[k + 0] = p.life / 20.0
        g_vec[k + 1] = len(p.hand) / 12.0
        g_vec[k + 2] = len(lands) / 20.0
        g_vec[k + 3] = sum(1 for c in lands if not c.tapped) / 20.0
        g_vec[k + 4] = sum(pool.values()) / 10.0
        for ci, c in enumerate(("W", "U", "B", "R", "G")):
            g_vec[k + 5 + ci] = pool.get(c, 0) / 10.0
        g_vec[k + 10] = p.mulligans / 7.0
        g_vec[k + 11] = float(bool(p.has_lost))
        k += _PER_PLAYER
    g_vec[k + 0] = g.turn_number / 40.0
    g_vec[k + 1] = float(g.active_player == viewer)
    g_vec[k + 2] = float(g.priority_player == viewer)
    g_vec[k + 3] = len(g.library) / 80.0
    g_vec[k + 4] = len(g.stack) / 6.0
    k += _GAME
    g_vec[k + V.STEP_INDEX.get(g.current_step, 0)] = 1.0
    k += V.N_STEPS
    atk = g.combat.attackers or {}
    am_attacking = any(a in g.players[viewer].battlefield for a in atk)
    am_defending = any(info.get("target") == viewer for info in atk.values())
    g_vec[k + 0] = float(am_attacking)
    g_vec[k + 1] = float(am_defending)
    g_vec[k + 2] = len(atk) / 20.0
    g_vec[k + 3] = sum(len(info.get("blockers", [])) for info in atk.values()) / 20.0
    k += _COMBAT
    pend = g.pending
    if pend is not None:
        pi = V.PENDING_INDEX.get(pend.type)
        if pi is not None:
            g_vec[k + pi] = 1.0
    k += V.N_PENDING
    if pend is not None and pend.type == "pay":
        ctx = pend.context or {}
        need = ctx.get("need", {})
        g_vec[k + 0] = need.get("U", 0) / 4.0
        g_vec[k + 1] = sum(need.values()) / 6.0
        g_vec[k + 2] = ctx.get("generic", 0) / 6.0
        g_vec[k + 3] = 1.0
    k += _PAY
    g_vec[k] = float(builder_progress)
    k += _BUILDER
    g_vec[k], g_vec[k + 1], g_vec[k + 2] = deckout_clock(g, viewer)   # viewer-oriented
    k += _CLOCK

    parts.append(g_vec)
    return np.concatenate(parts).astype(np.float32)
