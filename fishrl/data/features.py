"""Feature encoders for the two outcome estimators, plus the hand-guesser label.

Three information levels feed the learning stack:
  * perspective (per-seat, fair) — :func:`fishrl.obs.encoder.encode_observation`
    (feeds the guesser and the actor; defined elsewhere).
  * privileged / god — :func:`encode_god`: every zone with full identity, including
    BOTH hands and the FULL ordered library. Feeds the privileged critic.
  * public / mutual — :func:`encode_public`: from :func:`state.spectator_view`
    (only what both players know). Feeds the public estimator.

Both estimator encoders are **p1-oriented** (seat-agnostic, fixed orientation) and
output features for a head that predicts P(p1 wins). The guesser label
:func:`opponent_hand_counts` is a god-state quantity used ONLY as a supervised
target — never as a model input.
"""
from __future__ import annotations

import numpy as np

from fishrl.forgetful_fish.state import _public_object, _is_land, spectator_view
from fishrl.obs.encoder import (CARD_F, _encode_card, _fill_zone, _gy_dicts, _gy_objs, _zone,
                                _zone_obj, deckout_clock)
from fishrl.obs import vocab as V

# ── god (privileged) layout ───────────────────────────────────────────────────
GOD_SLOTS = {
    "p1_hand": 12, "p2_hand": 12, "p1_bf": 34, "p2_bf": 34,
    "graveyard": 32, "exile": 8, "stack": 6, "library": 64,
}
_GOD_ROWS = sum(GOD_SLOTS.values())
_GOD_PER_PLAYER = 12       # life, hand_count, lands, untapped, pool_total, pool×5, mulligans, has_lost
_GOD_GAME = 5              # turn, active_is_p1, priority_is_p1, library_count, stack_depth
# The critic needs the deckout clock too: it sees library size only as len/80.0, the same
# smooth float the actor cannot extract parity from -- so without this it cannot learn the
# value of a parity flip, and GAE cannot credit the draw that causes one. p1-oriented.
_CLOCK = 3                 # library parity, next_drawer_is_p1, p1_decks_first
GOD_DIM = _GOD_ROWS * CARD_F + _GOD_PER_PLAYER * 2 + _GOD_GAME + _CLOCK

# ── public (mutual-knowledge) layout ──────────────────────────────────────────
PUB_SLOTS = {
    "p1_hand": 12, "p2_hand": 12, "p1_bf": 34, "p2_bf": 34,
    "graveyard": 32, "exile": 8, "stack": 6, "library": 8,
}
_PUB_ROWS = sum(PUB_SLOTS.values())
_PUB_PER_PLAYER = 12
_PUB_GAME = 5
PUB_DIM = _PUB_ROWS * CARD_F + _PUB_PER_PLAYER * 2 + _PUB_GAME + _CLOCK

# ── hands (critic_view="hands", 2026-08-21) layout ────────────────────────────
# The public layout with BOTH hands fully visible and the mutual-knowledge library
# rows dropped. Benchmarked on 88.7k v1.3-mirror states (3 seeds): Brier .183 vs
# public .190 vs god .188 -- hands carry the value the library order drowned; the
# known-top slots added nothing. Same per-player / game / clock tail as public.
HANDS_SLOTS = {
    "p1_hand": 12, "p2_hand": 12, "p1_bf": 34, "p2_bf": 34,
    "graveyard": 32, "exile": 8, "stack": 6,
}
_HANDS_ROWS = sum(HANDS_SLOTS.values())
HANDS_DIM = _HANDS_ROWS * CARD_F + _PUB_PER_PLAYER * 2 + _PUB_GAME + _CLOCK
PUBLIC_FAMILY = ("public", "hands")     # critic views that ride the pub_feat slot


# ── per-name count block (Config.obs_counts, 2026-08-22) ──────────────────────
# The entity encoder pools each zone by masked mean + max: a per-name FRACTION and a
# presence bit, never an absolute count -- and the scalar tail carries no graveyard or
# exile count. So "how many Dandâns / Lapses are left in the deck" was not linearly
# recoverable from any view (the same failure class as library parity from len/80).
# When enabled, a COUNT_DIM block is appended to the globals tail of the actor input
# (after the bookkeeper belief) and of the hands critic's features:
#   actor : per-name cards the VIEWER cannot see (library + unknown opponent hand),
#           = deck copies - visible - known, the bookkeeper's own `remaining` vector
#   hands : per-name cards in the LIBRARY (exact; both hands are visible to it)
#   both  : + graveyard count / 40, exile count / 8
# Layout is versioned in the checkpoint config (obs_counts); the toggle is set by
# build_models / the collector worker init, like set_public_view. Adding it to a
# trained net is an in-place zero-column widening of the head's first Linear
# (fishrl/train/widen_counts.py) -- function-identical at the seam, no restart.
COUNT_DIM = V.N_NAMES + 2
_COUNTS = False


# ── FoF split-context block (Config.obs_split, 2026-08-26) ────────────────────
# During a Fact-or-Fiction resolution the five revealed cards leave the library
# (they live only in g.pending.context) and the splitter's in-progress pile
# arrangement lived only in the env-side CompoundBuilder — so NEITHER the actor
# NOR the critic could see what was being split: measured at it=122.7k, critic V
# is EXACTLY flat across the five PICK toggles (mean |dV| 0.0000, n=1,587) with
# the whole ~0.14 swing landing after the opponent's pile choice, and the 0-5
# degenerate-split rate sat frozen (17% vs h1.3 / 44% mirror) across three
# checkpoints while the critic's pricing of it sharpened. The builder now mirrors
# its live arrangement into the pending context (engine's own update_fof_split +
# an env-owned "assigned" key), and this block encodes it:
#   [4]        who acts: split by p1 / split by p2 / choose by p1 / choose by p2
#              (viewer-oriented me/opp for the actor's tail)
#   [N_NAMES]  pile-1 per-name counts (assigned so far / offered pile 1)
#   [N_NAMES]  pile-2 per-name counts
#   [N_NAMES]  still-unassigned per-name counts (zeros at fof_choose)
# All zeros outside a FoF resolution — so a zero-column widening of a trained net
# (fishrl/train/widen_split.py) is function-identical everywhere else, exactly
# like the count block. Appended AFTER the count block on the actor's belief tail
# and the hands critic's features; versioned in the checkpoint config (obs_split).
SPLIT_DIM = 4 + 3 * V.N_NAMES
_SPLIT = False


def set_count_block(enabled: bool) -> None:
    global _COUNTS
    _COUNTS = bool(enabled)


def count_block_on() -> bool:
    return _COUNTS


def set_split_block(enabled: bool) -> None:
    global _SPLIT
    _SPLIT = bool(enabled)


def split_block_on() -> bool:
    return _SPLIT


def split_live(g) -> bool:
    """True while a fof_split arrangement is in progress — the one pending whose
    context (and therefore the split block) changes WITHIN a decision_id, so the
    collectors' compound-substep encode dedupe must not reuse the cached row."""
    p = getattr(g, "pending", None)
    return _SPLIT and p is not None and p.type == "fof_split"


def split_context_block(g, viewer: str | None = None) -> np.ndarray:
    """The SPLIT_DIM block for the current state. `viewer=None` gives the critic's
    p1-oriented flags; a seat gives the actor's me/opp orientation."""
    out = np.zeros(SPLIT_DIM, dtype=np.float32)
    pend = getattr(g, "pending", None)
    if pend is None or pend.type not in ("fof_split", "fof_choose"):
        return out
    ctx = pend.context or {}
    mine = (pend.player == ("p1" if viewer is None else viewer))
    out[(0 if mine else 1) + (0 if pend.type == "fof_split" else 2)] = 1.0
    if pend.type == "fof_split":
        revealed = ctx.get("revealed", ())
        assigned = set(ctx.get("assigned", ()))          # env-owned mirror of the builder
        pile2 = set(i for i in ctx.get("pile2", ()) if i in assigned)
        pile1 = [i for i in revealed if i in assigned and i not in pile2]
        unassigned = [i for i in revealed if i not in assigned]
    else:
        pile1, pile2 = ctx.get("pile1_ids", ()), ctx.get("pile2_ids", ())
        unassigned = ()
    _name_counts_into(g, pile1, out[4:4 + V.N_NAMES])
    _name_counts_into(g, pile2, out[4 + V.N_NAMES:4 + 2 * V.N_NAMES])
    _name_counts_into(g, unassigned, out[4 + 2 * V.N_NAMES:4 + 3 * V.N_NAMES])
    return out


# ── decision-context pack (Config.obs_ctx, 2026-08-26 observability audit) ────
# The audit (journal 2026-08-26-observability-audit.md) aliasing-proved four more
# blind spots: stack-spell TARGETS invisible to actor+critic+mask (an opponent
# Spray at my fish vs my land = bit-identical encodings; 28.9 response rows/g),
# Mystical Tutor searches blind (eligible mapping unencoded beyond the top-8
# library rows), compound ARRANGEMENTS beyond FoF blind (scry/reorder/putback:
# only a progress scalar; a Ponder reorder has only its first pick informed),
# and the blocker focus env-side only. This pack closes them:
#   CTX (shared, appended to the actor belief tail viewer-oriented and to the
#        hands critic p1-oriented):
#     [2 x 25]  top-two stack objects' targets: name one-hot(20) + is-player +
#               controller-is-viewer + is-bf-creature + is-bf-land + is-on-stack
#     [20]      search_library eligible per-name counts (searcher/critic only —
#               the library is hidden from the non-searcher)
#     [61]      builder arrangement: pileA counts(20) + pileB counts(20) +
#               last-placed one-hot(21) — mirrored into pending.context by the
#               CompoundBuilder for scry (top/bottom), reorder/putback/bottom/
#               discard (placed order) and declare_attackers; zeros during
#               fof_split (obs_split's block owns that pending)
#     [2]       blocker focus: has_focus + (focused attacker index+1)/CMP_K
#   CRITIC_CTX_EXTRA (hands tail only — the actor already carries these in its
#   perspective globals): step one-hot + pending one-hot + combat(4, p1-
#   oriented) + pay(5 incl. payer_is_p1).
# All zeros outside the relevant pendings, so widen_ctx.py's zero-column
# widening is function-identical at the seam (the obs_split pattern). The same
# flag gates two env-side MASK-ORDER remaps (masking.pick_list): search_library
# and choose_graveyard eligibles become name-sorted, giving PICK_SINGLE indices
# stable learnable semantics (the old order was engine/library order — blind —
# and the graveyard one desynced from the value-first encoded rows at gy>32).
CTX_STACK_OBJ = V.N_NAMES + 5
CTX_DIM = 2 * CTX_STACK_OBJ + V.N_NAMES + (2 * V.N_NAMES + V.N_NAMES + 1) + 2
CRITIC_CTX_EXTRA = V.N_STEPS + V.N_PENDING + 4 + 5
_CTX = False

# pendings whose builder arrangement is mirrored into context (fof_split is
# handled by the obs_split block; declare_blockers mirrors focus + used blockers)
CTX_BUILDER_TYPES = ("scry", "reorder", "putback", "bottom", "discard",
                     "declare_attackers", "declare_blockers")


def set_ctx_block(enabled: bool) -> None:
    global _CTX
    _CTX = bool(enabled)


def ctx_block_on() -> bool:
    return _CTX


def ctx_live(g) -> bool:
    """True while a pending whose CONTEXT mutates within one decision_id is up
    (builder mirrors + blocker focus) — the collectors' compound-substep encode
    dedupe must re-encode these rows, like split_live."""
    p = getattr(g, "pending", None)
    return _CTX and p is not None and p.type in CTX_BUILDER_TYPES


def _target_slot(g, so, viewer: str, out: np.ndarray) -> None:
    """One stack object's target into a CTX_STACK_OBJ slice (zeros if untargeted)."""
    tgts = getattr(so, "targets", None) or []
    if not tgts:
        return
    t = tgts[0]
    o = V.N_NAMES
    if t.get("type") == "player":
        out[o + 0] = 1.0
        out[o + 1] = float(t.get("id") == viewer)
        return
    iid = t.get("id")
    obj = g.objects.get(iid)
    if obj is None:
        return
    idx = V.NAME_INDEX.get(obj.name)
    if idx is not None:
        out[idx] = 1.0
    out[o + 1] = float(getattr(obj, "controller", None) == viewer)
    on_bf = any(iid in g.players[p].battlefield for p in ("p1", "p2"))
    tl = (obj.type_line or "")
    out[o + 2] = float(on_bf and "Creature" in tl)
    out[o + 3] = float(on_bf and "Land" in tl)
    out[o + 4] = float(any(s.source_instance_id == iid for s in g.stack))


def ctx_block(g, viewer: str | None = None) -> np.ndarray:
    """The CTX_DIM pack for the current state. `viewer=None` gives the critic's
    p1 orientation; a seat gives the actor's."""
    out = np.zeros(CTX_DIM, dtype=np.float32)
    vw = viewer or "p1"
    # stack targets: top two objects (top last)
    for slot, so in enumerate(reversed(g.stack[-2:])):
        base = slot * CTX_STACK_OBJ
        _target_slot(g, so, vw, out[base:base + CTX_STACK_OBJ])
    k = 2 * CTX_STACK_OBJ
    pend = getattr(g, "pending", None)
    ctx = (pend.context or {}) if pend is not None else {}
    # search eligibility (hidden info: searcher's eyes only; the critic sees it)
    if (pend is not None and pend.type == "search_library"
            and (viewer is None or pend.player == viewer)):
        _name_counts_into(g, ctx.get("eligible", ()), out[k:k + V.N_NAMES])
    k += V.N_NAMES
    # builder arrangement (the acting seat's own in-progress state; critic sees it)
    if (pend is not None and pend.type in CTX_BUILDER_TYPES
            and (viewer is None or pend.player == viewer)):
        _name_counts_into(g, ctx.get("placed_a", ()), out[k:k + V.N_NAMES])
        _name_counts_into(g, ctx.get("placed_b", ()), out[k + V.N_NAMES:k + 2 * V.N_NAMES])
        last = ctx.get("last_placed")
        lo = g.objects.get(last) if last else None
        li = V.NAME_INDEX.get(lo.name) if lo is not None else None
        out[k + 2 * V.N_NAMES + (li if li is not None else V.N_NAMES)] = 1.0
        if pend.type == "declare_blockers":
            focus = ctx.get("focus")
            if focus is not None:
                out[k + 3 * V.N_NAMES + 1] = 1.0
                out[k + 3 * V.N_NAMES + 2] = (int(focus) + 1) / 20.0
    return out


def critic_ctx_extra(g) -> np.ndarray:
    """Hands-critic-only context: step/pending one-hots + p1-oriented combat +
    pay sub-state (the actor's perspective globals already carry all of these)."""
    out = np.zeros(CRITIC_CTX_EXTRA, dtype=np.float32)
    si = V.STEP_INDEX.get(g.current_step, 0)
    out[si] = 1.0
    k = V.N_STEPS
    pend = getattr(g, "pending", None)
    if pend is not None:
        pi = V.PENDING_INDEX.get(pend.type)
        if pi is not None:
            out[k + pi] = 1.0
    k += V.N_PENDING
    atk = g.combat.attackers or {}
    out[k + 0] = float(any(a in g.players["p1"].battlefield for a in atk))
    out[k + 1] = float(any(info.get("target") == "p1" for info in atk.values()))
    out[k + 2] = len(atk) / 20.0
    out[k + 3] = sum(len(info.get("blockers", [])) for info in atk.values()) / 20.0
    k += 4
    if pend is not None and pend.type == "pay":
        ctx = pend.context or {}
        need = ctx.get("need", {})
        out[k + 0] = need.get("U", 0) / 4.0
        out[k + 1] = sum(need.values()) / 6.0
        out[k + 2] = ctx.get("generic", 0) / 6.0
        out[k + 3] = 1.0
        out[k + 4] = float(pend.player == "p1")
    return out


def belief_dim() -> int:
    """Width of the actor's belief tail: the 20-dim bookkeeper (+ COUNT_DIM when on,
    + SPLIT_DIM when on, + CTX_DIM when on)."""
    return (V.N_NAMES + (COUNT_DIM if _COUNTS else 0) + (SPLIT_DIM if _SPLIT else 0)
            + (CTX_DIM if _CTX else 0))


def hands_dim() -> int:
    return (HANDS_DIM + (COUNT_DIM if _COUNTS else 0) + (SPLIT_DIM if _SPLIT else 0)
            + ((CTX_DIM + CRITIC_CTX_EXTRA) if _CTX else 0))


def pub_dim_for(view: str) -> int:
    return hands_dim() if view == "hands" else PUB_DIM


# Index of `turn_number / 40` in a critic feature vector: the game block follows the
# per-player block (12 x 2) right after the card rows, for every view.
_TURN_OFF = 12 * 2


def turn_index_for(view: str) -> int:
    rows = {"god": _GOD_ROWS, "public": _PUB_ROWS, "hands": _HANDS_ROWS}[view]
    return rows * CARD_F + _TURN_OFF


# Per-turn calibration buckets (telemetry): (label, lo, hi) inclusive turn ranges.
TURN_BUCKETS = (("t1_10", 1, 10), ("t11_20", 11, 20), ("t21_30", 21, 30), ("t31p", 31, 999))
TURN_MAX = 40        # per-turn telemetry: turns 1..TURN_MAX-1 individually, TURN_MAX+ pooled


def _zone_counts_tail(g) -> list:
    return [len(g.graveyard) / 40.0, len(g.exile) / 8.0]


def library_counts(g) -> np.ndarray:
    """Per-name count of cards in the shared library (exact; the hands critic's block)."""
    out = np.zeros(V.N_NAMES, dtype=np.float32)
    _name_counts_into(g, (s.instance_id for s in g.library), out)
    return out


def opponent_hand_counts(g, viewer: str) -> np.ndarray:
    """Per-name count of the opponent's hand (the guesser's supervised target)."""
    opp = "p2" if viewer == "p1" else "p1"
    counts = np.zeros(V.N_NAMES, dtype=np.float32)
    for iid in g.players[opp].hand:
        idx = V.NAME_INDEX.get(g.objects[iid].name)
        if idx is not None:
            counts[idx] += 1.0
    return counts


def _name_idx_of(g) -> dict:
    """Per-game iid -> vocab name-index cache (lazy; -2 marks no-index/missing).
    A card's NAME never changes (text changes rewrite type/oracle text only), so the
    mapping is stable for the instance's lifetime. Plain attr, not serialized --
    same pattern as `_bk_copies`."""
    ni = getattr(g, "_bk_nameidx", None)
    if ni is None:
        ni = {}
        g._bk_nameidx = ni
    return ni


def _name_counts_into(g, iids, out: np.ndarray) -> None:
    ni = _name_idx_of(g)
    obj = g.objects
    for iid in iids:
        idx = ni.get(iid, -1)
        if idx == -1:                       # not cached yet (-2 caches a definite miss)
            o = obj.get(iid)
            idx = -2 if o is None else V.NAME_INDEX.get(o.name, -2)
            ni[iid] = idx
        if idx >= 0:
            out[idx] += 1.0


def bookkeeper_counts(g, viewer: str) -> np.ndarray:
    """The analytic hand bookkeeper: per-name EXPECTED counts of the opponent's hand,
    computed purely from what `viewer` can see (belief_mode="bookkeeper"'s fill for the
    actor's 20-dim belief slot — the zero-parameter replacement for the HandGuesser).

    Formula (the Guesser Deposition's "perfect bookkeeper"):
        known + (handn - known_total) * remaining / remaining_total
    where, per card name,
      * known      = opponent hand cards the viewer has been shown (known_by),
      * remaining  = deck copies - visible - known, with visible = every card whose
                     identity the viewer can see AND that cannot be in the opponent's
                     hand (own hand, both battlefields, graveyard, exile, stack
                     sources, viewer-known library slots),
      * handn      = the opponent's public hand size.
    remaining is non-negative by construction (each physical card is counted once and
    `visible` excludes the opponent's hand). If the viewer accounts for every unseen
    card (remaining_total == 0) the proportional term is zero and the output is
    exactly `known`."""
    opp = "p2" if viewer == "p1" else "p1"
    copies = getattr(g, "_bk_copies", None)
    if copies is None:                      # per-game cache; plain attr, not serialized
        copies = np.zeros(V.N_NAMES, dtype=np.float32)
        for o in g.objects.values():
            idx = V.NAME_INDEX.get(o.name)
            if idx is not None:
                copies[idx] += 1.0
        g._bk_copies = copies
    known = np.zeros(V.N_NAMES, dtype=np.float32)
    visible = np.zeros(V.N_NAMES, dtype=np.float32)
    obj = g.objects
    for iid in g.players[opp].hand:
        o = obj.get(iid)
        if o is not None and viewer in (o.known_by or []):
            idx = V.NAME_INDEX.get(o.name)
            if idx is not None:
                known[idx] += 1.0
    _name_counts_into(g, g.players[viewer].hand, visible)
    for pid in ("p1", "p2"):
        _name_counts_into(g, g.players[pid].battlefield, visible)
    _name_counts_into(g, g.graveyard, visible)
    _name_counts_into(g, g.exile, visible)
    _name_counts_into(g, (s.source_instance_id for s in g.stack), visible)
    _name_counts_into(g, (s.instance_id for s in g.library
                          if s.known_by.get(viewer)), visible)
    fill = len(g.players[opp].hand) - float(known.sum())
    remaining = np.maximum(copies - visible - known, 0.0)
    rtot = float(remaining.sum())
    if fill <= 0.0 or rtot <= 0.0:
        belief = known
    else:
        belief = (known + fill * remaining / rtot).astype(np.float32)
    if not _COUNTS and not _SPLIT and not _CTX:
        return belief
    # count block: what the viewer cannot see, by name (library + unknown opp hand);
    # split block: the live FoF pile arrangement; ctx pack: stack targets / search
    # eligibility / builder arrangement / blocker focus (all viewer-oriented)
    parts = [belief]
    if _COUNTS:
        parts += [remaining, np.asarray(_zone_counts_tail(g), dtype=np.float32)]
    if _SPLIT:
        parts.append(split_context_block(g, viewer))
    if _CTX:
        parts.append(ctx_block(g, viewer))
    return np.concatenate(parts).astype(np.float32, copy=False)


def _player_scalars(life, hand_count, bf_cards, pool, mulligans, has_lost) -> list:
    lands = [c for c in bf_cards if "land" in (c.get("type_line") or "").lower()]
    out = [life / 20.0, hand_count / 12.0, len(lands) / 20.0,
           sum(1 for c in lands if not c.get("tapped")) / 20.0,
           sum(pool.values()) / 10.0]
    out += [pool.get(c, 0) / 10.0 for c in ("W", "U", "B", "R", "G")]
    out += [mulligans / 7.0, float(bool(has_lost))]
    return out


def _obj_dicts(g, ids) -> list:
    return [_public_object(g.objects[iid]) for iid in ids if iid in g.objects]


def encode_god_ref(g) -> np.ndarray:
    """REFERENCE god encoder (dict path through _public_object) -- the contract `encode_god`
    must stay bit-identical to. Kept as the equivalence oracle (test_encoder_equivalence)."""
    bf = {pid: _obj_dicts(g, g.players[pid].battlefield) for pid in ("p1", "p2")}  # reused below
    parts = [
        _zone(_obj_dicts(g, g.players["p1"].hand), GOD_SLOTS["p1_hand"], "p1"),
        _zone(_obj_dicts(g, g.players["p2"].hand), GOD_SLOTS["p2_hand"], "p1"),
        _zone(bf["p1"], GOD_SLOTS["p1_bf"], "p1"),
        _zone(bf["p2"], GOD_SLOTS["p2_bf"], "p1"),
        _zone(_gy_dicts(_obj_dicts(g, g.graveyard), GOD_SLOTS["graveyard"]), GOD_SLOTS["graveyard"], "p1"),
        _zone(_obj_dicts(g, g.exile), GOD_SLOTS["exile"], "p1"),
        _zone([_public_object(g.objects[s.source_instance_id])
               for s in g.stack if s.source_instance_id in g.objects],
              GOD_SLOTS["stack"], "p1"),
        _zone(_obj_dicts(g, [s.instance_id for s in g.library]), GOD_SLOTS["library"], "p1"),
    ]
    gv = []
    for pid in ("p1", "p2"):
        p = g.players[pid]
        gv += _player_scalars(p.life, len(p.hand), bf[pid], p.mana_pool, p.mulligans, p.has_lost)
    gv += [g.turn_number / 40.0, float(g.active_player == "p1"),
           float(g.priority_player == "p1"), len(g.library) / 80.0, len(g.stack) / 6.0]
    gv += list(deckout_clock(g, "p1"))                # deckout clock (p1-oriented)
    parts.append(np.asarray(gv, dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


def _player_scalars_obj(life, hand_count, bf_objs, pool, mulligans, has_lost) -> list:
    """Object-native twin of `_player_scalars` (reads CardInstance attrs, no dict)."""
    lands = [o for o in bf_objs if "land" in (o.type_line or "").lower()]
    out = [life / 20.0, hand_count / 12.0, len(lands) / 20.0,
           sum(1 for o in lands if not o.tapped) / 20.0,
           sum(pool.values()) / 10.0]
    out += [pool.get(c, 0) / 10.0 for c in ("W", "U", "B", "R", "G")]
    out += [mulligans / 7.0, float(bool(has_lost))]
    return out


def encode_god(g) -> np.ndarray:
    """Privileged, fully-observed, p1-oriented feature vector (GOD_DIM). Fast object-native path:
    god sees full identity in every zone (no visibility filter), so it reads CardInstance objects
    directly and skips the _public_object/_ability_view dict build. Encodes into ONE
    preallocated vector (no per-zone alloc / concatenate / astype). Bit-identical to
    `encode_god_ref` -- guarded by test_encoder_equivalence."""
    obj = g.objects
    bf = {pid: [obj[iid] for iid in g.players[pid].battlefield if iid in obj] for pid in ("p1", "p2")}
    out = np.zeros(GOD_DIM, dtype=np.float32)
    rows = out[:_GOD_ROWS * CARD_F].reshape(_GOD_ROWS, CARD_F)

    b = _fill_zone(rows, 0, [obj[iid] for iid in g.players["p1"].hand if iid in obj],
                   GOD_SLOTS["p1_hand"], "p1")
    b = _fill_zone(rows, b, [obj[iid] for iid in g.players["p2"].hand if iid in obj],
                   GOD_SLOTS["p2_hand"], "p1")
    b = _fill_zone(rows, b, bf["p1"], GOD_SLOTS["p1_bf"], "p1")
    b = _fill_zone(rows, b, bf["p2"], GOD_SLOTS["p2_bf"], "p1")
    b = _fill_zone(rows, b, _gy_objs([obj[iid] for iid in g.graveyard if iid in obj],
                                     GOD_SLOTS["graveyard"]), GOD_SLOTS["graveyard"], "p1")
    b = _fill_zone(rows, b, [obj[iid] for iid in g.exile if iid in obj],
                   GOD_SLOTS["exile"], "p1")
    b = _fill_zone(rows, b, [obj[s.source_instance_id] for s in g.stack
                             if s.source_instance_id in obj], GOD_SLOTS["stack"], "p1")
    _fill_zone(rows, b, [obj[s.instance_id] for s in g.library if s.instance_id in obj],
               GOD_SLOTS["library"], "p1")

    gv = []
    for pid in ("p1", "p2"):
        p = g.players[pid]
        gv += _player_scalars_obj(p.life, len(p.hand), bf[pid], p.mana_pool, p.mulligans, p.has_lost)
    gv += [g.turn_number / 40.0, float(g.active_player == "p1"),
           float(g.priority_player == "p1"), len(g.library) / 80.0, len(g.stack) / 6.0]
    gv += list(deckout_clock(g, "p1"))                # deckout clock (p1-oriented)
    out[_GOD_ROWS * CARD_F:] = gv
    return out


def encode_public_ref(g) -> np.ndarray:
    """REFERENCE public encoder (dict path through spectator_view) -- the contract
    `encode_public` must stay bit-identical to (test_encoder_equivalence)."""
    view = spectator_view(g)
    pl = view["players"]

    def known_only(cards):
        return [c for c in cards if c.get("known")]

    parts = [
        _zone(pl["p1"].get("hand", []), PUB_SLOTS["p1_hand"], "p1"),
        _zone(pl["p2"].get("hand", []), PUB_SLOTS["p2_hand"], "p1"),
        _zone(pl["p1"].get("battlefield", []), PUB_SLOTS["p1_bf"], "p1"),
        _zone(pl["p2"].get("battlefield", []), PUB_SLOTS["p2_bf"], "p1"),
        _zone(_gy_dicts(view.get("graveyard", []), PUB_SLOTS["graveyard"]), PUB_SLOTS["graveyard"], "p1"),
        _zone(view.get("exile", []), PUB_SLOTS["exile"], "p1"),
        _zone([s.get("card") for s in view.get("stack", [])], PUB_SLOTS["stack"], "p1"),
        _zone(known_only(view.get("library", [])), PUB_SLOTS["library"], "p1"),
    ]
    gv = []
    for pid in ("p1", "p2"):
        pv = pl[pid]
        gv += _player_scalars(pv.get("life", 0), pv.get("hand_count", 0),
                              pv.get("battlefield", []), pv.get("mana_pool", {}),
                              g.players[pid].mulligans, pv.get("has_lost"))
    gv += [g.turn_number / 40.0, float(g.active_player == "p1"),
           float(g.priority_player == "p1"), len(g.library) / 80.0, len(g.stack) / 6.0]
    gv += list(deckout_clock(g, "p1"))                # deckout clock (p1-oriented)
    parts.append(np.asarray(gv, dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


# The public estimator is DIAGNOSTIC-ONLY (never feeds the policy/advantages). When a run
# disables it (Config.train_public=False) we skip the per-decision encode entirely and buffer
# a shared zero vector instead -- ~40us/decision saved on the collection hot path, at the cost
# of the pub calibration telemetry. Toggled once per process (main + each pcollect worker).
_PUBLIC_ENCODING = True
_ZERO_PUB = np.zeros(PUB_DIM, dtype=np.float32)
_PUB_VIEW = "public"


def set_public_view(view: str) -> None:
    """Which public-family encoding `encode_critic_pub` produces: "public" (mutual
    knowledge, legacy PublicEstimator / v3 critic) or "hands" (both hands visible).
    Set from cfg.critic_view at trainer init, in the collector workers and by the
    eval loaders; defaults to "public" so legacy checkpoints need no call."""
    global _PUB_VIEW
    if view not in PUBLIC_FAMILY:
        raise ValueError(f"public view must be one of {PUBLIC_FAMILY}, got {view!r}")
    _PUB_VIEW = view


def public_view() -> str:
    return _PUB_VIEW


def encode_critic_pub(g) -> np.ndarray:
    """The pub_feat row for the configured public-family view."""
    return encode_hands(g) if _PUB_VIEW == "hands" else encode_public(g)


def encode_hands(g) -> np.ndarray:
    """p1-oriented feature vector (HANDS_DIM): both hands in full, both
    battlefields, graveyard, exile and the stack; no library rows. Gated by the
    same set_public_encoding switch as encode_public."""
    if not _PUBLIC_ENCODING:
        return np.zeros(hands_dim(), dtype=np.float32)
    obj = g.objects
    bf = {pid: [obj[iid] for iid in g.players[pid].battlefield if iid in obj]
          for pid in ("p1", "p2")}
    out = np.zeros(hands_dim(), dtype=np.float32)
    rows = out[:_HANDS_ROWS * CARD_F].reshape(_HANDS_ROWS, CARD_F)
    b = _fill_zone(rows, 0, [obj[iid] for iid in g.players["p1"].hand if iid in obj],
                   HANDS_SLOTS["p1_hand"], "p1")
    b = _fill_zone(rows, b, [obj[iid] for iid in g.players["p2"].hand if iid in obj],
                   HANDS_SLOTS["p2_hand"], "p1")
    b = _fill_zone(rows, b, bf["p1"], HANDS_SLOTS["p1_bf"], "p1")
    b = _fill_zone(rows, b, bf["p2"], HANDS_SLOTS["p2_bf"], "p1")
    b = _fill_zone(rows, b, _gy_objs([obj[iid] for iid in g.graveyard if iid in obj],
                                     HANDS_SLOTS["graveyard"]), HANDS_SLOTS["graveyard"], "p1")
    b = _fill_zone(rows, b, [obj[iid] for iid in g.exile if iid in obj],
                   HANDS_SLOTS["exile"], "p1")
    _fill_zone(rows, b, [obj.get(s.source_instance_id) for s in g.stack],
               HANDS_SLOTS["stack"], "p1")
    gv = []
    for pid in ("p1", "p2"):
        p = g.players[pid]
        gv += _player_scalars_obj(p.life, len(p.hand), bf[pid], p.mana_pool,
                                  p.mulligans, p.has_lost)
    gv += [g.turn_number / 40.0, float(g.active_player == "p1"),
           float(g.priority_player == "p1"), len(g.library) / 80.0, len(g.stack) / 6.0]
    gv += list(deckout_clock(g, "p1"))
    if _COUNTS:
        gv += list(library_counts(g)) + _zone_counts_tail(g)
    k = _HANDS_ROWS * CARD_F
    out[k:k + len(gv)] = gv
    k += len(gv)
    if _SPLIT:
        out[k:k + SPLIT_DIM] = split_context_block(g)          # p1-oriented (critic)
        k += SPLIT_DIM
    if _CTX:
        out[k:k + CTX_DIM] = ctx_block(g)
        out[k + CTX_DIM:k + CTX_DIM + CRITIC_CTX_EXTRA] = critic_ctx_extra(g)
    return out


def set_public_encoding(enabled: bool) -> None:
    global _PUBLIC_ENCODING
    _PUBLIC_ENCODING = bool(enabled)


def encode_public(g) -> np.ndarray:
    """Mutual-knowledge, p1-oriented feature vector (PUB_DIM). Fast object-native
    path re-deriving spectator_view's filters (engine objects read-only):

      * a HAND card is mutual knowledge iff the NON-owner knows it,
      * both battlefields / graveyard / exile / stack are public,
      * a LIBRARY slot shows iff BOTH players know it.

    Bit-identical to `encode_public_ref` -- guarded by test_encoder_equivalence."""
    if not _PUBLIC_ENCODING:
        return _ZERO_PUB                                   # public head disabled -> skip the encode
    obj = g.objects

    def hand_objs(pid, other):
        return [obj[iid] if other in (obj[iid].known_by or []) else None
                for iid in g.players[pid].hand]

    bf = {pid: [obj[iid] for iid in g.players[pid].battlefield if iid in obj]
          for pid in ("p1", "p2")}
    out = np.zeros(PUB_DIM, dtype=np.float32)
    rows = out[:_PUB_ROWS * CARD_F].reshape(_PUB_ROWS, CARD_F)

    b = _fill_zone(rows, 0, hand_objs("p1", "p2"), PUB_SLOTS["p1_hand"], "p1")
    b = _fill_zone(rows, b, hand_objs("p2", "p1"), PUB_SLOTS["p2_hand"], "p1")
    b = _fill_zone(rows, b, bf["p1"], PUB_SLOTS["p1_bf"], "p1")
    b = _fill_zone(rows, b, bf["p2"], PUB_SLOTS["p2_bf"], "p1")
    b = _fill_zone(rows, b, _gy_objs([obj[iid] for iid in g.graveyard if iid in obj],
                                     PUB_SLOTS["graveyard"]), PUB_SLOTS["graveyard"], "p1")
    b = _fill_zone(rows, b, [obj[iid] for iid in g.exile if iid in obj],
                   PUB_SLOTS["exile"], "p1")
    b = _fill_zone(rows, b, [obj.get(s.source_instance_id) for s in g.stack],
                   PUB_SLOTS["stack"], "p1")
    _fill_zone(rows, b, [obj[s.instance_id] for s in g.library
                         if s.known_by.get("p1") and s.known_by.get("p2")],
               PUB_SLOTS["library"], "p1")

    gv = []
    for pid in ("p1", "p2"):
        p = g.players[pid]
        gv += _player_scalars_obj(p.life, len(p.hand), bf[pid], p.mana_pool,
                                  p.mulligans, p.has_lost)
    gv += [g.turn_number / 40.0, float(g.active_player == "p1"),
           float(g.priority_player == "p1"), len(g.library) / 80.0, len(g.stack) / 6.0]
    gv += list(deckout_clock(g, "p1"))                # deckout clock (p1-oriented)
    out[_PUB_ROWS * CARD_F:] = gv
    return out
