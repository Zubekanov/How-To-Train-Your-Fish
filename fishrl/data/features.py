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
from fishrl.obs.encoder import (CARD_F, _encode_card, _fill_zone, _zone, _zone_obj,
                                deckout_clock)
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


def pub_dim_for(view: str) -> int:
    return HANDS_DIM if view == "hands" else PUB_DIM


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
    if fill <= 0.0:
        return known
    remaining = np.maximum(copies - visible - known, 0.0)
    rtot = float(remaining.sum())
    if rtot <= 0.0:
        return known
    return (known + fill * remaining / rtot).astype(np.float32)


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
        _zone(_obj_dicts(g, g.graveyard), GOD_SLOTS["graveyard"], "p1"),
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
    b = _fill_zone(rows, b, [obj[iid] for iid in g.graveyard if iid in obj],
                   GOD_SLOTS["graveyard"], "p1")
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
        _zone(view.get("graveyard", []), PUB_SLOTS["graveyard"], "p1"),
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
_ZERO_HANDS = np.zeros(HANDS_DIM, dtype=np.float32)
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
        return _ZERO_HANDS
    obj = g.objects
    bf = {pid: [obj[iid] for iid in g.players[pid].battlefield if iid in obj]
          for pid in ("p1", "p2")}
    out = np.zeros(HANDS_DIM, dtype=np.float32)
    rows = out[:_HANDS_ROWS * CARD_F].reshape(_HANDS_ROWS, CARD_F)
    b = _fill_zone(rows, 0, [obj[iid] for iid in g.players["p1"].hand if iid in obj],
                   HANDS_SLOTS["p1_hand"], "p1")
    b = _fill_zone(rows, b, [obj[iid] for iid in g.players["p2"].hand if iid in obj],
                   HANDS_SLOTS["p2_hand"], "p1")
    b = _fill_zone(rows, b, bf["p1"], HANDS_SLOTS["p1_bf"], "p1")
    b = _fill_zone(rows, b, bf["p2"], HANDS_SLOTS["p2_bf"], "p1")
    b = _fill_zone(rows, b, [obj[iid] for iid in g.graveyard if iid in obj],
                   HANDS_SLOTS["graveyard"], "p1")
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
    out[_HANDS_ROWS * CARD_F:] = gv
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
    b = _fill_zone(rows, b, [obj[iid] for iid in g.graveyard if iid in obj],
                   PUB_SLOTS["graveyard"], "p1")
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
