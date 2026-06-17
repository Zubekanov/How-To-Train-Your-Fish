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
from fishrl.obs.encoder import CARD_F, _encode_card, _zone
from fishrl.obs import vocab as V

# ── god (privileged) layout ───────────────────────────────────────────────────
GOD_SLOTS = {
    "p1_hand": 12, "p2_hand": 12, "p1_bf": 20, "p2_bf": 20,
    "graveyard": 32, "exile": 8, "stack": 6, "library": 64,
}
_GOD_ROWS = sum(GOD_SLOTS.values())
_GOD_PER_PLAYER = 12       # life, hand_count, lands, untapped, pool_total, pool×5, mulligans, has_lost
_GOD_GAME = 5              # turn, active_is_p1, priority_is_p1, library_count, stack_depth
GOD_DIM = _GOD_ROWS * CARD_F + _GOD_PER_PLAYER * 2 + _GOD_GAME

# ── public (mutual-knowledge) layout ──────────────────────────────────────────
PUB_SLOTS = {
    "p1_hand": 12, "p2_hand": 12, "p1_bf": 20, "p2_bf": 20,
    "graveyard": 32, "exile": 8, "stack": 6, "library": 8,
}
_PUB_ROWS = sum(PUB_SLOTS.values())
_PUB_PER_PLAYER = 12
_PUB_GAME = 5
PUB_DIM = _PUB_ROWS * CARD_F + _PUB_PER_PLAYER * 2 + _PUB_GAME


def opponent_hand_counts(g, viewer: str) -> np.ndarray:
    """Per-name count of the opponent's hand (the guesser's supervised target)."""
    opp = "p2" if viewer == "p1" else "p1"
    counts = np.zeros(V.N_NAMES, dtype=np.float32)
    for iid in g.players[opp].hand:
        idx = V.NAME_INDEX.get(g.objects[iid].name)
        if idx is not None:
            counts[idx] += 1.0
    return counts


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


def encode_god(g) -> np.ndarray:
    """Privileged, fully-observed, p1-oriented feature vector (GOD_DIM)."""
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
    parts.append(np.asarray(gv, dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


def encode_public(g) -> np.ndarray:
    """Mutual-knowledge, p1-oriented feature vector (PUB_DIM) from spectator_view."""
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
    parts.append(np.asarray(gv, dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)
