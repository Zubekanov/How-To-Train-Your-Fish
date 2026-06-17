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

import numpy as np

from fishrl.forgetful_fish.state import current_view
from fishrl.obs import vocab as V

# ── per-card feature vector ───────────────────────────────────────────────────
# name one-hot (N_NAMES) + unknown bit, type flags(5), stats(4), perm flags(3), known bit
CARD_F = V.N_NAMES + 1 + 5 + 4 + 3 + 1

# ── fixed zone slot counts (pad / truncate) ───────────────────────────────────
SLOTS = {
    "own_hand": 12, "opp_hand": 12, "own_bf": 20, "opp_bf": 20,
    "graveyard": 32, "exile": 8, "stack": 6, "library": 8,
}
_ZONE_ROWS = sum(SLOTS.values())

# ── global / decision scalar block ────────────────────────────────────────────
_PER_PLAYER = 12          # life, hand_count, lands, untapped, pool_total, pool×5, mulligans, has_lost
_GAME = 5                 # turn, active_is_self, priority_is_self, library_count, stack_depth
_COMBAT = 4               # am_attacking, am_defending, n_attackers, n_blockers
_PAY = 4                  # need_U, need_total, generic, in_pay
_BUILDER = 1              # compound-builder progress (injected by the env)
_GLOBALS = _PER_PLAYER * 2 + _GAME + V.N_STEPS + _COMBAT + V.N_PENDING + _PAY + _BUILDER

OBS_DIM = _ZONE_ROWS * CARD_F + _GLOBALS


def _encode_card(card: dict | None, viewer: str) -> np.ndarray:
    v = np.zeros(CARD_F, dtype=np.float32)
    if not card or card.get("known") is False or "name" not in card:
        v[V.N_NAMES] = 1.0          # "unknown/hidden" bit
        return v
    name = card.get("name", "")
    idx = V.NAME_INDEX.get(name)
    if idx is not None:
        v[idx] = 1.0
    else:
        v[V.N_NAMES] = 1.0
    o = V.N_NAMES + 1
    tl = (card.get("type_line") or "").lower()
    v[o + 0] = float("land" in tl)
    v[o + 1] = float("creature" in tl)
    v[o + 2] = float("instant" in tl)
    v[o + 3] = float("sorcery" in tl)
    v[o + 4] = float("basic land" in tl and "island" in tl)
    o += 5
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
    return v


def _zone(cards: list, n: int, viewer: str) -> np.ndarray:
    rows = np.zeros((n, CARD_F), dtype=np.float32)
    for i, card in enumerate(cards[:n]):
        rows[i] = _encode_card(card, viewer)
    return rows.reshape(-1)


def encode_observation(g, viewer: str, builder_progress: float = 0.0) -> np.ndarray:
    """Fixed-size float32 observation for `viewer`. `builder_progress` is the
    fraction of a compound decision already specified (0 when none active)."""
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

    parts.append(g_vec)
    return np.concatenate(parts).astype(np.float32)
