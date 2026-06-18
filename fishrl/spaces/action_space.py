"""Flat discrete action space for the Forgetful Fish environment.

The engine exposes a heterogeneous, imperative decision API (play / tap / cast /
declare attackers / scry / split Fact-or-Fiction / ...). We project all of it onto
a single fixed ``Discrete(N)`` head with a length-N legality mask. Each "block"
below owns a contiguous slice of the id space and maps to one engine operation.

Cards are addressed by **zone-slot index** — the ordinal position of a card in the
engine's own ordered zone list (hand / battlefield) or in a decision's context
list (``g.pending.context``). The environment resolves a slot to the per-game UUID
``instance_id`` at apply time, so observation row k and action slot k always refer
to the same card.

Compound decisions (scry, declare-blockers, Fact-or-Fiction split, ...) are driven
through a small shared sub-action alphabet (``PICK_A`` / ``PICK_B`` / ``COMMIT`` /
``SHUFFLE``) accumulated in an env-side builder (see :mod:`fishrl.spaces.compound`);
the engine's completion function is only called once the builder is well-formed.
"""
from __future__ import annotations

# ── Capacities (sized generously above realistic maxima for the 80-card pool) ──
HAND = 12          # own hand slots (mulligan-7 + draws; Brainstorm peaks ~10)
BF = 34            # own battlefield slots (lands + creatures)
PICK_K = 64        # single-pick context list (library search / graveyard / names)
CMP_K = 20         # compound working-list size (attackers/blockers ≤ creatures; hand ≤ 12)
COLORS = ("W", "U", "B", "R", "G")
ABIL_SLOTS = 2     # max activated abilities per permanent (Svyelunite / Surgical Bay)
BASICS = ("Plains", "Island", "Swamp", "Mountain", "Forest")

# ── Block layout: ordered (name, size). Offsets are derived below. ─────────────
_LAYOUT = [
    # priority / pay actions
    ("PASS", 1),
    ("END_TURN", 1),
    ("PLAY_HAND", HAND),          # play hand slot i (primary mode for modal spells)
    ("PLAY_HAND_ALT", HAND),      # play hand slot i (secondary mode — Vision Charm land)
    ("CYCLE_HAND", HAND),         # cycle hand slot i (Lonely Sandbar)
    ("TAP_LAND", BF),             # tap battlefield slot i (float at priority / pay into cost)
    ("ACTIVATE", BF * ABIL_SLOTS),  # activate battlefield slot i, ability (i % ABIL_SLOTS)
    ("ALLOC_MANA", len(COLORS)),  # spend a floating WUBRG during payment
    ("CANCEL_PAY", 1),            # abort the pending payment
    # atomic decision singles
    ("PLAY_ORDER", 2),            # choose_play_order: 0=first, 1=second
    ("MULLIGAN", 2),              # mulligan: 0=keep, 1=mulligan
    ("FOF_CHOOSE", 2),            # fof_choose: 0=pile1, 1=pile2
    ("TEXT_CHANGE", len(BASICS) * len(BASICS)),  # choose_text_change: from*5 + to
    ("PICK_SINGLE", PICK_K),      # pick one item from the pending's single-pick list
    ("PICK_NONE", 1),             # decline a "may" pick / find nothing
    ("TARGET_CANCEL", 1),         # cancel a choose_targets cast
    # compound builder sub-actions
    ("PICK_A", CMP_K),            # primary append/toggle (top / pile1 / order / blocker)
    ("PICK_B", CMP_K),            # secondary (scry bottom / pile2 / select attacker)
    ("COMMIT", 1),               # finalize a builder (attackers / blockers)
    ("SHUFFLE", 1),              # reorder: shuffle instead of keeping an order
]

# name -> (offset, size)
OFFSETS: dict[str, tuple[int, int]] = {}
_o = 0
for _name, _size in _LAYOUT:
    OFFSETS[_name] = (_o, _size)
    _o += _size
N = _o  # total action-space size


def block(name: str) -> tuple[int, int]:
    """(offset, size) of a named block."""
    return OFFSETS[name]


def aid(name: str, local: int = 0) -> int:
    """Global action id for local index `local` within block `name`."""
    off, size = OFFSETS[name]
    if not (0 <= local < size):
        raise IndexError(f"{name}[{local}] out of range (size {size})")
    return off + local


def decode(action: int) -> tuple[str, int]:
    """Map a global action id back to (block_name, local_index)."""
    for name, (off, size) in OFFSETS.items():
        if off <= action < off + size:
            return name, action - off
    raise IndexError(f"action {action} out of range [0,{N})")


def text_change_pair(local: int) -> tuple[str, str]:
    """Decode a TEXT_CHANGE local index into (from_type, to_type)."""
    return BASICS[local // len(BASICS)], BASICS[local % len(BASICS)]
