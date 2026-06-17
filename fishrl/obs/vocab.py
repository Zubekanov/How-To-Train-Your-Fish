"""Fixed vocabularies for observation encoding.

The card-name vocabulary is derived from the packaged decklist at import time
(the two Dandân printings collapse to one name), so it tracks the deck data while
staying stable within a build. Steps and pending-decision types are fixed lists.
"""
from __future__ import annotations

from fishrl.forgetful_fish import engine as E
from fishrl.forgetful_fish.cards import load_decklist

# Card-name vocabulary (sorted, unique). Falls back to a known 20-name list if the
# data file is somehow absent so the obs layout stays fixed.
_FALLBACK_NAMES = [
    "Accumulated Knowledge", "Brainstorm", "Crystal Spray", "Dandân",
    "Day's Undoing", "Fact or Fiction", "Halimar Depths", "Island",
    "Lonely Sandbar", "Memory Lapse", "Metamorphose", "Mind Bend",
    "Mystic Sanctuary", "Mystical Tutor", "Ponder", "Predict",
    "Svyelunite Temple", "Temple of Epiphany", "The Surgical Bay", "Vision Charm",
]
_names = sorted({c.get("name", "") for c in load_decklist() if c.get("name")})
CARD_NAMES = _names if _names else list(_FALLBACK_NAMES)
NAME_INDEX = {n: i for i, n in enumerate(CARD_NAMES)}
N_NAMES = len(CARD_NAMES)

# Turn steps, plus a pregame slot ("") for the opening before any phase.
STEPS = [""] + list(E._STEPS)
STEP_INDEX = {s: i for i, s in enumerate(STEPS)}
N_STEPS = len(STEPS)

# Every pending decision type a controlled seat can receive.
PENDING_TYPES = [
    "priority", "pay", "choose_play_order", "mulligan", "bottom", "discard",
    "scry", "reorder", "putback", "search_library", "fof_split", "fof_choose",
    "name_card", "choose_text_change", "put_from_hand", "choose_graveyard",
    "choose_targets", "declare_attackers", "declare_blockers", "order_triggers",
]
PENDING_INDEX = {t: i for i, t in enumerate(PENDING_TYPES)}
N_PENDING = len(PENDING_TYPES)
