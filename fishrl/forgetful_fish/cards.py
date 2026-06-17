"""Load the Forgetful Fish decklist (card data) for building games.

Vendored from Website_Dev for the fishrl ML environment. The decklist is shipped
alongside the package at fishrl/data/fish_cards.json (committed, not a gitignored
build artifact as in the source repo).
"""
from __future__ import annotations

import json
from pathlib import Path

# fishrl/forgetful_fish/cards.py -> fishrl/data/fish_cards.json
_DATA_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "fish_cards.json"
)


def load_decklist() -> list:
    """Return the decklist (list of card dicts), or [] if the data file is absent."""
    try:
        return json.loads(_DATA_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
