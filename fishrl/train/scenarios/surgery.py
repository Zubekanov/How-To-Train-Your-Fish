"""State surgery for the manufactured scenarios. We RELOCATE existing, already-legal
card instances rather than fabricating new ones — no card_key/oracle_text/effect
registration to get wrong, so the result stays a legal engine state. The library is
SHARED (``g.library``, index 0 = top); both seats draw from it.
"""
from __future__ import annotations

from fishrl.forgetful_fish import engine as E


def pool_all_zones(g) -> list:
    """Empty EVERY zone and return all instance_ids — the basis for a full
    manufacture (redistribute the whole deck into a constructed start-state)."""
    pool = []
    for pid in g.players:
        pool += g.players[pid].hand; g.players[pid].hand = []
        pool += g.players[pid].battlefield; g.players[pid].battlefield = []
    pool += [s.instance_id for s in g.library]; g.library = []
    pool += list(g.graveyard); g.graveyard = []
    pool += list(g.exile); g.exile = []
    pool += [s.source_instance_id for s in g.stack if s.source_instance_id]
    g.stack = []
    return pool


def make_engine_heuristic(g, seat: str) -> None:
    """Hand `seat` to the engine's heuristic AI (it then auto-draws + casts, while
    the other seat stays controlled). Verified to take effect mid-game on a snapshot."""
    g.players[seat].is_ai = True
    g.players[seat].ai_profile = "heuristic"
