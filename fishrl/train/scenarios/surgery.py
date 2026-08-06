"""State surgery for the manufactured scenarios. We RELOCATE existing, already-legal
card instances rather than fabricating new ones — no card_key/oracle_text/effect
registration to get wrong, so the result stays a legal engine state. The library is
SHARED (``g.library``, index 0 = top); both seats draw from it.

Every relocation goes through `scrub` so a card carries no state from wherever the
snapshot happened to leave it (stale opponent-hand knowledge, marked damage,
counters, text-changing effects).
"""
from __future__ import annotations

from fishrl.forgetful_fish.state import LibrarySlot, revert_text_changes


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


def scrub(o) -> None:
    """Reset the state a relocated card carries from wherever the snapshot left it:
    nobody knows it any more (card-level ``known_by`` feeds the opponent-hand
    visibility filter in the obs encoder — library knowledge is LibrarySlot-level
    and untouched here), it bears no damage or counters, and any text-changing
    effects (Vision Charm / Mind Bend / Crystal Spray) revert to the printed text."""
    o.known_by = []
    o.damage_marked = 0
    o.counters = {}
    revert_text_changes(o)


def is_land(o) -> bool:
    return "Land" in (o.type_line or "")


def is_island(o) -> bool:
    """Island-TYPED via type_line (Island, Mystic Sanctuary) — NOT merely
    blue-producing; Halimar Depths et al. are plain 'Land'."""
    return "Island" in (o.type_line or "")


def put_battlefield(g, seat: str, iid: str, tapped: bool = False) -> None:
    """Put `iid` onto `seat`'s battlefield: scrubbed, controlled by `seat`,
    untapped by default and NOT summoning-sick (ready to act)."""
    o = g.objects[iid]
    scrub(o)
    o.tapped = tapped
    o.controller = seat
    o.entered_this_turn = False
    g.players[seat].battlefield.append(iid)


def put_hand(g, seat: str, iid: str) -> None:
    """Append `iid` to `seat`'s hand, scrubbed and controlled by `seat`."""
    o = g.objects[iid]
    scrub(o)
    o.controller = seat
    g.players[seat].hand.append(iid)


def put_graveyard(g, iid: str) -> None:
    scrub(g.objects[iid])
    g.graveyard.append(iid)


def put_exile(g, iid: str) -> None:
    scrub(g.objects[iid])
    g.exile.append(iid)


def rebuild_library(g, iids, top=None, top_known_by=None) -> None:
    """Rebuild the shared library as fresh, unknown LibrarySlots (index 0 = top).
    `top` (optional) becomes the top card; `top_known_by` (e.g. ``("p1",)``) marks
    that SLOT known to those seats — slot-level knowledge is what the obs encoder
    uses for library visibility, so the card itself is still scrubbed."""
    g.library = []
    if top is not None:
        kb = {"p1": False, "p2": False}
        for pid in (top_known_by or ()):
            kb[pid] = True
        scrub(g.objects[top])
        g.library.append(LibrarySlot(instance_id=top, known_by=kb))
    for iid in iids:
        scrub(g.objects[iid])
        g.library.append(LibrarySlot(instance_id=iid,
                                     known_by={"p1": False, "p2": False}))


def make_engine_heuristic(g, seat: str, profile: str = "heuristic") -> None:
    """Hand `seat` to the engine's heuristic AI (it then auto-draws + casts, while
    the other seat stays controlled). Verified to take effect mid-game on a snapshot.
    `profile` picks the vendored AI version ("heuristic" = v1.0, "heuristic_1_1",
    "heuristic_1_2", "heuristic_1_3"); an unknown name would silently mean v1.0,
    so validate here."""
    from fishrl.forgetful_fish.engine import _HEURISTIC_PROFILES
    if profile not in _HEURISTIC_PROFILES:
        raise ValueError(f"unknown heuristic profile {profile!r}; have {_HEURISTIC_PROFILES}")
    g.players[seat].is_ai = True
    g.players[seat].ai_profile = profile
