"""Minimal, validated state surgery for scenarios whose setup must be precise
(the hand-authored half of the hybrid). We RELOCATE existing, already-legal card
instances rather than fabricating new ones — no card_key/oracle_text/effect-
registration to get wrong, so the result stays a legal engine state.

The library is SHARED (``g.library``, index 0 = top); both seats draw from it.
"""
from __future__ import annotations

from fishrl.forgetful_fish import engine as E
from fishrl.forgetful_fish.state import LibrarySlot


def basics(g, seat: str, subtype: str = "Island") -> int:
    """Count `seat`'s lands of a basic subtype (tapped or not) — they untap on that
    seat's untap step, so this gauges the mana it will have on its own turn."""
    return sum(1 for iid in g.players[seat].battlefield
               if iid in g.objects and subtype in (g.objects[iid].type_line or ""))


def clear_battlefield_creatures(g) -> int:
    """Move every creature off both battlefields into the graveyard (a legal zone),
    e.g. to force a creatureless deckout race. Returns how many were moved."""
    moved = 0
    for pid in g.players:
        for iid in list(g.players[pid].battlefield):
            o = g.objects.get(iid)
            if o is not None and E._is_creature(o):
                g.players[pid].battlefield.remove(iid)
                g.graveyard.append(iid)
                moved += 1
    return moved


def trim_library(g, n: int) -> int:
    """Shrink the shared library to its top `n` cards, exiling the rest (instances
    stay valid in a real zone). Returns how many were removed."""
    removed = 0
    while len(g.library) > n:
        slot = g.library.pop()          # from the bottom
        g.exile.append(slot.instance_id)
        removed += 1
    return removed


def clear_hand_of(g, name: str, seat: str) -> int:
    """Move every `name` out of `seat`'s hand to the bottom of the shared library, so
    a placed-on-top copy is the only one in play. Returns how many were moved."""
    moved = 0
    for iid in list(g.players[seat].hand):
        if g.objects.get(iid) is not None and g.objects[iid].name == name:
            g.players[seat].hand.remove(iid)
            g.library.append(LibrarySlot(instance_id=iid, known_by={"p1": False, "p2": False}))
            moved += 1
    return moved


def library_names(g) -> list:
    return [g.objects[s.instance_id].name for s in g.library]


def find_library_slot(g, name: str) -> int | None:
    """Index of the first library slot holding a card called `name`, or None."""
    for i, slot in enumerate(g.library):
        o = g.objects.get(slot.instance_id)
        if o is not None and o.name == name:
            return i
    return None


def move_library_card_to_top(g, name: str) -> str | None:
    """Move the first `name` in the library to the top (index 0), known to both.
    Returns its instance_id, or None if not present."""
    i = find_library_slot(g, name)
    if i is None:
        return None
    slot = g.library.pop(i)
    slot.known_by = {"p1": True, "p2": True}
    g.library.insert(0, slot)
    return slot.instance_id


def move_library_card_to_hand(g, name: str, seat: str) -> str | None:
    """Pull the first `name` out of the library into `seat`'s hand (a held tool).
    Returns its instance_id, or None if not present."""
    i = find_library_slot(g, name)
    if i is None:
        return None
    slot = g.library.pop(i)
    iid = slot.instance_id
    g.players[seat].hand.append(iid)
    g.objects[iid].controller = seat
    g.objects[iid].known_by = [seat]
    return iid


def untapped_basics(g, seat: str, subtype: str = "Island") -> int:
    return sum(1 for iid in g.players[seat].battlefield
               if iid in g.objects and subtype in (g.objects[iid].type_line or "")
               and not g.objects[iid].tapped)


def make_engine_heuristic(g, seat: str) -> None:
    """Hand `seat` to the engine's heuristic AI (it then auto-draws + casts, while
    the other seat stays controlled). Verified to take effect mid-game on a snapshot."""
    g.players[seat].is_ai = True
    g.players[seat].ai_profile = "heuristic"


def on_stack(g, iid: str) -> bool:
    return any(so.source_instance_id == iid and so.kind == "spell" for so in g.stack)


def zone_of(g, iid: str) -> str | None:
    for pid in g.players:
        if iid in g.players[pid].hand:
            return f"hand:{pid}"
        if iid in g.players[pid].battlefield:
            return f"battlefield:{pid}"
    if iid in g.graveyard:
        return "graveyard"
    if iid in g.exile:
        return "exile"
    if any(s.instance_id == iid for s in g.library):
        return "library"
    return None
