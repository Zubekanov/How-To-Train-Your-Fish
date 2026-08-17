"""Guided text-change masking (Config.text_change_mode; fishrl.spaces.masking).

The 5x5 choose_text_change block collapses to {EFFECT, NO-OP} in "guided" mode and
{EFFECT} in "auto": EFFECT = the type written on the targeted card -> the first
canonical type absent from its controller's permanents; NO-OP = a from-type written
on none of the targets (provably inert). "full" must stay byte-identical to the
legacy mask, and guided may only ever SHRINK the legal set.
"""
from __future__ import annotations

import numpy as np
import pytest

from fishrl.forgetful_fish import engine as E
from fishrl.forgetful_fish.cards import load_decklist
from fishrl.spaces import action_space as A
from fishrl.spaces import masking


@pytest.fixture(autouse=True)
def _restore_mode():
    yield
    masking.set_text_change_mode("full")


def _game_with_pending_text_change():
    """A real game paused on choose_text_change for p1, targeting an Island that
    p2 controls (the canonical Mind-Bend removal shape)."""
    g = E.new_multiplayer_game(load_decklist(), p1_name="p1", p2_name="p2", seed=5)
    E.choose_play_order(g, g.pending.player, "first")
    guard = 0
    while g.pending is not None and g.pending.type == "mulligan" and guard < 8:
        guard += 1
        E.mulligan_decision(g, g.pending.player, "keep")
    # move an Island from the library onto p2's battlefield
    slot = next(s for s in g.library if g.objects[s.instance_id].name == "Island")
    g.library.remove(slot)
    isl = slot.instance_id
    g.objects[isl].controller = "p2"
    g.players["p2"].battlefield.append(isl)
    # a source card for the text change (any instant in p1's hand works as the prop)
    src = g.players["p1"].hand[0]
    if "tc_test_noop" not in E._AFTER_TEXT_CHANGE:
        E.register_after_text_change("tc_test_noop")(lambda g, p, frm, to, ctx: None)
    assert E.start_text_change(g, "p1", src, then="tc_test_noop", change_targets=[isl])
    assert g.pending is not None and g.pending.type == "choose_text_change"
    return g


def _tc_ids(mask) -> set:
    off, size = A.block("TEXT_CHANGE")
    return {i for i in range(size) if mask[off + i]}


def test_full_mode_is_the_legacy_25_minus_diagonal():
    g = _game_with_pending_text_change()
    masking.set_text_change_mode("full")
    ids = _tc_ids(masking.atomic_mask(g, "p1"))
    assert len(ids) == 20                       # 5x5 minus the no-op diagonal
    for li in ids:
        frm, to = A.text_change_pair(li)
        assert frm != to


def test_guided_mode_is_effect_plus_noop_and_a_subset_of_full():
    g = _game_with_pending_text_change()
    masking.set_text_change_mode("full")
    full = _tc_ids(masking.atomic_mask(g, "p1"))
    masking.set_text_change_mode("guided")
    ids = _tc_ids(masking.atomic_mask(g, "p1"))
    assert ids <= full and len(ids) == 2
    pairs = {A.text_change_pair(li) for li in ids}
    # EFFECT: Island (written on the target) -> Plains (first type absent from
    # p2's permanents, which are Island-only)
    assert ("Island", "Plains") in pairs
    # NO-OP: a from-type written on none of the targets
    noop = next(p for p in pairs if p != ("Island", "Plains"))
    assert noop[0] != "Island"
    # both survive the engine's own validation lists
    ctx = g.pending.context
    for frm, to in pairs:
        assert frm in ctx["from_types"] and to in ctx["to_types"]


def test_auto_mode_is_effect_only():
    g = _game_with_pending_text_change()
    masking.set_text_change_mode("auto")
    ids = _tc_ids(masking.atomic_mask(g, "p1"))
    assert len(ids) == 1
    assert A.text_change_pair(next(iter(ids))) == ("Island", "Plains")


def test_guided_falls_back_to_full_when_nothing_to_analyse():
    g = _game_with_pending_text_change()
    g.pending.context["change_targets"] = []    # nothing the helper can read
    masking.set_text_change_mode("guided")
    ids = _tc_ids(masking.atomic_mask(g, "p1"))
    assert len(ids) == 20                       # never invents; falls back to full


def test_guided_effect_is_accepted_by_the_engine():
    g = _game_with_pending_text_change()
    masking.set_text_change_mode("guided")
    ids = _tc_ids(masking.atomic_mask(g, "p1"))
    li = next(i for i in ids if A.text_change_pair(i) == ("Island", "Plains"))
    frm, to = A.text_change_pair(li)
    assert E.complete_text_change(g, "p1", frm, to)   # the mask contract holds
    assert g.pending is None or g.pending.type != "choose_text_change"
