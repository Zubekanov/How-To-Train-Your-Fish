"""Mana-affordability gating and the removal of the reversible 'undo' actions.

The agent may only attempt casts/activations it can actually pay for (counting
sac-for-mana like Svyelunite Temple, and excluding an ability's self-tap source like
The Surgical Bay), it is never offered CANCEL_PAY / TARGET_CANCEL, and a committed
payment can never strand (so it is always completable without a cancel)."""
import numpy as np

from fishrl.forgetful_fish import engine as E
from fishrl.forgetful_fish.state import (
    GameState, PlayerState, CardInstance, PendingDecision,
)
from fishrl.spaces import action_space as A
from fishrl.spaces import masking as M


def _game():
    g = GameState()
    g.players = {"p1": PlayerState(pid="p1"), "p2": PlayerState(pid="p2")}
    g.priority_player = "p1"
    return g


def _put(g, name, type_line, pid="p1", oracle_text="", tapped=False):
    iid = f"{name}#{len(g.objects)}"
    g.objects[iid] = CardInstance(instance_id=iid, name=name, type_line=type_line,
                                  oracle_text=oracle_text, controller=pid, owner=pid,
                                  tapped=tapped, entered_this_turn=False)
    g.players[pid].battlefield.append(iid)
    return iid


def _pay_pending(g, need, generic, source=None):
    g.pending = PendingDecision(type="pay", player="p1",
                                context={"need": dict(need), "generic": generic,
                                         "ops": [], "source": source})


# ── affordability counts sac-for-mana and excludes self-tapping sources ────────
def test_affordable_counts_svyelunite_sacrifice():
    g = _game()
    _put(g, "Svyelunite Temple", "Land")             # {T}:U  OR  {T},Sac:UU
    assert M._affordable(g, "p1", {}, 2)             # can make 2 via sacrifice
    assert not M._affordable(g, "p1", {}, 3)


def test_surgical_bay_ability_needs_two_other_lands():
    g = _game()
    bay = _put(g, "The Surgical Bay", "Land")        # {1}{U},{T},Sac: draw  (cost 2)
    _put(g, "Island", "Basic Land - Island")
    # the bay taps/sacs itself, so it can't help pay its own cost: one other land is short
    assert not M._affordable(g, "p1", {}, 2, exclude_iid=bay)
    _put(g, "Island", "Basic Land - Island")
    assert M._affordable(g, "p1", {}, 2, exclude_iid=bay)   # two other lands -> affordable


def test_affordable_is_colour_aware():
    """A {U} cost (e.g. cycling Lonely Sandbar) can't be paid by a land whose basic
    type was changed to Mountain (taps for R) -- otherwise the agent commits to an
    unpayable cost with no cancel and the pay mask goes empty."""
    g = _game()
    _put(g, "Island", "Basic Land - Mountain")       # text-changed -> taps for {R}
    assert not M._affordable(g, "p1", {"U": 1}, 0)
    _put(g, "Island", "Basic Land - Island")
    assert M._affordable(g, "p1", {"U": 1}, 0)


# ── the reversible 'undo' actions are never offered ───────────────────────────
def test_cancel_pay_never_offered():
    g = _game()
    _put(g, "Island", "Basic Land - Island")
    _pay_pending(g, {}, 1)
    m = M.atomic_mask(g, "p1")
    assert m[A.aid("CANCEL_PAY")] == 0
    assert m.sum() >= 1                              # ...but the payment is still completable


def test_target_cancel_never_offered():
    g = _game()
    a, b = _put(g, "Fish", "Creature"), _put(g, "Fish", "Creature")
    g.pending = PendingDecision(type="choose_targets", player="p1",
                                context={"legal": [a, b], "count": 1})
    m = M.atomic_mask(g, "p1")
    assert m[A.aid("TARGET_CANCEL")] == 0
    assert m[A.aid("PICK_SINGLE", 0)] == 1 and m[A.aid("PICK_SINGLE", 1)] == 1


# ── a committed payment can never strand (no-strand pay mask) ──────────────────
def test_pay_mask_no_strand_offers_only_the_sacrifice():
    g = _game()
    _put(g, "Svyelunite Temple", "Land")             # sole source; must make UU to pay {2}
    _pay_pending(g, {}, 2)
    m = M.atomic_mask(g, "p1")
    assert m[A.aid("ACTIVATE", 1)] == 1              # the {T},Sac:UU ability (index 1)
    assert m[A.aid("ACTIVATE", 0)] == 0             # tap-for-one would strand -> suppressed
    assert m[A.aid("TAP_LAND", 0)] == 0            # land-tap-for-one would strand too
    assert m[A.aid("CANCEL_PAY")] == 0


def test_pay_mask_two_islands_both_tappable():
    g = _game()
    _put(g, "Island", "Basic Land - Island")
    _put(g, "Island", "Basic Land - Island")
    _pay_pending(g, {}, 2)
    m = M.atomic_mask(g, "p1")
    assert m[A.aid("TAP_LAND", 0)] == 1 and m[A.aid("TAP_LAND", 1)] == 1


# ── floating mana empties when the phase moves (CR 500.4) ──────────────────────
def test_floating_mana_empties_on_step_advance():
    g = _game()
    g.current_step = "main1"
    g.players["p1"].mana_pool = {"U": 2}
    E._advance_step(g)
    assert g.players["p1"].mana_pool == {}
