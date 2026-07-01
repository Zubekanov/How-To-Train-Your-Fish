"""Mana-affordability gating and the removal of the reversible 'undo' actions.

The agent may only attempt casts/activations it can actually pay for (counting
sac-for-mana like Svyelunite Temple, and excluding an ability's self-tap source like
The Surgical Bay), it is never offered CANCEL_PAY / TARGET_CANCEL, and a committed
payment can never strand (so it is always completable without a cancel)."""
import numpy as np

from fishrl.forgetful_fish import engine as E
from fishrl.forgetful_fish.state import (
    GameState, PlayerState, CardInstance, PendingDecision, PERMANENT_ABILITIES,
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


def test_surgical_bay_ability_cost_is_colour_aware():
    """The Surgical Bay's draw is {1}{U} (one generic AND one blue), paid from OTHER
    sources since it taps/sacrifices itself. So it needs a blue pip plus one more mana
    -- two RED lands cannot pay it (no blue), contrary to a flat 'two generic'."""
    g = _game()
    bay = _put(g, "The Surgical Bay", "Land")
    colored, generic = E._parse_cost(PERMANENT_ABILITIES["The Surgical Bay"][1]["cost"])
    assert (colored, generic) == ({"U": 1}, 1)       # not ({}, 2)
    _put(g, "Island", "Basic Land - Mountain")       # {R}
    _put(g, "Island", "Basic Land - Mountain")       # {R}
    assert not M._affordable(g, "p1", colored, generic, exclude_iid=bay)   # no blue
    _put(g, "Island", "Basic Land - Island")         # {U}
    assert M._affordable(g, "p1", colored, generic, exclude_iid=bay)       # blue + generic


def test_affordable_is_colour_aware():
    """A {U} cost (e.g. cycling Lonely Sandbar, or any blue spell) can't be paid by a
    land whose basic type was changed to Mountain (taps for R) -- otherwise the agent
    commits to an unpayable cost with no cancel and the pay mask goes empty."""
    g = _game()
    _put(g, "Island", "Basic Land - Mountain")       # text-changed -> taps for {R}
    assert not M._affordable(g, "p1", {"U": 1}, 0)
    _put(g, "Island", "Basic Land - Island")
    assert M._affordable(g, "p1", {"U": 1}, 0)


def test_affordable_checks_colour_not_just_total():
    """Affordability is by colour, not total mana: {U}{U} is unpayable with one blue
    and one red source even though the total is two. (Guards every gate that routes a
    coloured cost through _affordable -- casts and cycling.)"""
    g = _game()
    _put(g, "Island", "Basic Land - Island")         # {U}
    _put(g, "Island", "Basic Land - Mountain")       # {R}
    assert not M._affordable(g, "p1", {"U": 2}, 0)   # two mana, but only one is blue
    assert M._affordable(g, "p1", {"U": 1}, 1)       # one blue + one generic -> ok


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


def test_pay_mask_required_source_may_use_smaller_option():
    """Per-option completability: Svyelunite Temple + Island paying {U}{U}. The Temple
    is REQUIRED (the Island alone can't cover {U}{U}), but its tap-for-one option still
    completes the cost (its 1 + the Island's 1 = {U}{U}), so it must be offered — the
    old required->MAX rule suppressed it and forced an unnecessary sacrifice the policy
    could never learn around."""
    g = _game()
    _put(g, "Svyelunite Temple", "Land")             # slot 0: {T}:U  OR  {T},Sac:UU
    _put(g, "Island", "Basic Land - Island")         # slot 1: {U}
    _pay_pending(g, {"U": 2}, 0)
    m = M.atomic_mask(g, "p1")
    assert m[A.aid("TAP_LAND", 0)] == 1              # Temple tap-for-one: completable now
    assert m[A.aid("ACTIVATE", 0)] == 1              # same option via the ability menu
    assert m[A.aid("ACTIVATE", 1)] == 1              # the sacrifice stays on offer too
    assert m[A.aid("TAP_LAND", 1)] == 1              # and the Island


def test_pay_mask_sole_temple_paying_uu_still_forces_sacrifice():
    """Sole-source case: Temple alone paying {U}{U}. Tap-for-one strands (no other
    source can add the second {U}), so only the sacrifice may be offered — the
    per-option rule must not be MORE permissive than the anti-stall invariant allows."""
    g = _game()
    _put(g, "Svyelunite Temple", "Land")
    _pay_pending(g, {"U": 2}, 0)
    m = M.atomic_mask(g, "p1")
    assert m[A.aid("ACTIVATE", 1)] == 1              # {T},Sac:UU completes
    assert m[A.aid("ACTIVATE", 0)] == 0              # tap-for-one would strand
    assert m[A.aid("TAP_LAND", 0)] == 0


def test_pay_mask_two_islands_both_tappable():
    g = _game()
    _put(g, "Island", "Basic Land - Island")
    _put(g, "Island", "Basic Land - Island")
    _pay_pending(g, {}, 2)
    m = M.atomic_mask(g, "p1")
    assert m[A.aid("TAP_LAND", 0)] == 1 and m[A.aid("TAP_LAND", 1)] == 1


# ── affordability counts only ADDRESSABLE mana (the {U}-spell strand) ──────────
def test_unaddressable_u_source_past_slot_cap_does_not_count():
    """A {U} source sitting past the action space's battlefield-slot cap (A.BF) has no
    TAP_LAND id, so the pay mask can't emit it. Affordability must NOT count it -- else
    a {U} spell is offered at priority and then strands with an empty pay mask (the bug
    seen ~31x in the 6h run). With the only blue source at slot 20, the cast is simply
    not offered."""
    g = _game()
    for _ in range(A.BF):                            # 20 addressable non-blue lands ({R})
        _put(g, "Island", "Basic Land - Mountain")
    u_iid = _put(g, "Island", "Basic Land - Island")  # the ONLY {U}, at slot 20 (>= A.BF)
    assert g.players["p1"].battlefield.index(u_iid) >= A.BF
    assert not M._affordable(g, "p1", {"U": 1}, 0)   # unaddressable blue doesn't count
    # and the pay mask for a committed {U} is empty -- which is why it must not be offered
    _pay_pending(g, {"U": 1}, 0)
    assert int(M.atomic_mask(g, "p1").sum()) == 0


def test_affordable_implies_nonempty_pay_mask():
    """The invariant that kills the strand: if a coloured cost is deemed affordable, the
    pay mask for that exact cost is non-empty (the committed payment is completable).
    Checked across a U source within the cap and a sac-for-UU source."""
    for setup, need, generic in [
        (lambda g: _put(g, "Island", "Basic Land - Island"), {"U": 1}, 0),       # in-cap blue
        (lambda g: _put(g, "Svyelunite Temple", "Land"), {}, 2),                  # sac -> UU
    ]:
        g = _game()
        setup(g)
        assert M._affordable(g, "p1", need, generic)
        _pay_pending(g, need, generic)
        assert int(M.atomic_mask(g, "p1").sum()) >= 1


# ── floating mana empties when the phase moves (CR 500.4) ──────────────────────
def test_floating_mana_empties_on_step_advance():
    g = _game()
    g.current_step = "main1"
    g.players["p1"].mana_pool = {"U": 2}
    E._advance_step(g)
    assert g.players["p1"].mana_pool == {}
