"""Compound builders must make monotonic progress, so the agent can't stall by
oscillating a combat selection forever (the declare_attackers PICK_A toggle was 87%
of a stalling game's decisions)."""
import numpy as np

from fishrl.spaces import action_space as A
from fishrl.spaces.compound import CompoundBuilder


def test_declare_attackers_is_add_only():
    b = CompoundBuilder(None, "p1", "declare_attackers", {"eligible": ["c0", "c1", "c2"]})
    m0 = b.mask()
    assert all(m0[A.aid("PICK_A", i)] for i in range(3)) and m0[A.aid("COMMIT")]
    b.feed(None, A.aid("PICK_A", 0))                 # attack with c0
    m1 = b.mask()
    assert m1[A.aid("PICK_A", 0)] == 0               # no reversible toggle-off of c0
    assert m1[A.aid("PICK_A", 1)] == 1 and m1[A.aid("COMMIT")]
    assert b.attacking == ["c0"]
    b.feed(None, A.aid("PICK_A", 1))
    b.feed(None, A.aid("PICK_A", 2))
    mf = b.mask()
    assert not any(mf[A.aid("PICK_A", i)] for i in range(3))   # all chosen -> only COMMIT left
    assert mf[A.aid("COMMIT")]


def test_declare_attackers_subactions_are_bounded():
    """Following any non-COMMIT legal action can only add attackers, so the builder
    finalises in at most (#creatures) steps -- no unbounded oscillation."""
    b = CompoundBuilder(None, "p1", "declare_attackers", {"eligible": ["c0", "c1", "c2"]})
    steps = 0
    while True:
        legal = np.flatnonzero(b.mask())
        pick = next((a for a in legal if A.decode(int(a))[0] != "COMMIT"), None)
        if pick is None:
            break
        b.feed(None, int(pick))
        steps += 1
        assert steps <= 3, "declare_attackers did not make monotonic progress"
    assert b.attacking == ["c0", "c1", "c2"]


def test_declare_blockers_focus_is_forward_only():
    b = CompoundBuilder(None, "p1", "declare_blockers",
                        {"attackers": ["a0", "a1"], "eligible": ["b0", "b1"]})
    assert b.mask()[A.aid("PICK_B", 0)] and b.mask()[A.aid("PICK_B", 1)]
    b.feed(None, A.aid("PICK_B", 0))                 # focus attacker 0
    m1 = b.mask()
    assert m1[A.aid("PICK_B", 0)] == 0               # can't re-select current (no-op loop)
    assert m1[A.aid("PICK_B", 1)] == 1               # only forward
