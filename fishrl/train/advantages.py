"""Per-seat credit assignment with a single, consistent sign convention.

Both the privileged critic and the public estimator output P(p1 wins). Each seat's
value/return/reward is expressed in that **seat's own frame** so one shared
policy+critic is consistent whether acting as p1 or p2:

    s(p1)=+1, s(p2)=-1
    V_seat = s(seat) * (2*P(p1 win) - 1)          # in [-1, +1]
    z_seat = +1 if winner==seat, -1 if winner==opp, 0 on draw/truncation

Because both seats read the SAME P(p1 win) with opposite sign, the critic is
zero-sum-consistent: V_p1(g) == -V_p2(g) for the same state. GAE is computed over
each seat's OWN ordered decision subsequence (reward is terminal-only).
"""
from __future__ import annotations

import numpy as np

SEAT_SIGN = {"p1": 1.0, "p2": -1.0}


def p1_winprob_to_seat_value(p_p1: float, seat: str) -> float:
    return SEAT_SIGN[seat] * (2.0 * float(p_p1) - 1.0)


def seat_outcome(winner, seat: str) -> float:
    if winner is None:
        return 0.0
    return 1.0 if winner == seat else -1.0


def gae(values: np.ndarray, rewards: np.ndarray, gamma: float, lam: float):
    """GAE over one seat's ordered subsequence. `values`/`rewards` are seat-frame;
    the bootstrap value after the last decision is 0 (the game has ended).
    Returns (advantages, returns)."""
    T = len(values)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in reversed(range(T)):
        v_next = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * v_next - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + values
