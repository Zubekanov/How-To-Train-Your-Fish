"""A simple, deterministic-ish scripted "attacker" opponent in the RL action space.

Strategy (intentionally naive and aggressive):
  * play a land every turn when one is in hand,
  * cast Dandan whenever it can be paid for,
  * always attack with every eligible creature (never holds back, never blocks),
  * when forced to discard, throw away a non-land / non-Dandan card first, else random.

It reads the game state + the env's legality mask and returns ONE legal action id, so
it plugs into the same AEC env the learned agents use (drive one seat with this, the
other with a policy). Unlike the engine's heuristic AI it needs no engine-side profile.

Useful as a TRANSITIVE evaluation anchor: it is a fixed policy (identical for every
arm), and its strength sits between random and the heuristic AI, so it discriminates in
the skill band the learned agents actually occupy.
"""
from __future__ import annotations

import numpy as np

from fishrl.forgetful_fish.state import _is_land
from fishrl.spaces import action_space as A

DANDAN = "Dandân"


def _first_legal_in(block: str, legal: set) -> int | None:
    off, size = A.block(block)
    for a in range(off, off + size):
        if a in legal:
            return a
    return None


def attacker_action(g, seat: str, mask: np.ndarray, rng: np.random.Generator) -> int:
    """Return one legal action id for `seat` under the current pending decision."""
    legal_arr = np.flatnonzero(mask)
    if legal_arr.size == 0:
        return 0                                   # mask contract should prevent this
    legal = set(int(a) for a in legal_arr)
    t = g.pending.type
    hand = g.players[seat].hand

    if t == "priority":
        # 1) play a land (special action, retains priority)
        for i in range(min(len(hand), A.HAND)):
            if A.aid("PLAY_HAND", i) in legal and _is_land(g.objects[hand[i]].type_line):
                return A.aid("PLAY_HAND", i)
        # 2) cast Dandan if affordable (mask only offers it when it is)
        for i in range(min(len(hand), A.HAND)):
            if A.aid("PLAY_HAND", i) in legal and g.objects[hand[i]].name == DANDAN:
                return A.aid("PLAY_HAND", i)
        # 3) otherwise PASS — advances through to combat (never END_TURN, which would
        #    skip the attack step)
        return A.aid("PASS") if A.aid("PASS") in legal else int(legal_arr[0])

    if t == "pay":
        for blk in ("TAP_LAND", "ALLOC_MANA", "ACTIVATE"):   # any legal payment completes it
            a = _first_legal_in(blk, legal)
            if a is not None:
                return a
        return int(legal_arr[0])

    if t == "declare_attackers":
        a = _first_legal_in("PICK_A", legal)        # add every eligible attacker...
        if a is not None:
            return a
        return A.aid("COMMIT") if A.aid("COMMIT") in legal else int(legal_arr[0])  # ...then swing

    if t == "declare_blockers":
        return A.aid("COMMIT") if A.aid("COMMIT") in legal else int(legal_arr[0])  # never block

    if t == "discard":
        off, _size = A.block("PICK_A")
        prefer, any_legal = [], []
        for a in sorted(legal):
            if not (off <= a < off + A.block("PICK_A")[1]):
                continue
            i = a - off
            any_legal.append(a)
            if i < len(hand):
                o = g.objects[hand[i]]
                if not _is_land(o.type_line) and o.name != DANDAN:
                    prefer.append(a)
        pool = prefer or any_legal
        return int(rng.choice(pool)) if pool else int(legal_arr[0])

    if t == "mulligan":
        return A.aid("MULLIGAN", 0) if A.aid("MULLIGAN", 0) in legal else int(legal_arr[0])  # keep
    if t == "choose_play_order":
        return A.aid("PLAY_ORDER", 0) if A.aid("PLAY_ORDER", 0) in legal else int(legal_arr[0])

    return int(legal_arr[0])                         # sane default for any other decision


def attacker_policy(seed: int = 0):
    """A reusable (g, seat, mask) -> action callable with its own RNG for discards."""
    rng = np.random.default_rng(seed)
    return lambda g, seat, mask: attacker_action(g, seat, mask, rng)
