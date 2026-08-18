"""STANDING equivalence guard for the object-native `_priority_mask`.

`_priority_mask` re-derives the view's can_play / can_cycle / modes /
ability-availability annotations straight from engine objects (skipping the full
`current_view` build, which was ~8% of collection wall). Like the encoder fast
paths, the risk is silent divergence from the view exactly where those
annotations are dynamic -- so every priority state of random self-play is
compared bit-for-bit against `_priority_mask_ref` (the original current_view
path, kept as the behavioural contract), for BOTH seats: the acting player and
the non-priority viewer (whose mask must stay pass-only per the timing rules).

Re-run on every engine re-vendor; if the vendored view logic changes, the lean
mask diverges and THIS test fails. Do not delete it.
"""
import numpy as np

from fishrl.env.aec_env import FishAEC
from fishrl.spaces import action_space as A
from fishrl.spaces.masking import _priority_mask, _priority_mask_ref


def _compare(g, viewer: str, where: str) -> np.ndarray:
    m_fast = np.zeros(A.N, dtype=np.int8)
    m_ref = np.zeros(A.N, dtype=np.int8)
    _priority_mask(g, viewer, m_fast)
    _priority_mask_ref(g, viewer, m_ref)
    if not np.array_equal(m_fast, m_ref):
        diff = [A.decode(int(i)) for i in np.flatnonzero(m_fast != m_ref)]
        raise AssertionError(f"{where}: object-native priority mask diverges from "
                             f"the current_view reference at {diff} (viewer={viewer})")
    return m_ref


def test_priority_mask_matches_reference():
    rng = np.random.default_rng(0)
    env = FishAEC(max_decisions=2000)
    compared = 0
    strata = {"play_hand": 0, "play_alt": 0, "cycle": 0, "activate": 0}
    for sd in range(8):
        env.reset(seed=sd)
        for agent in env.agent_iter(max_iter=12000):
            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
                continue
            obs = env.observe(agent)
            pend = env.g.pending
            if pend is not None and pend.type == "priority":
                m = _compare(env.g, agent, f"seed={sd} acting")
                opp = "p2" if agent == "p1" else "p1"
                _compare(env.g, opp, f"seed={sd} non-acting")
                strata["play_hand"] += int(m[A.aid("PLAY_HAND", 0):A.aid("PLAY_HAND", 0) + A.HAND].any())
                strata["play_alt"] += int(m[A.aid("PLAY_HAND_ALT", 0):A.aid("PLAY_HAND_ALT", 0) + A.HAND].any())
                strata["cycle"] += int(m[A.aid("CYCLE_HAND", 0):A.aid("CYCLE_HAND", 0) + A.HAND].any())
                strata["activate"] += int(m[A.aid("ACTIVATE", 0):A.aid("ACTIVATE", 0) + A.BF * A.ABIL_SLOTS].any())
                compared += 1
            legal = np.flatnonzero(obs["action_mask"])
            env.step(int(rng.choice(legal)))
    assert compared > 300, f"too few priority states compared: {compared}"
    # A green run must certify the non-trivial branch (playable hand cards, i.e. the
    # affordability + targets logic) was actually exercised, not just PASS states.
    assert strata["play_hand"] > 0, f"playable-hand stratum never exercised ({strata})"
