"""Self-play integration: games terminate, rewards are zero-sum, masks are sound."""
import numpy as np

from fishrl.forgetful_fish.state import GameState
from fishrl.env.apply import apply_atomic
from fishrl.selfplay.pettingzoo_api import env as make_env, raw_env
from fishrl.spaces import action_space as A
from fishrl.spaces.compound import COMPOUND_TYPES
from fishrl.spaces.masking import atomic_mask


def _clone(g):
    return GameState.from_dict(g.to_dict())


def test_random_selfplay_terminates_zero_sum():
    rng = np.random.default_rng(0)
    for seed in range(12):
        e = raw_env()
        e.reset(seed=seed)
        steps = 0
        while e.agents:
            a = e.agent_selection
            if e.terminations[a] or e.truncations[a]:
                e.step(None)
                continue
            mask = e.observe(a)["action_mask"]
            legal = np.flatnonzero(mask)
            assert legal.size > 0, f"empty mask at {e.g.pending.type}"
            e.step(int(rng.choice(legal)))
            steps += 1
            assert steps < 8000, "game failed to terminate"
        assert e.g.result["status"] != "ongoing"


def test_rewards_delivered_via_api():
    rng = np.random.default_rng(7)
    e = make_env()
    e.reset(seed=1)
    totals = {"p1": 0.0, "p2": 0.0}
    for agent in e.agent_iter(max_iter=20000):
        obs, rew, term, trunc, _ = e.last()
        totals[agent] += rew
        if term or trunc:
            e.step(None)
        else:
            e.step(int(rng.choice(np.flatnonzero(obs["action_mask"]))))
    assert totals["p1"] + totals["p2"] == 0.0
    assert {totals["p1"], totals["p2"]} == {1.0, -1.0}


def test_atomic_mask_is_sound():
    """Every unmasked atomic action is accepted by the engine (tested on clones)."""
    rng = np.random.default_rng(3)
    checked = 0
    for seed in range(4):
        e = raw_env()
        e.reset(seed=seed)
        steps = 0
        while e.agents and steps < 1500:
            a = e.agent_selection
            if e.terminations[a] or e.truncations[a]:
                e.step(None)
                continue
            pend = e.g.pending
            if e._builder is None and pend.type not in COMPOUND_TYPES:
                mask = atomic_mask(e.g, a)
                for act in np.flatnonzero(mask):
                    g2 = _clone(e.g)
                    assert apply_atomic(g2, a, int(act)), \
                        f"{A.decode(int(act))} rejected for {pend.type}"
                    checked += 1
            mask = e.observe(a)["action_mask"]
            e.step(int(rng.choice(np.flatnonzero(mask))))
            steps += 1
    assert checked > 200       # exercised a meaningful number of (state, action) pairs


def test_compound_types_are_exercised():
    """Compound builders actually run (and finalize) in real games."""
    rng = np.random.default_rng(5)
    seen = set()
    for seed in range(20):
        e = raw_env()
        e.reset(seed=seed)
        steps = 0
        while e.agents and steps < 4000:
            a = e.agent_selection
            if e.terminations[a] or e.truncations[a]:
                e.step(None)
                continue
            if e.g.pending.type in COMPOUND_TYPES:
                seen.add(e.g.pending.type)
            e.step(int(rng.choice(np.flatnonzero(e.observe(a)["action_mask"]))))
            steps += 1
    assert seen, "no compound decisions encountered across 20 games"
