"""Config.scenario_selfplay_frac (2026-08-27): scenario games roll self-play —
both seats the current policy — with probability frac, instead of handing p2 to
the v1.3 engine seat. The roll is the episode rng's FIRST draw, so the
constructed board is seed-identical across modes and the trainer can recompute
the mode from the game seed alone (selfplay_mode)."""
from __future__ import annotations

import numpy as np
import pytest

from fishrl.spaces import action_space as A
from fishrl.train.scenarios import (ScenarioEnv, get_scenario, scenario_selfplay,
                                    selfplay_mode, set_scenario_selfplay)
from fishrl.train.scenarios.constructed import FishWar


@pytest.fixture(autouse=True)
def _reset():
    yield
    set_scenario_selfplay(0.0)


def _board_key(g):
    return (tuple(s.instance_id for s in g.library),
            tuple(g.players["p1"].hand), tuple(g.players["p2"].hand),
            tuple(g.players["p1"].battlefield), tuple(g.players["p2"].battlefield),
            g.turn_number, g.current_step)


def test_roll_controls_engine_seat_and_board_is_mode_invariant():
    scn = FishWar(); scn.pool_n = 6
    set_scenario_selfplay(0.0)
    g_script = scn.sample(np.random.default_rng(7))
    assert g_script.players["p2"].is_ai and g_script.players["p2"].ai_profile == "heuristic_1_3"
    assert not g_script.players["p1"].is_ai
    set_scenario_selfplay(1.0)
    g_self = scn.sample(np.random.default_rng(7))
    assert not g_self.players["p2"].is_ai, "self-play roll must skip make_engine_heuristic"
    # same seed -> byte-identical board either mode (the roll is a dedicated draw)
    assert _board_key(g_script) == _board_key(g_self)


def test_mode_recomputable_from_seed():
    scn = FishWar(); scn.pool_n = 6
    set_scenario_selfplay(0.5)
    hits = 0
    for seed in range(40):
        g = scn.sample(np.random.default_rng(seed))
        rolled_self = not g.players["p2"].is_ai
        assert rolled_self == selfplay_mode(seed), seed
        hits += rolled_self
    assert 5 < hits < 35                       # ~half at frac=0.5
    assert scenario_selfplay() == 0.5
    assert selfplay_mode(3, frac=0.0) is False and selfplay_mode(3, frac=1.0) is True


def test_selfplay_scenario_surfaces_both_seats():
    scn = FishWar(); scn.pool_n = 6
    set_scenario_selfplay(1.0)
    env = ScenarioEnv(scn, max_decisions=400)
    env.reset(seed=11)
    rng = np.random.default_rng(0)
    seats = set()
    for agent in env.agent_iter(max_iter=600):
        if env.terminations[agent] or env.truncations[agent]:
            env.step(None); continue
        seats.add(agent)
        if seats == {"p1", "p2"}:
            break
        mask = env.observe(agent)["action_mask"]
        env.step(int(rng.choice(np.flatnonzero(mask))))
    assert seats == {"p1", "p2"}, "both seats must reach the policy under self-play"
    # script mode: p2 never surfaces (the engine auto-plays it)
    set_scenario_selfplay(0.0)
    env.reset(seed=11)
    seats = set()
    for i, agent in enumerate(env.agent_iter(max_iter=200)):
        if env.terminations[agent] or env.truncations[agent]:
            env.step(None); continue
        seats.add(agent)
        if i >= 60:
            break
        mask = env.observe(agent)["action_mask"]
        env.step(int(rng.choice(np.flatnonzero(mask))))
    assert seats == {"p1"}, "engine seat must keep p2 off the policy path"


def test_registered_scenarios_keep_engine_seat_default():
    # the roll composes with EVERY constructed scenario (engine_seat == 'p2');
    # frac=0 must reproduce today's behaviour exactly
    set_scenario_selfplay(0.0)
    scn = get_scenario("response_window")
    g = scn.sample(np.random.default_rng(3))
    assert g.players["p2"].is_ai and g.players["p2"].ai_profile == "heuristic_1_3"
