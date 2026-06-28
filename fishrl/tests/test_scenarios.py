"""Scenario-curriculum harness tests: the base env is unchanged, pools build from
real play, the terminators give the right terminal result, and scenario winners
propagate through the belief wrapper + collector into the PPO buffer."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fishrl.env.aec_env import FishAEC
from fishrl.env.driver import terminal_winner
from fishrl.spaces import action_space as A
from fishrl.train.scenarios import ScenarioEnv, sample_scenario_name, scenario_names
from fishrl.train.scenarios.board_presence import BoardPresenceScenario, _creatures
from fishrl.train.scenarios.deckout import DeckoutScenario, LIBRARY
from fishrl.train.scenarios.establish_clock import EstablishClockScenario
from fishrl.train.scenarios.free_attack import FreeAttackScenario
from fishrl.train.scenarios.known_threat import THREAT, TOOL, KnownThreatScenario


def _small(scn_cls):
    s = scn_cls()
    s.pool_n, s.pool_max_games = 6, 3000
    return s


def _legal(env):
    return np.flatnonzero(env.observe(env.agent_selection)["action_mask"])


def _play_random(env, rng, max_steps=4000):
    n = 0
    while env.agents and n < max_steps:
        n += 1
        s = env.agent_selection
        if env.terminations[s] or env.truncations[s]:
            env.step(None)
            continue
        ls = _legal(env)
        if ls.size == 0:
            env.step(None)
            continue
        env.step(int(rng.choice(ls)))


# ── base env is untouched by the hook ────────────────────────────────────────
def test_base_env_terminal_override_is_none_and_winner_matches():
    env = FishAEC(max_decisions=600)
    env.reset(seed=1)
    assert env._terminal_override() is None
    _play_random(env, np.random.default_rng(0))
    assert env.winner == terminal_winner(env.g)
    assert env.winner in ("p1", "p2", None)


# ── pools build from real play and every state satisfies the predicate ───────
@pytest.mark.parametrize("scn_cls", [FreeAttackScenario, EstablishClockScenario,
                                     BoardPresenceScenario, DeckoutScenario])
def test_pool_builds_and_states_match_predicate(scn_cls):
    scn = _small(scn_cls)
    assert scn.ensure_pool() >= 1
    for g in scn._pool:
        assert scn.predicate(SimpleNamespace(g=g))


# ── free-attack: attacking wins, declining loses (the deliberate hard rule) ───
def test_free_attack_win_on_attack():
    scn = _small(FreeAttackScenario)
    env = ScenarioEnv(scn, max_decisions=600)
    env.reset(seed=0)
    assert env.g.pending.type == "declare_attackers" and env.agent_selection == "p1"
    picks = [int(a) for a in _legal(env) if A.decode(int(a))[0] == "PICK_A"]
    assert picks, "expected at least one eligible attacker to declare"
    env.step(picks[0])                 # declare the attacker
    env.step(A.aid("COMMIT"))          # finalize -> combat resolves
    assert env.winner == "p1"


def test_free_attack_loss_on_decline():
    scn = _small(FreeAttackScenario)
    env = ScenarioEnv(scn, max_decisions=600)
    env.reset(seed=0)
    env.step(A.aid("COMMIT"))          # declare ZERO attackers -> unused attacker
    assert env.winner == "p2"


# ── establish-a-clock: terminates with a structural race result ──────────────
def test_establish_clock_terminates_with_result():
    scn = _small(EstablishClockScenario)
    env = ScenarioEnv(scn, max_decisions=1500)
    env.reset(seed=0)
    _play_random(env, np.random.default_rng(3))
    assert not env.agents or all(env.terminations.values()) or all(env.truncations.values())
    assert env.winner in ("p1", "p2", None)


# ── scenario winner flows through belief wrapper + collector into the buffer ──
def test_scenario_winner_propagates_to_buffer():
    from fishrl.train.belief_env import BeliefAugmentedEnv
    from fishrl.train.collector import collect_games, random_act_fn

    scn = _small(FreeAttackScenario)
    senv = BeliefAugmentedEnv(None, belief=False, env=ScenarioEnv(scn, max_decisions=600))
    buf = collect_games(senv, random_act_fn(np.random.default_rng(0)), 4, 0,
                        critic=None, max_decisions=600)
    assert buf.steps
    # free-attack always resolves to a terminal win/loss (never a None draw)
    assert all(s.winner in ("p1", "p2") for s in buf.steps)


# ── known-threat denial: surgery is sound and denial flips the outcome ───────
def _drive_known_threat(env, rng, counter: bool):
    """Play p1 to maximally PASS, optionally countering the threat with the held
    Memory Lapse whenever it is on the stack."""
    guard = 0
    while env.agents and guard < 5000:
        guard += 1
        a = env.agent_selection
        if env.terminations[a] or env.truncations[a]:
            env.step(None)
            continue
        ls = _legal(env)
        if ls.size == 0:
            env.step(None)
            continue
        g = env.g
        act = None
        if counter and g.pending and g.pending.type == "choose_targets" \
                and g.pending.context.get("name") == TOOL:
            for idx, tid in enumerate(g.pending.context.get("legal", [])):
                if g.objects.get(tid) and g.objects[tid].name == THREAT:
                    cand = [x for x in ls if A.decode(int(x)) == ("PICK_SINGLE", idx)]
                    act = int(cand[0]) if cand else None
        if counter and act is None and g.pending and g.pending.type == "priority" \
                and any(s.kind == "spell" and s.controller == "p2"
                        and g.objects[s.source_instance_id].name == THREAT for s in g.stack):
            ml = [i for i in g.players["p1"].hand if g.objects[i].name == TOOL]
            if ml:
                hi = g.players["p1"].hand.index(ml[0])
                cand = [x for x in ls if A.decode(int(x)) == ("PLAY_HAND", hi)]
                act = int(cand[0]) if cand else None
        if act is None:
            passes = [x for x in ls if A.decode(int(x))[0] == "PASS"]
            act = int(passes[0]) if passes else int(rng.choice(ls))
        env.step(act)
    return env.winner


def test_known_threat_surgery_preconditions():
    scn = _small(KnownThreatScenario)
    env = ScenarioEnv(scn, max_decisions=800)
    env.reset(seed=0)
    g = env.g
    assert g.library and g.objects[g.library[0].instance_id].name == THREAT   # threat on top
    assert any(g.objects[i].name == TOOL for i in g.players["p1"].hand)        # learner holds the counter
    assert not any(g.objects[i].name == THREAT for i in g.players["p2"].hand)  # p2's copies cleared
    assert g.players["p2"].is_ai                                              # opponent is engine-driven


def _reset(scn, seed):
    env = ScenarioEnv(scn, max_decisions=800)
    env.reset(seed=seed)
    return env


def test_known_threat_denial_flips_outcome():
    scn = _small(KnownThreatScenario)
    seeds = range(8)
    passive = [_drive_known_threat(_reset(scn, s), np.random.default_rng(s), counter=False)
               for s in seeds]
    counter = [_drive_known_threat(_reset(scn, s), np.random.default_rng(s), counter=True)
               for s in seeds]
    # undefended the threat lands (p2 wins); countering denies it for more seeds.
    assert passive.count("p2") >= 1, "threat should land when p1 is passive"
    assert counter.count("p1") > passive.count("p1"), "countering must deny more than passivity"


# ── board presence: a real fight that ends decisively on an empty board ──────
def test_board_presence_fight_is_decisive():
    scn = _small(BoardPresenceScenario)
    env = ScenarioEnv(scn, max_decisions=800)
    env.reset(seed=0)
    assert _creatures(env.g, "p1") >= 1 and _creatures(env.g, "p2") >= 1   # both start with board
    winners = []
    for s in range(6):
        e = _reset(scn, s)
        _play_random(e, np.random.default_rng(s))
        winners.append(e.winner)
    assert all(w in ("p1", "p2", None) for w in winners)
    assert any(w in ("p1", "p2") for w in winners)        # the fight resolves decisively


# ── deckout: surgery forces a creatureless, trimmed-library deckout race ──────
def test_deckout_surgery_and_resolution():
    scn = _small(DeckoutScenario)
    env = ScenarioEnv(scn, max_decisions=800)
    env.reset(seed=0)
    g = env.g
    assert _creatures(g, "p1") == 0 and _creatures(g, "p2") == 0   # battlefields cleared
    assert len(g.library) <= LIBRARY                              # library trimmed
    _play_random(env, np.random.default_rng(0))
    assert env.winner in ("p1", "p2", None)                       # resolves to a terminal


# ── registry / weighted sampling ─────────────────────────────────────────────
def test_sample_scenario_name_respects_weights():
    rng = np.random.default_rng(0)
    only = {"free_attack": 1.0, "establish_clock": 0.0}
    picks = {sample_scenario_name(only, rng) for _ in range(50)}
    assert picks == {"free_attack"}
    assert set(scenario_names()) == {"free_attack", "establish_clock", "known_threat",
                                     "board_presence", "deckout"}
