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
from fishrl.train.scenarios.deckout import DeckoutScenario
from fishrl.train.scenarios.known_threat import (THREAT, TOOL, KnownThreatScenario,
                                                 KnownThreatRandomScenario)


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
@pytest.mark.parametrize("scn_cls", [BoardPresenceScenario, DeckoutScenario])
def test_pool_builds_and_states_match_predicate(scn_cls):
    scn = _small(scn_cls)
    assert scn.ensure_pool() >= 1
    for g in scn._pool:
        assert scn.predicate(SimpleNamespace(g=g))


# ── free-attack hard rule (env-level, all training): short declare -> loss ────
def _install(g, enforce):
    import copy
    from fishrl.env.driver import STOPS_MODES
    from fishrl.forgetful_fish import engine as E
    env = FishAEC(max_decisions=600, enforce_free_attack=enforce)
    g = copy.deepcopy(g)
    for pid in env.possible_agents:
        E.set_player_stops(g, pid, STOPS_MODES["default"])
    env.g = g
    env.agents = list(env.possible_agents)
    env.rewards = {a: 0.0 for a in env.agents}
    env._cumulative_rewards = {a: 0.0 for a in env.agents}
    env.terminations = {a: False for a in env.agents}
    env.truncations = {a: False for a in env.agents}
    env.infos = {a: {} for a in env.agents}
    env._builder = None
    env._decisions = 0
    env._scenario_result = None
    env._forced_result = None
    env.agent_selection = env.possible_agents[0]
    env._refresh()
    return env


def test_free_attack_rule_short_declare_loses():
    from fishrl.forgetful_fish import engine as E
    from fishrl.train.scenarios.pool import build_snapshots

    def pred(env):
        g = env.g
        p = g.pending
        return (p is not None and p.type == "declare_attackers" and p.player == "p1"
                and len(p.context.get("eligible", [])) >= 1
                and not any(E._is_creature(g.objects[i]) for i in g.players["p2"].battlefield
                            if i in g.objects))
    states, _ = build_snapshots(pred, 3, seed=101, max_games=2500)
    assert states, "expected a free-attack state in the budget"
    g = states[0]
    # declaring ZERO with the rule ON -> the seat loses; rule OFF -> no forced loss
    env = _install(g, True)
    env.step(A.aid("COMMIT"))
    assert env.winner == "p2"
    env = _install(g, False)
    env.step(A.aid("COMMIT"))
    assert env.winner != "p2" or terminal_winner(env.g) == "p2"
    # declaring ALL eligible -> no forced loss
    env = _install(g, True)
    while True:
        ls = _legal(env)
        picks = [a for a in ls if A.decode(int(a))[0] == "PICK_A"]
        if not picks:
            break
        env.step(int(picks[0]))
    env.step(A.aid("COMMIT"))
    assert env._forced_result is None


# ── scenario winner flows through belief wrapper + collector into the buffer ──
def test_scenario_winner_propagates_to_buffer():
    from fishrl.train.belief_env import BeliefAugmentedEnv
    from fishrl.train.collector import collect_games, random_act_fn

    scn = _small(BoardPresenceScenario)
    senv = BeliefAugmentedEnv(None, belief=False, env=ScenarioEnv(scn, max_decisions=600))
    buf = collect_games(senv, random_act_fn(np.random.default_rng(0)), 4, 0,
                        critic=None, max_decisions=600)
    assert buf.steps
    assert all(s.winner in ("p1", "p2", None) for s in buf.steps)


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


def test_known_threat_manufacture():
    from fishrl.forgetful_fish import engine as E
    scn = _small(KnownThreatScenario)
    env = ScenarioEnv(scn, max_decisions=800)
    env.reset(seed=0)
    g = env.g
    # the bot: exactly 10 Islands, empty hand, no creatures, engine-driven
    isl = sum(1 for i in g.players["p2"].battlefield if g.objects[i].name == "Island")
    assert isl == 10 and len(g.players["p2"].hand) == 0
    assert not any(E._is_creature(g.objects[i]) for i in g.players["p2"].battlefield)
    assert g.players["p2"].is_ai
    # the agent also has 10 Islands (ample mana to answer)
    assert sum(1 for i in g.players["p1"].battlefield if g.objects[i].name == "Island") == 10
    # the threat on top, the agent holding its counter (curated grip)
    assert g.objects[g.library[0].instance_id].name == THREAT
    assert any(g.objects[i].name == TOOL for i in g.players["p1"].hand)


def test_known_threat_random_manufacture():
    scn = _small(KnownThreatRandomScenario)
    env = ScenarioEnv(scn, max_decisions=800)
    env.reset(seed=0)
    g = env.g
    # same denial frame: bot with 10 Islands + empty hand, agent with 10 Islands,
    # the threat on top of the shared library...
    assert sum(1 for i in g.players["p2"].battlefield if g.objects[i].name == "Island") == 10
    assert len(g.players["p2"].hand) == 0
    assert sum(1 for i in g.players["p1"].battlefield if g.objects[i].name == "Island") == 10
    assert g.objects[g.library[0].instance_id].name == THREAT
    # ...but a RANDOM 7-card grip rather than the curated counter + manipulation.
    assert len(g.players["p1"].hand) == 7


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


# ── board presence: manufactured 4-life / one-creature fight, then deckout ───
def test_board_presence_manufacture_and_resolves():
    from fishrl.forgetful_fish import engine as E
    scn = _small(BoardPresenceScenario)
    env = ScenarioEnv(scn, max_decisions=2000)
    env.reset(seed=0)
    g = env.g
    assert _creatures(g, "p1") == 1 and _creatures(g, "p2") == 1     # exactly one creature each
    assert g.players["p1"].life == 4 and g.players["p2"].life == 4   # 4 life each
    assert _lands(g, "p1") == _lands(g, "p2") and 4 <= _lands(g, "p1") <= 10  # equal random Islands
    assert sum(E._is_creature(g.objects[s.instance_id]) for s in g.library) == 0  # creatureless deck
    # no cards are dropped: non-Island lands stay in the library, so every instance
    # is accounted for across the zones (only creatures leave, to exile).
    accounted = (len(g.library) + len(g.graveyard) + len(g.exile) + len(g.stack)
                 + sum(len(g.players[p].hand) + len(g.players[p].battlefield) for p in ("p1", "p2")))
    assert accounted == len(g.objects)
    assert any("Island" not in (g.objects[s.instance_id].type_line or "")
               and "Land" in (g.objects[s.instance_id].type_line or "") for s in g.library)
    winners = []
    for s in range(6):
        e = ScenarioEnv(scn, max_decisions=2000)
        e.reset(seed=s)
        _play_random(e, np.random.default_rng(s))
        winners.append(e.winner)
    assert all(w in ("p1", "p2", None) for w in winners)
    assert any(w in ("p1", "p2") for w in winners)        # resolves decisively


# ── deckout: surgery forces a creatureless, trimmed-library deckout race ──────
def _all_creatures(g):
    from fishrl.forgetful_fish import engine as E
    ids = ([s.instance_id for s in g.library]
           + g.players["p1"].hand + g.players["p2"].hand
           + g.players["p1"].battlefield + g.players["p2"].battlefield)
    return sum(1 for i in ids if g.objects.get(i) and E._is_creature(g.objects[i]))


def _lands(g, seat):
    return sum(1 for i in g.players[seat].battlefield
               if "Land" in (g.objects[i].type_line or ""))


def test_deckout_manufacture_and_resolves():
    scn = _small(DeckoutScenario)
    for s in range(4):
        env = ScenarioEnv(scn, max_decisions=1500)
        env.reset(seed=s)
        g = env.g
        assert _lands(g, "p1") == _lands(g, "p2") and 4 <= _lands(g, "p1") <= 10  # equal random 4..10
        assert len(g.players["p1"].hand) == 7 and len(g.players["p2"].hand) == 7  # 7-card hands
        assert len(g.library) <= 40                  # ~40-card library (rest in graveyard)
        assert _all_creatures(g) == 0                # truly creatureless: every zone
    env = ScenarioEnv(scn, max_decisions=1500)
    env.reset(seed=0)
    _play_random(env, np.random.default_rng(0))
    assert env.winner in ("p1", "p2", None)


# ── registry / weighted sampling ─────────────────────────────────────────────
def test_sample_scenario_name_respects_weights():
    rng = np.random.default_rng(0)
    only = {"known_threat": 1.0, "board_presence": 0.0, "deckout": 0.0}
    picks = {sample_scenario_name(only, rng) for _ in range(50)}
    assert picks == {"known_threat"}
    assert set(scenario_names()) == {"known_threat", "known_threat_random",
                                     "board_presence", "deckout"}
