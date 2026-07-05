"""Heuristic versions behind separate engine profiles: "heuristic" (v1.0 —
the run's long-standing opponent, eval anchor, scenario bot), "heuristic_1_1"
(the testbench line frozen at its original release) and "heuristic_1_2" (the
current testbench line) — the latter two PFSP pool opponents only. The dispatch
must route each profile to its own module, and every versioned bot must play
clean full games under THIS repo's engine."""
from __future__ import annotations

import numpy as np

from fishrl.forgetful_fish import ai, ai_v1_1, ai_v1_2
from fishrl.forgetful_fish import engine as E
from fishrl.forgetful_fish.cards import load_decklist
from fishrl.opponents.heuristic import HeuristicMatch
from fishrl.train.config import Config
from fishrl.train.pfsp import PFSPLeague

VERSIONED = {"heuristic_1_1": ai_v1_1, "heuristic_1_2": ai_v1_2}


def test_engine_dispatch_routes_profiles_to_modules():
    g = E.new_sandbox_game(load_decklist(), seed=0, ai_profile="heuristic")
    assert g.players["p2"].ai_profile == "heuristic"
    assert E._ai_mod(g, "p2") is ai
    assert E._heuristic(g, "p2")
    for profile, mod in VERSIONED.items():
        gv = E.new_sandbox_game(load_decklist(), seed=0, ai_profile=profile)
        assert gv.players["p2"].ai_profile == profile
        assert E._ai_mod(gv, "p2") is mod
        # every version counts as "a heuristic" for the engine's AI gates
        assert E._heuristic(gv, "p2")
    # an unknown profile falls back to the default heuristic
    gx = E.new_sandbox_game(load_decklist(), seed=0, ai_profile="bogus")
    assert gx.players["p2"].ai_profile == "heuristic"
    # argless call (module-level uses) is v1.0
    assert E._ai_mod() is ai


def _play_random_match(profile, seed, max_steps=3000):
    match = HeuristicMatch(max_decisions=800, profile=profile)
    obs = match.reset(seed=seed)
    rng = np.random.default_rng(seed)
    done, n = False, 0
    while not done and n < max_steps:
        n += 1
        legal = np.flatnonzero(obs["action_mask"])
        if legal.size == 0:
            break
        obs, _r, done, _i = match.step(int(rng.choice(legal)))
    return match


def test_versioned_heuristics_play_full_games_under_this_engine():
    # A random p1 vs each versioned bot must resolve (or hit the cap) without the
    # AI stalling or raising — the profile self-check inside each module's
    # resolve_pending must accept its own profile or p2's decisions silently stop
    # resolving.
    for profile in VERSIONED:
        for seed in range(3):
            match = _play_random_match(profile, seed)
            w = match.g.result.get("winner")
            assert w in ("p1", "p2", None)
            # the bot actually acted: it played lands / took game actions (its
            # battlefield is non-empty by game end in any non-degenerate game)
            assert match.g.players["p2"].battlefield or w is not None


def test_league_includes_all_heuristics_and_pool_routes_them():
    cfg = Config()
    league = PFSPLeague.from_config(cfg)
    kinds = {m.kind for m in league.members()}
    assert {"random", "attacker", "heuristic", "heuristic_1_1", "heuristic_1_2"} <= kinds
    # engine-driven kinds must NOT go through collect_vs_opponent
    import pytest
    from fishrl.train.collector import collect_vs_opponent
    from fishrl.train.pfsp import LeagueMember
    for kind in ("heuristic", "heuristic_1_1", "heuristic_1_2"):
        with pytest.raises(ValueError):
            collect_vs_opponent(None, LeagueMember(name=kind, kind=kind), 1, 0)


def test_league_checkpoint_restore_matches_by_name_across_new_anchor():
    # A checkpoint saved before the versioned heuristics existed restores onto the
    # full league: saved EMAs land on their names, the new anchors stay fresh.
    old = PFSPLeague(anchors=[], selves=__import__("collections").deque(maxlen=0))
    from fishrl.train.pfsp import LeagueMember
    old.anchors = [LeagueMember(name=k, kind=k) for k in ("random", "attacker", "heuristic")]
    old.update(old.anchors[2], False)                    # heuristic EMA moves off 0.5
    state = old.state_dict()

    new = PFSPLeague.from_config(Config())
    new.load_state_dict(state)
    by = {m.name: m for m in new.anchors}
    assert by["heuristic"].games == 1 and by["heuristic"].wr < 0.5
    for fresh in ("heuristic_1_1", "heuristic_1_2"):
        assert by[fresh].games == 0 and by[fresh].wr == 0.5
