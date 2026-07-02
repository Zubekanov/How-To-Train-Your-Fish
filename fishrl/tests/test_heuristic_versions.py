"""Two heuristic versions behind separate engine profiles: "heuristic" (v1.0 —
the run's long-standing opponent, eval anchor, scenario bot) and "heuristic_1_1"
(the stronger testbench line, a PFSP pool opponent only). The dispatch must route
each profile to its own module, and v1.1 must play clean full games under THIS
repo's engine."""
from __future__ import annotations

import numpy as np

from fishrl.forgetful_fish import ai, ai_v1_1
from fishrl.forgetful_fish import engine as E
from fishrl.forgetful_fish.cards import load_decklist
from fishrl.opponents.heuristic import HeuristicMatch
from fishrl.train.config import Config
from fishrl.train.pfsp import PFSPLeague


def test_engine_dispatch_routes_profiles_to_modules():
    g = E.new_sandbox_game(load_decklist(), seed=0, ai_profile="heuristic")
    assert g.players["p2"].ai_profile == "heuristic"
    assert E._ai_mod(g, "p2") is ai
    g11 = E.new_sandbox_game(load_decklist(), seed=0, ai_profile="heuristic_1_1")
    assert g11.players["p2"].ai_profile == "heuristic_1_1"
    assert E._ai_mod(g11, "p2") is ai_v1_1
    # both count as "a heuristic" for the engine's AI gates
    assert E._heuristic(g, "p2") and E._heuristic(g11, "p2")
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


def test_v1_1_plays_full_games_under_this_engine():
    # A random p1 vs the v1.1 bot must resolve (or hit the cap) without the AI
    # stalling or raising — the profile self-check inside ai_v1_1.resolve_pending
    # must accept its own profile or p2's decisions silently stop resolving.
    for seed in range(3):
        match = _play_random_match("heuristic_1_1", seed)
        w = match.g.result.get("winner")
        assert w in ("p1", "p2", None)
        # the bot actually acted: it played lands / took game actions (its
        # battlefield is non-empty by game end in any non-degenerate game)
        assert match.g.players["p2"].battlefield or w is not None


def test_league_includes_both_heuristics_and_pool_routes_them():
    cfg = Config()
    league = PFSPLeague.from_config(cfg)
    kinds = {m.kind for m in league.members()}
    assert {"random", "attacker", "heuristic", "heuristic_1_1"} <= kinds
    # engine-driven kinds must NOT go through collect_vs_opponent
    import pytest
    from fishrl.train.collector import collect_vs_opponent
    from fishrl.train.pfsp import LeagueMember
    for kind in ("heuristic", "heuristic_1_1"):
        with pytest.raises(ValueError):
            collect_vs_opponent(None, LeagueMember(name=kind, kind=kind), 1, 0)


def test_league_checkpoint_restore_matches_by_name_across_new_anchor():
    # A checkpoint saved before heuristic_1_1 existed restores onto the new
    # 4-anchor league: saved EMAs land on their names, the new anchor stays fresh.
    old = PFSPLeague(anchors=[], selves=__import__("collections").deque(maxlen=0))
    from fishrl.train.pfsp import LeagueMember
    old.anchors = [LeagueMember(name=k, kind=k) for k in ("random", "attacker", "heuristic")]
    old.update(old.anchors[2], False)                    # heuristic EMA moves off 0.5
    state = old.state_dict()

    new = PFSPLeague.from_config(Config())
    new.load_state_dict(state)
    by = {m.name: m for m in new.anchors}
    assert by["heuristic"].games == 1 and by["heuristic"].wr < 0.5
    assert by["heuristic_1_1"].games == 0 and by["heuristic_1_1"].wr == 0.5
