"""The scenario_frac > 0 branch of train(): scenario games merge into the PPO
buffer, the scenario league's per-scenario EMA updates, and league state
round-trips through the checkpoint. Previously this branch ran only on the live
service."""
from __future__ import annotations

import numpy as np

from fishrl.train import scenarios as scn_mod
from fishrl.train.config import Config
from fishrl.train.pfsp import PFSPLeague
from fishrl.train.scenarios.board_presence import BoardPresenceScenario
from fishrl.train.train_loop import build_models, train


def _small_pool(name: str, scn_cls):
    """Seed the scenario singleton cache with a small-pool instance so the train
    smoke test doesn't build the full 200-snapshot pool."""
    s = scn_cls()
    s.pool_n, s.pool_max_games = 4, 3000
    scn_mod._INSTANCES[name] = s
    return s


def _only(name: str) -> dict:
    """Weights whitelisting ONE scenario. Explicit zeros matter: a name missing
    from scenario_weights defaults to weight 1.0 (so newly registered scenarios
    auto-join in production), so a single-entry dict does NOT exclude the rest."""
    from fishrl.train.scenarios import scenario_names
    return {n: (1.0 if n == name else 0.0) for n in scenario_names()}


def test_train_scenario_branch_runs_and_updates_league():
    _small_pool("board_presence", BoardPresenceScenario)
    cfg = Config(iters=2, games_per_iter=4, warmup_games=2, warmup_epochs=1,
                 max_decisions=300, report_winrate_games=0,
                 pool_frac=0.0, league_size=0,
                 scenario_frac=0.5,
                 scenario_weights=_only("board_presence"))   # single scenario -> small pool only
    m = build_models(cfg)
    train(cfg, m, log=lambda *a, **k: None)
    # the branch ran: the (module-cached) scenario instance has a built pool
    assert scn_mod._INSTANCES["board_presence"]._pool


def test_train_scenarios_in_pool_mode():
    # Scenarios as MAIN-league members: with no other anchors and no past-self ring,
    # every pool draw is a scenario game, so the pool path must build the scenario
    # pool, count games into scen_mix (not opp_mix), and update the member's EMA.
    _small_pool("board_presence", BoardPresenceScenario)
    cfg = Config(iters=2, games_per_iter=4, warmup_games=2, warmup_epochs=1,
                 max_decisions=300, report_winrate_games=0,
                 pool_frac=0.5, pfsp_anchors=(), league_size=0,
                 scenarios_in_pool=True, scenario_frac=0.0,
                 scenario_weights=_only("board_presence"))
    lines = []
    m = build_models(cfg)
    train(cfg, m, log=lines.append)
    assert scn_mod._INSTANCES["board_presence"]._pool
    # the final report carries the [scenario] line with the pool-mode counts/wr
    scen_lines = [ln for ln in lines if ln.startswith("[scenario")]
    assert scen_lines and "board_presence=" in scen_lines[-1]
    # 2 iters x 2 pool games each -> 4 scenario games counted
    assert "games=4" in scen_lines[-1]


def test_scenarios_in_pool_overrides_carveout():
    # Both set: pool mode wins; the carve-out must not double-spend the budget.
    _small_pool("board_presence", BoardPresenceScenario)
    cfg = Config(iters=1, games_per_iter=4, warmup_games=2, warmup_epochs=1,
                 max_decisions=300, report_winrate_games=0,
                 pool_frac=0.5, pfsp_anchors=(), league_size=0,
                 scenarios_in_pool=True, scenario_frac=0.5,
                 scenario_weights=_only("board_presence"))
    lines = []
    train(cfg, build_models(cfg), log=lines.append)
    scen_lines = [ln for ln in lines if ln.startswith("[scenario")]
    # pool mode: 2 of 4 games (pool_frac), NOT 2 (carve-out) + 1 (pool) = 3
    assert scen_lines and "games=2" in scen_lines[-1]


def test_scenario_league_state_roundtrip():
    cfg = Config(scenario_frac=0.5,
                 scenario_weights={"board_presence": 1.0, "deckout": 1.0})
    league = PFSPLeague.scenario_league(cfg, ["board_presence", "deckout"])
    league.update(league.anchors[0], True)
    league.update(league.anchors[0], True)
    league.update(league.anchors[1], False)
    state = league.state_dict()

    fresh = PFSPLeague.scenario_league(cfg, ["board_presence", "deckout"])
    fresh.load_state_dict(state)
    assert fresh.anchors[0].games == 2 and fresh.anchors[0].wr > 0.5
    assert fresh.anchors[1].games == 1 and fresh.anchors[1].wr < 0.5

    # membership drift: a scenario added after the save keeps a fresh EMA
    grown = PFSPLeague.scenario_league(
        Config(scenario_weights={"board_presence": 1.0, "deckout": 1.0, "brand_new": 1.0}),
        ["board_presence", "deckout", "brand_new"])
    grown.load_state_dict(state)
    assert grown.anchors[2].wr == 0.5 and grown.anchors[2].games == 0
