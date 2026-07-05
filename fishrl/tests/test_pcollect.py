"""Parallel collection (collect_workers > 0) must be a wall-clock change only:
the same games are sampled (same RNG streams/seeds/seat parity as the serial
branch), every buffer comes back, and the telemetry counts match the serial
path. Spawning real worker processes costs a few seconds -- kept to one test."""
from __future__ import annotations

import re

from fishrl.train.config import Config
from fishrl.train.train_loop import build_models, train


def _run(workers: int) -> dict:
    cfg = Config(iters=2, games_per_iter=6, warmup_games=2, warmup_epochs=1,
                 max_decisions=300, report_winrate_games=0,
                 pool_frac=0.5, league_size=2,
                 pfsp_anchors=("random", "attacker", "heuristic"),
                 collect_workers=workers, seed=0)
    lines: list = []
    train(cfg, build_models(cfg), log=lines.append)
    status = next(line for line in reversed(lines) if line.startswith("[status"))
    league = next((line for line in reversed(lines) if line.startswith("[league")), "")
    out = {"games": int(re.search(r"games=(\d+)", status).group(1)),
           "T": int(re.search(r"T=(\d+)", status).group(1)),
           # member NAMES only (a set: the summary is sorted by EMA win-rate,
           # which legitimately differs run-to-run with game outcomes)
           "members": frozenset(re.findall(r"([\w@.]+)=\d\.\d\d\(", league))}
    m = re.search(r"games=(\d+) trained", league)
    out["league_games"] = int(m.group(1)) if m else 0
    return out


def test_parallel_matches_serial_game_accounting():
    serial = _run(0)
    par = _run(2)
    # Identical sampling decisions -> identical game counts, same league members
    # seen in the same composition. Transitions differ (independent action-RNG
    # streams) but must exist in quantity.
    assert par["games"] == serial["games"] == 12          # 2 iters x 6 games
    assert par["league_games"] == serial["league_games"]
    assert par["members"] == serial["members"]
    assert par["T"] > 200 and serial["T"] > 200
