"""Parallel collection (collect_workers > 0) must be a wall-clock change only:
the same games are sampled (same RNG streams/seeds/seat parity as the serial
branch), every buffer comes back, and the telemetry counts match the serial
path. Spawning real worker processes costs a few seconds -- kept to one test."""
from __future__ import annotations

import re

import pytest

from fishrl.train.config import Config
from fishrl.train.pcollect import ParallelCollector, parse_affinity
from fishrl.train.train_loop import build_models, train


def _run(workers: int, affinity: str = "") -> dict:
    cfg = Config(iters=2, games_per_iter=6, warmup_games=2, warmup_epochs=1,
                 max_decisions=300, report_winrate_games=0,
                 pool_frac=0.5, league_size=2,
                 pfsp_anchors=("random", "attacker", "heuristic"),
                 collect_workers=workers, collect_affinity=affinity, seed=0)
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


def test_affinity_spec_parses_strictly():
    assert parse_affinity("") == [] and parse_affinity(None) == []
    assert parse_affinity("0,2,4") == [0, 2, 4]
    assert parse_affinity("4, 2,2") == [2, 4]            # order/dupes normalized
    with pytest.raises(ValueError):                      # typo fails at startup,
        parse_affinity("0,x")                            # never silently unpinned


def test_pipeline_flag_ignored_without_workers():
    """--pipeline-collect on a serial trainer must warn and fall back, never crash
    (the ODROID path if the flag ever leaks into train.args)."""
    cfg = Config(iters=1, games_per_iter=2, warmup_games=1, warmup_epochs=1,
                 max_decisions=100, report_winrate_games=0, pool_frac=0.0,
                 collect_workers=0, pipeline_collect=True, seed=3)
    lines: list = []
    train(cfg, build_models(cfg), log=lines.append)
    assert any("--pipeline-collect ignored" in line for line in lines)
    status = next(line for line in reversed(lines) if line.startswith("[status"))
    assert int(re.search(r"games=(\d+)", status).group(1)) == 2


def test_pipelined_run_completes_with_full_game_accounting():
    """Pipelined collection changes WHEN games are played (overlapped with the
    update, one-update-stale weights), never HOW MANY: every iteration's games
    must all arrive and be booked. League evolution legitimately differs from
    serial (sampling sees EMAs one iteration late), so only counts are pinned."""
    cfg = Config(iters=3, games_per_iter=4, warmup_games=2, warmup_epochs=1,
                 max_decisions=200, report_winrate_games=0,
                 pool_frac=0.5, league_size=2,
                 pfsp_anchors=("random", "attacker"),
                 collect_workers=2, pipeline_collect=True, seed=1)
    lines: list = []
    train(cfg, build_models(cfg), log=lines.append)
    assert any("PIPELINED" in line for line in lines)
    status = next(line for line in reversed(lines) if line.startswith("[status"))
    assert int(re.search(r"games=(\d+)", status).group(1)) == 12   # 3 iters x 4 games
    assert int(re.search(r"T=(\d+)", status).group(1)) > 100


def test_gather_rebuilds_a_poisoned_pool(monkeypatch):
    """A worker death (e.g. Windows WinError 1450 mid-send) poisons the whole
    ProcessPoolExecutor: every later submit/result raises BrokenProcessPool. The
    old retry resubmitted to the DEAD pool and re-raised, ending a 150h+ run.
    _retry_chunk must rebuild the pool and replay the chunk instead."""
    import pickle
    import time
    import zlib

    from concurrent.futures.process import BrokenProcessPool

    monkeypatch.setattr(time, "sleep", lambda *a, **k: None)   # skip the 10s backoff

    cfg = Config(device="cpu", collect_workers=2, max_decisions=200,
                 pool_frac=0.0, seed=0)
    m = build_models(cfg)
    pc = ParallelCollector(cfg, 2)
    try:
        specs = [{"kind": "self", "seed": 7, "idx": 0}]
        pc.gather(pc.submit(m, specs, it=1), 1)        # happy path primes _last_blob

        poisoned = pc._ex                              # simulate the worker-death poisoning
        poisoned.shutdown(wait=False, cancel_futures=True)
        blob = pc._retry_chunk(specs, seed=7,
                               err=BrokenProcessPool("simulated worker death"))
        got = pickle.loads(zlib.decompress(blob))
        assert pc._ex is not poisoned                  # the pool was rebuilt, not reused
        assert got and got[0][0] == 0 and got[0][1].steps   # (idx, buffer) with real steps
    finally:
        pc.close()


def test_retry_chunk_honors_stop_during_backoff(monkeypatch):
    """A stop requested while a resource storm holds the collector in recovery must
    break out promptly (CollectorStopped), not wait out the whole backoff -- that is
    what makes the End-session button work during a WinError 1450 storm."""
    import time

    from concurrent.futures.process import BrokenProcessPool

    from fishrl.train.pcollect import CollectorStopped

    monkeypatch.setattr(time, "sleep", lambda *a, **k: None)   # backoff must not gate the test

    cfg = Config(device="cpu", collect_workers=2, max_decisions=100, pool_frac=0.0, seed=0)
    pc = ParallelCollector(cfg, 2, should_stop=lambda: True)    # stop is already requested
    try:
        with pytest.raises(CollectorStopped):
            pc._retry_chunk([{"kind": "self", "seed": 0, "idx": 0}], seed=0,
                            err=BrokenProcessPool("storm"))
    finally:
        pc.close()


def test_parallel_matches_serial_game_accounting():
    serial = _run(0)
    # affinity "0,1": LPs that exist on any machine -- exercises the pin path in
    # the real workers (placement only; results must still match serial).
    par = _run(2, affinity="0,1")
    # Identical sampling decisions -> identical game counts, same league members
    # seen in the same composition. Transitions differ (independent action-RNG
    # streams) but must exist in quantity.
    assert par["games"] == serial["games"] == 12          # 2 iters x 6 games
    assert par["league_games"] == serial["league_games"]
    assert par["members"] == serial["members"]
    assert par["T"] > 200 and serial["T"] > 200


def test_streamed_run_completes_with_full_game_accounting():
    """Streamed collection: one task per game, stream_depth sets in flight, games
    consumed in completion order. Every iteration still books exactly
    games_per_iter games; the pool is left holding stream_depth sets' worth."""
    cfg = Config(iters=3, games_per_iter=4, warmup_games=2, warmup_epochs=1,
                 max_decisions=200, report_winrate_games=0,
                 pool_frac=0.5, league_size=2,
                 pfsp_anchors=("random", "attacker"),
                 collect_workers=2, pipeline_collect=True, collect_stream=True,
                 stream_depth=2, seed=1)
    lines: list = []
    train(cfg, build_models(cfg), log=lines.append)
    assert any("STREAMED" in line for line in lines)
    status = next(line for line in reversed(lines) if line.startswith("[status"))
    assert int(re.search(r"games=(\d+)", status).group(1)) == 12   # 3 iters x 4 games
    assert int(re.search(r"T=(\d+)", status).group(1)) > 100


def test_take_returns_completion_order_and_keeps_rest():
    cfg = Config(device="cpu", collect_workers=2, max_decisions=120, pool_frac=0.0, seed=0)
    m = build_models(cfg)
    pc = ParallelCollector(cfg, 2)
    try:
        specs = [{"kind": "self", "seed": 100 + i} for i in range(5)]
        items = pc.submit(m, specs, it=1, per_game=True)
        assert len(items) == 5 and all(len(c) == 1 for _f, c, _s in items)
        got, rest = pc.take(items, 3)
        assert len(got) == 3 and len(rest) == 2
        assert {i for i, _b in got}.isdisjoint({items.index(r) for r in rest})
        got2, rest2 = pc.take(rest, 2)
        assert len(got2) == 2 and rest2 == []
        assert all(len(b.games) == 1 for _i, b in got + got2)
    finally:
        pc.close()
