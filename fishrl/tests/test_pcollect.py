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
        from fishrl.train.pcollect import _decode
        got = _decode(blob)
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


def test_weights_ring_versions_and_torn_read_guard():
    """Tasks carry a ring stamp, not the blob; an overwritten slot resolves to the
    newest version instead of a torn read."""
    from fishrl.train.pcollect import _ring_read
    cfg = Config(device="cpu", collect_workers=1, max_decisions=60, pool_frac=0.0, seed=0)
    m = build_models(cfg)
    pc = ParallelCollector(cfg, 1)
    try:
        refs = [pc._ring_write(bytes([i]) * (1000 + i)) for i in range(6)]   # 6 writes, 4 slots
        assert [r[-1] for r in refs] == [1, 2, 3, 4, 5, 6]
        tag, name, nbuf, cap, _ = refs[-1]
        assert _ring_read(name, nbuf, cap, 6) == bytes([5]) * 1005
        assert _ring_read(name, nbuf, cap, 3) == bytes([2]) * 1002          # still resident
        newest = _ring_read(name, nbuf, cap, 1)                             # overwritten by 5
        assert newest == bytes([5]) * 1005
        # end-to-end: a real game through the ring path
        specs = [{"kind": "self", "seed": 3}]
        bufs = pc.gather(pc.submit(m, specs, it=2), 1)
        assert len(bufs[0].games) == 1
    finally:
        pc.close()


def test_pack_unpack_roundtrip_is_exact():
    import numpy as np
    from fishrl.data.buffer import RolloutBuffer, Step
    from fishrl.train.pcollect import _pack_buf, _unpack_buf
    god0 = np.zeros(3, dtype=np.float32)
    buf = RolloutBuffer()
    for i in range(5):
        buf.add(Step(seat="p1" if i % 2 == 0 else "p2", x_act=np.full(4, i, np.float32),
                     mask=np.array([1, 0, i % 2], np.int8), action=i, logp=-0.1 * i, value=0.2 * i,
                     god_feat=god0, pub_feat=np.full(2, -i, np.float32),
                     guess_in=np.zeros(2, np.float32), cnt_target=np.ones(2, np.float32),
                     winner="p1" if i < 3 else None, game_id=7, truncated=i == 4, deckout_end=i == 1))
    buf.games = ["p1"]; buf.meta = [{"truncated": False}]
    back = _unpack_buf(_pack_buf(buf))
    assert back.games == buf.games and back.meta == buf.meta and len(back.steps) == 5
    for a, b in zip(buf.steps, back.steps):
        for f in ("seat", "action", "logp", "value", "winner", "game_id", "truncated", "deckout_end"):
            assert getattr(a, f) == getattr(b, f), f
        for f in ("x_act", "mask", "god_feat", "pub_feat", "guess_in", "cnt_target"):
            assert np.array_equal(getattr(a, f), getattr(b, f)), f
    assert back.steps[0].god_feat is back.steps[4].god_feat          # shared zero row survives
    assert len(_unpack_buf(_pack_buf(RolloutBuffer())).steps) == 0


@pytest.mark.skipif(not __import__("os").name == "posix", reason="POSIX shared memory only")
def test_shm_transport_roundtrip():
    import numpy as np
    from fishrl.data.buffer import RolloutBuffer, Step
    from fishrl.train.pcollect import _decode, _pack_buf, _shm_pack
    buf = RolloutBuffer()
    for i in range(4):
        buf.add(Step(seat="p1", x_act=np.full(6, i, np.float32), mask=np.array([1, 0], np.int8),
                     action=i, logp=-0.5, value=0.1, god_feat=np.zeros(3, np.float32),
                     pub_feat=np.full(2, -i, np.float32), guess_in=np.zeros(2, np.float32),
                     cnt_target=np.ones(2, np.float32), winner="p2", game_id=0))
    buf.games = ["p2"]
    blob = _shm_pack([(0, _pack_buf(buf)), (-1, {"gap": 0.0, "play": 1.0, "pid": 1})])
    got = _decode(blob)
    assert got[0][0] == 0 and got[1][0] == -1
    back = got[0][1]
    assert back.cols and np.array_equal(back.column("x_act"), np.stack([s.x_act for s in buf.steps]))
    assert np.array_equal(back.steps[3].pub_feat, np.full(2, -3, np.float32))


@pytest.mark.skipif(not __import__("os").name == "posix", reason="POSIX shared memory only")
def test_shm_pool_roundtrip_and_recycle():
    cfg = Config(device="cpu", collect_workers=2, max_decisions=120, pool_frac=0.0, seed=0,
                 shm_pool_blocks=3, shm_block_mb=16)
    m = build_models(cfg)
    pc = ParallelCollector(cfg, 2)
    try:
        assert len(pc._pool) == 3
        specs = [{"kind": "self", "seed": 50 + i} for i in range(5)]   # > pool: fallback path too
        bufs = pc.gather(pc.submit(m, specs, it=1), 5)
        ids = [c["_blk"] for b in bufs for c in b.cols if "_blk" in c]
        assert 1 <= len(ids) <= 3 and len(set(ids)) == len(ids)
        merged = bufs[0]
        for b in bufs[1:]:
            merged.merge(b)
        batch = merged.compute(0.99, 0.95)
        assert batch["x_act"].shape[0] == len(merged.steps)
        got = merged.release()
        assert sorted(got) == sorted(ids)
        pc.recycle(got)
        bufs = pc.gather(pc.submit(m, specs[:2], it=2), 2)             # blocks reusable
        assert all(len(b.games) == 1 for b in bufs)
    finally:
        pc.close()
