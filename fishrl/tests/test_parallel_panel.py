"""The parallel out-of-band panel must be reproducible and well-formed.

parallel_panel splits each anchor's games into contiguous, seed-aligned chunks across worker
processes and seeds torch per chunk, so for a fixed checkpoint + worker count it returns the
SAME panel every time (hour-over-hour deltas then reflect the policy, not action-sampling
noise). This guards reproducibility, the seat-balanced frozen aggregation, and the metadata
passthrough. Process-spawning + game rollouts make it slow, so it is opt-in like the other
train()-style tests."""
import json
import os

import pytest

from fishrl.eval.metrics import seat_diag_rates
from fishrl.eval.parallel_panel import (BEST, BEST_META, _even_chunks, _maybe_save_best,
                                        best_score, find_harvest, parallel_panel,
                                        plan_topup)
from fishrl.train import checkpoint as ckpt
from fishrl.train.config import Config
from fishrl.train.train_loop import _model_state, build_models

slow = pytest.mark.skipif(
    not os.environ.get("FISHRL_SLOW_TESTS"),
    reason="spawns worker processes + runs rollouts; set FISHRL_SLOW_TESTS=1 to run")


def _write_ckpt(path: str, cfg: Config):
    m = build_models(cfg)
    frozen = build_models(cfg)
    ckpt.save_checkpoint(path, {
        "format": ckpt.FORMAT,
        "config": {"seed": cfg.seed, "encoders": {n: cfg.enc_for(n)
                   for n in ("actor", "critic", "guesser", "public")},
                   "use_belief": cfg.use_belief, "critic_hidden": cfg.critic_hidden},
        "done": 3, "frozen_it": 1, "elapsed": 12.0,
        "models": _model_state(m), "frozen": _model_state(frozen),
    })
    return m, frozen


def test_even_chunks_partition_and_parity():
    # Even counts (seat parity), contiguous cover, correct total -- across odd/even splits.
    for n, k in [(100, 6), (8, 3), (12, 4), (50, 7), (2, 4)]:
        ch = _even_chunks(n, k)
        assert sum(c for _, c in ch) == n
        assert [s for s, _ in ch] == [sum(c for _, c in ch[:i]) for i in range(len(ch))]
        assert all(c % 2 == 0 for _, c in ch)            # each chunk internally seat-balanced


def test_seat_diag_rates_sums_chunks_into_correct_rates():
    # Two worker chunks' counters sum, then convert to rates. Decided is the seat/
    # play-draw denominator; games is first_player's; choose_total is the choice's.
    chunks = [
        {"p1_wins": 3, "p2_wins": 7, "play_wins": 6, "draw_wins": 4,
         "choose_first": 1, "choose_total": 5, "fp_p1": 6, "decided": 10, "games": 12},
        {"p1_wins": 2, "p2_wins": 8, "play_wins": 5, "draw_wins": 5,
         "choose_first": 0, "choose_total": 5, "fp_p1": 6, "decided": 10, "games": 12},
    ]
    agg = {k: sum(c[k] for c in chunks) for k in chunks[0]}
    r = seat_diag_rates(agg)
    assert r["seat_p1_wr"] == 5 / 20 and r["seat_p2_wr"] == 15 / 20   # learned p2 skew
    assert r["seat_p1_wr"] + r["seat_p2_wr"] == 1.0                   # decided games split
    assert r["play_wr"] == 11 / 20 and r["draw_wr"] == 9 / 20
    assert r["choose_first_frac"] == 1 / 10                           # over choose_total
    assert r["first_player_p1_frac"] == 12 / 24                       # ~0.5 sanity (over games)
    assert r["decided"] == 20 and r["n_games"] == 24


def test_seat_diag_rates_tolerates_no_choice_decisions():
    # A window with no choose_play_order decisions must not divide by zero.
    r = seat_diag_rates({"p1_wins": 1, "p2_wins": 1, "play_wins": 1, "draw_wins": 1,
                         "choose_first": 0, "choose_total": 0, "fp_p1": 1,
                         "decided": 2, "games": 2})
    assert r["choose_first_frac"] != r["choose_first_frac"]           # NaN, not a crash


def test_best_score_is_the_minimum_present_anchor():
    # The score is the LOWEST scripted-anchor wr (the hardest opponent) with its name;
    # anchors a row doesn't carry (pre-that-heuristic panels) are skipped, not zeroed.
    assert best_score({"random": 0.9, "attacker": 0.8, "heuristic": 0.5,
                       "heuristic13": 0.04}) == (0.04, "heuristic13")
    assert best_score({"heuristic": 0.5, "heuristic13": None}) == (0.5, "heuristic")
    assert best_score({}) == (float("-inf"), "none")


def test_maybe_save_best_keeps_the_highest_maximin(tmp_path):
    # best.pt rolls only when the MAXIMIN anchor wr strictly improves; the saved payload
    # is exactly the one evaluated (distinct sentinels), and best.json records the score.
    d = str(tmp_path)
    bpt, bjson = os.path.join(d, BEST), os.path.join(d, BEST_META)

    assert _maybe_save_best(d, {"tag": "a"}, {"heuristic": 0.40, "heuristic13": 0.10}) is True
    meta = json.load(open(bjson))
    assert os.path.exists(bpt) and meta["best_score"] == 0.10
    assert meta["best_low_anchor"] == "heuristic13"
    import torch
    assert torch.load(bpt, weights_only=False)["tag"] == "a"

    # a HIGHER v1.0 wr with a lower minimum is WORSE under the maximin -> ignored
    assert _maybe_save_best(d, {"tag": "b"}, {"heuristic": 0.60, "heuristic13": 0.05}) is False
    assert json.load(open(bjson))["best_score"] == 0.10
    assert torch.load(bpt, weights_only=False)["tag"] == "a"                 # unchanged

    assert _maybe_save_best(d, {"tag": "c"}, {"heuristic": 0.40, "heuristic13": 0.10}) is False
    assert _maybe_save_best(d, {"tag": "d"}, {"heuristic": 0.45, "heuristic13": 0.12}) is True
    assert json.load(open(bjson))["best_score"] == 0.12
    assert torch.load(bpt, weights_only=False)["tag"] == "d"


def test_maybe_save_best_supersedes_legacy_v10_keyed_meta(tmp_path):
    # A best.json written under the old rule (keyed on heuristic v1.0, no best_score,
    # never measured on the newer anchors) cannot compete under the maximin: the first
    # panel after the metric change re-seeds the ratchet even with a "worse" v1.0 wr.
    d = str(tmp_path)
    with open(os.path.join(d, BEST_META), "w") as f:
        json.dump({"heuristic": 0.60, "heuristic11": 0.42, "heuristic12": 0.27}, f)
    assert _maybe_save_best(d, {"tag": "n"}, {"heuristic": 0.35, "heuristic13": 0.04}) is True
    assert json.load(open(os.path.join(d, BEST_META)))["best_score"] == 0.04


TARGETS = {"heuristic": 100, "heuristic11": 100, "heuristic12": 100,
           "heuristic13": 100, "attacker": 50, "random": 30}


def test_plan_topup_adds_harvest_on_top_of_a_full_panel():
    # Harvest ADDS samples; it never shrinks the top-up. The panel always plays the
    # full target, so the combined estimate is over (n_train + target) games -- that
    # is what tightens the win-rate curves. Attacker rounds UP to even (seat pairs).
    p = plan_topup(TARGETS, {"heuristic": [34, 40], "attacker": [5, 9], "random": [30, 31],
                             "heuristic12": [11, 126]})
    assert p["heuristic"] == (100, 34, 40)           # full 100 top-up + 40 harvested
    assert p["attacker"] == (50, 5, 9)               # target 50 already even
    assert p["random"] == (30, 30, 31)               # still topped up: pool starves it
    assert p["heuristic11"] == (100, 0, 0)           # nothing harvested -> full panel
    assert p["heuristic12"] == (100, 11, 126)        # over-target harvest is kept, not dropped
    assert p["heuristic13"] == (100, 0, 0)           # new anchor, nothing harvested yet


def test_plan_topup_attacker_rounds_odd_target_up_to_even():
    p = plan_topup({**TARGETS, "attacker": 41}, None)
    assert p["attacker"][0] == 42


def test_plan_topup_no_harvest_is_the_full_panel():
    p = plan_topup(TARGETS, None)
    assert {k: v[0] for k, v in p.items()} == TARGETS
    assert all(v[1] == 0 and v[2] == 0 for v in p.values())


def test_find_harvest_sums_every_unconsumed_window():
    # Evals run more often than reports, so a report can land while another eval is in
    # flight. All rows above the high-water mark are summed -- none are dropped.
    now = 1_000_000.0
    a = {"it": 40, "wall_time": now - 300, "wr_train": {"heuristic": [1, 2]}}
    b = {"it": 42, "wall_time": now - 60, "wr_train": {"heuristic": [3, 5], "random": [1, 1]}}
    wr, it = find_harvest({"reports": [a, b], "evals": []}, 7200, now)
    assert it == 42                                   # new high-water mark
    assert wr["heuristic"] == [4, 7]                  # 1+3 wins of 2+5 games (both windows)
    assert wr["random"] == [1, 1]


def test_find_harvest_guards():
    now = 1_000_000.0
    fresh = {"it": 42, "wall_time": now - 60,
             "wr_train": {"heuristic": [1, 2], "heuristic11": [0, 0],
                          "attacker": [0, 1], "random": [1, 1]}}
    old = {"it": 30, "wall_time": now - 9_999, "wr_train": {"heuristic": [9, 9]}}

    # a stale row is skipped, the usable one is still harvested
    wr, it = find_harvest({"reports": [old, fresh], "evals": []}, 7200, now)
    assert it == 42 and wr["heuristic"] == [1, 2]     # `old` excluded by max_age
    # no wr_train rows at all (pre-harvest trainer) -> no harvest
    assert find_harvest({"reports": [{"it": 1}], "evals": []}, 7200, now) == (None, None)
    # every row too old (stalled trainer) -> no harvest
    assert find_harvest({"reports": [old], "evals": []}, 7200, now) == (None, None)
    # at/below the high-water mark -> no harvest (a window is never double-counted)
    stats = {"reports": [fresh], "evals": [{"it": 50, "harvest_from": 42}]}
    assert find_harvest(stats, 7200, now) == (None, None)
    # the mark only blocks what it covers: a LATER report is still harvested
    later = {"it": 43, "wall_time": now - 30, "wr_train": {"heuristic": [2, 3]}}
    stats = {"reports": [fresh, later], "evals": [{"it": 50, "harvest_from": 42}]}
    wr, it = find_harvest(stats, 7200, now)
    assert it == 43 and wr["heuristic"] == [2, 3]


def test_harvest_counting_convention():
    # Denominator counts EVERY game (draw/truncation winner=None included); wins are
    # strict decisions for the learner's seat -- the eval convention, so the panel can
    # pool these counts with its own top-up games.
    from fishrl.train.train_loop import _harvest_count
    entry = [0, 0]
    _harvest_count(entry, ["p1"], "p1")              # win
    _harvest_count(entry, ["p2"], "p1")              # loss
    _harvest_count(entry, [None], "p1")              # draw/truncation: game, not win
    _harvest_count(entry, ["p2"], "p2")              # win from the p2 seat
    assert entry == [2, 4]


def test_follow_lock_helpers(tmp_path):
    # wait_for_lock sees a held lock immediately; sleep_while_locked returns False
    # (stop following) once the holder lets go, and cuts the sleep short doing so.
    import time as _time

    from fishrl.eval.parallel_panel import sleep_while_locked, wait_for_lock
    from fishrl.train.locks import hold_lockfile, unlock

    p = str(tmp_path / "trainer.lock")
    assert wait_for_lock(p, timeout_s=0.3, poll_s=0.05) is False   # nobody home
    h = hold_lockfile(p)
    try:
        assert wait_for_lock(p, timeout_s=5, poll_s=0.05) is True
        assert sleep_while_locked(p, interval_s=0.2, poll_s=0.05) is True  # still live
    finally:
        unlock(h)
        h.close()
    t0 = _time.perf_counter()
    assert sleep_while_locked(p, interval_s=30, poll_s=0.05) is False      # released
    assert _time.perf_counter() - t0 < 5                                   # woke early


@slow
def test_parallel_panel_reproducible_and_wellformed(tmp_path):
    cfg = Config(seed=0)
    path = str(tmp_path / "latest.pt")
    _write_ckpt(path, cfg)

    N, MD = 8, 30
    a = parallel_panel(path, n_games=N, max_workers=3, max_decisions=MD)
    b = parallel_panel(path, n_games=N, max_workers=3, max_decisions=MD)

    for k in ("random", "attacker", "heuristic", "frozen"):
        assert 0.0 <= a[k] <= 1.0
        assert a[k] == pytest.approx(b[k], abs=1e-9), f"{k} not reproducible: {a[k]} != {b[k]}"
    assert a["it"] == 3 and a["frozen_it"] == 1 and a["n"] == N
