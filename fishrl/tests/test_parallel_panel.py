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

from fishrl.eval.parallel_panel import (BEST, BEST_META, _even_chunks, _maybe_save_best,
                                        find_harvest, parallel_panel, plan_topup)
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


def test_maybe_save_best_keeps_the_highest_heuristic(tmp_path):
    # best.pt rolls only when heuristic win-rate strictly improves; the saved payload is exactly
    # the one evaluated (we pass distinct sentinels), and best.json tracks the winning rate.
    d = str(tmp_path)
    bpt, bjson = os.path.join(d, BEST), os.path.join(d, BEST_META)

    assert _maybe_save_best(d, {"tag": "a"}, {"heuristic": 0.40}) is True   # seeds the best
    assert os.path.exists(bpt) and json.load(open(bjson))["heuristic"] == 0.40
    import torch
    assert torch.load(bpt, weights_only=False)["tag"] == "a"

    assert _maybe_save_best(d, {"tag": "b"}, {"heuristic": 0.30}) is False   # worse -> ignored
    assert json.load(open(bjson))["heuristic"] == 0.40
    assert torch.load(bpt, weights_only=False)["tag"] == "a"                 # unchanged

    assert _maybe_save_best(d, {"tag": "c"}, {"heuristic": 0.40}) is False   # tie -> not better
    assert _maybe_save_best(d, {"tag": "d"}, {"heuristic": 0.55}) is True    # better -> rolls
    assert json.load(open(bjson))["heuristic"] == 0.55
    assert torch.load(bpt, weights_only=False)["tag"] == "d"


TARGETS = {"heuristic": 100, "heuristic11": 100, "heuristic12": 100,
           "attacker": 50, "random": 30}


def test_plan_topup_deficits_and_even_rounding():
    # Deficit = target - harvested games; attacker rounds UP to even (seat pairs);
    # over-target harvest -> 0 deficit; missing anchors -> full target.
    p = plan_topup(TARGETS, {"heuristic": [34, 40], "attacker": [5, 9], "random": [30, 31],
                             "heuristic12": [11, 126]})
    assert p["heuristic"] == (60, 34, 40)
    assert p["attacker"] == (42, 5, 9)               # 41 -> 42, even
    assert p["random"] == (0, 30, 31)                # already past target: no games
    assert p["heuristic11"] == (100, 0, 0)           # nothing harvested -> full panel
    assert p["heuristic12"] == (0, 11, 126)          # v1.2 over-target: harvest only


def test_plan_topup_no_harvest_is_the_full_panel():
    p = plan_topup(TARGETS, None)
    assert {k: v[0] for k, v in p.items()} == TARGETS
    assert all(v[1] == 0 and v[2] == 0 for v in p.values())


def test_find_harvest_guards():
    now = 1_000_000.0
    fresh = {"it": 42, "wall_time": now - 60,
             "wr_train": {"heuristic": [1, 2], "heuristic11": [0, 0],
                          "attacker": [0, 1], "random": [1, 1]}}
    old = {"it": 30, "wall_time": now - 9_999, "wr_train": {"heuristic": [9, 9]}}

    # newest usable row wins
    wr, it = find_harvest({"reports": [old, fresh], "evals": []}, 7200, now)
    assert it == 42 and wr["heuristic"] == [1, 2]
    # no wr_train rows at all (pre-harvest trainer) -> no harvest
    assert find_harvest({"reports": [{"it": 1}], "evals": []}, 7200, now) == (None, None)
    # newest row too old (stalled trainer) -> no harvest
    assert find_harvest({"reports": [old], "evals": []}, 7200, now) == (None, None)
    # already consumed by an earlier eval -> no harvest (never double-count a window)
    stats = {"reports": [fresh], "evals": [{"it": 50, "harvest_from": 42}]}
    assert find_harvest(stats, 7200, now) == (None, None)


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
