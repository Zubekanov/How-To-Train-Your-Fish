"""Checkpoint durability + resume continuity.

Pure-module tests (atomic write, milestone rotation) are instant; the train()-based tests
use a tiny config (few warmup games, short games, 1-game win-rate panel) to stay fast.
"""
import glob
import os

import pytest

from fishrl.train import checkpoint as ckpt
from fishrl.train.config import Config
from fishrl.train.train_loop import build_models, train

# The train()-based integration tests spin up warmup + a win-rate panel (~40s) and so are
# opt-in, matching the project's preference for a fast default suite. The pure-module tests
# below (atomic write, milestone rotation) are instant and always run.
slow = pytest.mark.skipif(
    not os.environ.get("FISHRL_SLOW_TESTS"),
    reason="slow train()-based test; set FISHRL_SLOW_TESTS=1 to run")


def _cfg(**kw) -> Config:
    # Aggressively tiny so the suite stays fast: short games, one warmup/collect game, and a
    # huge report interval so only ONE final win-rate panel runs per train() call (the panel
    # is the expensive part). checkpoint_every_seconds=0 still saves latest.pt every iter.
    base = dict(warmup_games=1, warmup_epochs=1, games_per_iter=1, minibatch=128,
                max_decisions=20, report_winrate_games=1,
                report_every_seconds=1e9, checkpoint_every_seconds=0.0)
    base.update(kw)
    return Config(**base)


def test_atomic_write_leaves_no_temp(tmp_path):
    path = str(tmp_path / "latest.pt")
    ckpt.save_checkpoint(path, {"done": 7})
    assert ckpt.load_checkpoint(path)["done"] == 7
    assert not os.path.exists(path + ".tmp")


def test_milestone_rotation_keeps_last_n(tmp_path):
    for done in range(1, 6):
        ckpt.save_milestone(str(tmp_path), done, {"done": done}, keep_last=3)
    steps = sorted(os.path.basename(x) for x in glob.glob(str(tmp_path / "step_*.pt")))
    assert steps == ["step_00000003.pt", "step_00000004.pt", "step_00000005.pt"]


def test_archives_are_permanent_and_survive_milestone_pruning(tmp_path):
    # Archives write atomically and load back...
    apath = ckpt.save_archive(str(tmp_path), 10_000, {"done": 10_000})
    assert os.path.basename(apath) == "archive_00010000.pt"
    assert ckpt.load_checkpoint(apath)["done"] == 10_000
    assert not os.path.exists(apath + ".tmp")
    # ...and the milestone prune (glob step_*.pt) never touches them, no matter
    # how many milestones roll past.
    for done in range(1, 8):
        ckpt.save_milestone(str(tmp_path), done, {"done": done}, keep_last=2)
    assert os.path.exists(apath)
    assert len(glob.glob(str(tmp_path / "step_*.pt"))) == 2


@slow
def test_resume_continues_iteration_and_restores_state(tmp_path):
    latest = ckpt.latest_path(str(tmp_path))
    cfg = _cfg(iters=1, seed=5)
    train(cfg, build_models(cfg), checkpoint_path=latest)
    p = ckpt.load_checkpoint(latest)
    assert p["done"] == 1
    assert p["optim"]["ppo"]["state"]            # Adam moment buffers were persisted
    assert set(p["rng"]) >= {"torch", "numpy"}
    assert p["warmup_done"] is True

    # resume to a higher cap -> continues the counter, does NOT restart from 0
    cfg2 = _cfg(iters=3, seed=5)
    train(cfg2, build_models(cfg2), resume_path=latest, checkpoint_path=latest)
    assert ckpt.load_checkpoint(latest)["done"] == 3
    # at most keep_last milestones survive
    assert len(glob.glob(str(tmp_path / "step_*.pt"))) <= cfg2.keep_last_checkpoints


@slow
def test_train_writes_stats_report(tmp_path):
    import json

    from fishrl.train import stats as stats_io

    latest = ckpt.latest_path(str(tmp_path))
    cfg = _cfg(iters=1, seed=5)                  # report_winrate_games=1 -> inline panel -> eval row
    train(cfg, build_models(cfg), checkpoint_path=latest)
    with open(stats_io.stats_path(str(tmp_path))) as f:
        data = json.load(f)
    assert len(data["reports"]) == 1            # the final emit() appended one datapoint
    rec = data["reports"][0]
    assert rec["it"] == 1 and rec["source"] == "live"
    for k in ("policy_loss", "critic_loss", "entropy", "gmae"):
        assert k in rec
    assert not any(k.startswith("wr_") for k in rec)   # win-rates are NOT on report rows
    assert len(data["evals"]) == 1              # the inline panel went to the eval array instead
    ev = data["evals"][0]
    assert ev["it"] == 1 and ev["source"] == "inline" and "heuristic" in ev


@slow
def test_resume_rejects_encoder_mismatch(tmp_path):
    latest = ckpt.latest_path(str(tmp_path))
    cfg = _cfg(iters=1, seed=1)
    train(cfg, build_models(cfg), checkpoint_path=latest)
    bad = _cfg(iters=2, seed=1, actor_encoder="entity")   # saved actor encoder was "flat"
    with pytest.raises(ValueError, match="encoder mismatch"):
        train(bad, build_models(bad), resume_path=latest, checkpoint_path=latest)
