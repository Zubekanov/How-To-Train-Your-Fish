"""Checkpoint durability + resume continuity.

Pure-module tests (atomic write, milestone rotation) are instant; the train()-based tests
use a tiny config (few warmup games, short games, 1-game win-rate panel) to stay fast.
"""
import glob
import os

import pytest
import torch

from fishrl.train import checkpoint as ckpt
from fishrl.train.config import Config
from fishrl.train.train_loop import _rng_state, _set_rng_state, build_models, train


def test_set_rng_state_survives_map_location_move():
    """Regression: resume loads the payload with map_location=cfg.device, so a
    --gpu resume hands _set_rng_state CUDA-moved ByteTensors -- torch requires
    CPU ones ("RNG state must be a torch.ByteTensor"). The restore must coerce;
    on CPU the coercion is a no-op (the ODROID path is bit-identical)."""
    rng = _rng_state("cpu")
    _set_rng_state(rng)                                  # CPU round trip
    if torch.cuda.is_available():
        moved = {"torch": rng["torch"].to("cuda"), "numpy": rng["numpy"]}
        _set_rng_state(moved)                            # the exact live crash case
        cuda_rng = _rng_state("cuda")
        cuda_rng["torch"] = cuda_rng["torch"].to("cuda")
        cuda_rng["cuda"] = [s.to("cuda") for s in cuda_rng["cuda"]]
        _set_rng_state(cuda_rng)
    # determinism survives the coercion
    before = _rng_state("cpu")
    a = torch.rand(4)
    _set_rng_state(before)
    b = torch.rand(4)
    assert torch.equal(a, b)

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


def test_config_from_checkpoint_lr_and_league_every():
    """lr_ppo / league_every are runtime knobs: legacy checkpoints (no keys) fall
    back to the defaults, and CLI-derived overrides win -- the plumbing behind
    --lr-ppo / --league-every being resume-tunable."""
    from fishrl.train.train_loop import config_from_checkpoint
    cd = {"encoders": {"actor": "flat", "critic": "flat"}}
    cfg = config_from_checkpoint(cd)
    assert cfg.lr_ppo == pytest.approx(3e-4)
    assert cfg.league_every == 1
    cfg2 = config_from_checkpoint(cd, lr_ppo=1.5e-4, league_every=8)
    assert cfg2.lr_ppo == pytest.approx(1.5e-4)
    assert cfg2.league_every == 8


@slow
def test_resume_reasserts_lr_ppo_onto_loaded_optimizer(tmp_path):
    """The optimizer state_dict carries the launch-time LR; a resume must re-assert
    the config's lr_ppo onto the loaded param_groups (otherwise --lr-ppo is inert
    on every resumed lineage -- the 300h-at-3e-4 trap)."""
    latest = ckpt.latest_path(str(tmp_path))
    cfg = _cfg(iters=1, seed=5)
    train(cfg, build_models(cfg), checkpoint_path=latest)
    saved = ckpt.load_checkpoint(latest)["optim"]["ppo"]["param_groups"][0]["lr"]
    assert saved == pytest.approx(cfg.lr_ppo)

    cfg2 = _cfg(iters=2, seed=5, lr_ppo=1.5e-4)
    train(cfg2, build_models(cfg2), resume_path=latest, checkpoint_path=latest)
    resumed = ckpt.load_checkpoint(latest)["optim"]["ppo"]["param_groups"][0]["lr"]
    assert resumed == pytest.approx(1.5e-4)


@slow
def test_league_every_spaces_past_self_snapshots(tmp_path):
    """league_every=N takes a past-self snapshot only every Nth status report.
    With report_every_seconds=0 every iteration reports, so over the same number
    of iterations the spaced run must bank at most half the selves of the
    every-report run (exact counts depend on the final flush emit)."""
    def selves_after(league_every, subdir):
        latest = ckpt.latest_path(str(tmp_path / subdir))
        cfg = _cfg(iters=4, seed=5, report_every_seconds=0.0,
                   report_winrate_games=0, league_size=8,
                   league_every=league_every)
        train(cfg, build_models(cfg), checkpoint_path=latest)
        return len(ckpt.load_checkpoint(latest)["league"]["selves"])

    dense = selves_after(1, "dense")
    sparse = selves_after(3, "sparse")
    assert dense >= 4                      # one per report (+ maybe the final emit)
    assert 1 <= sparse <= (dense + 2) // 3  # only every 3rd report snapshots


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
    # win-rate POINT ESTIMATES are not on report rows (they live in "evals"); the
    # harvested [wins, games] counts (wr_train, since the relay work) are allowed.
    assert not any(k.startswith("wr_") for k in rec if k != "wr_train")
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


@slow
def test_v3_report_row_by_turn_calibration(tmp_path):
    """v3 rows carry BOTH by-turn calibration series: critic_turn stays the last
    batch's (the high-variance visual read), critic_turn_win aggregates every batch
    scored in the window (3 iterations here) -- so its totals can only be larger."""
    import json

    from fishrl.train import stats as stats_io

    latest = ckpt.latest_path(str(tmp_path))
    # max_decisions must let games FINISH: truncated games carry no outcome (valid=0),
    # and with none finished estimator_metrics returns {"n": 0} -> critic_turn null.
    cfg = _cfg(iters=3, seed=5, warmup_games=0, report_winrate_games=0,
               max_decisions=800, belief_mode="bookkeeper", critic_view="public")
    train(cfg, build_models(cfg), checkpoint_path=latest)
    with open(stats_io.stats_path(str(tmp_path))) as f:
        data = json.load(f)
    rec = data["reports"][-1]
    assert rec["iters"] == 3
    ct, ctw = rec["critic_turn"], rec["critic_turn_win"]
    # per-batch: bucket keys came off the SAME batch (turn-0 rows pool in slots only)
    bucket_n = sum(rec[f"critic_n_{lbl}"] for lbl in ("t1_10", "t11_20", "t21_30", "t31p"))
    assert sum(ct["n"]) > 0 and bucket_n <= sum(ct["n"])
    # window series covers the per-batch one and never loses rows
    assert sum(ctw["n"]) >= sum(ct["n"])
    for arr in (ct, ctw):
        assert all((b is None) == (n == 0) for n, b in zip(arr["n"], arr["brier"]))
        assert all(b is None or 0.0 <= b <= 1.0 for b in arr["brier"])
