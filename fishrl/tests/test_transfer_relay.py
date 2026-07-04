"""transfer export/import as the relay handoff: ownership released on export,
claimed on import, stale zips refused, and a live trainer blocks export.

Uses a real (tiny-model) checkpoint because export sanity-loads latest.pt."""
import os

import pytest

from fishrl.train import checkpoint as ckpt
from fishrl.train import ownership as o
from fishrl.train.config import Config
from fishrl.train.locks import hold_lockfile, unlock
from fishrl.train.train_loop import _model_state, build_models
from fishrl.transfer import export, import_run


@pytest.fixture(scope="module")
def payload():
    cfg = Config(seed=0)
    m, frozen = build_models(cfg), build_models(cfg)
    return {
        "format": ckpt.FORMAT,
        "config": {"seed": cfg.seed, "encoders": {n: cfg.enc_for(n)
                   for n in ("actor", "critic", "guesser", "public")},
                   "use_belief": cfg.use_belief, "critic_hidden": cfg.critic_hidden},
        "done": 5, "elapsed": 60.0, "frozen_it": 5, "warmup_done": True,
        "models": _model_state(m), "frozen": _model_state(frozen),
        "optim": {}, "rng": {},
    }


def _src(tmp_path, payload, name="src"):
    d = str(tmp_path / name)
    ckpt.save_checkpoint(os.path.join(d, "latest.pt"), payload)
    return d


def test_handoff_round_trip_with_ownership(tmp_path, payload):
    src, dst = _src(tmp_path, payload), str(tmp_path / "dst")
    o.claim(src, "odroid")
    z = str(tmp_path / "run.zip")

    assert export(src, z, False, False) == 0
    s = o.read(src)
    assert s["state"] == "released" and s["generation"] == 1     # turn handed over

    assert import_run(z, dst, force=False, merge_stats=False) == 0
    s = o.read(dst)
    assert s["state"] == "active" and s["generation"] == 1       # turn taken here
    assert o.check(dst) is None                                  # this host may train
    assert os.path.exists(os.path.join(dst, "latest.pt"))


def test_stale_zip_refused_then_force_stale(tmp_path, payload):
    src = _src(tmp_path, payload)
    z1, z2 = str(tmp_path / "one.zip"), str(tmp_path / "two.zip")
    o.claim(src, "pc")
    assert export(src, z1, False, False) == 0                    # generation 1
    o.claim(src)                                                 # resume the lineage here
    assert export(src, z2, False, False) == 0                    # generation 2

    dst = str(tmp_path / "dst")
    assert import_run(z2, dst, force=False, merge_stats=False) == 0
    # z1 is now behind the lineage: refuse, even with --force
    assert import_run(z1, dst, force=True, merge_stats=False) == 1
    assert o.read(dst)["generation"] == 2                        # stamp untouched
    # the explicit rewind escape hatch
    assert import_run(z1, dst, force=True, merge_stats=False, force_stale=True) == 0
    assert o.read(dst)["generation"] == 1


def test_export_refused_while_trainer_live(tmp_path, payload):
    src = _src(tmp_path, payload)
    h = hold_lockfile(o.trainer_lock_path(src))                  # "trainer running"
    try:
        assert export(src, str(tmp_path / "x.zip"), False, False) == 1
        assert not os.path.exists(str(tmp_path / "x.zip"))
        # --allow-live: snapshot copy, ownership NOT released
        o.claim(src, "pc")
        assert export(src, str(tmp_path / "y.zip"), False, False, allow_live=True) == 0
        assert o.read(src)["state"] == "active"
    finally:
        unlock(h)
        h.close()
