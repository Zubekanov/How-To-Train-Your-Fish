"""fishrl.relay orchestration over the LocalSim backend: the pull and handback
legs must walk the ownership protocol exactly like the manual runbook, and a
failure in either leg must leave at most one owner."""
import os

import pytest

from fishrl.relay import LocalSim, _local_owner_active, handback, parse, pull
from fishrl.train import checkpoint as ckpt
from fishrl.train import ownership as o
from fishrl.train.config import Config
from fishrl.train.locks import hold_lockfile, unlock
from fishrl.train.train_loop import _model_state, build_models


def test_parse_own_flags_never_leak_into_trainer_args():
    """Regression: argparse.REMAINDER once swallowed --ckpt-dir/--local-remote
    into the trainer passthrough, silently falling back to the REAL server
    defaults. Relay flags must bind to relay wherever they appear; only tokens
    after the literal -- are trainer args."""
    args, extra = parse(["train", "--ckpt-dir", "L", "--local-remote", "S",
                         "--", "--reserve-cores", "2", "--collect-workers", "8"])
    assert args.ckpt_dir == "L" and args.local_remote == "S"
    assert extra == ["--reserve-cores", "2", "--collect-workers", "8"]

    args, extra = parse(["train"])
    assert args.local_remote is None and extra == []

    args, extra = parse(["handback", "--host", "odroid-lan"])
    assert args.cmd == "handback" and extra == []


@pytest.fixture(scope="module")
def payload():
    cfg = Config(seed=0)
    m, frozen = build_models(cfg), build_models(cfg)
    return {
        "format": ckpt.FORMAT,
        "config": {"seed": cfg.seed, "encoders": {n: cfg.enc_for(n)
                   for n in ("actor", "critic", "guesser", "public")},
                   "use_belief": cfg.use_belief, "critic_hidden": cfg.critic_hidden},
        "done": 7, "elapsed": 30.0, "frozen_it": 7, "warmup_done": True,
        "models": _model_state(m), "frozen": _model_state(frozen),
        "optim": {}, "rng": {},
    }


@pytest.fixture()
def server(tmp_path, payload):
    d = str(tmp_path / "server")
    ckpt.save_checkpoint(os.path.join(d, "latest.pt"), payload)
    o.claim(d, "odroid-sim")
    return d


def test_pull_then_handback_walks_ownership(tmp_path, server):
    local = str(tmp_path / "local")
    sim = LocalSim(server)

    pull(sim, local)
    assert _local_owner_active(local)                    # our turn
    assert o.read(server)["state"] == "released"         # server's turn is over
    assert os.path.exists(os.path.join(local, "latest.pt"))

    handback(sim, local)
    assert o.read(server)["state"] == "active"           # server has it back
    assert o.read(local)["state"] == "released"          # we gave it up
    assert not _local_owner_active(local)
    assert o.read(server)["generation"] == o.read(local)["generation"]


def test_pull_skipped_marker_when_we_already_own(tmp_path, server):
    # train_cycle's skip condition: _local_owner_active is the guard.
    local = str(tmp_path / "local")
    pull(LocalSim(server), local)
    assert _local_owner_active(local) is True
    o.release(local)                                     # after a handback...
    assert _local_owner_active(local) is False           # ...we no longer skip


def test_handback_refused_while_local_trainer_live(tmp_path, server):
    local = str(tmp_path / "local")
    sim = LocalSim(server)
    pull(sim, local)
    h = hold_lockfile(o.trainer_lock_path(local))        # "trainer still running"
    try:
        with pytest.raises(SystemExit):
            handback(sim, local)
        assert _local_owner_active(local)                # nothing was given up
        assert o.read(server)["state"] == "released"     # and nothing arrived there
    finally:
        unlock(h)
        h.close()


def test_push_stats_is_telemetry_only(tmp_path, server):
    # push-stats merges rows into the server's stats.json but must move NOTHING
    # else: no ownership change, no checkpoint change.
    import json

    from fishrl.relay import push_stats
    from fishrl.train import stats as stats_io

    local = str(tmp_path / "local")
    stats_io.append_eval(local, {"it": 80000, "heuristic": 0.25, "wall_time": 1.0})
    stats_io.append_eval(str(tmp_path / "server"), {"it": 85000, "heuristic": 0.3,
                                                    "wall_time": 2.0})
    owner_before = o.read(server)
    lat = os.path.getmtime(os.path.join(server, "latest.pt"))

    push_stats(LocalSim(server), local)

    merged = json.load(open(os.path.join(server, "stats.json")))
    assert sorted(e["it"] for e in merged["evals"]) == [80000, 85000]
    assert o.read(server) == owner_before                        # untouched
    assert os.path.getmtime(os.path.join(server, "latest.pt")) == lat


def test_pull_from_empty_server_aborts_cleanly(tmp_path):
    with pytest.raises(SystemExit):
        pull(LocalSim(str(tmp_path / "empty")), str(tmp_path / "local"))
    assert not os.path.exists(os.path.join(str(tmp_path / "local"), "latest.pt"))


def test_handoffs_stamp_and_clear_the_telemetry_redirect(tmp_path, server):
    """Each leg leaves peer.json (the new owner's dashboard URL) on the side
    the lineage LEFT, and the import on the receiving side clears any stale
    pointer there -- so at most one side ever points away, and it points true."""
    local = str(tmp_path / "local")
    sim = LocalSim(server)

    pull(sim, local, serve_url="http://192.168.4.99:8765/")
    p = o.read_peer(server)                              # server now points at us
    assert p["url"] == "http://192.168.4.99:8765/" and p["host"] == o.this_host()
    assert o.read_peer(local) is None                    # active owner points nowhere

    handback(sim, local, remote_serve_url="http://192.168.4.28:8765/")
    assert o.read_peer(local)["url"] == "http://192.168.4.28:8765/"   # we point home
    assert o.read_peer(server) is None                   # import_back's claim cleared it

    pull(sim, local, serve_url="http://192.168.4.99:8765/")
    assert o.read_peer(local) is None                    # our import cleared OUR pointer
    assert o.read_peer(server)["url"] == "http://192.168.4.99:8765/"
