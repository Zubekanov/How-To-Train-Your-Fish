"""Batched-inference server (fishrl.train.inference): the served proxies must be
drop-in equal to the local nets (same weights, same math on a CPU server), the
per-iteration weight push must be exact and ACKed, concurrent clients must not
cross answers, and a dead/absent server must degrade to a timeout the workers
can catch — never a hang."""
from __future__ import annotations

import pickle
import threading

import numpy as np
import pytest
import torch

from fishrl.models.guesser import HandGuesser
from fishrl.models.policy import ACTOR_IN, MaskedActor
from fishrl.obs.encoder import OBS_DIM
from fishrl.obs import vocab as V
from fishrl.spaces import action_space as A
from fishrl.train.inference import (InferenceServer, InferenceTimeout, ServedActor,
                                    ServedGuesser, _Client, slot_bytes)

LITE = {"hidden": (32, 32), "actor_hidden": (32, 32), "card_dim": 16,
        "enc_actor": "entity", "enc_guesser": "flat", "device": "cpu"}


def _nets(seed: int = 0):
    torch.manual_seed(seed)
    actor = MaskedActor(LITE["actor_hidden"], LITE["enc_actor"], LITE["card_dim"])
    guesser = HandGuesser(LITE["hidden"], LITE["enc_guesser"], LITE["card_dim"])
    actor.eval(), guesser.eval()
    return actor, guesser


def _blob(actor, guesser) -> bytes:
    return pickle.dumps({"actor": actor.state_dict(), "guesser": guesser.state_dict()})


def _inputs(rng, n=6):
    xs = torch.from_numpy(rng.random((n, ACTOR_IN), dtype=np.float32))
    masks = torch.from_numpy(
        (rng.random((n, A.N)) < 0.3).astype(np.float32))
    masks[:, 0] = 1.0                                     # never an empty mask
    ps = torch.from_numpy(rng.random((n, OBS_DIM), dtype=np.float32))
    gs = torch.from_numpy(rng.random((n, V.N_NAMES), dtype=np.float32))
    return xs, masks, ps, gs


@pytest.fixture(scope="module")
def server():
    srv = InferenceServer(LITE, workers=3)
    yield srv
    srv.close()


def test_served_equals_local_and_weight_swap(server):
    actor, guesser = _nets(0)
    server.push_weights(_blob(actor, guesser))
    cl = _Client(server.shm.name, server.nslots, timeout=30.0)
    sa, sg = ServedActor(cl), ServedGuesser(cl)
    rng = np.random.default_rng(1)
    xs, masks, ps, gs = _inputs(rng)
    try:
        with torch.no_grad():
            for i in range(xs.shape[0]):
                want = actor.log_probs(xs[i:i + 1], masks[i:i + 1])
                got = sa.log_probs(xs[i:i + 1], masks[i:i + 1])
                fin = torch.isfinite(want)
                assert torch.allclose(got[fin], want[fin], atol=1e-5)
                assert torch.equal(torch.isfinite(got), fin)   # same legal support
                want_g = guesser(ps[i:i + 1], gs[i:i + 1])
                assert torch.allclose(sg(ps[i:i + 1], gs[i:i + 1]), want_g, atol=1e-5)
        # swap to different weights: answers must track the NEW nets exactly
        actor2, guesser2 = _nets(7)
        server.push_weights(_blob(actor2, guesser2))
        with torch.no_grad():
            want = actor2.log_probs(xs[:1], masks[:1])
            got = sa.log_probs(xs[:1], masks[:1])
            fin = torch.isfinite(want)
            assert torch.allclose(got[fin], want[fin], atol=1e-5)
            old = actor.log_probs(xs[:1], masks[:1])
            assert not torch.allclose(got[fin], old[fin], atol=1e-4)
    finally:
        cl.release()


def test_concurrent_clients_get_their_own_answers(server):
    actor, guesser = _nets(3)
    server.push_weights(_blob(actor, guesser))
    rng = np.random.default_rng(2)
    xs, masks, _, _ = _inputs(rng, n=3)
    errs = []

    def hammer(i: int):
        try:
            cl = _Client(server.shm.name, server.nslots, timeout=30.0)
            sa = ServedActor(cl)
            with torch.no_grad():
                want = actor.log_probs(xs[i:i + 1], masks[i:i + 1])
                fin = torch.isfinite(want)
                for _ in range(50):
                    got = sa.log_probs(xs[i:i + 1], masks[i:i + 1])
                    assert torch.allclose(got[fin], want[fin], atol=1e-5)
            cl.release()
        except Exception as e:                            # noqa: BLE001
            errs.append(e)

    ts = [threading.Thread(target=hammer, args=(i,)) for i in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errs, errs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_graphed_server_matches_local():
    """The live server runs CUDA-graph replays; their outputs drift from eager
    only by kernel-selection float noise (~4e-4 measured) — assert they track
    the local nets through a weight swap within that tolerance."""
    srv = InferenceServer(LITE | {"device": "cuda"}, workers=1)
    try:
        actor, guesser = _nets(5)
        srv.push_weights(_blob(actor, guesser))
        cl = _Client(srv.shm.name, srv.nslots, timeout=60.0)
        sa, sg = ServedActor(cl), ServedGuesser(cl)
        rng = np.random.default_rng(9)
        xs, masks, ps, gs = _inputs(rng, n=4)
        with torch.no_grad():
            for i in range(4):
                want = actor.log_probs(xs[i:i + 1], masks[i:i + 1])
                got = sa.log_probs(xs[i:i + 1], masks[i:i + 1])
                assert torch.allclose(got, want, rtol=1e-3, atol=1e-2)
                assert torch.allclose(sg(ps[i:i + 1], gs[i:i + 1]),
                                      guesser(ps[i:i + 1], gs[i:i + 1]),
                                      rtol=1e-3, atol=1e-2)
            actor2, guesser2 = _nets(6)
            srv.push_weights(_blob(actor2, guesser2))     # in-place load keeps graphs valid
            want = actor2.log_probs(xs[:1], masks[:1])
            assert torch.allclose(sa.log_probs(xs[:1], masks[:1]), want,
                                  rtol=1e-3, atol=1e-2)
        cl.release()
    finally:
        srv.close()


def test_no_server_times_out_not_hangs():
    from multiprocessing import shared_memory
    shm = shared_memory.SharedMemory(create=True, size=4 * slot_bytes())
    try:
        buf = np.ndarray((shm.size // 4,), dtype=np.int32, buffer=shm.buf)
        buf[:] = 0
        cl = _Client(shm.name, 4, timeout=0.3)
        sa = ServedActor(cl)
        with pytest.raises(InferenceTimeout):
            sa.log_probs(torch.zeros(1, ACTOR_IN), torch.ones(1, A.N))
        cl.release()
    finally:
        shm.close()
        shm.unlink()


def test_end_to_end_parallel_collection_with_server():
    """Real ParallelCollector path: 2 workers + a CPU server, mirror + heuristic
    specs. Games must complete and produce valid, mask-legal buffers."""
    from types import SimpleNamespace

    from fishrl.train.config import Config
    from fishrl.train.pcollect import ParallelCollector

    cfg = Config(device="cpu", collect_workers=2, infer_server=True,
                 use_belief=True, max_decisions=120,
                 hidden=(32, 32), actor_hidden=(32, 32), card_dim=16,
                 actor_encoder="entity", encoder="flat")
    actor, guesser = _nets(11)
    m = SimpleNamespace(actor=actor, guesser=guesser)
    pcol = ParallelCollector(cfg, 2)
    try:
        specs = [{"kind": "self", "seed": 100 + i} for i in range(2)] + \
                [{"kind": "heuristic", "profile": "heuristic_1_2", "seed": 300}]
        bufs = pcol.collect(m, specs, it=0)
        assert len(bufs) == 3
        for buf in bufs:
            assert len(buf.games) == 1
            for s in buf.steps:
                assert s.mask[s.action] == 1, "illegal action recorded"
    finally:
        pcol.close()
