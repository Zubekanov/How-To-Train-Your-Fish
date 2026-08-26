"""Central batched inference for parallel collection.

Collection is 8 worker processes each running batch-1 actor/guesser forwards on
CPU. Measured (eval/profile_discrete + contention benches, 2026-08): a batch-1
actor forward streams ~14 MB of fp32 weights, and under 8-way load that is
DRAM-bound — 818 us solo degrades to ~3.7 ms, while the same forwards batched
8-wide cost 814 us *per decision*. So instead of every worker streaming the
weights per decision, ONE server process holds the learner's actor + guesser
(on the trainer's device — the GPU idles at ~28% during collection) and answers
all workers' forwards in whatever batch is pending each poll.

Wiring is duck-typed: workers swap their local nets for :class:`ServedActor` /
:class:`ServedGuesser`, which satisfy exactly the surface the collectors use
(``log_probs(x, mask)`` / ``forward(persp, prev)`` / ``parameters()`` for
`device_of`), so collector.py and belief_env.py are untouched. Only the
LEARNER's nets are served; frozen past-self opponents stay batch-1 local (a
minority of forwards, and their weights change per spec).

Transport is a shared-memory slot per worker (request payload + int32 state
word; x86 store ordering makes payload-then-state safe) polled by the server;
weights arrive over a control pipe once per iteration — the SAME pre-pickled
blob the workers get, pushed and ACKed in ``ParallelCollector.submit`` before
any of that iteration's games start, so served forwards are never staler than
the worker-local path they replace. On any server timeout the worker falls
back to its local nets (weights are still shipped to workers regardless), so
a dead server degrades throughput, never a run.

Sampling stays in the workers (same per-chunk torch RNG streams as before);
the served log-probs come from the update device, which makes the stored
old_logp MORE consistent with the PPO recompute than the historic CPU-collect/
CUDA-update split, not less.

Opt-in via Config.infer_server / --infer-server (default off: serial path and
ODROID service byte-identical).
"""
from __future__ import annotations

import atexit
import os
import pickle
import time
from multiprocessing import shared_memory

import numpy as np
import torch

# slot states (header word 1)
_IDLE, _REQ_ACTOR, _REQ_GUESS, _RESP = 0, 1, 2, 3
_HDR_I32 = 4                    # per-slot int32 header words: [pid, state, _, _]


def _dims() -> tuple:
    from fishrl.models.policy import actor_in
    from fishrl.obs.encoder import OBS_DIM
    from fishrl.obs import vocab as V
    from fishrl.spaces import action_space as A
    ACTOR_IN = actor_in()                   # live width (count block, if the run has it)
    pay = max(ACTOR_IN + A.N, OBS_DIM + V.N_NAMES)          # request floats
    resp = max(A.N, V.N_NAMES)                              # response floats
    return ACTOR_IN, OBS_DIM, V.N_NAMES, A.N, pay, resp


def slot_bytes() -> int:
    *_ , pay, resp = _dims()
    return _HDR_I32 * 4 + (pay + resp) * 4


def _views(shm, nslots: int):
    """(header int32 (nslots, 4), payload f32 (nslots, PAY), resp f32 (nslots, RESP))."""
    *_, pay, resp = _dims()
    hdr = np.ndarray((nslots, _HDR_I32), dtype=np.int32, buffer=shm.buf)
    off = nslots * _HDR_I32 * 4
    req = np.ndarray((nslots, pay), dtype=np.float32, buffer=shm.buf, offset=off)
    rsp = np.ndarray((nslots, resp), dtype=np.float32, buffer=shm.buf,
                     offset=off + nslots * pay * 4)
    return hdr, req, rsp


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            import ctypes
            SYNCHRONIZE = 0x00100000
            h = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, pid)
            if not h:
                return False
            ctypes.windll.kernel32.CloseHandle(h)
            return True
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class InferenceTimeout(RuntimeError):
    """Server did not answer inside the deadline; caller should go local."""


# ── weight transport ──────────────────────────────────────────────────────────
# NUMPY ONLY, never torch tensors: plain-pickled torch CPU tensors LEAK their
# storage on unpickle (torch 2.12.1 -- C-level, gc-immune, perfectly linear;
# measured ~49 MB per 12M-param state dict). This killed the 2026-08-05 session:
# the server unpickled one blob per iteration and reached 155 GB of commit in
# ~4 h (pagefile exhaustion -> request timeouts -> trainer MemoryError). The
# collector workers run the SAME unpickle per chunk and had been leaking too --
# silently capped near 10 GB each by max_tasks_per_child recycling, visible as
# the "hour-scale allocator creep" pcollect's recycle comment records. Numpy
# arrays round-trip clean (measured flat over 300 iterations).

def np_state(net) -> dict:
    """state_dict as numpy arrays: the weight-transport format."""
    return {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}


def load_np_state(net, state: dict) -> None:
    """Inverse of np_state. from_numpy is a zero-copy view; load_state_dict
    copies it into the parameters, so nothing of the blob outlives the call."""
    net.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()})


class _Client:
    """One worker's slot: claim on construction, release at exit."""

    def __init__(self, shm_name: str, nslots: int, timeout: float = 5.0):
        self.shm = shared_memory.SharedMemory(name=shm_name)
        self.nslots = nslots
        self.timeout = timeout
        self.hdr, self.req, self.rsp = _views(self.shm, nslots)
        self.slot = self._claim()
        atexit.register(self.release)

    def _claim(self) -> int:
        pid = os.getpid()
        for sweep in range(2):
            for s in range(self.nslots):
                cur = int(self.hdr[s, 0])
                # sweep 0 takes free slots; sweep 1 reclaims dead claimants
                # (recycled workers that died without releasing)
                if cur == 0 or (sweep == 1 and not _pid_alive(cur)):
                    self.hdr[s, 0] = pid
                    time.sleep(0.002)                       # settle a racing writer
                    if int(self.hdr[s, 0]) == pid:
                        self.hdr[s, 1] = _IDLE
                        return s
        raise RuntimeError("no free inference slot")

    def release(self) -> None:
        try:
            if self.hdr is not None and int(self.hdr[self.slot, 0]) == os.getpid():
                self.hdr[self.slot, 1] = _IDLE
                self.hdr[self.slot, 0] = 0
            # Drop the numpy views BEFORE closing: on Windows, unmapping while a
            # view still references shm.buf access-violates at interpreter exit.
            self.hdr = self.req = self.rsp = None
            self.shm.close()
        except Exception:                                   # noqa: BLE001 -- exiting anyway
            pass

    def _roundtrip(self, kind: int, payload: np.ndarray, n_out: int) -> np.ndarray:
        s = self.slot
        self.req[s, :payload.shape[0]] = payload
        self.hdr[s, 1] = kind                               # payload first, state last
        deadline = time.perf_counter() + self.timeout
        spins = 0
        # Spin-wait with sleep(0) yields ONLY: on Windows any nonzero sleep rounds
        # up to the ~1 ms scheduler tick, which alone would double the cost of the
        # forward being awaited. The worker is blocked on this answer anyway, so
        # burning its own LP while waiting costs nothing it could otherwise use.
        while int(self.hdr[s, 1]) != _RESP:
            spins += 1
            if spins & 0xFF == 0:
                time.sleep(0)                               # yield, never a timed sleep
                if time.perf_counter() > deadline:
                    self.hdr[s, 1] = _IDLE                  # withdraw the request
                    raise InferenceTimeout(f"slot {s}: no response in {self.timeout}s")
        out = self.rsp[s, :n_out].copy()
        self.hdr[s, 1] = _IDLE
        return out


class ServedActor(torch.nn.Module):
    """Duck-types MaskedActor's inference surface through the server."""

    def __init__(self, client: _Client):
        super().__init__()
        self.client = client
        self._dev = torch.nn.Parameter(torch.zeros(1), requires_grad=False)  # device_of -> cpu
        self.ACTOR_IN, _, _, self.A_N, *_ = _dims()

    def log_probs(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        payload = np.concatenate([
            np.asarray(x, dtype=np.float32).reshape(-1),
            np.asarray(mask, dtype=np.float32).reshape(-1)])
        out = self.client._roundtrip(_REQ_ACTOR, payload, self.A_N)
        return torch.from_numpy(out).unsqueeze(0)           # (1, A.N) like the local net


class ServedGuesser(torch.nn.Module):
    """Duck-types HandGuesser's inference surface through the server."""

    def __init__(self, client: _Client):
        super().__init__()
        self.client = client
        self._dev = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        _, self.OBS_DIM, self.N_NAMES, *_ = _dims()

    def forward(self, perspective: torch.Tensor, prev_guess: torch.Tensor) -> torch.Tensor:
        payload = np.concatenate([
            np.asarray(perspective, dtype=np.float32).reshape(-1),
            np.asarray(prev_guess, dtype=np.float32).reshape(-1)])
        out = self.client._roundtrip(_REQ_GUESS, payload, self.N_NAMES)
        return torch.from_numpy(out).unsqueeze(0)


# ── server process ────────────────────────────────────────────────────────────

class _Graphed:
    """CUDA-graph the batch-B forward of a 2-input net, per power-of-two batch
    bucket. The eager entity forward is ~60 kernel LAUNCHES (~0.54 ms whatever
    the batch); a graph replay of the same math is ~0.22 ms — and launch cost is
    exactly what the serve loop is bound by. Buffers are static per bucket;
    requests are padded up to the bucket (pad rows compute garbage that is never
    read back). Weight pushes stay valid because load_state_dict copies IN PLACE
    into the parameter storages the capture recorded. Replay output drifts from
    eager only by kernel-selection float noise (measured max ~4e-4 on log-probs
    — the same order as the CPU-collect vs CUDA-update mismatch PPO already
    absorbs)."""

    _BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128)      # 120-worker box: one replay per poll

    def __init__(self, fn, in_dims: tuple, out_dim: int, dev):
        self.fn, self.in_dims, self.out_dim, self.dev = fn, in_dims, out_dim, dev
        self.graphs: dict = {}

    def _capture(self, B: int):
        ins = [torch.zeros(B, d, device=self.dev) for d in self.in_dims]
        for _ in range(3):                                  # warm autotuned kernels
            self.fn(*ins)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = self.fn(*ins)
        return g, ins, out

    def run(self, arrays: list) -> "np.ndarray":
        """arrays: per-input (n, dim) float32 numpy. Returns (n, out_dim)."""
        n = arrays[0].shape[0]
        B = next(b for b in self._BUCKETS if b >= min(n, self._BUCKETS[-1]))
        if n > B:                                           # oversize: split
            head = self.run([a[:B] for a in arrays])
            tail = self.run([a[B:] for a in arrays])
            return np.concatenate([head, tail])
        if B not in self.graphs:
            self.graphs[B] = self._capture(B)
        g, ins, out = self.graphs[B]
        for buf, a in zip(ins, arrays):
            buf[:n].copy_(torch.from_numpy(a), non_blocking=True)
        g.replay()
        return out[:n].cpu().numpy()


def _serve_main(shm_name: str, nslots: int, conn, lite: dict) -> None:
    """Poll slots, batch pending requests per net, answer. Weights and stop
    arrive on `conn`; every weights message is ACKed after it is applied."""
    import multiprocessing
    import threading

    torch.set_num_threads(2)                # tensor glue only; the math is on `dev`
    parent = multiprocessing.parent_process()
    if parent is not None:                  # die with the trainer, however it exits
        def _die():
            parent.join()
            os._exit(0)
        threading.Thread(target=_die, daemon=True).start()

    from fishrl.data import features
    from fishrl.models.guesser import HandGuesser
    from fishrl.models.policy import MaskedActor, actor_in

    features.set_count_block(lite.get("obs_counts", False))
    features.set_split_block(lite.get("obs_split", False))
    dev = lite["device"] if (not str(lite["device"]).startswith("cuda")
                             or torch.cuda.is_available()) else "cpu"
    ah = tuple(lite.get("actor_hidden") or lite["hidden"])
    actor = MaskedActor(ah, lite["enc_actor"], int(lite.get("card_dim", 64)),
                        in_dim=actor_in()).to(dev)
    has_guesser = lite.get("belief_mode", "guesser") == "guesser"
    guesser = None
    if has_guesser:
        guesser = HandGuesser(tuple(lite["hidden"]), lite["enc_guesser"],
                              int(lite.get("card_dim", 64))).to(dev)
        guesser.eval()
    actor.eval()
    have_weights = False

    ACTOR_IN, OBS_DIM, N_NAMES, A_N, *_ = _dims()
    graphed_a = graphed_g = None
    if str(dev).startswith("cuda"):
        graphed_a = _Graphed(actor.log_probs, (ACTOR_IN, A_N), A_N, dev)
        if guesser is not None:
            graphed_g = _Graphed(guesser, (OBS_DIM, N_NAMES), N_NAMES, dev)
    shm = shared_memory.SharedMemory(name=shm_name)
    hdr, req, rsp = _views(shm, nslots)
    states = None
    idle = 0
    try:
        while True:
            while conn.poll():
                msg = conn.recv()
                if msg[0] == "stop":
                    return
                if msg[0] == "weights":
                    state = pickle.loads(msg[1])            # numpy arrays only (see np_state)
                    load_np_state(actor, state["actor"])
                    if guesser is not None and "guesser" in state:
                        load_np_state(guesser, state["guesser"])
                    have_weights = True
                    conn.send(("ok",))
            states = hdr[:, 1]
            a_idx = np.flatnonzero(states == _REQ_ACTOR)
            g_idx = np.flatnonzero(states == _REQ_GUESS)
            if (a_idx.size == 0 and g_idx.size == 0) or not have_weights:
                # Busy-poll while collection is live: a timed sleep here costs the
                # Windows ~1 ms tick PER REQUEST. Only back off after a long idle
                # stretch (between iterations / trainer paused) to spare a core.
                idle += 1
                if idle > 50_000:
                    time.sleep(0.001)
                continue
            idle = 0
            with torch.no_grad():
                if a_idx.size:
                    xa = req[a_idx, :ACTOR_IN].copy()
                    mka = req[a_idx, ACTOR_IN:ACTOR_IN + A_N].copy()
                    if graphed_a is not None:
                        out = graphed_a.run([xa, mka])
                    else:
                        out = actor.log_probs(torch.from_numpy(xa).to(dev),
                                              torch.from_numpy(mka).to(dev)).cpu().numpy()
                    for j, s in enumerate(a_idx):
                        rsp[s, :A_N] = out[j]
                        hdr[s, 1] = _RESP                   # payload first, state last
                if g_idx.size and guesser is not None:
                    pa = req[g_idx, :OBS_DIM].copy()
                    pga = req[g_idx, OBS_DIM:OBS_DIM + N_NAMES].copy()
                    if graphed_g is not None:
                        out = graphed_g.run([pa, pga])
                    else:
                        out = guesser(torch.from_numpy(pa).to(dev),
                                      torch.from_numpy(pga).to(dev)).cpu().numpy()
                    for j, s in enumerate(g_idx):
                        rsp[s, :N_NAMES] = out[j]
                        hdr[s, 1] = _RESP
                elif g_idx.size:                            # no guesser in this run: zeros
                    for s in g_idx:
                        rsp[s, :N_NAMES] = 0.0
                        hdr[s, 1] = _RESP
    finally:
        del hdr, req, rsp, states                # views must die before the unmap
        shm.close()


class InferenceServer:
    """Main-process handle: owns the shared memory + server process lifecycle."""

    def __init__(self, lite: dict, workers: int):
        import multiprocessing
        self.nslots = workers + 8                # headroom for recycled workers
        size = self.nslots * slot_bytes()
        self.shm = shared_memory.SharedMemory(create=True, size=size)
        np.ndarray((size // 4,), dtype=np.int32, buffer=self.shm.buf)[:] = 0
        ctx = multiprocessing.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(
            target=_serve_main,
            args=(self.shm.name, self.nslots,
                  child, {k: lite[k] for k in
                          ("hidden", "actor_hidden", "card_dim",
                           "enc_actor", "enc_guesser")}
                  | {"device": lite["device"], "obs_counts": lite.get("obs_counts", False),
                     "obs_split": lite.get("obs_split", False)}),
            daemon=True)
        self.proc.start()

    def push_weights(self, blob: bytes, timeout: float = 60.0) -> None:
        """Apply this iteration's learner weights and wait for the ACK, so no
        game of the iteration can be served with stale weights."""
        self.conn.send(("weights", blob))
        if not self.conn.poll(timeout):
            raise InferenceTimeout("server did not ack weights")
        assert self.conn.recv() == ("ok",)

    def close(self) -> None:
        try:
            self.conn.send(("stop",))
            self.proc.join(timeout=5)
        except Exception:                                   # noqa: BLE001
            pass
        if self.proc.is_alive():
            self.proc.terminate()
        try:
            self.shm.close()
            self.shm.unlink()
        except Exception:                                   # noqa: BLE001
            pass
