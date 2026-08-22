"""Parallel local game collection: fan one iteration's games across persistent
worker processes.

Collection is per-decision, batch-1, single-threaded Python — on a many-core
box it uses one core while the rest idle (measured 62% of iteration wall-clock
at 20 cores). This module keeps a pool of collector processes alive across
iterations and ships them the CURRENT learner weights with every batch of game
specs, so the training semantics are unchanged: every game in an iteration is
played by this iteration's policy (strictly on-policy, exactly like the serial
path), transitions come back as ordinary RolloutBuffers merged in spec order,
and ALL bookkeeping (league EMAs, harvest counts, telemetry) stays on the main
thread. What changes is wall-clock only.

Costs shipped per iteration: the learner actor+guesser state_dicts to each
worker chunk (a few MB of pickled CPU tensors — trivial against a multi-second
iteration), plus a frozen past-self's weights riding with any pool-game spec
against one. Workers pin torch to one thread each (parallelism is across
processes, the eval panel's proven pattern) and collect on CPU regardless of
the trainer's device — the update phase, not collection, is what CUDA
accelerates.

Opt-in via Config.collect_workers / --collect-workers (default 0 = the serial
path, byte-identical for the ODROID service).
"""
from __future__ import annotations

import os
import pickle
import time
import zlib

import numpy as np
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait as _fwait
from concurrent.futures.process import BrokenProcessPool
from types import SimpleNamespace

# ── worker side ───────────────────────────────────────────────────────────────
_G: dict = {}


def parse_affinity(spec: str) -> list:
    """'0,2,4' -> [0, 2, 4]; ''/None -> []. Raises ValueError on junk so a typo
    fails the trainer at startup, not silently unpinned."""
    if not spec:
        return []
    return sorted({int(tok) for tok in str(spec).split(",") if tok.strip() != ""})


def _apply_affinity(lps: list) -> None:
    """Restrict THIS process to the given logical processors. Every worker gets
    the same mask; with >= as many LPs as workers the scheduler settles one
    each. Purpose: hybrid Intel parts under Windows 10 (whose scheduler is not
    hybrid-aware) drift collector workers onto E-cores measured 2.3x slower per
    decision. Best-effort: an invalid mask warns rather than kills the worker."""
    if not lps:
        return
    try:
        import os
        if hasattr(os, "sched_setaffinity"):            # POSIX
            os.sched_setaffinity(0, set(lps))
            return
        import ctypes                                   # Windows
        mask = 0
        for lp in lps:
            mask |= 1 << lp
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        k32.SetProcessAffinityMask.restype = ctypes.c_size_t
        if not k32.SetProcessAffinityMask(k32.GetCurrentProcess(), mask):
            raise OSError(f"SetProcessAffinityMask({mask:#x}) failed")
    except Exception as e:                              # noqa: BLE001 -- best-effort
        print(f"[pcollect] affinity {lps} not applied ({e}); worker runs unpinned",
              flush=True)


def _winit(lite: dict) -> None:
    """Once per worker: skeleton nets to load state into (learner + one opponent
    slot) and the fixed collection settings. Torch pinned to 1 thread.

    Scenario snapshot pools are built HERE, up front and in parallel across the
    workers -- built lazily instead, each worker stalls ~4s the first time it is
    handed each scenario type, which shows up as unpredictable mid-training
    iteration spikes (measured: recurring 4-5s collect outliers)."""
    import multiprocessing
    import os
    import threading

    import torch

    from fishrl.models.guesser import HandGuesser
    from fishrl.models.policy import MaskedActor, actor_in

    # Die with the parent. A gracefully-stopped trainer shuts the pool down, but
    # a hard-killed one (TerminateProcess) leaves the workers orphaned -- observed
    # as 16 stray torch processes after two killed runs. parent.join() returns the
    # moment the parent exits, however it exits.
    parent = multiprocessing.parent_process()
    if parent is not None:
        def _die_with_parent():
            parent.join()
            os._exit(0)
        threading.Thread(target=_die_with_parent, daemon=True).start()

    torch.set_num_threads(1)
    _apply_affinity(lite.get("affinity") or [])
    from fishrl.data import features
    from fishrl.spaces import masking
    # Match the trainer's gates exactly: public encoding is on iff the diagnostic
    # wants it OR the critic eats it; the text-change mask mode mirrors the run's.
    features.set_public_encoding(lite.get("train_public", True)
                                 or lite.get("critic_view", "god") in features.PUBLIC_FAMILY)
    if lite.get("critic_view", "god") in features.PUBLIC_FAMILY:
        features.set_public_view(lite["critic_view"])
    masking.set_text_change_mode(lite.get("text_change_mode", "full"))
    features.set_count_block(lite.get("obs_counts", False))
    _G["owner_pid"] = int(lite.get("owner_pid", 0))
    hidden = tuple(lite["hidden"])                        # guesser head width
    ah = tuple(lite.get("actor_hidden", hidden))          # actor head width (may differ)
    cd = int(lite.get("card_dim", 64))                    # entity card-embedding width
    has_guesser = lite.get("belief_mode", "guesser") == "guesser"
    _G.update(
        lite=lite,
        actor=MaskedActor(ah, lite["enc_actor"], cd, in_dim=actor_in()),
        guesser=HandGuesser(hidden, lite["enc_guesser"], cd) if has_guesser else None,
        opp_actor=MaskedActor(ah, lite["enc_actor"], cd, in_dim=actor_in()),
        opp_guesser=HandGuesser(hidden, lite["enc_guesser"], cd) if has_guesser else None,
    )
    if lite.get("serve_shm"):
        # Batched-inference client: the LEARNER's forwards go to the central
        # server (fishrl.train.inference); local nets stay loaded as the
        # fallback and for frozen opponents. A failed claim just means this
        # worker runs local -- never fatal.
        try:
            from fishrl.train.inference import ServedActor, ServedGuesser, _Client
            cl = _Client(lite["serve_shm"], lite["serve_nslots"])
            _G.update(served_actor=ServedActor(cl), served_guesser=ServedGuesser(cl))
        except Exception as e:                            # noqa: BLE001 -- degrade, don't die
            print(f"[infer] worker slot claim failed ({e!r}); running local", flush=True)
    if lite["scenario_names"]:
        from fishrl.train.scenarios import get_scenario
        for name in lite["scenario_names"]:
            get_scenario(name).ensure_pool()


def _collect_one(spec: dict):
    """One game per the spec; mirrors the serial train-loop branches exactly.
    With a live inference server, the learner's nets are the served proxies;
    an InferenceTimeout marks the server dead for this worker (sticky) and
    replays the game with the local nets."""
    from fishrl.train.inference import InferenceTimeout

    if _G.get("served_actor") is not None and not _G.get("serve_dead"):
        try:
            return _collect_one_with(spec, _G["served_actor"], _G["served_guesser"])
        except InferenceTimeout as e:
            _G["serve_dead"] = True
            print(f"[infer] server timeout ({e}); worker falls back to local nets",
                  flush=True)
    return _collect_one_with(spec, _G["actor"], _G["guesser"])


def _collect_one_with(spec: dict, actor, guesser):
    from fishrl.train.belief_env import BeliefAugmentedEnv
    from fishrl.train.collector import (actor_act_fn, collect_games,
                                        collect_heuristic_games, collect_vs_opponent)

    lite = _G["lite"]
    kind = spec["kind"]
    bmode = lite.get("belief_mode",
                     "guesser" if lite.get("use_belief", True) else "none")
    cview = lite.get("critic_view", "god")
    if kind == "self":
        benv = BeliefAugmentedEnv(guesser, mode=bmode,
                                  max_decisions=lite["max_decisions"])
        return collect_games(benv, actor_act_fn(actor), 1, spec["seed"],
                             critic=None, max_decisions=lite["max_decisions"],
                             critic_view=cview)
    if kind == "scenario":
        from fishrl.train.scenarios import ScenarioEnv, get_scenario
        senv = BeliefAugmentedEnv(
            guesser, mode=bmode,
            env=ScenarioEnv(get_scenario(spec["name"]),
                            max_decisions=lite["max_decisions"]))
        return collect_games(senv, actor_act_fn(actor), 1, spec["seed"],
                             critic=None, max_decisions=lite["max_decisions"],
                             critic_view=cview)
    if kind == "heuristic":
        return collect_heuristic_games(guesser, actor, 1, spec["seed"], critic=None,
                                       belief_mode=bmode,
                                       max_decisions=lite["max_decisions"],
                                       profile=spec["profile"], critic_view=cview)
    # kind == "opponent": random / attacker / frozen past-self
    okind = spec["okind"]
    models = None
    if okind == "self":
        from fishrl.train.inference import load_np_state
        a_state, g_state = spec["opp_state"]           # frozen nets ride with the spec
        load_np_state(_G["opp_actor"], a_state)
        if g_state is not None and _G["opp_guesser"] is not None:
            load_np_state(_G["opp_guesser"], g_state)
        models = SimpleNamespace(actor=_G["opp_actor"], guesser=_G["opp_guesser"])
    member = SimpleNamespace(kind=okind, models=models)
    learner = SimpleNamespace(actor=actor, guesser=guesser)
    return collect_vs_opponent(learner, member, 1, spec["seed"], critic=None,
                               belief_mode=bmode,
                               max_decisions=lite["max_decisions"],
                               learner_seat=spec["lseat"], critic_view=cview)


_WRING = {"shm": None, "ver": -1}
_WT = {"last_done": None}          # worker-side timing: when the previous task returned


def _ring_read(name: str, nbuf: int, cap: int, ver: int) -> bytes:
    """Read the learner blob stamped `ver` (or newer) from the weights ring. The
    writer stamps [ver, len] AFTER the bytes; we re-check the stamp after the copy
    so a buffer being overwritten underneath us is detected and re-read from the
    newest slot (a newer blob is always acceptable: served forwards are newer still)."""
    from multiprocessing import shared_memory
    if _WRING["shm"] is None:
        _WRING["shm"] = shared_memory.SharedMemory(name=name)
    buf = _WRING["shm"].buf
    hdr = np.ndarray((nbuf, 2), dtype=np.int64, buffer=buf)            # [ver, len] per slot
    base = nbuf * 16
    for _attempt in range(50):
        v, n = int(hdr[ver % nbuf, 0]), int(hdr[ver % nbuf, 1])
        if v < ver:                                                     # not written yet
            time.sleep(0.001); continue
        if v > ver:                                                     # overwritten: take newest
            ver = int(hdr[:, 0].max())
            continue
        off = base + (ver % nbuf) * cap
        data = bytes(buf[off:off + n])
        if int(hdr[ver % nbuf, 0]) == v:
            return data
    raise RuntimeError("weights ring: could not get a consistent read")


def _pack_buf(buf) -> dict:
    """Columnar transport form of a per-game RolloutBuffer: a few stacked arrays
    instead of ~300 Step objects. Unpickling stacked arrays is a memcpy; unpickling
    Step objects was ~30 ms/game on the trainer's main thread (3.6 s/iteration of
    120 games, serial, on the thread the executor's manager also needs)."""
    st = buf.steps
    if not st:
        return {"n": 0, "games": list(buf.games), "meta": list(buf.meta)}
    god_shared = all(x.god_feat is st[0].god_feat for x in st)
    return {
        "n": len(st),
        "seat": np.array([x.seat == "p1" for x in st], dtype=np.bool_),
        "x_act": np.stack([x.x_act for x in st]),
        "mask": np.stack([x.mask for x in st]),
        "action": np.array([x.action for x in st], dtype=np.int64),
        "logp": np.array([x.logp for x in st], dtype=np.float64),
        "value": np.array([x.value for x in st], dtype=np.float64),
        "god": st[0].god_feat if god_shared else np.stack([x.god_feat for x in st]),
        "god_shared": god_shared,
        "pub": np.stack([x.pub_feat for x in st]),
        "guess_in": np.stack([x.guess_in for x in st]),
        "cnt": np.stack([x.cnt_target for x in st]),
        "winner": [x.winner for x in st],
        "game_id": np.array([x.game_id for x in st], dtype=np.int64),
        "truncated": np.array([x.truncated for x in st], dtype=np.bool_),
        "deckout_end": np.array([x.deckout_end for x in st], dtype=np.bool_),
        "games": list(buf.games), "meta": list(buf.meta),
    }


def _unpack_buf(d: dict):
    from fishrl.data.buffer import RolloutBuffer, Step
    buf = RolloutBuffer()
    buf.games = list(d["games"]); buf.meta = list(d["meta"])
    n = int(d["n"])
    if n == 0:
        return buf
    god = d["god"]
    for i in range(n):
        buf.steps.append(Step(
            seat="p1" if d["seat"][i] else "p2", x_act=d["x_act"][i], mask=d["mask"][i],
            action=int(d["action"][i]), logp=float(d["logp"][i]), value=float(d["value"][i]),
            god_feat=god if d["god_shared"] else god[i], pub_feat=d["pub"][i],
            guess_in=d["guess_in"][i], cnt_target=d["cnt"][i], winner=d["winner"][i],
            game_id=int(d["game_id"][i]), truncated=bool(d["truncated"][i]),
            deckout_end=bool(d["deckout_end"][i])))
    buf.cols = [d]                                 # columnar backing (see RolloutBuffer.cols)
    return buf


_SHM_SEQ = [0]
_SHM_TRANSPORT = os.name == "posix"           # POSIX shm persists until unlink; Windows
_SHM_KEYS = ("x_act", "pub", "mask", "guess_in", "cnt", "god")   # frees it with the last handle


def sweep_stale_shm(root: str = "/dev/shm") -> int:
    """Unlink fishrl_<owner>_<worker>_<n> blocks whose owner trainer is dead."""
    if os.name != "posix" or not os.path.isdir(root):
        return 0
    n = 0
    for fn in os.listdir(root):
        if not fn.startswith("fishrl_"):
            continue
        try:
            owner = int(fn.split("_")[1])
        except (IndexError, ValueError):
            continue
        alive = True
        try:
            os.kill(owner, 0)
        except ProcessLookupError:
            alive = False
        except PermissionError:
            alive = True
        if owner == os.getpid() or not alive:
            try:
                os.unlink(os.path.join(root, fn)); n += 1
            except OSError:
                pass
    if n:
        print(f"[pcollect] swept {n} stale shared-memory block(s)", flush=True)
    return n


def _shm_pack(items: list) -> bytes:
    """Put every big array of the chunk's packed games into ONE shared-memory block
    and return a small pickle carrying its name + descriptors. The trainer's decode
    is then a memcpy per array (GIL released) instead of unpickling ~20 MB/game."""
    from multiprocessing import resource_tracker, shared_memory
    descs, total = [], 0
    for _idx, d in items:
        if not isinstance(d, dict) or "x_act" not in d:
            continue
        for k in _SHM_KEYS:
            a = np.ascontiguousarray(d[k])
            descs.append((d, k, a, total)); total += a.nbytes
    if total == 0:
        return pickle.dumps({"shm": None, "items": items}, protocol=pickle.HIGHEST_PROTOCOL)
    # name carries the TRAINER's pid (shipped in _G) so a restart can sweep blocks whose
    # owner died with results unread (a STOP mid-flight leaked ~14 GB per restart)
    _SHM_SEQ[0] += 1
    name = f"fishrl_{_G.get('owner_pid', 0)}_{os.getpid()}_{_SHM_SEQ[0]}"
    shm = shared_memory.SharedMemory(name=name, create=True, size=total)
    for d, k, a, off in descs:
        view = np.frombuffer(shm.buf, dtype=a.dtype, count=a.size, offset=off).reshape(a.shape)
        view[...] = a
        d[k] = ("shm", off, a.shape, a.dtype.str)
    del view, descs                                # views of shm.buf must die before close()
    name = shm.name
    try:                                          # the TRAINER unlinks; this worker must
        resource_tracker.unregister(shm._name, "shared_memory")   # not reap it on recycle
    except Exception:                             # noqa: BLE001
        pass
    shm.close()
    return pickle.dumps({"shm": name, "items": items}, protocol=pickle.HIGHEST_PROTOCOL)


def _decode(blob: bytes) -> list:
    """Result blob -> [(idx, RolloutBuffer | timing dict)]. Both the zlib path and
    the shm memcpy path release the GIL, so gather/take run this in a thread pool."""
    if blob[:1] == b"x":                          # zlib header: pickled-arrays transport
        items = pickle.loads(zlib.decompress(blob))
    else:
        from multiprocessing import shared_memory
        msg = pickle.loads(blob)
        items = msg["items"]
        if msg["shm"] is not None:
            # Zero-copy: the column arrays are VIEWS into the block; the one copy left
            # is column()'s assembly into the batch. Lifetime: unlink now (POSIX keeps
            # the mapping valid), hand the memoryview + mmap to the arrays, and strip
            # them off the SharedMemory object so its close()/__del__ never sees
            # exported pointers. The mapping is released when the last view dies
            # (the RolloutBuffer of this iteration).
            shm = shared_memory.SharedMemory(name=msg["shm"])
            buf = shm.buf
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
            shm._buf = None
            shm._mmap = None
            shm.close()                               # just the fd; mapping stays
            for _idx, d in items:
                if not isinstance(d, dict):
                    continue
                for k in _SHM_KEYS:
                    v = d.get(k)
                    if isinstance(v, tuple) and v and v[0] == "shm":
                        _t, off, shape, dt = v
                        d[k] = np.frombuffer(buf, dtype=np.dtype(dt),
                                             count=int(np.prod(shape)), offset=off).reshape(shape)
            del buf
    out = []
    for idx, item in items:
        out.append((idx, _unpack_buf(item) if idx >= 0 else item))
    return out


def _collect_chunk(learner_blob, specs: list, torch_seed: int) -> bytes:
    """Load this iteration's learner weights once, then play the chunk's games.
    Returns [(spec_index, RolloutBuffer), ...] as a COMPRESSED pickle. The
    weights arrive PRE-PICKLED: the main process serializes the 14 MB state
    dict once per iteration instead of once per chunk (the executor then only
    memcpys bytes per submit). The result is compressed HERE because the raw
    buffers are large (~90 KB/step, dominated by mostly-zero obs vectors that
    zlib crushes ~10x) -- shrinking both the queue traffic and the commit
    spike whose allocation failure once killed a whole session (MemoryError
    inside the executor's own result pickling, where no user code can catch)."""
    import torch
    from fishrl.train.inference import load_np_state
    torch.manual_seed(torch_seed)                      # reproducible action sampling per chunk
    if isinstance(learner_blob, tuple):                # ("ring", name, nbuf, cap, ver): weights
        _tag, name, nbuf, cap, ver = learner_blob      # live in shared memory, one write per
        if _WRING["ver"] >= ver:                       # iteration; skip if already at/above
            learner_blob = None
        else:
            learner_blob = _ring_read(name, nbuf, cap, ver)
            _WRING["ver"] = ver
    if learner_blob is not None:
        learner_state = pickle.loads(learner_blob)     # numpy arrays only (see np_state):
        load_np_state(_G["actor"], learner_state["actor"])    # plain-pickled torch tensors
        if "guesser" in learner_state and _G["guesser"] is not None:   # leak their storage
            load_np_state(_G["guesser"], learner_state["guesser"])     # on loads
    t0 = time.perf_counter()
    gap = (t0 - _WT["last_done"]) if _WT["last_done"] is not None else 0.0
    out = [(spec["idx"], _pack_buf(_collect_one(spec))) for spec in specs]
    play = time.perf_counter() - t0
    # worker timing rides along as a pseudo-row: gap = idle between tasks (task
    # feeding / result draining), play = inside the games (engine + forwards)
    out.append((-1, {"gap": gap, "play": play, "pid": os.getpid()}))
    if _SHM_TRANSPORT:
        blob = _shm_pack(out)
    else:
        blob = zlib.compress(pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL), 1)
    _WT["last_done"] = time.perf_counter()
    return blob


# ── main-process side ─────────────────────────────────────────────────────────

def _cpu_state(net) -> dict:
    """Weight-transport snapshot. NUMPY, not torch: see inference.np_state for
    the unpickle leak this dodges (workers' recycle creep + the server's fatal
    unbounded version of the same)."""
    from fishrl.train.inference import np_state
    return np_state(net)


class CollectorStopped(Exception):
    """Raised out of gather/collect when a stop is requested DURING recovery, so a
    resource storm's backoff doesn't swallow the End-session button: the trainer
    only polls STOP at the iteration boundary, and a multi-minute pool-recovery
    would otherwise hold the loop there for minutes. The trainer catches this and
    checkpoints-and-exits like any other stop."""


class ParallelCollector:
    """Persistent collector pool. `collect(m, specs, it)` plays every spec with
    the CURRENT weights of `m` and returns the RolloutBuffers in spec order."""

    def __init__(self, cfg, workers: int, should_stop=None):
        self.workers = workers
        self._should_stop = should_stop           # () -> bool: honored during recovery backoff
        self._last_blob: bytes = b""
        scen_names: list = []
        if cfg.scenario_frac > 0 or cfg.scenarios_in_pool:
            from fishrl.train.scenarios import scenario_names
            scen_names = [n for n in scenario_names()
                          if cfg.scenario_weights.get(n, 1.0) > 0]
        lite = {"hidden": tuple(cfg.hidden),
                # actor may be sized differently from the guesser (actor_hidden), and
                # both entity nets need card_dim -- workers must build the SAME shapes
                # or the shipped weights won't load into them.
                "actor_hidden": tuple(cfg.head_hidden("actor")), "card_dim": cfg.card_dim,
                "enc_actor": cfg.enc_for("actor"), "enc_guesser": cfg.enc_for("guesser"),
                "use_belief": cfg.use_belief, "max_decisions": cfg.max_decisions,
                "train_public": cfg.train_public,
                "belief_mode": cfg.belief_mode, "critic_view": cfg.critic_view,
                "text_change_mode": cfg.text_change_mode,
                "obs_counts": bool(getattr(cfg, "obs_counts", False)),
                "scenario_names": scen_names,
                "owner_pid": os.getpid(),
                "affinity": parse_affinity(getattr(cfg, "collect_affinity", ""))}
        self._server = None
        if getattr(cfg, "infer_server", False):
            from fishrl.train.inference import InferenceServer
            self._server = InferenceServer(lite | {"device": cfg.device}, workers)
            lite = lite | {"serve_shm": self._server.shm.name,
                           "serve_nslots": self._server.nslots}
            print(f"[infer] batched inference server on {cfg.device} "
                  f"({self._server.nslots} slots)", flush=True)
        self._lite = lite                              # kept so a poisoned pool can be rebuilt
        sweep_stale_shm()
        # Workers are single-threaded BY DESIGN (one game each, batch-1 / served
        # forwards). torch.set_num_threads(1) in _winit is too late for the OpenBLAS /
        # OpenMP pools numpy spins up at import: a host env of OMP_NUM_THREADS=62
        # (the Vast.ai image, 2026-08-22) gave 120 workers x 62 BLAS threads, every
        # tiny per-decision numpy op paying a 62-thread futex barrier (workers at
        # ~19% CPU, asleep in futex_wait). Spawned children inherit os.environ, so
        # pin it HERE, before the pool exists. The parent's own pools are untouched.
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[k] = "1"
        self._ex = self._new_executor()
        # Weights ring: the per-iteration learner blob is written ONCE into shared
        # memory and tasks carry a version stamp, instead of every task hauling the
        # ~10 MB blob through the executor's single feeder pipe (120 tasks/iter =
        # 1.2 GB/iter serialised behind the main thread's GIL -- workers sat at ~19%
        # CPU blocked on that pipe, 2026-08-22). 4 slots so in-flight tasks from
        # older sets still find their stamp; a worker that misses reads the newest.
        self._ring = None
        self._ring_n = 4
        self._ring_cap = 0
        self._ring_ver = 0
        self.timing = {"gap_s": 0.0, "play_s": 0.0, "n": 0}
        self._dec = ThreadPoolExecutor(max_workers=12, thread_name_prefix="pcollect-decode")

    def _new_executor(self) -> ProcessPoolExecutor:
        # Recycle each worker after this many chunks. Historically this capped an
        # "hour-scale allocator creep" -- root-caused 2026-08-05 as the torch
        # tensor-unpickle leak (~19 MB per chunk's weights load; see
        # inference.np_state), now fixed at the source by the numpy transport
        # format. Recycling stays as a cheap backstop against whatever leaks
        # next: a fresh process costs one slow re-warm iteration every `recycle`
        # iters (all workers hit the cap together -- round-robin, one chunk each).
        try:
            return ProcessPoolExecutor(max_workers=self.workers,
                                       initializer=_winit, initargs=(self._lite,),
                                       max_tasks_per_child=512)
        except TypeError:                              # python < 3.11: no recycling
            return ProcessPoolExecutor(max_workers=self.workers,
                                       initializer=_winit, initargs=(self._lite,))

    @staticmethod
    def opp_state_for(member) -> tuple:
        """A frozen past-self's nets as CPU state_dicts, shipped WITH its spec.
        Workers are anonymous under ProcessPoolExecutor, so a ship-once cache
        can't guarantee coverage -- and at 1-2 pool games per iteration, a few
        MB alongside the per-chunk learner shipment is noise. The guesser leg is
        None for a bookkeeper-mode past-self (nothing to freeze)."""
        g = getattr(member.models, "guesser", None)
        return (_cpu_state(member.models.actor), _cpu_state(g) if g is not None else None)

    def submit(self, m, specs: list, it: int, per_game: bool = False) -> list:
        """Ship the CURRENT weights of `m` and start the specs' games on the pool;
        returns [(future, chunk, seed), ...] work items (gather needs the args to
        resubmit a failed chunk). Round-robin chunking; one weights shipment per
        chunk. Split from `gather` so a pipelined trainer can overlap the games
        with the GPU update -- the weights are snapshotted HERE, so what the games
        are played with is fixed at submit time regardless of later updates to `m`."""
        for i, s in enumerate(specs):
            s["idx"] = i
        learner_blob = pickle.dumps(
            {"actor": _cpu_state(m.actor),
             **({"guesser": _cpu_state(m.guesser)} if m.guesser is not None else {})},
            protocol=pickle.HIGHEST_PROTOCOL)          # serialize ONCE, memcpy per chunk
        self._last_blob = learner_blob                 # for gather's resubmit path
        ring_ref = self._ring_write(learner_blob)
        if self._server is not None:
            # Same blob, applied + ACKed before any chunk starts: served forwards
            # are exactly as fresh as the worker-local weights they replace.
            self._server.push_weights(learner_blob)
        items = []
        chunks = ([[s] for s in specs] if per_game else
                  [specs[w::self.workers] for w in range(self.workers)])
        for w, chunk in enumerate(chunks):
            if chunk:
                seed = (it * 1009 + w) % (2**31)
                items.append((self._ex.submit(_collect_chunk, ring_ref, chunk, seed),
                              chunk, seed))
        return items

    def _ring_write(self, blob: bytes) -> tuple:
        """Write `blob` into the next ring slot; returns the task-side reference."""
        from multiprocessing import shared_memory
        if self._ring is None or len(blob) > self._ring_cap:
            if self._ring is not None:
                try:
                    self._ring.close(); self._ring.unlink()
                except Exception:                      # noqa: BLE001
                    pass
            self._ring_cap = int(len(blob) * 1.25) + 4096
            self._ring = shared_memory.SharedMemory(
                create=True, size=self._ring_n * 16 + self._ring_n * self._ring_cap)
            np.ndarray((self._ring_n, 2), dtype=np.int64, buffer=self._ring.buf)[:] = -1
        self._ring_ver += 1
        ver = self._ring_ver
        slot = ver % self._ring_n
        hdr = np.ndarray((self._ring_n, 2), dtype=np.int64, buffer=self._ring.buf)
        hdr[slot, 0] = -1                              # invalidate while writing
        off = self._ring_n * 16 + slot * self._ring_cap
        self._ring.buf[off:off + len(blob)] = blob
        hdr[slot, 1] = len(blob)
        hdr[slot, 0] = ver                             # bytes first, stamp last
        return ("ring", self._ring.name, self._ring_n, self._ring_cap, ver)

    def gather(self, items: list, n_specs: int) -> list:
        """Block on the work items and return the RolloutBuffers in spec order.
        Any chunk failure is retried (see `_retry_chunk`) rather than propagated,
        so a transient Windows resource spike doesn't end a multi-day run."""
        out: dict = {}
        blobs = []
        for fut, chunk, seed in items:
            try:
                blobs.append(fut.result())
            except Exception as e:                     # noqa: BLE001 -- retry ANY chunk failure
                blobs.append(self._retry_chunk(chunk, seed, e))
        for decoded in self._dec.map(_decode, blobs):
            for idx, buf in decoded:
                if idx < 0:
                    self._note_timing(buf); continue
                out[idx] = buf
        return [out[i] for i in range(n_specs)]

    def _note_timing(self, t: dict) -> None:
        self.timing["gap_s"] += t["gap"]; self.timing["play_s"] += t["play"]; self.timing["n"] += 1

    def pop_timing(self) -> dict:
        """Mean worker gap/play seconds per task since the last call."""
        t, n = self.timing, max(self.timing["n"], 1)
        out = {"worker_gap_s": t["gap_s"] / n, "worker_play_s": t["play_s"] / n, "worker_tasks": t["n"]}
        self.timing = {"gap_s": 0.0, "play_s": 0.0, "n": 0}
        return out

    def take(self, items: list, n: int) -> tuple:
        """Streamed gather: block until `n` of the single-game work items have
        finished and return ([(idx_in_items, RolloutBuffer), ...] in completion
        order, remaining items). Failures retry like `gather`."""
        pending = {fut: i for i, (fut, _c, _s) in enumerate(items)}
        got: list = []
        decoding: list = []                            # (i, future-of-decode)
        while len(got) + len(decoding) < n and pending:
            done, _ = _fwait(list(pending), return_when=FIRST_COMPLETED)
            # several may land together: take at most n overall, the rest stay pending
            for fut in sorted(done, key=pending.get)[:n - len(got) - len(decoding)]:
                i = pending.pop(fut)
                _f, chunk, seed = items[i]
                try:
                    blob = fut.result()
                except Exception as e:                 # noqa: BLE001 -- retry ANY failure
                    blob = self._retry_chunk(chunk, seed, e)
                decoding.append((i, self._dec.submit(_decode, blob)))
        for i, df in decoding:
            for idx, buf in df.result():
                if idx < 0:
                    self._note_timing(buf); continue
                got.append((i, buf))
        rest = [items[i] for i in sorted(pending.values())]
        return got, rest

    # Escalating backoff (~3.9 min total) so recovery OUTLASTS a resource storm:
    # the eval-panel burst that triggers WinError 1450 lasts "a couple of minutes",
    # and the old 2x10s retry gave up inside it and crashed the run.
    _RETRY_BACKOFFS = (10, 20, 40, 60, 60, 60)

    def _sleep_or_stop(self, seconds: float) -> bool:
        """Sleep up to `seconds`, but wake early (returning True) the moment a stop
        is requested -- so the End-session button is honored during a recovery storm,
        not only at the loop boundary the trainer polls."""
        import time as _t
        if self._should_stop is None:
            _t.sleep(seconds)
            return False
        for _ in range(int(seconds * 10)):
            if self._should_stop():
                return True
            _t.sleep(0.1)
        return self._should_stop()

    def _retry_chunk(self, chunk: list, seed: int, err: Exception) -> bytes:
        """Replay one failed chunk instead of losing the run to it. Two Windows
        failure modes seen under system pressure (an eval-panel burst adds ~14 GB
        of worker processes for a couple of minutes):
          * a TRANSIENT allocation/pipe failure (MemoryError, or WinError 1450
            'insufficient system resources' mid-send) -- a gc + backoff clears it;
          * that same mid-write failure KILLING a worker, which poisons the whole
            ProcessPoolExecutor (BrokenProcessPool) so every later submit/result on
            it raises 'pool not usable' -- the pool must be REBUILT before the retry
            can land. The old code resubmitted to the dead pool and re-raised, which
            is exactly what turned a blip into a crash after 150h.
        Backoff escalates to outlast the storm; a rebuild costs one worker re-warm
        and the recovery iteration replays its chunks serially, but the run survives.
        A stop requested mid-backoff raises CollectorStopped so the trainer can exit
        gracefully rather than wait out the whole schedule."""
        import gc
        for attempt, wait in enumerate(self._RETRY_BACKOFFS, 1):
            broken = isinstance(err, BrokenProcessPool) or bool(getattr(self._ex, "_broken", None))
            print(f"[pcollect] chunk of {len(chunk)} game(s) failed ({err!r}); "
                  f"{'rebuilding pool + ' if broken else ''}"
                  f"retry {attempt}/{len(self._RETRY_BACKOFFS)} after {wait}s", flush=True)
            gc.collect()
            if self._sleep_or_stop(wait):
                raise CollectorStopped()
            if broken:
                try:
                    self._ex.shutdown(wait=False, cancel_futures=True)
                except Exception:                      # noqa: BLE001 -- a dead pool may refuse
                    pass
                self._ex = self._new_executor()
            try:
                return self._ex.submit(_collect_chunk, self._last_blob, chunk, seed).result()
            except Exception as e:                     # noqa: BLE001
                err = e
        raise err                                      # exhausted the backoff -> let the run restart

    def collect(self, m, specs: list, it: int) -> list:
        """Synchronous submit+gather: play every spec with the CURRENT weights of
        `m` (strictly on-policy, exactly like the serial path)."""
        return self.gather(self.submit(m, specs, it), len(specs))

    def close(self) -> None:
        if self._ring is not None:
            try:
                self._ring.close(); self._ring.unlink()
            except Exception:                          # noqa: BLE001
                pass
            self._ring = None
        self._ex.shutdown(wait=False, cancel_futures=True)
        self._dec.shutdown(wait=False)
        if self._server is not None:
            self._server.close()
