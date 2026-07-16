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

import pickle
import zlib
from concurrent.futures import ProcessPoolExecutor
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
    from fishrl.models.policy import MaskedActor

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
    features.set_public_encoding(lite.get("train_public", True))   # match the trainer's public gate
    hidden = tuple(lite["hidden"])                        # guesser head width
    ah = tuple(lite.get("actor_hidden", hidden))          # actor head width (may differ)
    cd = int(lite.get("card_dim", 64))                    # entity card-embedding width
    _G.update(
        lite=lite,
        actor=MaskedActor(ah, lite["enc_actor"], cd),
        guesser=HandGuesser(hidden, lite["enc_guesser"], cd),
        opp_actor=MaskedActor(ah, lite["enc_actor"], cd),
        opp_guesser=HandGuesser(hidden, lite["enc_guesser"], cd),
    )
    if lite["scenario_names"]:
        from fishrl.train.scenarios import get_scenario
        for name in lite["scenario_names"]:
            get_scenario(name).ensure_pool()


def _collect_one(spec: dict):
    """One game per the spec; mirrors the serial train-loop branches exactly."""
    from fishrl.train.belief_env import BeliefAugmentedEnv
    from fishrl.train.collector import (actor_act_fn, collect_games,
                                        collect_heuristic_games, collect_vs_opponent)

    lite = _G["lite"]
    actor, guesser = _G["actor"], _G["guesser"]
    kind = spec["kind"]
    if kind == "self":
        benv = BeliefAugmentedEnv(guesser, belief=lite["use_belief"],
                                  max_decisions=lite["max_decisions"])
        return collect_games(benv, actor_act_fn(actor), 1, spec["seed"],
                             critic=None, max_decisions=lite["max_decisions"])
    if kind == "scenario":
        from fishrl.train.scenarios import ScenarioEnv, get_scenario
        senv = BeliefAugmentedEnv(
            guesser, belief=lite["use_belief"],
            env=ScenarioEnv(get_scenario(spec["name"]),
                            max_decisions=lite["max_decisions"]))
        return collect_games(senv, actor_act_fn(actor), 1, spec["seed"],
                             critic=None, max_decisions=lite["max_decisions"])
    if kind == "heuristic":
        return collect_heuristic_games(guesser, actor, 1, spec["seed"], critic=None,
                                       use_belief=lite["use_belief"],
                                       max_decisions=lite["max_decisions"],
                                       profile=spec["profile"])
    # kind == "opponent": random / attacker / frozen past-self
    okind = spec["okind"]
    models = None
    if okind == "self":
        a_state, g_state = spec["opp_state"]           # frozen nets ride with the spec
        _G["opp_actor"].load_state_dict(a_state)
        _G["opp_guesser"].load_state_dict(g_state)
        models = SimpleNamespace(actor=_G["opp_actor"], guesser=_G["opp_guesser"])
    member = SimpleNamespace(kind=okind, models=models)
    learner = SimpleNamespace(actor=actor, guesser=guesser)
    return collect_vs_opponent(learner, member, 1, spec["seed"], critic=None,
                               use_belief=lite["use_belief"],
                               max_decisions=lite["max_decisions"],
                               learner_seat=spec["lseat"])


def _collect_chunk(learner_blob: bytes, specs: list, torch_seed: int) -> bytes:
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
    torch.manual_seed(torch_seed)                      # reproducible action sampling per chunk
    learner_state = pickle.loads(learner_blob)
    _G["actor"].load_state_dict(learner_state["actor"])
    _G["guesser"].load_state_dict(learner_state["guesser"])
    out = [(spec["idx"], _collect_one(spec)) for spec in specs]
    return zlib.compress(pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL), 1)


# ── main-process side ─────────────────────────────────────────────────────────

def _cpu_state(net) -> dict:
    return {k: v.detach().cpu() for k, v in net.state_dict().items()}


class ParallelCollector:
    """Persistent collector pool. `collect(m, specs, it)` plays every spec with
    the CURRENT weights of `m` and returns the RolloutBuffers in spec order."""

    def __init__(self, cfg, workers: int):
        self.workers = workers
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
                "scenario_names": scen_names,
                "affinity": parse_affinity(getattr(cfg, "collect_affinity", ""))}
        # Recycle each worker after this many chunks: a fresh process resets the
        # hour-scale allocator creep of long-lived torch workers (each worker is
        # ~1.7 GB commit at BIRTH -- torch's runtime alone is 1.55 GB -- and only
        # grows). All workers hit the cap on the same iteration (round-robin, one
        # chunk each), so expect one slow re-warm iteration every `recycle` iters.
        try:
            self._ex = ProcessPoolExecutor(max_workers=workers,
                                           initializer=_winit, initargs=(lite,),
                                           max_tasks_per_child=512)
        except TypeError:                              # python < 3.11: no recycling
            self._ex = ProcessPoolExecutor(max_workers=workers,
                                           initializer=_winit, initargs=(lite,))

    @staticmethod
    def opp_state_for(member) -> tuple:
        """A frozen past-self's nets as CPU state_dicts, shipped WITH its spec.
        Workers are anonymous under ProcessPoolExecutor, so a ship-once cache
        can't guarantee coverage -- and at 1-2 pool games per iteration, a few
        MB alongside the per-chunk learner shipment is noise."""
        return (_cpu_state(member.models.actor), _cpu_state(member.models.guesser))

    def submit(self, m, specs: list, it: int) -> list:
        """Ship the CURRENT weights of `m` and start the specs' games on the pool;
        returns [(future, chunk, seed), ...] work items (gather needs the args to
        resubmit a failed chunk). Round-robin chunking; one weights shipment per
        chunk. Split from `gather` so a pipelined trainer can overlap the games
        with the GPU update -- the weights are snapshotted HERE, so what the games
        are played with is fixed at submit time regardless of later updates to `m`."""
        for i, s in enumerate(specs):
            s["idx"] = i
        learner_blob = pickle.dumps(
            {"actor": _cpu_state(m.actor), "guesser": _cpu_state(m.guesser)},
            protocol=pickle.HIGHEST_PROTOCOL)          # serialize ONCE, memcpy per chunk
        self._last_blob = learner_blob                 # for gather's resubmit path
        items = []
        for w, chunk in enumerate(specs[w::self.workers] for w in range(self.workers)):
            if chunk:
                seed = (it * 1009 + w) % (2**31)
                items.append((self._ex.submit(_collect_chunk, learner_blob, chunk, seed),
                              chunk, seed))
        return items

    def gather(self, items: list, n_specs: int) -> list:
        """Block on the work items and return the RolloutBuffers in spec order.

        One retry per chunk: the observed failure mode is a TRANSIENT MemoryError
        under system commit pressure (an eval-panel burst adds ~14 GB of worker
        processes for a couple of minutes), and losing a whole session to one
        unlucky allocation is far worse than replaying one chunk 10s later."""
        out: dict = {}
        for fut, chunk, seed in items:
            try:
                blob = fut.result()
            except Exception as e:                     # noqa: BLE001 -- retry ANY chunk failure once
                import gc
                import time as _t
                print(f"[pcollect] chunk of {len(chunk)} game(s) failed ({e!r}); "
                      f"retrying once in 10s", flush=True)
                gc.collect()
                _t.sleep(10)
                blob = self._ex.submit(_collect_chunk, self._last_blob,
                                       chunk, seed).result()
            for idx, buf in pickle.loads(zlib.decompress(blob)):
                out[idx] = buf
        return [out[i] for i in range(n_specs)]

    def collect(self, m, specs: list, it: int) -> list:
        """Synchronous submit+gather: play every spec with the CURRENT weights of
        `m` (strictly on-policy, exactly like the serial path)."""
        return self.gather(self.submit(m, specs, it), len(specs))

    def close(self) -> None:
        self._ex.shutdown(wait=False, cancel_futures=True)
