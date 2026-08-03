"""Very discretised profile of the LIVE training configuration.

The existing profilers predate the current deployment shape: profile_train times the
serial phases, profile_collection buckets an untrained flat-policy collection. The
live mainline is neither — it is 8 pinned collector workers playing 8 games per
iteration (1 game per chunk) under --pipeline-collect, an entity actor at trained
weights (~230 decisions/game), and a CUDA update. Its stats.json only records the
coarse collect_s/update_s pair, and under pipeline `collect_s` is the residual
STALL waiting for workers, not where the time actually goes.

This module opens that up with three lenses, each selectable via --sections:

  A  iteration anatomy   Replays live-shaped iterations (same spec mix as the last
                         stats windows: 3 mirror / 3 past-self / 1 heuristic_1_2 /
                         1 scenario) through a real ParallelCollector with the live
                         affinity mask, and times every main-thread component the
                         trainer's two timers lump together: spec build (frozen-net
                         pickling), learner-weights pickle, per-chunk worker wall
                         (min/mean/max -> the max-of-8 imbalance IS the pipelined
                         stall), result decompress+unpickle, buffer merge, critic
                         fill, GAE, ppo_update, aux_update.
  B  worker microscope   cProfile of the same game mix played serially at the live
                         weights (1 torch thread, live affinity mask), bucketed per
                         decision: engine step, mask build, perspective encode,
                         god/public features, guesser forward, actor forward split
                         into entity-encoder vs head, tensor building, buffer append.
  C  update microscope   The CUDA update at a real batch: H2D copy, ppo_update and
                         aux_update wall (synchronized), plus a torch.profiler op
                         table attributing the CUDA time (matmuls fwd/bwd, optimizer,
                         copies, softmax/exp glue).

    python -m fishrl.eval.profile_discrete                     # all three sections
    python -m fishrl.eval.profile_discrete --sections a --iters 15

Run with the trainer STOPPED: a concurrent trainer or eval panel oversubscribes the
box and corrupts every number here.
"""
from __future__ import annotations

import argparse
import cProfile
import pickle
import pstats
import time
import zlib

import numpy as np

# live deployment shape (fishrl-pc.bat / deploy/train.args)
LIVE_WORKERS = 8
LIVE_AFFINITY = "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
# live spec mix per 8-game iteration, from the last stats windows (opp_self 3.0,
# opp_past 2.5, heuristic family 1.4, scenario 0.9 -> rounded to 8 whole games)
MIX = [("self", 3), ("pastself", 3), ("heuristic_1_2", 1), ("scenario", 1)]
SCEN_NAMES = ("deckout", "survive_lethal")     # alternated for the scenario slot


def _load(ckpt_dir: str, device: str):
    from fishrl.train.checkpoint import latest_path, load_checkpoint
    from fishrl.train.train_loop import (_load_model_state, build_models,
                                         config_from_checkpoint)
    pl = load_checkpoint(latest_path(ckpt_dir), map_location="cpu")
    cfg = config_from_checkpoint(pl["config"], device=device)
    m = build_models(cfg)
    _load_model_state(m, pl["models"])
    for net in (m.actor, m.critic, m.guesser, m.public):
        net.eval()
    return cfg, m, int(pl.get("done", 0))


def _specs(m, it: int, cfg) -> list:
    """One live-shaped iteration of specs. Past-self opponents use the CURRENT
    weights as the frozen nets — identical forward cost, and it exercises the real
    per-spec state_dict shipping the live trainer pays for every pool game."""
    from fishrl.train.pcollect import _cpu_state
    specs, seed = [], cfg.seed + 1000 + it * 8
    gi = 0
    for kind, n in MIX:
        for _ in range(n):
            s = seed + gi
            if kind == "self":
                specs.append({"kind": "self", "seed": s})
            elif kind == "pastself":
                specs.append({"kind": "opponent", "okind": "self", "seed": s,
                              "lseat": "p1" if gi % 2 == 0 else "p2",
                              "opp_state": (_cpu_state(m.actor), _cpu_state(m.guesser))})
            elif kind == "scenario":
                specs.append({"kind": "scenario", "seed": s,
                              "name": SCEN_NAMES[it % len(SCEN_NAMES)]})
            else:
                specs.append({"kind": "heuristic", "profile": kind, "seed": s})
            gi += 1
    return specs


def _sync(dev: str) -> None:
    if dev.startswith("cuda"):
        import torch
        torch.cuda.synchronize()


class _T:
    """Accumulating wall-clock buckets."""

    def __init__(self):
        self.acc: dict = {}

    def add(self, key: str, dt: float) -> None:
        self.acc[key] = self.acc.get(key, 0.0) + dt

    def time(self, key: str, fn, *a, **kw):
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        self.add(key, time.perf_counter() - t0)
        return out


# ── section A: iteration anatomy through a real worker pool ──────────────────

def section_a(cfg, m, done: int, iters: int, warmup: int,
              workers: int = LIVE_WORKERS, affinity: str = LIVE_AFFINITY) -> None:
    import torch

    from fishrl.data.buffer import RolloutBuffer
    from fishrl.train.collector import fill_critic_values
    from fishrl.train.pcollect import ParallelCollector
    from fishrl.train.ppo import aux_update, ppo_update

    dev = cfg.device
    from fishrl.train.scenarios import scenario_names
    cfg.collect_workers, cfg.collect_affinity = workers, affinity
    cfg.scenarios_in_pool = True                     # workers pre-build ONLY these pools
    cfg.scenario_weights = {n: (1.0 if n in SCEN_NAMES else 0.0)
                            for n in scenario_names()}
    pcol = ParallelCollector(cfg, workers)
    opt = torch.optim.Adam(list(m.actor.parameters()) + list(m.critic.parameters()),
                           lr=cfg.lr_ppo)
    opt_g = torch.optim.Adam(m.guesser.parameters(), lr=cfg.lr_guesser)
    opt_p = torch.optim.Adam(m.public.parameters(), lr=cfg.lr_public)

    t = _T()
    chunk_walls: list = []                           # per-iter [per-chunk wall, ...]
    T_total = games_total = 0
    print(f"\n=== A. iteration anatomy | {workers} workers, affinity [{affinity}], "
          f"mix {dict(MIX)} | {warmup} warmup + {iters} measured iters ===", flush=True)

    for it in range(warmup + iters):
        measure = it >= warmup
        li = _T()
        specs = li.time("spec_build", _specs, m, done + it, cfg)
        t0 = time.perf_counter()
        items = pcol.submit(m, specs, done + it)     # learner pickle + chunk submits
        li.add("submit", time.perf_counter() - t0)
        finish: dict = {}
        for i, (fut, _c, _s) in enumerate(items):
            fut.add_done_callback(lambda f, i=i: finish.setdefault(i, time.perf_counter()))
        blobs = []
        for i, (fut, _c, _s) in enumerate(items):
            blobs.append(fut.result())
        walls = [finish[i] - t0 for i in range(len(items))]
        li.add("worker_wall_max", max(walls))
        t0 = time.perf_counter()
        parts = [pickle.loads(zlib.decompress(b)) for b in blobs]
        li.add("unpack", time.perf_counter() - t0)
        buf = RolloutBuffer()
        t0 = time.perf_counter()
        out = {}
        for part in parts:
            for idx, gbuf in part:
                out[idx] = gbuf
        for i in range(len(specs)):
            buf.merge(out[i])
        li.add("merge", time.perf_counter() - t0)
        t0 = time.perf_counter()
        fill_critic_values(buf, m.critic)
        _sync(dev)
        li.add("critic_fill", time.perf_counter() - t0)
        t0 = time.perf_counter()
        batch = buf.compute(cfg.gamma, cfg.lam)
        li.add("gae", time.perf_counter() - t0)
        t0 = time.perf_counter()
        ppo_update(batch, m.actor, m.critic, opt, cfg, cfg.ent_end, rng_seed=it)
        _sync(dev)
        li.add("ppo", time.perf_counter() - t0)
        t0 = time.perf_counter()
        aux_update(batch, m.guesser, m.public, opt_g, opt_p, cfg.aux_steps)
        _sync(dev)
        li.add("aux", time.perf_counter() - t0)
        if measure:
            for k, v in li.acc.items():
                t.add(k, v)
            chunk_walls.append(walls)
            T_total += len(buf)
            games_total += len(specs)
        print(f"  iter {it - warmup if measure else 'w'}: T={len(buf)} "
              f"worker max={max(walls):.2f}s mean={np.mean(walls):.2f}s "
              f"ppo={li.acc['ppo']:.2f}s", flush=True)

    pcol.close()
    n = iters
    all_walls = np.array([w for ws in chunk_walls for w in ws])
    maxes = np.array([max(ws) for ws in chunk_walls])
    update = sum(t.acc[k] for k in ("critic_fill", "gae", "ppo", "aux"))
    over = sum(t.acc[k] for k in ("spec_build", "submit", "unpack", "merge"))
    stall = float(np.mean(np.maximum(0.0, maxes - (update + over) / n)))

    print(f"\n  {T_total} transitions / {games_total} games over {n} iters "
          f"({T_total / games_total:.0f}/game)")
    print(f"  {'component':<30}{'s/iter':>9}   note")

    def row(label, s, note=""):
        print(f"  {label:<30}{s:>9.3f}   {note}")

    row("spec_build (frozen pickles)", t.acc["spec_build"] / n, "3x past-self state_dicts")
    row("submit (learner pickle)", t.acc["submit"] / n)
    row("worker wall  mean chunk", float(all_walls.mean()), "one game per chunk")
    row("worker wall  MAX chunk", float(maxes.mean()),
        f"imbalance x{float(maxes.mean() / all_walls.mean()):.2f} — the collect bound")
    row("unpack (zlib+pickle)", t.acc["unpack"] / n)
    row("merge buffers", t.acc["merge"] / n)
    row("critic_fill (batched fwd)", t.acc["critic_fill"] / n)
    row("gae (buf.compute)", t.acc["gae"] / n)
    row("ppo_update", t.acc["ppo"] / n)
    row("aux_update", t.acc["aux"] / n)
    row("-> update+overhead total", (update + over) / n)
    row("-> est. pipelined stall", stall, "max-chunk minus hideable work")
    row("-> est. pipelined iter", stall + (update + over) / n,
        f"~{3600.0 / max(stall + (update + over) / n, 1e-9):.0f} it/h")
    per_dec = all_walls.sum() * 1e3 / T_total
    print(f"\n  worker cost: {per_dec:.2f} ms/decision "
          f"({T_total / all_walls.sum():.0f} dec/s aggregate across the pool)")


# ── section B: worker-side per-decision microscope ────────────────────────────

def _play_mix_serial(cfg, m, iters: int, base_it: int) -> int:
    """The same game mix Section A ships to workers, played serially in-process
    (this is exactly what one worker does, minus the pickle shipping). Returns
    the number of labelled transitions for per-decision normalization."""
    from types import SimpleNamespace

    from fishrl.train.belief_env import BeliefAugmentedEnv
    from fishrl.train.collector import (actor_act_fn, collect_games,
                                        collect_heuristic_games, collect_vs_opponent)
    n = 0
    for it in range(iters):
        seed = cfg.seed + 1000 + (base_it + it) * 8
        gi = 0
        for kind, cnt in MIX:
            for _ in range(cnt):
                s = seed + gi
                if kind == "self":
                    benv = BeliefAugmentedEnv(m.guesser, belief=cfg.use_belief,
                                              max_decisions=cfg.max_decisions)
                    buf = collect_games(benv, actor_act_fn(m.actor), 1, s,
                                        critic=None, max_decisions=cfg.max_decisions)
                elif kind == "pastself":
                    member = SimpleNamespace(kind="self", models=SimpleNamespace(
                        actor=m.actor, guesser=m.guesser))
                    buf = collect_vs_opponent(m, member, 1, s, critic=None,
                                              use_belief=cfg.use_belief,
                                              max_decisions=cfg.max_decisions,
                                              learner_seat="p1" if gi % 2 == 0 else "p2")
                elif kind == "scenario":
                    from fishrl.train.scenarios import ScenarioEnv, get_scenario
                    senv = BeliefAugmentedEnv(
                        m.guesser, belief=cfg.use_belief,
                        env=ScenarioEnv(get_scenario(SCEN_NAMES[it % len(SCEN_NAMES)]),
                                        max_decisions=cfg.max_decisions))
                    buf = collect_games(senv, actor_act_fn(m.actor), 1, s,
                                        critic=None, max_decisions=cfg.max_decisions)
                else:
                    buf = collect_heuristic_games(m.guesser, m.actor, 1, s,
                                                  critic=None, use_belief=cfg.use_belief,
                                                  max_decisions=cfg.max_decisions,
                                                  profile=kind)
                n += len(buf)
                gi += 1
    return n


def _cum(st: pstats.Stats, file_sub: str, func: str) -> float:
    total = 0.0
    for (fn, _ln, name), (_cc, _nc, _tt, ct, _callers) in st.stats.items():
        if name == func and file_sub in fn.replace("\\", "/"):
            total += ct
    return total


def _tot_by(st: pstats.Stats, file_sub: str) -> float:
    total = 0.0
    for (fn, _ln, _name), (_cc, _nc, tt, _ct, _callers) in st.stats.items():
        if file_sub in fn.replace("\\", "/"):
            total += tt
    return total


def section_b(cfg, m, done: int, iters: int) -> None:
    import torch

    from fishrl.train.pcollect import _apply_affinity, parse_affinity
    from fishrl.train.scenarios import get_scenario

    torch.set_num_threads(1)                        # a worker's exact torch shape
    _apply_affinity(parse_affinity(LIVE_AFFINITY))
    for name in SCEN_NAMES:
        get_scenario(name).ensure_pool()            # exclude one-time pool builds

    print(f"\n=== B. worker microscope | serial, 1 torch thread, live weights, "
          f"{iters} iteration(s) of the live mix ===", flush=True)
    pr = cProfile.Profile()
    pr.enable()
    n_dec = _play_mix_serial(cfg, m, iters, done)
    pr.disable()
    st = pstats.Stats(pr)
    total = st.total_tt

    guesser_fwd = _cum(st, "fishrl/models/guesser.py", "forward")
    actor_path = (_cum(st, "fishrl/train/collector.py", "act")
                  + _cum(st, "fishrl/train/collector.py", "_belief_guess"))
    enc_fwd = _cum(st, "fishrl/models/entity_encoder.py", "forward")
    observe = _cum(st, "fishrl/train/belief_env.py", "observe")
    step = _cum(st, "fishrl/env/aec_env.py", "step")
    persp = _cum(st, "fishrl/obs/encoder.py", "encode_observation")
    god = _cum(st, "fishrl/data/features.py", "encode_god")
    pub = _cum(st, "fishrl/data/features.py", "encode_public")
    cnt = _cum(st, "fishrl/data/features.py", "opponent_hand_counts")
    mask = _cum(st, "fishrl/spaces/masking.py", "legal_mask")
    engine_self = _tot_by(st, "forgetful_fish/engine.py")
    state_self = _tot_by(st, "forgetful_fish/state.py")
    ai_self = sum(_tot_by(st, f"forgetful_fish/{f}.py") for f in ("ai", "ai_v1_1", "ai_v1_2"))

    def row(label, t, denom=None):
        print(f"  {label:<38} {t * 1e3 / n_dec:8.3f} ms/dec  {100 * t / (denom or total):5.1f}%")

    print(f"  {n_dec} decisions in {total:.1f}s -> {total * 1e3 / n_dec:.3f} ms/decision "
          f"({n_dec / total:.0f} dec/s single worker)\n")
    print("  NETWORK (cumulative, incl. torch internals):")
    row("actor act() + opp/belief forwards", actor_path)
    row("  +- entity encoder forward", enc_fwd)
    row("guesser forward", guesser_fwd)
    print("  ENV / ENGINE / FEATURES:")
    row("env.step (engine advance)", step)
    row("  +- engine.py self-time", engine_self)
    row("  +- state.py self-time (views)", state_self)
    row("  +- vendored ai*.py self-time", ai_self)
    row("legal_mask build", mask)
    row("observe (perspective+belief wrap)", observe)
    row("  +- encode_observation", persp)
    row("encode_god / public / hand counts", god + pub + cnt)

    skip = ("site-packages/torch", "/torch/", "<built-in")
    items = [((fn, name), tt) for (fn, _ln, name), (_cc, _nc, tt, _ct, _cl)
             in st.stats.items()
             if not any(s in fn.replace("\\", "/") for s in skip)]
    items.sort(key=lambda kv: kv[1], reverse=True)
    print("\n  TOP 25 BY SELF-TIME (non-torch):")
    for (fn, name), tt in items[:25]:
        mod = fn.replace("\\", "/").split("/")[-1]
        print(f"    {tt * 1e3 / n_dec:7.3f} ms/dec  {100 * tt / total:5.1f}%  {mod}:{name}")
    torch_self = sum(tt for (fn, _ln, _n), (_cc, _nc, tt, _ct, _cl) in st.stats.items()
                     if "torch" in fn.replace("\\", "/") or fn == "~")
    print(f"\n  torch+builtins self-time total: {torch_self * 1e3 / n_dec:.3f} ms/dec "
          f"({100 * torch_self / total:.1f}%)")


# ── section C: update microscope at a real batch ──────────────────────────────

def section_c(cfg, m, done: int, reps: int) -> None:
    import torch

    from fishrl.train.collector import fill_critic_values
    from fishrl.train.ppo import aux_update, ppo_update

    dev = cfg.device
    print(f"\n=== C. update microscope | device={dev} | one live-shaped batch ===",
          flush=True)
    torch.set_num_threads(6)                     # the live trainer's main-process shape
    t = _T()
    from fishrl.data.buffer import RolloutBuffer
    from fishrl.train.belief_env import BeliefAugmentedEnv
    from fishrl.train.collector import actor_act_fn, collect_games
    buf = RolloutBuffer()
    benv = BeliefAugmentedEnv(m.guesser, belief=cfg.use_belief,
                              max_decisions=cfg.max_decisions)
    buf.merge(collect_games(benv, actor_act_fn(m.actor), 8, cfg.seed + 999_000,
                            critic=None, max_decisions=cfg.max_decisions))
    print(f"  batch: {len(buf)} transitions (8 mirror games)")

    _sync(dev)
    for _ in range(reps):
        t.time("critic_fill", fill_critic_values, buf, m.critic)
        _sync(dev)
    t.acc["critic_fill"] /= reps
    batch = t.time("gae", buf.compute, cfg.gamma, cfg.lam)

    keys = ("x_act", "mask", "action", "old_logp", "adv", "god", "y_p1", "valid")
    t0 = time.perf_counter()
    _b = {k: batch[k].to(dev, non_blocking=True) for k in keys}
    _sync(dev)
    t.add("h2d_copy", time.perf_counter() - t0)
    del _b

    opt = torch.optim.Adam(list(m.actor.parameters()) + list(m.critic.parameters()),
                           lr=cfg.lr_ppo)
    opt_g = torch.optim.Adam(m.guesser.parameters(), lr=cfg.lr_guesser)
    opt_p = torch.optim.Adam(m.public.parameters(), lr=cfg.lr_public)
    ppo_update(batch, m.actor, m.critic, opt, cfg, cfg.ent_end)   # warm the kernels
    _sync(dev)
    for r in range(reps):
        t0 = time.perf_counter()
        ppo_update(batch, m.actor, m.critic, opt, cfg, cfg.ent_end, rng_seed=r)
        _sync(dev)
        t.add("ppo_update", time.perf_counter() - t0)
    t.acc["ppo_update"] /= reps
    for _ in range(reps):
        t0 = time.perf_counter()
        aux_update(batch, m.guesser, m.public, opt_g, opt_p, cfg.aux_steps)
        _sync(dev)
        t.add("aux_update", time.perf_counter() - t0)
    t.acc["aux_update"] /= reps

    M = batch["x_act"].shape[0]
    steps = cfg.ppo_epochs * ((M + cfg.minibatch - 1) // cfg.minibatch)
    print(f"  {'component':<26}{'s/iter':>9}   note")
    for k, note in (("critic_fill", "batched value pass"),
                    ("gae", "CPU, per-game segments"),
                    ("h2d_copy", "batch to device, one-off"),
                    ("ppo_update", f"{cfg.ppo_epochs} epochs x mb{cfg.minibatch} = {steps} steps"),
                    ("aux_update", f"{cfg.aux_steps} guesser+public steps")):
        print(f"  {k:<26}{t.acc[k]:>9.3f}   {note}")
    print(f"  ppo per minibatch step: {t.acc['ppo_update'] * 1e3 / steps:.1f} ms")

    if dev.startswith("cuda"):
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            ppo_update(batch, m.actor, m.critic, opt, cfg, cfg.ent_end, rng_seed=99)
            torch.cuda.synchronize()
        print("\n  TOP 15 OPS BY CUDA TIME (one ppo_update):")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt-dir", default="checkpoints-v2")
    ap.add_argument("--sections", default="a,b,c")
    ap.add_argument("--iters", type=int, default=12, help="measured iters (section A)")
    ap.add_argument("--warmup-iters", type=int, default=2)
    ap.add_argument("--b-iters", type=int, default=2, help="mix repetitions (section B)")
    ap.add_argument("--reps", type=int, default=5, help="update repetitions (section C)")
    ap.add_argument("--workers", type=int, default=LIVE_WORKERS,
                    help="pool size for section A (contention experiments)")
    ap.add_argument("--affinity", default=LIVE_AFFINITY,
                    help="worker affinity mask for section A")
    ap.add_argument("--cpu", action="store_true", help="force CPU for the update section")
    args = ap.parse_args()

    import torch
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    cfg, m, done = _load(args.ckpt_dir, device)
    secs = {s.strip().lower() for s in args.sections.split(",")}
    print(f"[profile_discrete] {args.ckpt_dir} @ it={done} | device={device} | "
          f"actor={cfg.enc_for('actor')}/{cfg.head_hidden('actor')} d={cfg.card_dim}")

    if "a" in secs:
        section_a(cfg, m, done, args.iters, args.warmup_iters,
                  workers=args.workers, affinity=args.affinity)
    if "b" in secs:
        # section B wants CPU nets (a worker's shape); rebuild on CPU if needed
        if device != "cpu":
            cfg_b, m_b, _ = _load(args.ckpt_dir, "cpu")
        else:
            cfg_b, m_b = cfg, m
        section_b(cfg_b, m_b, done, args.b_iters)
    if "c" in secs:
        section_c(cfg, m, done, args.reps)


if __name__ == "__main__":
    main()
