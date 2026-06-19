"""Granular per-phase profiler for one self-play training iteration.

The live status line reports only iterations/hour (and, when enabled, eval time); it says
nothing about WHERE the per-iteration wall-clock goes. This module breaks a single training
iteration into its phases and reports each one's wall-clock, share of the iteration, and
throughput:

    collect   -- self-play rollouts (the per-decision loop: engine step + feature encode +
                 actor forward). Suspected dominant cost; the engine is pure Python.
    actor_fwd -- the batch-1 actor forward inside collect (measured via a wrapper, so it is a
                 SUBSET of collect, not an extra phase).
    critic    -- the batched post-collection critic value pass (fill_critic_values).
    gae       -- buf.compute(gamma, lam): advantage / return computation.
    ppo       -- ppo_update: K epochs of minibatched actor+critic SGD.
    aux       -- aux_update: guesser (Poisson) + public (BCE) SGD steps.

It profiles the CURRENT policy by loading latest.pt (per-iteration cost depends on game
length, which grows as the policy improves), or a freshly built model if no checkpoint exists.
`--cprofile` additionally runs cProfile over one collection to attribute the collect loop down
to the function level (engine vs encode vs torch).

    python -m fishrl.eval.profile_train --ckpt-dir checkpoints --iters 5
    python -m fishrl.eval.profile_train --iters 3 --cprofile

Note on threads: by default this honours OMP_NUM_THREADS etc. Run it with the SAME caps as the
service (6) and with the trainer STOPPED for a clean read -- a concurrent torch-bound trainer
oversubscribes the box and inflates the ppo/aux phases most.
"""
from __future__ import annotations

import argparse
import os
import time
from contextlib import contextmanager

NETS = ("actor", "critic", "guesser", "public")


@contextmanager
def _timed(acc: dict, key: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        acc[key] = acc.get(key, 0.0) + (time.perf_counter() - t0)


def _build_from_ckpt(ckpt_path: str):
    """Rebuild models + config from a checkpoint (matching parallel_panel's loader)."""
    from fishrl.train.checkpoint import load_checkpoint
    from fishrl.train.config import Config
    from fishrl.train.train_loop import _load_model_state, build_models

    pl = load_checkpoint(ckpt_path, map_location="cpu")
    cfg_dict = pl["config"]
    per_net = {f"{n}_encoder": cfg_dict["encoders"][n] for n in NETS}
    cfg = Config(seed=cfg_dict["seed"], use_belief=cfg_dict.get("use_belief", True),
                 critic_hidden=tuple(cfg_dict.get("critic_hidden", (512, 512, 256))), **per_net)
    m = build_models(cfg)
    _load_model_state(m, pl["models"])
    return cfg, m, int(pl.get("done", 0))


def _timed_act_fn(actor, acc: dict):
    """Wrap actor_act_fn so the batch-1 forward time is accumulated into acc['actor_fwd']."""
    from fishrl.train.collector import actor_act_fn
    inner = actor_act_fn(actor)

    def act(obs):
        t0 = time.perf_counter()
        out = inner(obs)
        acc["actor_fwd"] = acc.get("actor_fwd", 0.0) + (time.perf_counter() - t0)
        return out
    return act


def profile(ckpt_dir: str, iters: int, warmup_iters: int, games_per_iter: int | None,
            run_cprofile: bool) -> None:
    import torch

    from fishrl.train.checkpoint import latest_path
    from fishrl.train.collector import collect_games, fill_critic_values
    from fishrl.train.belief_env import BeliefAugmentedEnv
    from fishrl.train.ppo import aux_update, ppo_update

    latest = latest_path(ckpt_dir)
    if os.path.exists(latest):
        cfg, m, done = _build_from_ckpt(latest)
        src = f"latest.pt @ it={done}"
    else:
        from fishrl.train.config import Config
        from fishrl.train.train_loop import build_models
        cfg = Config()
        m = build_models(cfg)
        src = "fresh build (no checkpoint)"

    gpi = games_per_iter or cfg.games_per_iter
    benv = BeliefAugmentedEnv(m.guesser, belief=cfg.use_belief, max_decisions=cfg.max_decisions)
    act = _timed_act_fn(m.actor, {})  # discarded; real one is rebound per iter below

    opt_ppo = torch.optim.Adam(list(m.actor.parameters()) + list(m.critic.parameters()), lr=cfg.lr_ppo)
    opt_g = torch.optim.Adam(m.guesser.parameters(), lr=cfg.lr_guesser)
    opt_p = torch.optim.Adam(m.public.parameters(), lr=cfg.lr_public)

    print(f"[profile] src={src} threads={torch.get_num_threads()} games/iter={gpi} "
          f"epochs={cfg.ppo_epochs} mb={cfg.minibatch} aux_steps={cfg.aux_steps} "
          f"warmup={warmup_iters} measured={iters}", flush=True)

    acc = {k: 0.0 for k in ("collect", "actor_fwd", "critic", "gae", "ppo", "aux")}
    tot_T = tot_games = 0
    base_seed = cfg.seed + 1000 + (done if os.path.exists(latest) else 0) * gpi

    for it in range(warmup_iters + iters):
        measure = it >= warmup_iters
        local: dict = {}                                  # per-iter timings (added into acc below)
        a = _timed_act_fn(m.actor, local)
        seed = base_seed + it * gpi
        with _timed(local, "collect"):
            buf = collect_games(benv, a, gpi, seed, critic=None, max_decisions=cfg.max_decisions)
        with _timed(local, "critic"):
            fill_critic_values(buf, m.critic)
        with _timed(local, "gae"):
            batch = buf.compute(cfg.gamma, cfg.lam)
        with _timed(local, "ppo"):
            ppo_update(batch, m.actor, m.critic, opt_ppo, cfg, cfg.ent_end)
        with _timed(local, "aux"):
            aux_update(batch, m.guesser, m.public, opt_g, opt_p, cfg.aux_steps)
        if measure:
            for k in acc:
                acc[k] += local.get(k, 0.0)
            tot_T += len(buf)
            tot_games += gpi
        print(f"  iter {it - warmup_iters if measure else 'w'}: "
              f"T={len(buf)} collect={local['collect']:.2f}s ppo={local['ppo']:.2f}s", flush=True)

    # Phase total = collect + critic + gae + ppo + aux (actor_fwd is a subset of collect).
    phase_total = sum(acc[k] for k in ("collect", "critic", "gae", "ppo", "aux"))
    per_iter = phase_total / iters
    print(f"\n[profile] {iters} iters, {tot_games} games, {tot_T} transitions "
          f"({tot_T / tot_games:.0f}/game)")
    print(f"[profile] iteration wall-clock = {per_iter:.3f}s  ->  {3600.0 / per_iter:.0f} iters/h "
          f"({tot_T / phase_total:.0f} transitions/s, {tot_games / phase_total:.2f} games/s)\n")

    order = ["collect", "actor_fwd", "critic", "gae", "ppo", "aux"]
    label = {"collect": "collect (rollouts)", "actor_fwd": "  +- actor_fwd (subset)",
             "critic": "critic fill (batched)", "gae": "gae / returns",
             "ppo": "ppo_update", "aux": "aux_update"}
    print(f"  {'phase':<24}{'s/iter':>10}{'% iter':>9}{'ms/game':>10}")
    for k in order:
        s_iter = acc[k] / iters
        pct = 100.0 * acc[k] / phase_total if k != "actor_fwd" else 100.0 * acc[k] / acc["collect"]
        suffix = " of collect" if k == "actor_fwd" else ""
        print(f"  {label[k]:<24}{s_iter:>10.3f}{pct:>8.1f}%{1000 * acc[k] / tot_games:>10.1f}{suffix}")
    # The collect remainder = engine step + feature encode + env overhead (collect - actor_fwd).
    rem = acc["collect"] - acc["actor_fwd"]
    print(f"  {'  +- engine+encode+env':<24}{rem / iters:>10.3f}{100.0 * rem / acc['collect']:>8.1f}%"
          f"{1000 * rem / tot_games:>10.1f} of collect")

    if run_cprofile:
        import cProfile
        import pstats

        print("\n[profile] cProfile of ONE collection (top 18 by tottime):")
        a = _timed_act_fn(m.actor, {})
        pr = cProfile.Profile()
        pr.enable()
        collect_games(benv, a, gpi, base_seed, critic=m.critic, max_decisions=cfg.max_decisions)
        pr.disable()
        pstats.Stats(pr).sort_stats("tottime").print_stats(18)


def main() -> None:
    ap = argparse.ArgumentParser(description="Granular per-phase training-iteration profiler.")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--iters", type=int, default=5, help="measured iterations")
    ap.add_argument("--warmup-iters", type=int, default=1, help="discarded warm-up iterations")
    ap.add_argument("--games-per-iter", type=int, default=None, help="override cfg.games_per_iter")
    ap.add_argument("--cprofile", action="store_true", help="also cProfile one collection")
    args = ap.parse_args()
    profile(args.ckpt_dir, args.iters, args.warmup_iters, args.games_per_iter, args.cprofile)


if __name__ == "__main__":
    main()
