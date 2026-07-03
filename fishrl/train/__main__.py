"""Self-play training entry point.

    python -m fishrl.train --iters 200 --encoder entity            # bounded run
    python -m fishrl.train --resume --iters 0                       # resume + run forever
    python -m fishrl.train --gpu --iters 200                        # on CUDA

Runs warmup + PPO self-play with crash-safe checkpointing to `<ckpt-dir>/latest.pt`. On
`--resume` it restores the latest checkpoint (models + optimizers + RNG + iteration counter)
and continues; `--iters 0` runs indefinitely until SIGTERM/SIGINT (graceful checkpoint).
"""
from __future__ import annotations

import argparse
import os

from fishrl.train.threads import DEFAULT_RESERVE, preconfigure

# BEFORE the imports below: BLAS/OpenMP read the *_NUM_THREADS env vars once,
# when torch loads its native libraries. fishrl.train.threads is torch-free.
_N_THREADS = preconfigure()

from fishrl.eval.metrics import collect_eval_batch, estimator_metrics, panel_winrates  # noqa: E402
from fishrl.train import checkpoint as ckpt                                            # noqa: E402
from fishrl.train.config import Config, resolve_device                                 # noqa: E402
from fishrl.train.train_loop import build_models, train                                # noqa: E402

NETS = ("actor", "critic", "guesser", "public")


def main():
    ap = argparse.ArgumentParser(description="Self-play training for fishrl.")
    ap.add_argument("--gpu", action="store_true", help="train on CUDA if available")
    ap.add_argument("--iters", type=int, default=100, help="iteration cap; <=0 runs unbounded")
    ap.add_argument("--games-per-iter", type=int, default=8)
    ap.add_argument("--warmup-games", type=int, default=64)
    ap.add_argument("--pool-frac", type=float, default=Config.pool_frac,
                    help="fraction of each iteration's games played vs a PFSP league "
                         "opponent (sampled by difficulty); 0 = pure self-play")
    ap.add_argument("--pfsp-mode", choices=["hard", "var"], default=Config.pfsp_mode,
                    help="opponent priority: 'hard' favours opponents you lose to, "
                         "'var' favours even matchups")
    ap.add_argument("--league-size", type=int, default=Config.league_size,
                    help="ring length of frozen past-self league members (0 = anchors only)")
    ap.add_argument("--scenario-pool", action="store_true", default=Config.scenarios_in_pool,
                    help="fold the scenarios into the main PFSP league (their play rate "
                         "floats with difficulty within --pool-frac) instead of the fixed "
                         "--scenario-frac carve-out; overrides --scenario-frac")
    ap.add_argument("--scenario-frac", type=float, default=Config.scenario_frac,
                    help="fraction of each iteration's games seeded from a curriculum "
                         "scenario start-state (0 = off). Mixed INTO self-play; reward stays "
                         "terminal. Judge progress on vs-heuristic WR, NOT scenario win-rate.")
    ap.add_argument("--enforce-free-attack", default=Config.enforce_free_attack,
                    action=argparse.BooleanOptionalAction,
                    help="hard rule across all training: declaring fewer than all eligible "
                         "attackers into an empty opposing board is an instant loss")
    ENC = ["flat", "entity", "attention"]
    ap.add_argument("--encoder", choices=ENC, default="flat",
                    help="base encoder for any net without a per-net override")
    # per-net overrides (None -> fall back to --encoder). The critic is off the
    # deployment path and batchable, so e.g. `--encoder flat --critic-encoder entity`
    # buys entity's calibration win at near-zero collection cost.
    ap.add_argument("--actor-encoder", choices=ENC, default=None)
    ap.add_argument("--critic-encoder", choices=ENC, default=None)
    ap.add_argument("--guesser-encoder", choices=ENC, default=None)
    ap.add_argument("--public-encoder", choices=ENC, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--resume", action="store_true",
                    help="resume from <ckpt-dir>/latest.pt if it exists")
    ap.add_argument("--fresh", action="store_true",
                    help="start fresh, overwriting an existing checkpoint")
    ap.add_argument("--report-every-seconds", type=float, default=3600.0)
    ap.add_argument("--report-winrate-games", type=int, default=30,
                    help="games per opponent in each status-line win-rate panel")
    ap.add_argument("--checkpoint-every-seconds", type=float, default=900.0)
    ap.add_argument("--archive-every-iters", type=int, default=Config.archive_every_iters,
                    help="write a permanent archive_{it}.pt every N iterations "
                         "(never pruned, unlike step_*.pt milestones); <=0 disables")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="optional wall-clock cap (across restarts); omit for indefinite")
    ap.add_argument("--reserve-cores", type=int, default=None,
                    help="keep this many PHYSICAL cores free of training threads "
                         f"(default {DEFAULT_RESERVE} unless *_NUM_THREADS is already set "
                         "in the environment). Applied before torch loads; parsed early.")
    ap.add_argument("--gui", action="store_true",
                    help="open the fishrl.monitor window (a read-only subprocess over "
                         "stats.json; closing it never affects training)")
    args = ap.parse_args()

    if _N_THREADS is not None:
        import torch
        torch.set_num_threads(_N_THREADS)             # env caps BLAS; this caps torch's own pool

    latest = ckpt.latest_path(args.ckpt_dir)
    have_ckpt = os.path.exists(latest)
    if have_ckpt and not args.resume and not args.fresh:
        ap.error(f"{latest} exists; pass --resume to continue or --fresh to overwrite it")
    resume = args.resume and have_ckpt
    if args.resume and not have_ckpt:
        print(f"[resume] no checkpoint at {latest}; starting fresh", flush=True)

    common = dict(device=resolve_device(args.gpu), iters=args.iters,
                  games_per_iter=args.games_per_iter, warmup_games=args.warmup_games,
                  pool_frac=args.pool_frac, pfsp_mode=args.pfsp_mode,
                  league_size=args.league_size, scenario_frac=args.scenario_frac,
                  scenarios_in_pool=args.scenario_pool,
                  enforce_free_attack=args.enforce_free_attack,
                  ckpt_dir=args.ckpt_dir, report_every_seconds=args.report_every_seconds,
                  report_winrate_games=args.report_winrate_games,
                  checkpoint_every_seconds=args.checkpoint_every_seconds,
                  archive_every_iters=args.archive_every_iters)
    if resume:
        # architecture + seed MUST match the saved weights -> take them from the checkpoint
        # (the --encoder* flags are ignored on resume).
        saved = ckpt.load_checkpoint(latest, map_location="cpu")["config"]
        per_net = {f"{n}_encoder": saved["encoders"][n] for n in NETS}
        cfg = Config(seed=saved["seed"], use_belief=saved.get("use_belief", True),
                     critic_hidden=tuple(saved.get("critic_hidden", (512, 512, 256))),
                     **per_net, **common)
    else:
        per_net = {f"{n}_encoder": getattr(args, f"{n}_encoder")
                   for n in NETS if getattr(args, f"{n}_encoder") is not None}
        cfg = Config(encoder=args.encoder, seed=args.seed, **per_net, **common)

    encs = {n: cfg.enc_for(n) for n in NETS}
    cap = "unbounded" if cfg.iters <= 0 else cfg.iters
    print(f"device: {cfg.device} | encoders: {encs} | iters: {cap} | "
          f"pool: {cfg.pool_frac:.2f} (pfsp={cfg.pfsp_mode}, league={cfg.league_size}) | "
          f"resume: {resume} | ckpt: {latest}", flush=True)

    monitor = None
    if args.gui:
        import subprocess
        import sys as _sys
        monitor = subprocess.Popen([_sys.executable, "-m", "fishrl.monitor",
                                    "--ckpt-dir", args.ckpt_dir])

    max_seconds = args.max_hours * 3600.0 if args.max_hours else None
    try:
        models = train(cfg, build_models(cfg), log=lambda s: print(s, flush=True),
                       max_seconds=max_seconds,
                       resume_path=latest if resume else None, checkpoint_path=latest)
        print(f"saved checkpoint -> {latest}", flush=True)

        # higher-game-count final readout for bounded runs (the in-loop status lines use fewer
        # games each). Skipped for the unbounded service, which exits via signal and shouldn't
        # spend tens of seconds on eval during shutdown.
        if cfg.iters > 0:
            # thread the run's belief/decision settings through — a belief-off run
            # evaluated with belief ON feeds live guesser output to an actor trained on zeros
            print("final win-rates:",
                  panel_winrates(models, n_games=100, use_belief=cfg.use_belief,
                                 max_decisions=cfg.max_decisions), flush=True)
            print("estimators:", estimator_metrics(models, collect_eval_batch(models, n_games=8)), flush=True)
    finally:
        if monitor is not None and monitor.poll() is None:
            monitor.terminate()                        # best-effort; monitor is read-only


if __name__ == "__main__":
    main()
