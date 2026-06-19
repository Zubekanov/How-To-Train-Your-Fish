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

from fishrl.eval.metrics import collect_eval_batch, estimator_metrics, panel_winrates
from fishrl.train import checkpoint as ckpt
from fishrl.train.config import Config, resolve_device
from fishrl.train.train_loop import build_models, train

NETS = ("actor", "critic", "guesser", "public")


def main():
    ap = argparse.ArgumentParser(description="Self-play training for fishrl.")
    ap.add_argument("--gpu", action="store_true", help="train on CUDA if available")
    ap.add_argument("--iters", type=int, default=100, help="iteration cap; <=0 runs unbounded")
    ap.add_argument("--games-per-iter", type=int, default=8)
    ap.add_argument("--warmup-games", type=int, default=64)
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
    ap.add_argument("--max-hours", type=float, default=None,
                    help="optional wall-clock cap (across restarts); omit for indefinite")
    args = ap.parse_args()

    latest = ckpt.latest_path(args.ckpt_dir)
    have_ckpt = os.path.exists(latest)
    if have_ckpt and not args.resume and not args.fresh:
        ap.error(f"{latest} exists; pass --resume to continue or --fresh to overwrite it")
    resume = args.resume and have_ckpt
    if args.resume and not have_ckpt:
        print(f"[resume] no checkpoint at {latest}; starting fresh", flush=True)

    common = dict(device=resolve_device(args.gpu), iters=args.iters,
                  games_per_iter=args.games_per_iter, warmup_games=args.warmup_games,
                  ckpt_dir=args.ckpt_dir, report_every_seconds=args.report_every_seconds,
                  report_winrate_games=args.report_winrate_games,
                  checkpoint_every_seconds=args.checkpoint_every_seconds)
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
          f"resume: {resume} | ckpt: {latest}", flush=True)

    max_seconds = args.max_hours * 3600.0 if args.max_hours else None
    models = train(cfg, build_models(cfg), log=lambda s: print(s, flush=True),
                   max_seconds=max_seconds,
                   resume_path=latest if resume else None, checkpoint_path=latest)
    print(f"saved checkpoint -> {latest}", flush=True)

    # higher-game-count final readout for bounded runs (the in-loop status lines use fewer
    # games each). Skipped for the unbounded service, which exits via signal and shouldn't
    # spend tens of seconds on eval during shutdown.
    if cfg.iters > 0:
        print("final win-rates:", panel_winrates(models, n_games=100), flush=True)
        print("estimators:", estimator_metrics(models, collect_eval_batch(models, n_games=8)), flush=True)


if __name__ == "__main__":
    main()
