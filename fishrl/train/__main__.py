"""Self-play training entry point.

    python -m fishrl.train --gpu --iters 200 --encoder entity

Runs warmup + PPO self-play, saves a checkpoint, and prints a quick evaluation.
"""
from __future__ import annotations

import argparse
import os

import torch

from fishrl.eval.metrics import collect_eval_batch, estimator_metrics, winrate_vs_random
from fishrl.train.config import Config, resolve_device
from fishrl.train.train_loop import build_models, train


def main():
    ap = argparse.ArgumentParser(description="Self-play training for fishrl.")
    ap.add_argument("--gpu", action="store_true", help="train on CUDA if available")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--games-per-iter", type=int, default=8)
    ap.add_argument("--warmup-games", type=int, default=64)
    ap.add_argument("--encoder", choices=["flat", "entity", "attention"], default="flat")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-dir", default="checkpoints")
    args = ap.parse_args()

    cfg = Config(device=resolve_device(args.gpu), iters=args.iters,
                 games_per_iter=args.games_per_iter, warmup_games=args.warmup_games,
                 encoder=args.encoder, seed=args.seed, ckpt_dir=args.ckpt_dir)
    print(f"device: {cfg.device} | encoder: {cfg.encoder} | iters: {cfg.iters}", flush=True)

    models = train(cfg, build_models(cfg), log=lambda s: print(s, flush=True))

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    path = os.path.join(cfg.ckpt_dir, "fishrl.pt")
    torch.save({"encoder": cfg.encoder,
                "actor": models.actor.state_dict(),
                "critic": models.critic.state_dict(),
                "guesser": models.guesser.state_dict(),
                "public": models.public.state_dict()}, path)
    print(f"saved checkpoint -> {path}", flush=True)
    print("win-rate vs random:", winrate_vs_random(models, n_games=20), flush=True)
    print("estimators:", estimator_metrics(models, collect_eval_batch(models, n_games=8)), flush=True)


if __name__ == "__main__":
    main()
