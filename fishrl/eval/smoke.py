"""Runnable end-to-end demo: train briefly, then report metrics.

    python -m fishrl.eval.smoke

Trains a small configuration on CPU and prints estimator calibration, guesser MAE,
and win-rates vs the random and heuristic baselines. Not a unit test (slow).
"""
from __future__ import annotations

import argparse

from fishrl.eval.metrics import (
    collect_eval_batch, estimator_metrics, guesser_mae,
    winrate_vs_heuristic, winrate_vs_random,
)
from fishrl.train.config import Config, resolve_device
from fishrl.train.train_loop import build_models, train


def main():
    ap = argparse.ArgumentParser(description="Short end-to-end training + metrics demo.")
    ap.add_argument("--gpu", action="store_true", help="train/evaluate on CUDA if available")
    args = ap.parse_args()
    device = resolve_device(args.gpu)
    print(f"device: {device}", flush=True)
    cfg = Config(warmup_games=24, warmup_epochs=3, iters=8, games_per_iter=6,
                 minibatch=256, max_decisions=2000, device=device)
    models = train(cfg, build_models(cfg))
    batch = collect_eval_batch(models, n_games=8)
    print("estimators:", estimator_metrics(models, batch))
    print("guesser MAE:", round(guesser_mae(models, batch), 4))
    print("win-rate vs random:   ", winrate_vs_random(models, n_games=20))
    print("win-rate vs heuristic:", winrate_vs_heuristic(models, n_games=20))


if __name__ == "__main__":
    main()
