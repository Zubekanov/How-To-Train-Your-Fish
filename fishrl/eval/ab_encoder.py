"""Flat vs entity encoder A/B on the privileged critic.

    python -m fishrl.eval.ab_encoder

Collects a fixed train batch and a held-out batch from random self-play, fits each
encoder's privileged critic to the terminal outcome (BCE), and reports held-out
Brier / accuracy and parameter counts. We compare flat-vs-entity head-to-head on
identical data rather than against an expected-zero loss: with a stochastic policy
and a hidden shared library the god-state value has an irreducible aleatoric floor,
so an absolute value-loss level is NOT a clean signal of encoder capacity.
"""
from __future__ import annotations

import numpy as np
import torch

from fishrl.models.estimators import PrivilegedCritic
from fishrl.models.guesser import HandGuesser
from fishrl.train.belief_env import BeliefAugmentedEnv
from fishrl.train.collector import collect_games, random_act_fn
from fishrl.train.losses import outcome_bce


def _log(msg: str) -> None:
    print(msg, flush=True)          # flush so progress shows live even when piped


def _batch(label, seed, n_games=6):
    _log(f"[collect] {label}: {n_games} random self-play games (seed {seed})...")
    benv = BeliefAugmentedEnv(HandGuesser(), max_decisions=2000)
    buf = collect_games(benv, random_act_fn(np.random.default_rng(seed)), n_games, seed,
                        critic=None, max_decisions=2000)
    b = buf.compute(0.997, 0.95)
    keep = b["valid"] > 0
    god, y = b["god"][keep], b["y_p1"][keep]
    _log(f"[collect] {label}: {god.shape[0]} decided transitions "
         f"(p1 win rate {float(y.mean()):.2f})")
    return god, y


def _fit_eval(encoder, train, holdout, steps=400, log_every=50):
    gx, gy = train
    hx, hy = holdout
    c = PrivilegedCritic(encoder=encoder)
    params = sum(pp.numel() for pp in c.parameters())
    _log(f"[fit] {encoder}: training privileged critic "
         f"({params:,} params) for {steps} steps...")
    opt = torch.optim.Adam(c.parameters(), lr=2e-3)
    ones = torch.ones_like(gy)
    for step in range(1, steps + 1):
        loss = outcome_bce(c(gx), gy, ones)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % log_every == 0 or step == steps:
            _log(f"[fit] {encoder}: step {step}/{steps}  train_bce {float(loss):.4f}")
    with torch.no_grad():
        p = torch.sigmoid(c(hx))
        brier = float(((p - hy) ** 2).mean())
        acc = float(((p > 0.5).float() == hy).float().mean())
    res = {"encoder": encoder, "holdout_brier": round(brier, 4),
           "holdout_acc": round(acc, 4), "params": params}
    _log(f"[result] {res}")
    return res


def main():
    _log("=== flat vs entity encoder A/B (privileged critic) ===")
    train = _batch("train", seed=0)
    holdout = _batch("holdout", seed=999)
    results = [_fit_eval(enc, train, holdout) for enc in ("flat", "entity")]
    _log("=== summary ===")
    for r in results:
        _log(str(r))
    winner = min(results, key=lambda r: r["holdout_brier"])["encoder"]
    _log(f"lower held-out Brier (better-calibrated critic): {winner}")


if __name__ == "__main__":
    main()
