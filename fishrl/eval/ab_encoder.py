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


def _batch(seed, n_games=6):
    benv = BeliefAugmentedEnv(HandGuesser(), max_decisions=2000)
    buf = collect_games(benv, random_act_fn(np.random.default_rng(seed)), n_games, seed,
                        critic=None, max_decisions=2000)
    b = buf.compute(0.997, 0.95)
    keep = b["valid"] > 0
    return b["god"][keep], b["y_p1"][keep]


def _fit_eval(encoder, train, holdout, steps=400):
    gx, gy = train
    hx, hy = holdout
    c = PrivilegedCritic(encoder=encoder)
    opt = torch.optim.Adam(c.parameters(), lr=2e-3)
    ones = torch.ones_like(gy)
    for _ in range(steps):
        loss = outcome_bce(c(gx), gy, ones)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        p = torch.sigmoid(c(hx))
        brier = float(((p - hy) ** 2).mean())
        acc = float(((p > 0.5).float() == hy).float().mean())
    params = sum(pp.numel() for pp in c.parameters())
    return {"encoder": encoder, "holdout_brier": round(brier, 4),
            "holdout_acc": round(acc, 4), "params": params}


def main():
    train = _batch(seed=0)
    holdout = _batch(seed=999)
    print(f"train n={train[0].shape[0]}  holdout n={holdout[0].shape[0]}")
    for enc in ("flat", "entity"):
        print(_fit_eval(enc, train, holdout))


if __name__ == "__main__":
    main()
