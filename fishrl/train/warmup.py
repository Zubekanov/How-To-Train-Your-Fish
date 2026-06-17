"""Phase-0 warmup: pretrain the beliefs and estimators on random self-play.

Before the policy depends on the hand-guess and the asymmetric critic, fit the
guesser (Poisson), privileged critic (BCE) and public estimator (BCE) on data
collected with a random masked policy, so PPO starts with a meaningful belief
input and a non-garbage baseline.
"""
from __future__ import annotations

import numpy as np
import torch

from fishrl.models import device_of
from fishrl.train.belief_env import BeliefAugmentedEnv
from fishrl.train.collector import collect_games, random_act_fn
from fishrl.train.losses import guesser_poisson, outcome_bce


def warmup(guesser, critic, public_est, cfg, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    benv = BeliefAugmentedEnv(guesser, max_decisions=cfg.max_decisions)
    buf = collect_games(benv, random_act_fn(rng), cfg.warmup_games, seed,
                        critic=None, max_decisions=cfg.max_decisions)
    batch = buf.compute(cfg.gamma, cfg.lam)
    M = batch["x_act"].shape[0]
    dev = device_of(critic)

    opt_g = torch.optim.Adam(guesser.parameters(), lr=cfg.lr_guesser)
    opt_c = torch.optim.Adam(critic.parameters(), lr=cfg.lr_ppo)
    opt_p = torch.optim.Adam(public_est.parameters(), lr=cfg.lr_public)

    idx = np.arange(M)
    last = {}
    for _ in range(cfg.warmup_epochs):
        rng.shuffle(idx)
        for s in range(0, M, cfg.minibatch):
            mb = idx[s:s + cfg.minibatch]
            persp, prev = batch["persp"][mb].to(dev), batch["prev_guess"][mb].to(dev)
            god, pub = batch["god"][mb].to(dev), batch["pub"][mb].to(dev)
            cnt, y, valid = batch["cnt"][mb].to(dev), batch["y_p1"][mb].to(dev), batch["valid"][mb].to(dev)
            gl = guesser_poisson(guesser(persp, prev), cnt)
            opt_g.zero_grad(); gl.backward(); opt_g.step()
            cl = outcome_bce(critic(god), y, valid)
            opt_c.zero_grad(); cl.backward(); opt_c.step()
            pl = outcome_bce(public_est(pub), y, valid)
            opt_p.zero_grad(); pl.backward(); opt_p.step()
            last = {"guesser": gl.detach().item(), "critic": cl.detach().item(),
                    "public": pl.detach().item()}
    return {"transitions": M, **last}
