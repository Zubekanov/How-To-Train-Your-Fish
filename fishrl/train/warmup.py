"""Phase-0 warmup: pretrain the beliefs and estimators on random self-play.

Before the policy depends on the hand-guess and the critic baseline, fit the
guesser (Poisson), critic (BCE) and public estimator (BCE) on data collected
with a random masked policy, so PPO starts with a meaningful belief input and a
non-garbage baseline. v3 (belief_mode="bookkeeper", critic_view="public"): the
guesser/public legs simply don't exist — the warmup collapses to critic-only,
trained on the public features.
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
    mode = getattr(cfg, "belief_mode", None) or ("guesser" if guesser is not None else "none")
    view = getattr(cfg, "critic_view", "god")
    benv = BeliefAugmentedEnv(guesser, mode=mode, max_decisions=cfg.max_decisions)
    buf = collect_games(benv, random_act_fn(rng), cfg.warmup_games, seed,
                        critic=None, max_decisions=cfg.max_decisions,
                        critic_view=view)
    batch = buf.compute(cfg.gamma, cfg.lam)
    M = batch["x_act"].shape[0]
    dev = device_of(critic)
    feat_key = "pub" if view == "public" else "god"

    opt_g = torch.optim.Adam(guesser.parameters(), lr=cfg.lr_guesser) if guesser is not None else None
    opt_c = torch.optim.Adam(critic.parameters(), lr=cfg.lr_ppo)
    opt_p = (torch.optim.Adam(public_est.parameters(), lr=cfg.lr_public)
             if public_est is not None else None)

    idx = np.arange(M)
    last = {}
    for _ in range(cfg.warmup_epochs):
        rng.shuffle(idx)
        for s in range(0, M, cfg.minibatch):
            mb = idx[s:s + cfg.minibatch]
            feat = batch[feat_key][mb].to(dev)
            y, valid = batch["y_p1"][mb].to(dev), batch["valid"][mb].to(dev)
            if opt_g is not None:
                persp, prev = batch["persp"][mb].to(dev), batch["prev_guess"][mb].to(dev)
                cnt = batch["cnt"][mb].to(dev)
                gl = guesser_poisson(guesser(persp, prev), cnt)
                opt_g.zero_grad(); gl.backward(); opt_g.step()
                last["guesser"] = gl.detach().item()
            cl = outcome_bce(critic(feat), y, valid)
            opt_c.zero_grad(); cl.backward(); opt_c.step()
            last["critic"] = cl.detach().item()
            if opt_p is not None:
                pub = batch["pub"][mb].to(dev)
                pl = outcome_bce(public_est(pub), y, valid)
                opt_p.zero_grad(); pl.backward(); opt_p.step()
                last["public"] = pl.detach().item()
    return {"transitions": M, **last}
