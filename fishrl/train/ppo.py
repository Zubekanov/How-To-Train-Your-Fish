"""PPO update (actor + privileged critic) and supervised aux-head updates."""
from __future__ import annotations

import numpy as np
import torch

from fishrl.models import device_of
from fishrl.train.losses import guesser_poisson, outcome_bce, ppo_actor_loss


def ppo_update(batch, actor, critic, opt, cfg, ent_coef, rng_seed: int = 0) -> dict:
    """K-epoch minibatched PPO over the actor and the privileged critic.

    The rollout batch lives on CPU (collection is CPU-bound); the fields the
    update touches are moved to the model's device ONCE up front, and the loss
    telemetry is accumulated as device tensors with a single host sync at the
    end. The old per-minibatch shape — 8 small `.to(dev)` copies plus 5
    `.item()` pipeline stalls per step, ~28 steps per update — serialized the
    GPU against launch latency. Same math, same minibatch order.

    `rng_seed` varies the minibatch shuffle per call (the trainer passes the
    iteration counter) — a fixed stream would replay the identical permutation
    every update."""
    dev = device_of(actor)
    keys = ("x_act", "mask", "action", "old_logp", "adv", "god", "y_p1", "valid")
    b = {k: batch[k].to(dev, non_blocking=True) for k in keys}
    M = b["x_act"].shape[0]
    idx = np.arange(M)
    rng = np.random.default_rng(rng_seed)
    acc = None                         # device-side [ploss, closs, ent, kl, cf] running sum
    n = 0
    for _ in range(cfg.ppo_epochs):
        rng.shuffle(idx)
        for s in range(0, M, cfg.minibatch):
            mb = torch.from_numpy(idx[s:s + cfg.minibatch]).to(dev)
            logp_all = actor.log_probs(b["x_act"][mb], b["mask"][mb])
            ploss, ent, kl, cf = ppo_actor_loss(
                logp_all, b["action"][mb], b["old_logp"][mb], b["adv"][mb], cfg.clip)
            critic_logit = critic(b["god"][mb])
            closs = outcome_bce(critic_logit, b["y_p1"][mb], b["valid"][mb])
            loss = ploss - ent_coef * ent + cfg.critic_coef * closs
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(critic.parameters()), cfg.grad_clip)
            opt.step()
            # kl/cf come back as detached device tensors (losses.py) -- stack
            # everything device-side; the ONLY host sync is the .tolist() below.
            step_stats = torch.stack([ploss.detach(), closs.detach(), ent.detach(), kl, cf])
            acc = step_stats if acc is None else acc + step_stats
            n += 1
    vals = (acc / max(n, 1)).tolist() if acc is not None else [0.0] * 5
    return {"policy_loss": vals[0], "critic_loss": vals[1], "entropy": vals[2],
            "approx_kl": vals[3], "clip_frac": vals[4], "n": n}


def aux_update(batch, guesser, public_est, opt_g, opt_p, steps: int) -> dict:
    """A few SGD steps on the guesser (Poisson) and public estimator (BCE).

    These are the only places the guesser and public estimator are trained — both
    purely supervised on the rollout's targets, never through the policy gradient.
    The guesser is thus an honest posterior; the public estimator is diagnostic
    (it is not the critic — only the privileged critic computes advantages)."""
    dev = device_of(guesser)
    persp, prev, cnt = batch["persp"].to(dev), batch["prev_guess"].to(dev), batch["cnt"].to(dev)
    pub, y_p1, valid = batch["pub"].to(dev), batch["y_p1"].to(dev), batch["valid"].to(dev)
    g_loss = p_loss = 0.0
    for _ in range(max(steps, 1)):
        gl = guesser_poisson(guesser(persp, prev), cnt)
        opt_g.zero_grad(); gl.backward(); opt_g.step()
        pl = outcome_bce(public_est(pub), y_p1, valid)
        opt_p.zero_grad(); pl.backward(); opt_p.step()
        g_loss, p_loss = gl.detach().item(), pl.detach().item()
    return {"guesser_loss": g_loss, "public_loss": p_loss}
