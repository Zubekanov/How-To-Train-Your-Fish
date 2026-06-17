"""PPO update (actor + privileged critic) and supervised aux-head updates."""
from __future__ import annotations

import numpy as np
import torch

from fishrl.train.losses import guesser_poisson, outcome_bce, ppo_actor_loss


def ppo_update(batch, actor, critic, opt, cfg, ent_coef) -> dict:
    """K-epoch minibatched PPO over the actor and the privileged critic."""
    M = batch["x_act"].shape[0]
    idx = np.arange(M)
    rng = np.random.default_rng(0)
    stats = {"policy_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "n": 0}
    for _ in range(cfg.ppo_epochs):
        rng.shuffle(idx)
        for s in range(0, M, cfg.minibatch):
            mb = idx[s:s + cfg.minibatch]
            logp_all = actor.log_probs(batch["x_act"][mb], batch["mask"][mb])
            ploss, ent, kl = ppo_actor_loss(
                logp_all, batch["mask"][mb], batch["action"][mb],
                batch["old_logp"][mb], batch["adv"][mb], cfg.clip)
            critic_logit = critic(batch["god"][mb])
            closs = outcome_bce(critic_logit, batch["y_p1"][mb], batch["valid"][mb])
            loss = ploss - ent_coef * ent + cfg.critic_coef * closs
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(critic.parameters()), cfg.grad_clip)
            opt.step()
            stats["policy_loss"] += ploss.detach().item()
            stats["critic_loss"] += closs.detach().item()
            stats["entropy"] += ent.detach().item()
            stats["approx_kl"] += float(kl)
            stats["n"] += 1
    for k in ("policy_loss", "critic_loss", "entropy", "approx_kl"):
        stats[k] /= max(stats["n"], 1)
    return stats


def aux_update(batch, guesser, public_est, opt_g, opt_p, steps: int) -> dict:
    """A few SGD steps on the guesser (Poisson) and public estimator (BCE)."""
    g_loss = p_loss = 0.0
    for _ in range(max(steps, 1)):
        pred = guesser(batch["persp"], batch["prev_guess"])
        gl = guesser_poisson(pred, batch["cnt"])
        opt_g.zero_grad(); gl.backward(); opt_g.step()
        pl = outcome_bce(public_est(batch["pub"]), batch["y_p1"], batch["valid"])
        opt_p.zero_grad(); pl.backward(); opt_p.step()
        g_loss, p_loss = gl.detach().item(), pl.detach().item()
    return {"guesser_loss": g_loss, "public_loss": p_loss}
