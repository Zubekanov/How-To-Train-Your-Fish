"""PPO update (actor + privileged critic) and supervised aux-head updates."""
from __future__ import annotations

import numpy as np
import torch

from fishrl.models import device_of
from fishrl.train.losses import guesser_poisson, outcome_bce, ppo_actor_loss


def ppo_update(batch, actor, critic, opt, cfg, ent_coef, rng_seed: int = 0) -> dict:
    """K-epoch minibatched PPO over the actor and the privileged critic.

    The rollout batch lives on CPU (collection is CPU-bound); each minibatch is
    moved to the model's device for the update. `rng_seed` varies the minibatch
    shuffle per call (the trainer passes the iteration counter) — a fixed stream
    would replay the identical permutation every update."""
    dev = device_of(actor)
    M = batch["x_act"].shape[0]
    idx = np.arange(M)
    rng = np.random.default_rng(rng_seed)
    stats = {"policy_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0,
             "clip_frac": 0.0, "n": 0}
    for _ in range(cfg.ppo_epochs):
        rng.shuffle(idx)
        for s in range(0, M, cfg.minibatch):
            mb = idx[s:s + cfg.minibatch]
            logp_all = actor.log_probs(batch["x_act"][mb].to(dev), batch["mask"][mb].to(dev))
            ploss, ent, kl, cf = ppo_actor_loss(
                logp_all, batch["action"][mb].to(dev),
                batch["old_logp"][mb].to(dev), batch["adv"][mb].to(dev), cfg.clip)
            critic_logit = critic(batch["god"][mb].to(dev))
            closs = outcome_bce(critic_logit, batch["y_p1"][mb].to(dev), batch["valid"][mb].to(dev))
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
            stats["clip_frac"] += float(cf)
            stats["n"] += 1
    for k in ("policy_loss", "critic_loss", "entropy", "approx_kl", "clip_frac"):
        stats[k] /= max(stats["n"], 1)
    return stats


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
