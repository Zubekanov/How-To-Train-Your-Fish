"""PPO update (actor + privileged critic) and supervised aux-head updates."""
from __future__ import annotations

import numpy as np
import torch

from fishrl.models import device_of
from fishrl.train.losses import guesser_poisson, outcome_bce, ppo_actor_loss


def ppo_update(batch, actor, critic, opt, cfg, ent_coef, rng_seed: int = 0,
               ref_actor=None, kl_ref_coef: float = 0.0, train_actor: bool = True) -> dict:
    """K-epoch minibatched PPO over the actor and the privileged critic.

    The rollout batch lives on CPU (collection is CPU-bound); the fields the
    update touches are moved to the model's device ONCE up front, and the loss
    telemetry is accumulated as device tensors with a single host sync at the
    end. The old per-minibatch shape — 8 small `.to(dev)` copies plus 5
    `.item()` pipeline stalls per step, ~28 steps per update — serialized the
    GPU against launch latency. Same math, same minibatch order.

    `rng_seed` varies the minibatch shuffle per call (the trainer passes the
    iteration counter) — a fixed stream would replay the identical permutation
    every update.

    BC-handoff extensions (defaults = historic behaviour exactly):
      * ``train_actor=False`` — critic-only gradient (the actor's telemetry is
        still computed, but no actor term enters the loss), used while the
        critic warms up against a freshly-cloned policy.
      * ``ref_actor`` + ``kl_ref_coef`` — adds ``kl_ref_coef * KL(ref || pi)``
        to the loss, anchoring early fine-tuning to the frozen teacher clone;
        the trainer anneals the coefficient to zero.

    v3 (Config.critic_view="public"): the critic reads the PUBLIC features and, when
    ``cfg.critic_deckout_aux > 0``, its deckout-winner aux head adds
    ``w * BCE(aux_logit, y_p1)`` over the deckout-ended rows (the parity-credit
    lever). At w=0 the head exists but contributes nothing.

    ``cfg.critic_epochs`` (2026-08-21): when > 0 the critic (and aux head) receive
    gradient only on the first ``critic_epochs`` of the ``ppo_epochs`` passes; the actor
    keeps all of them. critic_loss / deckout_aux_loss are averaged over the critic's
    passes only."""
    dev = device_of(actor)
    critic_view = getattr(cfg, "critic_view", "god")
    public_family = critic_view in ("public", "hands")
    feat_key = "pub" if public_family else "god"
    aux_w = float(getattr(cfg, "critic_deckout_aux", 0.0)) if public_family else 0.0
    keys = ("x_act", "mask", "action", "old_logp", "adv", feat_key, "y_p1", "valid")
    if aux_w > 0.0:
        keys = keys + ("deckout_valid",)
    consist_w = float(getattr(cfg, "critic_consistency", 0.0) or 0.0)
    nxt_all = None
    if consist_w > 0.0 and public_family and "det_next" in batch:
        from fishrl.data.features import turn_index_for
        ti = turn_index_for(critic_view)
        nxt_all = batch["det_next"].clone()
        ok = nxt_all >= 0
        # same-turn filter: a turn change hides a draw (stochastic) -- not a consistency pair
        same = torch.zeros_like(ok)
        same[ok] = batch[feat_key][ok, ti] == batch[feat_key][nxt_all[ok], ti]
        nxt_all[~same] = -1
        nxt_all = nxt_all.to(dev, non_blocking=True)
    else:
        consist_w = 0.0
    td_mix = float(getattr(cfg, "critic_td_mix", 0.0) or 0.0)
    if td_mix > 0.0 and "ret_p1" in batch:
        keys = keys + ("ret_p1",)
    else:
        td_mix = 0.0
    b = {k: batch[k].to(dev, non_blocking=True) for k in keys}
    M = b["x_act"].shape[0]
    idx = np.arange(M)
    rng = np.random.default_rng(rng_seed)
    acc = None                 # device-side [ploss, closs, ent, kl, cf, kl_ref] running sum
    n = 0
    zero = torch.zeros((), device=dev)
    critic_epochs = int(getattr(cfg, "critic_epochs", 0) or 0)
    acc_c = None               # device-side [closs, aux] over the critic's epochs only
    n_c = 0
    # gradient-noise-scale probe (epoch 0, actor grads, pre-clip): sum of the per-minibatch
    # gradients and of their squared norms. B_noise = tr(Sigma)/|G|^2 is the critical batch
    # -- a batch far above it means games/iter has slack; near it, smaller batches cost.
    actor_params = [p for p in actor.parameters() if p.requires_grad]
    g_sum = None
    g_sq = 0.0
    g_n = 0
    for epoch in range(cfg.ppo_epochs):
        train_critic = critic_epochs <= 0 or epoch < critic_epochs
        if not train_actor and not train_critic:
            break                                  # nothing left to fit this update
        rng.shuffle(idx)
        for s in range(0, M, cfg.minibatch):
            mb = torch.from_numpy(idx[s:s + cfg.minibatch]).to(dev)
            logp_all = actor.log_probs(b["x_act"][mb], b["mask"][mb])
            ploss, ent, kl, cf = ppo_actor_loss(
                logp_all, b["action"][mb], b["old_logp"][mb], b["adv"][mb], cfg.clip)
            aux_loss = None
            if train_critic:
                if aux_w > 0.0:
                    critic_logit, aux_logit = critic.forward_with_aux(b[feat_key][mb])
                    aux_loss = outcome_bce(aux_logit, b["y_p1"][mb], b["deckout_valid"][mb])
                else:
                    critic_logit = critic(b[feat_key][mb])
                y_c = b["y_p1"][mb]
                if td_mix > 0.0:
                    soft = (b["ret_p1"][mb].clamp(-1.0, 1.0) + 1.0) * 0.5
                    y_c = (1.0 - td_mix) * y_c + td_mix * soft
                closs = outcome_bce(critic_logit, y_c, b["valid"][mb])
                consist = zero
                if consist_w > 0.0:
                    nxt = nxt_all[mb]
                    has = nxt >= 0
                    if int(has.sum()) > 0:
                        p_here = torch.sigmoid(critic_logit[has])
                        p_next = torch.sigmoid(critic(b[feat_key][nxt[has]]))
                        consist = ((p_here - p_next) ** 2).mean()
                        closs = closs + consist_w * consist
            else:
                closs = zero
                consist = zero
            if train_actor:
                loss = ploss - ent_coef * ent + cfg.critic_coef * closs
            else:                                  # handoff warmup: critic-only gradient
                loss = cfg.critic_coef * closs
            if aux_loss is not None and torch.isfinite(aux_loss):
                loss = loss + aux_w * aux_loss
            kl_ref = zero
            if train_actor and ref_actor is not None and kl_ref_coef > 0.0:
                with torch.no_grad():
                    ref_logp = ref_actor.log_probs(b["x_act"][mb], b["mask"][mb])
                ref_p = ref_logp.exp()             # illegal entries underflow to exactly 0
                diff = torch.where(ref_p > 0, ref_logp - logp_all,
                                   torch.zeros_like(logp_all))
                kl_ref = (ref_p * diff).sum(-1).mean()
                loss = loss + kl_ref_coef * kl_ref
            opt.zero_grad()
            loss.backward()
            if train_actor and epoch == 0:
                with torch.no_grad():
                    flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                                      for p in actor_params])
                    g_sum = flat.clone() if g_sum is None else g_sum.add_(flat)
                    g_sq = g_sq + float(flat.dot(flat))
                    g_n += 1
            torch.nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(critic.parameters()), cfg.grad_clip)
            opt.step()
            # kl/cf come back as detached device tensors (losses.py) -- stack
            # everything device-side; the ONLY host sync is the .tolist() below.
            step_stats = torch.stack([ploss.detach(), ent.detach(), kl, cf, kl_ref.detach()])
            acc = step_stats if acc is None else acc + step_stats
            n += 1
            if train_critic:
                aux_stat = (aux_loss.detach() if aux_loss is not None
                            and torch.isfinite(aux_loss) else zero)
                cs = torch.stack([closs.detach(), aux_stat, consist.detach()])
                acc_c = cs if acc_c is None else acc_c + cs
                n_c += 1
    vals = (acc / max(n, 1)).tolist() if acc is not None else [0.0] * 5
    cvals = (acc_c / max(n_c, 1)).tolist() if acc_c is not None else [0.0, 0.0, 0.0]
    gns = {"gns_tr_sigma": float("nan"), "gns_g2": float("nan"), "gns_b": float("nan")}
    if g_n >= 2 and M > cfg.minibatch:
        b_mb = float(min(cfg.minibatch, M))
        e_gb2 = g_sq / g_n                                  # E|g_b|^2
        gB2 = float(g_sum.dot(g_sum)) / (g_n * g_n)         # |mean of the minibatch grads|^2
        tr_sigma = (e_gb2 - gB2) / (1.0 / b_mb - 1.0 / M)
        g2 = gB2 - tr_sigma / M
        gns = {"gns_tr_sigma": tr_sigma, "gns_g2": g2,
               "gns_b": (tr_sigma / g2) if g2 > 0 else float("inf")}
    return {"policy_loss": vals[0], "critic_loss": cvals[0], "entropy": vals[1],
            "approx_kl": vals[2], "clip_frac": vals[3], "kl_teacher": vals[4],
            "deckout_aux_loss": cvals[1], "critic_consist_loss": cvals[2],
            "n": n, "n_critic": n_c, **gns}


def aux_update(batch, guesser, public_est, opt_g, opt_p, steps: int,
               train_public: bool = True) -> dict:
    """A few SGD steps on the guesser (Poisson) and public estimator (BCE).

    These are the only places the guesser and public estimator are trained — both
    purely supervised on the rollout's targets, never through the policy gradient.
    The guesser is thus an honest posterior; the public estimator is diagnostic
    (it is not the critic — only the privileged critic computes advantages). When
    `train_public` is False the public head is skipped entirely (its buffered features
    are zeros; see features.set_public_encoding) and public_loss is NaN."""
    train_g = guesser is not None
    train_p = train_public and public_est is not None
    dev = device_of(guesser if train_g else public_est)
    if train_g:
        persp, prev, cnt = (batch["persp"].to(dev), batch["prev_guess"].to(dev),
                            batch["cnt"].to(dev))
    g_loss, p_loss = 0.0, float("nan")
    if train_p:
        pub, y_p1, valid = batch["pub"].to(dev), batch["y_p1"].to(dev), batch["valid"].to(dev)
    for _ in range(max(steps, 1)):
        if train_g:
            gl = guesser_poisson(guesser(persp, prev), cnt)
            opt_g.zero_grad(); gl.backward(); opt_g.step()
            g_loss = gl.detach().item()
        if train_p:
            pl = outcome_bce(public_est(pub), y_p1, valid)
            opt_p.zero_grad(); pl.backward(); opt_p.step()
            p_loss = pl.detach().item()
    return {"guesser_loss": g_loss, "public_loss": p_loss}
