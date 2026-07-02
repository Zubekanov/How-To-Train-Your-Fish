"""Loss functions for the policy, critic, and auxiliary heads."""
from __future__ import annotations

import torch
import torch.nn.functional as Fn

from fishrl.models.policy import MaskedActor


def ppo_actor_loss(logp_all, action, old_logp, adv, clip):
    """Masked PPO clip objective + mean entropy (over legal actions) + the clip
    fraction (share of ratios outside 1±clip — the standard "how hard is the
    trust region working" health signal)."""
    logp = logp_all.gather(-1, action.unsqueeze(-1)).squeeze(-1)
    ratio = torch.exp(logp - old_logp)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
    policy_loss = -torch.min(unclipped, clipped).mean()
    entropy = MaskedActor.entropy(logp_all).mean()
    approx_kl = (old_logp - logp).mean().detach()
    clip_frac = ((ratio - 1.0).abs() > clip).float().mean().detach()
    return policy_loss, entropy, approx_kl, clip_frac


def outcome_bce(logit, y_p1, valid):
    """BCE of P(p1 win) vs the terminal winner, masked to decided games."""
    per = Fn.binary_cross_entropy_with_logits(logit, y_p1, reduction="none")
    w = valid.sum().clamp(min=1.0)
    return (per * valid).sum() / w


def guesser_poisson(pred_counts, target_counts):
    """Poisson NLL for non-negative expected counts (log_input=False)."""
    return Fn.poisson_nll_loss(pred_counts, target_counts, log_input=False, reduction="mean")
