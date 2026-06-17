"""Orchestration: warmup, then iterate {collect self-play, PPO, aux updates}.

The guesser is frozen during each iteration's collection + PPO update (stable actor
input distribution) and slow-refreshed by `aux_update` between iterations.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from fishrl.models import device_of
from fishrl.models.estimators import PrivilegedCritic, PublicEstimator
from fishrl.models.guesser import HandGuesser
from fishrl.models.policy import MaskedActor
from fishrl.train.belief_env import BeliefAugmentedEnv
from fishrl.train.collector import actor_act_fn, collect_games
from fishrl.train.config import Config
from fishrl.train.ppo import aux_update, ppo_update


@dataclass
class Models:
    actor: MaskedActor
    critic: PrivilegedCritic
    guesser: HandGuesser
    public: PublicEstimator


def build_models(cfg: Config) -> Models:
    torch.manual_seed(cfg.seed)
    dev = cfg.device
    return Models(
        MaskedActor(cfg.hidden, cfg.enc_for("actor")).to(dev),
        PrivilegedCritic(cfg.critic_hidden, cfg.enc_for("critic")).to(dev),
        HandGuesser(cfg.hidden, cfg.enc_for("guesser")).to(dev),
        PublicEstimator(cfg.hidden, cfg.enc_for("public")).to(dev),
    )


def train(cfg: Config, models: Models | None = None, log=print,
          max_seconds: float | None = None) -> Models:
    """Run warmup + PPO self-play. If `max_seconds` is set, stop once that wall-clock
    budget is spent (cfg.iters is then just an upper cap) -- this is how fixed-wall-
    clock head-to-heads are run. The entropy schedule anneals over the budget (by
    elapsed fraction) rather than over iters so it doesn't depend on the iter count."""
    import time
    from fishrl.train.warmup import warmup
    m = models or build_models(cfg)
    w = warmup(m.guesser, m.critic, m.public, cfg, seed=cfg.seed)
    log(f"[warmup] {w}")

    opt_ppo = torch.optim.Adam(list(m.actor.parameters()) + list(m.critic.parameters()), lr=cfg.lr_ppo)
    opt_g = torch.optim.Adam(m.guesser.parameters(), lr=cfg.lr_guesser)
    opt_p = torch.optim.Adam(m.public.parameters(), lr=cfg.lr_public)
    benv = BeliefAugmentedEnv(m.guesser, belief=cfg.use_belief, max_decisions=cfg.max_decisions)

    start = time.perf_counter()
    for it in range(cfg.iters):
        elapsed = time.perf_counter() - start
        if max_seconds is not None and elapsed > max_seconds:
            log(f"[stop] wall-clock budget {max_seconds:.0f}s reached at iter {it}")
            break
        seed = cfg.seed + 1000 + it * cfg.games_per_iter
        buf = collect_games(benv, actor_act_fn(m.actor), cfg.games_per_iter, seed,
                            critic=m.critic, max_decisions=cfg.max_decisions)
        batch = buf.compute(cfg.gamma, cfg.lam)
        if max_seconds is not None:                      # anneal entropy over the budget
            frac = min(elapsed / max_seconds, 1.0)
            ent = cfg.ent_start + frac * (cfg.ent_end - cfg.ent_start)
        else:
            ent = cfg.ent_coef(it)
        ppo_stats = ppo_update(batch, m.actor, m.critic, opt_ppo, cfg, ent)
        aux_stats = aux_update(batch, m.guesser, m.public, opt_g, opt_p, cfg.aux_steps)
        log(f"[iter {it}] T={len(buf)} "
            f"pi={ppo_stats['policy_loss']:.3f} V={ppo_stats['critic_loss']:.3f} "
            f"H={ppo_stats['entropy']:.3f} kl={ppo_stats['approx_kl']:.4f} "
            f"guess={aux_stats['guesser_loss']:.3f} pub={aux_stats['public_loss']:.3f}")
    return m


def policy_callable(models: Models, guesser=None):
    """Greedy/sampled masked policy for evaluation. The returned env factory wraps
    FishAEC with the (trained) guesser so eval observations match training."""
    g = guesser or models.guesser

    dev = device_of(models.actor)

    def act(obs):
        x = torch.as_tensor(obs["observation"], dtype=torch.float32).unsqueeze(0).to(dev)
        m = torch.as_tensor(obs["action_mask"], dtype=torch.float32).unsqueeze(0).to(dev)
        with torch.no_grad():
            logp = models.actor.log_probs(x, m)[0]
        return int(torch.multinomial(logp.exp(), 1))     # sample (argmax is degenerate)
    return act, g
