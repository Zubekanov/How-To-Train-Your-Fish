"""Evaluation: estimator calibration, guesser accuracy, and win-rates.

The privileged estimator (full info) should be better-calibrated than the public
one; the policy's win-rate vs the random and heuristic baselines should rise with
training. Evaluation runs the actor through the belief augmentation so its inputs
match training time.
"""
from __future__ import annotations

import numpy as np
import torch

from fishrl.models import device_of
from fishrl.obs import vocab as V
from fishrl.train.belief_env import BeliefAugmentedEnv
from fishrl.train.collector import actor_act_fn, collect_games, random_act_fn


def collect_eval_batch(models, n_games=8, seed=10_000, max_decisions=2000):
    benv = BeliefAugmentedEnv(models.guesser, max_decisions=max_decisions)
    buf = collect_games(benv, actor_act_fn(models.actor), n_games, seed,
                        critic=models.critic, max_decisions=max_decisions)
    return buf.compute(0.997, 0.95)


def estimator_metrics(models, batch) -> dict:
    keep = batch["valid"] > 0
    y = batch["y_p1"][keep]
    if y.numel() == 0:
        return {"n": 0}
    with torch.no_grad():
        priv = models.critic.p1_winprob(batch["god"][keep].to(device_of(models.critic))).cpu()
        pub = models.public.p1_winprob(batch["pub"][keep].to(device_of(models.public))).cpu()

    def acc(p):
        return float(((p > 0.5).float() == y).float().mean())

    def brier(p):
        return float(((p - y) ** 2).mean())

    return {"n": int(y.numel()),
            "priv_acc": acc(priv), "pub_acc": acc(pub),
            "priv_brier": brier(priv), "pub_brier": brier(pub)}


def guesser_mae(models, batch) -> float:
    dev = device_of(models.guesser)
    with torch.no_grad():
        pred = models.guesser(batch["persp"].to(dev), batch["prev_guess"].to(dev)).cpu()
    return float((pred - batch["cnt"]).abs().mean())


def act_from_actor(actor, obs, greedy: bool = False) -> int:
    """Sample (default) or greedily pick a legal action. Sampling is the right
    default: greedy argmax of a high-entropy masked policy is degenerate (it
    deterministically repeats one action — usually a no-op pass — and loses)."""
    dev = device_of(actor)
    x = torch.as_tensor(obs["observation"], dtype=torch.float32).unsqueeze(0).to(dev)
    m = torch.as_tensor(obs["action_mask"], dtype=torch.float32).unsqueeze(0).to(dev)
    with torch.no_grad():
        logp = actor.log_probs(x, m)[0]
    if greedy:
        return int(torch.argmax(logp))
    return int(torch.multinomial(logp.exp(), 1))


def winrate_vs_random(models, n_games=20, seed=0, max_decisions=2000,
                      use_belief=True) -> float:
    rng = np.random.default_rng(seed)
    benv = BeliefAugmentedEnv(models.guesser, belief=use_belief, max_decisions=max_decisions)
    wins = 0
    for i in range(n_games):
        benv.reset(seed=seed + i)
        for agent in benv.agent_iter(max_iter=max_decisions * 6):
            if benv.terminations[agent] or benv.truncations[agent]:
                benv.step(None)
                continue
            obs = benv.observe(agent)
            if agent == "p1":
                a = act_from_actor(models.actor, obs)
            else:
                a = int(rng.choice(np.flatnonzero(obs["action_mask"])))
            benv.step(a)
        wins += int(benv.g.result.get("winner") == "p1")
    return wins / n_games


def winrate_vs_heuristic(models, n_games=20, seed=0, max_decisions=2000,
                         use_belief=True) -> float:
    """The trained actor (p1, belief-augmented) vs the engine heuristic AI (p2).
    With use_belief=False the belief channel is fed zeros (no guesser), matching a
    belief-off-trained actor."""
    from fishrl.opponents.heuristic import HeuristicMatch
    z = np.zeros(V.N_NAMES, dtype=np.float32)
    wins = 0
    for i in range(n_games):
        m = HeuristicMatch(max_decisions=max_decisions)
        obs = m.reset(seed=seed + i)
        prev = z.copy()
        done, r, guard = False, 0.0, 0
        while not done and guard < max_decisions * 6:
            guard += 1
            persp = obs["observation"]
            if use_belief:
                gdev = device_of(models.guesser)
                with torch.no_grad():
                    guess = models.guesser(
                        torch.as_tensor(persp).unsqueeze(0).to(gdev),
                        torch.as_tensor(prev).unsqueeze(0).to(gdev)).squeeze(0).cpu().numpy().astype(np.float32)
                prev = guess
            else:
                guess = z
            aug = {"observation": np.concatenate([persp, guess]).astype(np.float32),
                   "action_mask": obs["action_mask"]}
            obs, r, done, _ = m.step(act_from_actor(models.actor, aug))
        wins += int(r > 0)
    return wins / n_games
