"""Self-play rollout collection through the belief-augmented env.

One acting function drives BOTH seats (shared policy). Each env step is one
transition (compound builder sub-steps included — the mask changes per sub-step,
so each is a distinct masked decision). Features are captured at decision time,
before the engine mutates the state; the winner label is back-filled per game.
"""
from __future__ import annotations

import numpy as np
import torch

from fishrl.data.buffer import RolloutBuffer, Step
from fishrl.data.features import encode_god, encode_public, opponent_hand_counts
from fishrl.models import device_of
from fishrl.obs.encoder import OBS_DIM
from fishrl.train.advantages import SEAT_SIGN


def random_act_fn(rng: np.random.Generator):
    def act(obs):
        legal = np.flatnonzero(obs["action_mask"])
        return int(rng.choice(legal)), 0.0
    return act


def actor_act_fn(actor):
    dev = device_of(actor)

    def act(obs):
        x = torch.as_tensor(obs["observation"], dtype=torch.float32).unsqueeze(0).to(dev)
        m = torch.as_tensor(obs["action_mask"], dtype=torch.float32).unsqueeze(0).to(dev)
        with torch.no_grad():
            logp_all = actor.log_probs(x, m)[0]
            p = logp_all.exp()
            a = int(torch.multinomial(p, 1))
        return a, float(logp_all[a])
    return act


def fill_critic_values(buf: RolloutBuffer, critic, batch: int = 8192) -> None:
    """Batched post-collection value pass for the privileged critic.

    The critic is the ONLY value head in the advantage loop (asymmetric actor-critic),
    but it is FIXED during collection -- it's updated only afterward in `ppo_update`.
    So computing each step's seat-frame value baseline in one batched forward over the
    buffered `god_feat` is numerically identical to the old per-decision batch=1 calls,
    while lifting the critic forward out of the per-decision hot loop (it was ~14% of
    collection wall-clock at batch=1). No extra state: god_feat is already buffered."""
    if not buf.steps:
        return
    dev = device_of(critic)
    god = np.stack([s.god_feat for s in buf.steps])
    sign = np.array([SEAT_SIGN[s.seat] for s in buf.steps], dtype=np.float32)
    out = np.empty(len(buf.steps), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(god), batch):
            gt = torch.as_tensor(god[i:i + batch], dtype=torch.float32).to(dev)
            p1 = critic.p1_winprob(gt).cpu().numpy()
            out[i:i + batch] = sign[i:i + batch] * (2.0 * p1 - 1.0)
    for s, v in zip(buf.steps, out):
        s.value = float(v)


def collect_games(belief_env, act_fn, n_games, base_seed, critic=None,
                  max_decisions=2000) -> RolloutBuffer:
    buf = RolloutBuffer()
    for gi in range(n_games):
        belief_env.reset(seed=base_seed + gi)
        start = len(buf)
        for agent in belief_env.agent_iter(max_iter=max_decisions * 6):
            if belief_env.terminations[agent] or belief_env.truncations[agent]:
                belief_env.step(None)
                continue
            obs = belief_env.observe(agent)
            x = obs["observation"]
            g = belief_env.g
            action, logp = act_fn(obs)
            buf.add(Step(
                seat=agent, x_act=x.astype(np.float32),
                mask=obs["action_mask"].astype(np.int8), action=action, logp=logp,
                value=0.0, god_feat=encode_god(g), pub_feat=encode_public(g),
                guess_in=x[OBS_DIM:].astype(np.float32),
                cnt_target=opponent_hand_counts(g, agent),
            ))
            belief_env.step(action)
        winner = belief_env.g.result.get("winner")
        for i in range(start, len(buf)):
            buf.steps[i].winner = winner
    if critic is not None:
        fill_critic_values(buf, critic)            # batched, off the per-decision path
    return buf
