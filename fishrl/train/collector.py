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
from fishrl.obs.encoder import OBS_DIM
from fishrl.train.advantages import p1_winprob_to_seat_value


def random_act_fn(rng: np.random.Generator):
    def act(obs):
        legal = np.flatnonzero(obs["action_mask"])
        return int(rng.choice(legal)), 0.0
    return act


def actor_act_fn(actor):
    def act(obs):
        x = torch.as_tensor(obs["observation"], dtype=torch.float32).unsqueeze(0)
        m = torch.as_tensor(obs["action_mask"], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logp_all = actor.log_probs(x, m)[0]
            p = logp_all.exp()
            a = int(torch.multinomial(p, 1))
        return a, float(logp_all[a])
    return act


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
            god = encode_god(g)
            value = 0.0
            if critic is not None:
                with torch.no_grad():
                    p_p1 = float(critic.p1_winprob(torch.as_tensor(god).unsqueeze(0))[0])
                value = p1_winprob_to_seat_value(p_p1, agent)
            action, logp = act_fn(obs)
            buf.add(Step(
                seat=agent, x_act=x.astype(np.float32),
                mask=obs["action_mask"].astype(np.int8), action=action, logp=logp,
                value=value, god_feat=god, pub_feat=encode_public(g),
                guess_in=x[OBS_DIM:].astype(np.float32),
                cnt_target=opponent_hand_counts(g, agent),
            ))
            belief_env.step(action)
        winner = belief_env.g.result.get("winner")
        for i in range(start, len(buf)):
            buf.steps[i].winner = winner
    return buf
