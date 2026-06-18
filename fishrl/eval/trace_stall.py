"""Load a (stalling) checkpoint and trace what the policy actually does in a long
game: which actions repeat, whether turns advance, and where the decisions pile up.

    python -m fishrl.eval.trace_stall --ckpt checkpoints/fishrl.pt --gpu
"""
from __future__ import annotations

import argparse
import collections

import numpy as np
import torch

from fishrl.models import device_of
from fishrl.spaces import action_space as A
from fishrl.train.belief_env import BeliefAugmentedEnv
from fishrl.train.collector import actor_act_fn
from fishrl.train.config import Config, resolve_device
from fishrl.train.train_loop import build_models


def _load(path, device):
    ck = torch.load(path, map_location=device)
    encs = ck.get("encoders") or {n: ck.get("encoder", "flat")
                                   for n in ("actor", "critic", "guesser", "public")}
    cfg = Config(device=device, actor_encoder=encs["actor"], critic_encoder=encs["critic"],
                 guesser_encoder=encs["guesser"], public_encoder=encs["public"])
    m = build_models(cfg)
    m.actor.load_state_dict(ck["actor"])
    m.guesser.load_state_dict(ck["guesser"])
    for net in (m.actor, m.guesser):
        net.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/fishrl.pt")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--threshold", type=int, default=3000, help="trace the first game longer than this")
    ap.add_argument("--max-games", type=int, default=20)
    args = ap.parse_args()
    device = resolve_device(args.gpu)
    m = _load(args.ckpt, device)
    act = actor_act_fn(m.actor)
    benv = BeliefAugmentedEnv(m.guesser, max_decisions=2000)

    for gi in range(args.max_games):
        benv.reset(seed=10_000 + gi)
        seq = []                                   # (turn, step, pend_type, action_name)
        for agent in benv.agent_iter(max_iter=2000 * 6):
            if benv.terminations[agent] or benv.truncations[agent]:
                benv.step(None)
                continue
            obs = benv.observe(agent)
            g = benv.g
            a, _ = act(obs)
            name, _i = A.decode(int(a))
            seq.append((g.turn_number, g.current_step, g.pending.type, name))
            benv.step(a)
        if len(seq) <= args.threshold:
            print(f"[game {gi}] {len(seq)} decisions (<= threshold), skipping", flush=True)
            continue

        # ---- analyse the long game ----
        turns = [s[0] for s in seq]
        print(f"\n=== game {gi}: {len(seq)} decisions ===", flush=True)
        print(f"turn_number span: {min(turns)}..{max(turns)}  "
              f"(distinct turns {len(set(turns))})  winner={g.result.get('winner')}", flush=True)
        # decisions bucketed by (pending.type, action name)
        by_act = collections.Counter((s[2], s[3]) for s in seq)
        print("top (pending, action) by count:", flush=True)
        for (pt, nm), c in by_act.most_common(10):
            print(f"  {c:7d}  {100*c/len(seq):5.1f}%  {pt:>14s} / {nm}", flush=True)
        # decisions per turn -- is it stuck in a few turns?
        per_turn = collections.Counter(turns)
        worst = per_turn.most_common(5)
        print("turns with the most decisions:", flush=True)
        for t, c in worst:
            # which step(s) within that turn
            steps = collections.Counter(s[1] for s in seq if s[0] == t)
            print(f"  turn {t}: {c} decisions; steps={dict(steps)}", flush=True)
        # a window of the repeating tail
        print("tail (last 30 decisions): turn/step/pending/action", flush=True)
        for s in seq[-30:]:
            print(f"  t{s[0]:>3} {s[1]:>16} {s[2]:>12} {s[3]}", flush=True)
        return
    print("no game exceeded the threshold", flush=True)


if __name__ == "__main__":
    main()
