"""Audit: cProfile decomposition of self-play collection with a REAL checkpoint
(the live v2 architecture), not the untrained default shapes profile_collection
uses. Buckets: engine stepping, observation/feature encoding, network forwards.

    python -m fishrl.eval.audit_profile --ckpt checkpoints-v2/latest.pt --games 20
"""
from __future__ import annotations

import argparse
import cProfile
import pstats


def _cum(st: pstats.Stats, file_sub: str, func: str) -> float:
    total = 0.0
    for (fn, _ln, name), (_cc, _nc, _tt, ct, _callers) in st.stats.items():
        if name == func and file_sub in fn.replace("\\", "/"):
            total += ct
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--affinity", default="")
    args = ap.parse_args()

    import torch
    from fishrl.train.pcollect import _apply_affinity, parse_affinity
    _apply_affinity(parse_affinity(args.affinity))
    torch.set_num_threads(1)

    from fishrl.eval.audit_bench import build
    from fishrl.train.belief_env import BeliefAugmentedEnv
    from fishrl.train.collector import actor_act_fn, collect_games

    models, _pl = build(args.ckpt)
    benv = BeliefAugmentedEnv(models.guesser, max_decisions=2000)
    act = actor_act_fn(models.actor)

    pr = cProfile.Profile()
    pr.enable()
    buf = collect_games(benv, act, args.games, 8_000_000, critic=models.critic,
                        max_decisions=2000)
    pr.disable()

    st = pstats.Stats(pr)
    total = st.total_tt
    n = len(buf.steps)
    engine = _cum(st, "forgetful_fish/engine.py", "step")
    persp = _cum(st, "obs/encoder.py", "encode_observation")
    god = _cum(st, "data/features.py", "encode_god")
    pub = _cum(st, "data/features.py", "encode_public")
    actor_f = _cum(st, "train/collector.py", "act")
    guesser_f = _cum(st, "train/belief_env.py", "observe")
    critic_f = _cum(st, "train/collector.py", "fill_critic_values")
    mask = _cum(st, "spaces/masking.py", "atomic_mask")
    print(f"games={args.games} steps={n} total_s={total:.1f} "
          f"({total / max(n, 1) * 1000:.2f} ms/decision)")
    for name, v in [("engine.step", engine), ("encode_observation (persp)", persp),
                    ("encode_god", god), ("encode_public", pub),
                    ("actor act (fwd+sample)", actor_f),
                    ("belief observe (guesser fwd + persp encode)", guesser_f),
                    ("critic batched values", critic_f),
                    ("atomic_mask", mask)]:
        print(f"  {name:44s} {v:7.2f}s  {v / total * 100:5.1f}%")


if __name__ == "__main__":
    main()
