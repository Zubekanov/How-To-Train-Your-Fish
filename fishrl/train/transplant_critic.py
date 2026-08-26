"""Replace a live checkpoint's CRITIC (weights + its Adam moments) with the critic
from an earlier checkpoint of the SAME lineage, in place. Actor, league, counters,
frozen snapshot, handoff_start: untouched -- this is a critic swap, not a reset.

    python -m fishrl.train.transplant_critic --src checkpoints-v3/best.pt --dst checkpoints-v3/latest.pt

Use case (2026-08-23): 2.3k iterations under a mis-specified critic_consistency
objective flattened the critic's cast signal; restoring the pre-change critic beats
waiting for BCE to undo it. Requires identical critic architecture in src and dst.
"""
from __future__ import annotations

import argparse
import os
import shutil

import torch

from fishrl.train import checkpoint as ckpt
from fishrl.train.train_loop import _load_model_state, build_models, config_from_checkpoint


def transplant(src: str, dst: str) -> dict:
    ps = ckpt.load_checkpoint(src, map_location="cpu")
    pd = ckpt.load_checkpoint(dst, map_location="cpu")
    cs = config_from_checkpoint(ps["config"], device="cpu")
    cd = config_from_checkpoint(pd["config"], device="cpu")
    for k in ("critic_view", "critic_encoder", "card_dim", "obs_counts", "obs_split", "obs_ctx", "belief_mode"):
        assert getattr(cs, k) == getattr(cd, k), f"config mismatch on {k}: {getattr(cs, k)} vs {getattr(cd, k)}"
    sc, dc = ps["models"]["critic"], pd["models"]["critic"]
    assert set(sc) == set(dc), "critic state_dict keys differ"
    for k in sc:
        assert sc[k].shape == dc[k].shape, f"shape mismatch at {k}"

    # optimizer: critic params follow the actor's in the positional Adam state
    m = build_models(cd)
    _load_model_state(m, pd["models"])
    n_actor = len(list(m.actor.parameters()))
    n_critic = len(list(m.critic.parameters()))
    opt_s, opt_d = ps["optim"]["ppo"], pd["optim"]["ppo"]
    state_d = dict(opt_d["state"])
    moved = 0
    for j in range(n_critic):
        idx = n_actor + j
        if idx in opt_s["state"]:
            state_d[idx] = opt_s["state"][idx]
            moved += 1
        elif idx in state_d:
            del state_d[idx]                         # src had no moments: start fresh
    out = dict(pd)
    out["models"] = dict(pd["models"], critic=sc)
    out["optim"] = dict(pd["optim"], ppo=dict(opt_d, state=state_d))
    out["critic_transplant"] = {"from": os.path.basename(src), "src_it": int(ps["done"]),
                                "at_it": int(pd["done"]), "adam_slots": moved}
    # round-trip: the transplanted critic loads and equals src bit-for-bit
    m2 = build_models(cd)
    _load_model_state(m2, out["models"])
    for k, v in m2.critic.state_dict().items():
        assert torch.equal(v, sc[k]), k
    for k, v in m2.actor.state_dict().items():
        assert torch.equal(v, pd["models"]["actor"][k]), k
    opt2 = torch.optim.Adam(list(m2.actor.parameters()) + list(m2.critic.parameters()), lr=cd.lr_ppo)
    opt2.load_state_dict(out["optim"]["ppo"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--out", default=None, help="default: overwrite --dst (with a .pre-transplant.pt backup)")
    args = ap.parse_args()
    out = transplant(args.src, args.dst)
    target = args.out or args.dst
    if target == args.dst:
        bak = f"{args.dst}.pre-transplant.pt"
        if os.path.exists(bak):
            raise SystemExit(f"backup {bak} exists -- refusing to overwrite it")
        shutil.copy2(args.dst, bak)
        print(f"[transplant] original kept at {bak}")
    ckpt.save_checkpoint(target, out)
    info = out["critic_transplant"]
    print(f"[transplant] {target}: critic from {info['from']} (it={info['src_it']}) into it={info['at_it']}, "
          f"{info['adam_slots']} Adam slots carried; actor/league/counters untouched", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
