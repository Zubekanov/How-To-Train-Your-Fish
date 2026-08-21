"""Replace a checkpoint's PPO critic IN PLACE -- same lineage, same actor, same
league, same counters; only the critic (and its slice of the PPO optimizer state)
starts fresh under a new critic_view.

    python -m fishrl.train.swap_critic --ckpt checkpoints-v3/latest.pt --critic-view hands

Writes <ckpt> (backing the original up to <ckpt>.pre-<view>.pt) with:
  * config.critic_view = the new view (architecture is read from the checkpoint on
    resume, so no launcher flag is needed);
  * models.critic / frozen.critic = a freshly initialised critic of the new view
    (the frozen-self anchor only needs its actor; the critic rides along so the
    resume loader's shapes match);
  * optim.ppo = the saved Adam state for the ACTOR's parameters, fresh for the
    critic's (Adam state is positional: actor params first, then critic -- the same
    order train_loop builds the optimizer in);
  * handoff_start = done, so the resume's --freeze-actor-iters / --kl-teacher-*
    count from the swap (train_loop, 2026-08-21) and the actor gets a critic-only
    warmup without `done` being reset.
Everything else (done, elapsed, league, scen_league, rng, frozen_it) is copied.
"""
from __future__ import annotations

import argparse
import os
import shutil

import torch

from fishrl.train import checkpoint as ckpt
from fishrl.train.train_loop import (_encoders, _load_model_state, _model_state, build_models,
                                     config_from_checkpoint)


def swap(path: str, view: str, seed: int = 0, aux: float | None = None) -> dict:
    pl = ckpt.load_checkpoint(path, map_location="cpu")
    old_cfg = config_from_checkpoint(pl["config"], device="cpu")
    if old_cfg.critic_view == view:
        raise SystemExit(f"{path} already has critic_view={view!r}")
    kw = dict(critic_view=view, device="cpu")
    if aux is not None:
        kw["critic_deckout_aux"] = aux
    new_cfg = config_from_checkpoint(pl["config"], **kw)

    old_m = build_models(old_cfg)
    _load_model_state(old_m, pl["models"])
    torch.manual_seed(seed)
    new_m = build_models(new_cfg)
    new_m.actor.load_state_dict(pl["models"]["actor"])          # actor verbatim
    for k, v in new_m.actor.state_dict().items():
        assert torch.equal(v, pl["models"]["actor"][k]), f"actor mismatch at {k}"
    if new_m.guesser is not None and "guesser" in pl["models"]:
        new_m.guesser.load_state_dict(pl["models"]["guesser"])

    # optimizer: keep the actor's Adam moments, fresh for the new critic
    n_actor = len(list(old_m.actor.parameters()))
    assert n_actor == len(list(new_m.actor.parameters()))
    old_opt = pl["optim"]["ppo"]
    new_opt = torch.optim.Adam(list(new_m.actor.parameters()) + list(new_m.critic.parameters()),
                               lr=new_cfg.lr_ppo)
    sd = new_opt.state_dict()
    kept = {}
    for i in range(n_actor):
        if i in old_opt["state"]:
            kept[i] = old_opt["state"][i]
    sd["state"] = kept
    # hyper-params of the (single) param group travel with the old state
    for key in ("lr", "betas", "eps", "weight_decay", "amsgrad"):
        if key in old_opt["param_groups"][0]:
            sd["param_groups"][0][key] = old_opt["param_groups"][0][key]
    new_opt.load_state_dict(sd)

    frozen = dict(pl["frozen"])
    frozen["critic"] = _model_state(new_m)["critic"]

    out = dict(pl)
    out["config"] = dict(pl["config"], critic_view=view, encoders=_encoders(new_cfg),
                         critic_deckout_aux=new_cfg.critic_deckout_aux)
    out["models"] = _model_state(new_m)
    out["frozen"] = frozen
    out["optim"] = dict(pl["optim"], ppo=new_opt.state_dict())
    out["handoff_start"] = int(pl["done"])
    out["critic_swap"] = {"from": old_cfg.critic_view, "to": view, "at_it": int(pl["done"])}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--critic-view", required=True, choices=["public", "hands", "god"])
    ap.add_argument("--deckout-aux", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default: overwrite --ckpt (with a .pre-<view>.pt backup)")
    args = ap.parse_args()
    out = swap(args.ckpt, args.critic_view, args.seed, args.deckout_aux)
    dst = args.out or args.ckpt
    if dst == args.ckpt:
        bak = f"{args.ckpt}.pre-{args.critic_view}.pt"
        if os.path.exists(bak):
            raise SystemExit(f"backup {bak} exists -- refusing to overwrite it")
        shutil.copy2(args.ckpt, bak)
        print(f"[swap] original kept at {bak}")
    ckpt.save_checkpoint(dst, out)
    # round-trip through the resume machinery
    back = ckpt.load_checkpoint(dst, map_location="cpu")
    cfg2 = config_from_checkpoint(back["config"])
    m2 = build_models(cfg2)
    _load_model_state(m2, back["models"])
    frozen2 = build_models(cfg2)
    _load_model_state(frozen2, back["frozen"])
    opt2 = torch.optim.Adam(list(m2.actor.parameters()) + list(m2.critic.parameters()), lr=cfg2.lr_ppo)
    opt2.load_state_dict(back["optim"]["ppo"])
    print(f"[swap] {dst}: critic_view {out['critic_swap']['from']} -> {args.critic_view} at it={back['done']}, "
          f"handoff_start={back['handoff_start']}, actor/league/rng/elapsed untouched; round-trip OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
