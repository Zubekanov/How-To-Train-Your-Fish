"""Add the decision-context pack (Config.obs_ctx) to a live checkpoint IN PLACE --
same lineage, same weights, same league, same counters; the nets gain
zero-initialised input columns (actor +CTX_DIM, critic +CTX_DIM+CRITIC_CTX_EXTRA).

    python -m fishrl.train.widen_ctx --ckpt checkpoints-v3/latest.pt

Why (2026-08-26 observability audit, journal 2026-08-26-observability-audit.md):
aliasing probes proved the nets blind to (1) stack-spell TARGETS (an opponent
Spray at my fish vs my land = bit-identical obs+mask; 28.9 response rows/game),
(2) search_library eligibility (Mystical Tutor fetches were effectively random —
the long-standing "Tutor value-negative" result), (3) compound ARRANGEMENTS
beyond FoF (scry/reorder/putback: a Ponder reorder had only its first pick
informed), (4) the blocker focus. The pack closes all four for both nets and
gives the hands critic the step/pending/combat/pay context the actor already
had. The same flag turns on the name-sorted PICK_SINGLE remap for
search_library / choose_graveyard (masking.pick_list) -- an action-SEMANTICS
change that costs nothing because the old picks were provably uninformed.

Blocks are all-zero outside the relevant pendings, so zero columns are
function-identical at the seam (the widen_counts / widen_split pattern).
handoff_start is left alone by default (--rearm-kl sets it to done).
"""
from __future__ import annotations

import argparse
import os
import shutil

import torch

from fishrl.data import features
from fishrl.train import checkpoint as ckpt
from fishrl.train.train_loop import _load_model_state, build_models, config_from_checkpoint
from fishrl.train.widen_counts import ACTOR_KEYS, CRITIC_KEYS, _widen_cols

K_ACTOR = features.CTX_DIM
K_CRITIC = features.CTX_DIM + features.CRITIC_CTX_EXTRA


def _widen_state(sd: dict, keys: tuple, k: int) -> dict:
    out = dict(sd)
    for key in keys:
        out[key] = _widen_cols(sd[key], k)
    return out


def widen(path: str, rearm_kl: bool = False) -> dict:
    pl = ckpt.load_checkpoint(path, map_location="cpu")
    old_cfg = config_from_checkpoint(pl["config"], device="cpu")
    if old_cfg.obs_ctx:
        raise SystemExit(f"{path} already has obs_ctx")
    if not old_cfg.obs_split:
        raise SystemExit("obs_ctx rides on top of obs_split (tail column order)")

    old_m = build_models(old_cfg)
    _load_model_state(old_m, pl["models"])
    old_params = list(old_m.actor.parameters()) + list(old_m.critic.parameters())
    widened_slots: dict[int, int] = {}
    actor_names = [n for n, _ in old_m.actor.named_parameters()]
    critic_names = [n for n, _ in old_m.critic.named_parameters()]
    for i, n in enumerate(actor_names):
        if n in ACTOR_KEYS:
            widened_slots[i] = K_ACTOR
    for j, n in enumerate(critic_names):
        if n in CRITIC_KEYS:
            widened_slots[len(actor_names) + j] = K_CRITIC
    assert len(widened_slots) == len(ACTOR_KEYS) + len(CRITIC_KEYS), widened_slots

    models = dict(pl["models"])
    models["actor"] = _widen_state(pl["models"]["actor"], ACTOR_KEYS, K_ACTOR)
    models["critic"] = _widen_state(pl["models"]["critic"], CRITIC_KEYS, K_CRITIC)
    frozen = dict(pl["frozen"])
    frozen["actor"] = _widen_state(pl["frozen"]["actor"], ACTOR_KEYS, K_ACTOR)
    if "critic" in frozen:
        frozen["critic"] = _widen_state(pl["frozen"]["critic"], CRITIC_KEYS, K_CRITIC)

    n_selves = 0
    leagues = {}
    for lk in ("league", "scen_league"):
        league = pl.get(lk)
        if league is None:
            continue
        league = dict(league)
        selves = []
        for s in league.get("selves", []):
            s = dict(s)
            s["actor"] = _widen_state(s["actor"], ACTOR_KEYS, K_ACTOR)
            selves.append(s); n_selves += 1
        league["selves"] = selves
        leagues[lk] = league

    opt = dict(pl["optim"]); ppo = dict(opt["ppo"]); state = dict(ppo["state"])
    for idx, k in widened_slots.items():
        if idx in state:
            st = dict(state[idx])
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if key in st and st[key].dim() == 2:
                    st[key] = _widen_cols(st[key], k)
            state[idx] = st
    ppo["state"] = state; opt["ppo"] = ppo

    out = dict(pl)
    out["config"] = dict(pl["config"], obs_ctx=True)
    out["models"] = models
    out["frozen"] = frozen
    out.update(leagues)
    out["optim"] = opt
    if rearm_kl:
        out["handoff_start"] = int(pl["done"])
    out["obs_ctx_widened"] = {"at_it": int(pl["done"]), "k_actor": K_ACTOR,
                              "k_critic": K_CRITIC, "league_selves": n_selves}

    # seam check: widened nets must reproduce the old outputs on padded input
    new_cfg = config_from_checkpoint(out["config"], device="cpu")
    new_m = build_models(new_cfg)                       # sets features.set_ctx_block(True)
    _load_model_state(new_m, out["models"])
    assert len(list(new_m.actor.parameters()) + list(new_m.critic.parameters())) == len(old_params)
    g = torch.Generator().manual_seed(0)
    xa = torch.randn(4, old_m.actor.in_dim, generator=g)
    xa_w = torch.cat([xa, torch.randn(4, K_ACTOR, generator=g)], dim=1)   # ANY value in the block
    xc = torch.randn(4, old_m.critic.enc.globals_dim + old_m.critic.enc.R * features.CARD_F, generator=g)
    xc_w = torch.cat([xc, torch.randn(4, K_CRITIC, generator=g)], dim=1)
    with torch.no_grad():
        a, b = old_m.actor(xa), new_m.actor(xa_w)
        assert torch.allclose(a, b, atol=2e-3, rtol=1e-4), float((a - b).abs().max())
        c, d = old_m.critic(xc), new_m.critic(xc_w)
        assert torch.allclose(c, d, atol=2e-3, rtol=1e-4), float((c - d).abs().max())
    features.set_ctx_block(False)                       # leave the process as we found it
    features.set_split_block(False); features.set_count_block(False)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None, help="default: overwrite --ckpt (with a .pre-ctx.pt backup)")
    ap.add_argument("--rearm-kl", action="store_true", help="set handoff_start=done (re-arm the KL anchor)")
    args = ap.parse_args()
    out = widen(args.ckpt, rearm_kl=args.rearm_kl)
    dst = args.out or args.ckpt
    if dst == args.ckpt:
        bak = f"{args.ckpt}.pre-ctx.pt"
        if os.path.exists(bak):
            raise SystemExit(f"backup {bak} exists -- refusing to overwrite it")
        shutil.copy2(args.ckpt, bak)
        print(f"[widen] original kept at {bak}")
    ckpt.save_checkpoint(dst, out)
    back = ckpt.load_checkpoint(dst, map_location="cpu")
    cfg2 = config_from_checkpoint(back["config"])
    m2 = build_models(cfg2)
    _load_model_state(m2, back["models"])
    opt2 = torch.optim.Adam(list(m2.actor.parameters()) + list(m2.critic.parameters()), lr=cfg2.lr_ppo)
    opt2.load_state_dict(back["optim"]["ppo"])
    info = back["obs_ctx_widened"]
    print(f"[widen] {dst}: obs_ctx on at it={back['done']} (+{info['k_actor']} actor / "
          f"+{info['k_critic']} critic inputs, {info['league_selves']} league selves widened), "
          f"handoff_start={back['handoff_start']}; round-trip OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
