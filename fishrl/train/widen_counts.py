"""Add the per-name count block (Config.obs_counts) to a live checkpoint IN PLACE --
same lineage, same weights, same league, same counters; the nets simply gain
COUNT_DIM zero-initialised input columns.

    python -m fishrl.train.widen_counts --ckpt checkpoints-v3/latest.pt

Why it is function-identical at the seam: the block is appended to the globals
tail of the actor input (after the bookkeeper belief) and of the hands critic's
features. In the entity encoder the globals bypass the card pooling and are
concatenated LAST into the head MLP's input, so the only parameters that see the
new floats are the trailing columns of the head's first Linear (and the critic's
aux head). Zero columns there make every output bit-identical to the old net;
the nets then learn to use the channel under the live KL anchor.

What is rewritten:
  * config.obs_counts = True (architecture key; read by config_from_checkpoint)
  * models.actor / frozen.actor / every league past-self actor: `net.0.weight`
    widened by COUNT_DIM zero columns
  * models.critic / frozen.critic (hands): `net.0.weight` and `aux_head.weight` widened
  * optim.ppo: the Adam moments of those tensors padded with zeros (param order is
    unchanged, so the positional state keys still line up)
  * handoff_start is left alone by default: nothing is reset, so no anchor is needed
    (--rearm-kl sets handoff_start = done to re-arm --kl-teacher-* / --freeze-actor-*
    from the widening, if wanted).
Everything else (done, elapsed, league EMAs, rng, frozen_it) is copied verbatim.
"""
from __future__ import annotations

import argparse
import os
import shutil

import torch

from fishrl.data import features
from fishrl.train import checkpoint as ckpt
from fishrl.train.train_loop import _load_model_state, build_models, config_from_checkpoint

K = features.COUNT_DIM


def _widen_cols(t: torch.Tensor, k: int) -> torch.Tensor:
    """Append k zero columns to a (out, in) weight / moment tensor."""
    return torch.cat([t, t.new_zeros(t.shape[0], k)], dim=1)


def _widen_state(sd: dict, keys: tuple) -> dict:
    out = dict(sd)
    for key in keys:
        out[key] = _widen_cols(sd[key], K)
    return out


ACTOR_KEYS = ("net.0.weight",)
CRITIC_KEYS = ("net.0.weight", "aux_head.weight")


def widen(path: str, rearm_kl: bool = False) -> dict:
    pl = ckpt.load_checkpoint(path, map_location="cpu")
    old_cfg = config_from_checkpoint(pl["config"], device="cpu")
    if old_cfg.obs_counts:
        raise SystemExit(f"{path} already has obs_counts")
    if old_cfg.belief_mode != "bookkeeper" or old_cfg.critic_view != "hands":
        raise SystemExit("obs_counts needs belief_mode=bookkeeper and critic_view=hands")

    # old nets (for the parameter order the Adam state is keyed on)
    old_m = build_models(old_cfg)
    _load_model_state(old_m, pl["models"])
    old_params = list(old_m.actor.parameters()) + list(old_m.critic.parameters())
    old_actor_sd = old_m.actor.state_dict()
    old_critic_sd = old_m.critic.state_dict()
    # which positional optimizer slots hold the widened tensors
    widened_slots: dict[int, int] = {}          # param index -> k
    actor_names = [n for n, _ in old_m.actor.named_parameters()]
    critic_names = [n for n, _ in old_m.critic.named_parameters()]
    for i, n in enumerate(actor_names):
        if n in ACTOR_KEYS:
            widened_slots[i] = K
    for j, n in enumerate(critic_names):
        if n in CRITIC_KEYS:
            widened_slots[len(actor_names) + j] = K
    assert len(widened_slots) == len(ACTOR_KEYS) + len(CRITIC_KEYS), widened_slots

    models = dict(pl["models"])
    models["actor"] = _widen_state(pl["models"]["actor"], ACTOR_KEYS)
    models["critic"] = _widen_state(pl["models"]["critic"], CRITIC_KEYS)
    frozen = dict(pl["frozen"])
    frozen["actor"] = _widen_state(pl["frozen"]["actor"], ACTOR_KEYS)
    if "critic" in frozen:
        frozen["critic"] = _widen_state(pl["frozen"]["critic"], CRITIC_KEYS)

    # league past-selves (actor-only in bookkeeper mode), in both league payloads
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
            s["actor"] = _widen_state(s["actor"], ACTOR_KEYS)
            selves.append(s); n_selves += 1
        league["selves"] = selves
        leagues[lk] = league

    # optimizer moments
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
    out["config"] = dict(pl["config"], obs_counts=True)
    out["models"] = models
    out["frozen"] = frozen
    out.update(leagues)
    out["optim"] = opt
    if rearm_kl:
        out["handoff_start"] = int(pl["done"])
    out["obs_counts_widened"] = {"at_it": int(pl["done"]), "k": K, "league_selves": n_selves}

    # seam check: the widened nets must reproduce the old outputs on zero-padded input
    new_cfg = config_from_checkpoint(out["config"], device="cpu")
    new_m = build_models(new_cfg)                       # sets features.set_count_block(True)
    _load_model_state(new_m, out["models"])
    assert len(list(new_m.actor.parameters()) + list(new_m.critic.parameters())) == len(old_params)
    g = torch.Generator().manual_seed(0)
    xa = torch.randn(4, old_m.actor.in_dim, generator=g)
    xa_w = torch.cat([xa, torch.randn(4, K, generator=g)], dim=1)   # ANY value in the block
    xc = torch.randn(4, old_m.critic.enc.globals_dim + old_m.critic.enc.R * features.CARD_F, generator=g)
    xc_w = torch.cat([xc, torch.randn(4, K, generator=g)], dim=1)
    # (tolerance is float32 matmul blocking noise on a 2k-wide Linear, ~1e-4 on logits of
    # magnitude ~10; the zero columns contribute exactly nothing)
    with torch.no_grad():
        a, b = old_m.actor(xa), new_m.actor(xa_w)
        assert torch.allclose(a, b, atol=2e-3, rtol=1e-4), float((a - b).abs().max())
        c, d = old_m.critic(xc), new_m.critic(xc_w)
        assert torch.allclose(c, d, atol=2e-3, rtol=1e-4), float((c - d).abs().max())
    features.set_count_block(False)                    # leave the process as we found it
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None, help="default: overwrite --ckpt (with a .pre-counts.pt backup)")
    ap.add_argument("--rearm-kl", action="store_true", help="set handoff_start=done (re-arm the KL anchor)")
    args = ap.parse_args()
    out = widen(args.ckpt, rearm_kl=args.rearm_kl)
    dst = args.out or args.ckpt
    if dst == args.ckpt:
        bak = f"{args.ckpt}.pre-counts.pt"
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
    info = back["obs_counts_widened"]
    print(f"[widen] {dst}: obs_counts on at it={back['done']} (+{info['k']} inputs, "
          f"{info['league_selves']} league selves widened), handoff_start={back['handoff_start']}; "
          f"round-trip OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
