"""Write the v3 seed checkpoint: v2's actor warm-started into the v3 architecture.

The v3 restart (bookkeeper belief + public critic + parity aux + guided text-change)
keeps the actor's shapes bit-identical — the 20-dim belief slot keeps its width, so
ACTOR_IN stays 6517 and A.N stays 285. That means the v2 actor's weights transfer
verbatim; only the critic (input god→public) and its aux head start fresh. This
script follows the imitate/bc bootstrap pattern: build a RESUME-COMPATIBLE seed
payload and drop it as <out>/latest.pt, so the standard `--resume` launcher boots it
with no new trainer logic. The handoff schedule (freeze-actor phase, KL-to-teacher,
lowered ent_start) rides in train.args, not here.

    python -m fishrl.train.bootstrap_v3 --src checkpoints-v2/best.pt --out checkpoints-v3
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from fishrl.train import checkpoint as ckpt
from fishrl.train.train_loop import (
    Models, _encoders, _model_state, _rng_state, _snapshot, build_models,
    config_from_checkpoint,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="checkpoints-v2/best.pt",
                    help="legacy checkpoint whose ACTOR seeds the run")
    ap.add_argument("--out", default="checkpoints-v3", help="target checkpoint dir")
    ap.add_argument("--text-change", choices=["full", "guided", "auto"], default="guided")
    ap.add_argument("--deckout-aux", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pl = ckpt.load_checkpoint(args.src, map_location="cpu")
    src_cfg = pl["config"]
    cfg = config_from_checkpoint(
        src_cfg,                              # arch fields (encoders/hidden/card_dim) from v2
        belief_mode="bookkeeper", critic_view="public",
        critic_deckout_aux=args.deckout_aux, text_change_mode=args.text_change,
        guesser_encoder=None, public_encoder=None,   # nets that no longer exist
        seed=args.seed, device="cpu",
    )
    m = build_models(cfg)
    assert m.guesser is None and m.public is None, "v3 build should be actor+critic only"

    # Warm start: the v2 actor's weights load verbatim (identical shapes).
    m.actor.load_state_dict(pl["models"]["actor"])
    src_sd = pl["models"]["actor"]
    for k, v in m.actor.state_dict().items():
        assert torch.equal(v, src_sd[k]), f"actor warm-start mismatch at {k}"

    frozen = _snapshot(m)                     # day-0 frozen-self anchor = v2-best
    opt_ppo = torch.optim.Adam(list(m.actor.parameters()) + list(m.critic.parameters()),
                               lr=cfg.lr_ppo)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    payload = {
        "format": ckpt.FORMAT,
        "config": {"seed": cfg.seed, "encoders": _encoders(cfg),
                   "use_belief": cfg.use_belief, "critic_hidden": list(cfg.critic_hidden),
                   "hidden": list(cfg.hidden),
                   "actor_hidden": (list(cfg.actor_hidden)
                                    if cfg.actor_hidden is not None else None),
                   "card_dim": cfg.card_dim,
                   "belief_mode": cfg.belief_mode, "critic_view": cfg.critic_view,
                   "critic_deckout_aux": cfg.critic_deckout_aux,
                   "text_change_mode": cfg.text_change_mode},
        "done": 0, "elapsed": 0.0, "frozen_it": 0,
        "warmup_done": True,                  # superseded by the freeze-actor phase
        "models": _model_state(m), "frozen": _model_state(frozen),
        "optim": {"ppo": opt_ppo.state_dict()},
        "rng": _rng_state("cpu"),
        "league": None, "scen_league": None,
        "bootstrap": {"src": os.path.abspath(args.src),
                      "src_it": int(pl.get("done", 0)),
                      "src_elapsed_h": float(pl.get("elapsed", 0.0)) / 3600.0},
    }
    os.makedirs(args.out, exist_ok=True)
    dst = ckpt.latest_path(args.out)
    if os.path.exists(dst):
        raise SystemExit(f"{dst} already exists — refusing to overwrite a live lineage")
    ckpt.save_checkpoint(dst, payload)
    print(f"[bootstrap] {dst}: actor <- {args.src} (it={pl.get('done')}, "
          f"{float(pl.get('elapsed', 0.0)) / 3600.0:.0f}h), critic fresh "
          f"(public view, aux={cfg.critic_deckout_aux}), text_change={cfg.text_change_mode}",
          flush=True)
    # round-trip sanity: the seed must load through the standard resume machinery
    back = ckpt.load_checkpoint(dst, map_location="cpu")
    cfg2 = config_from_checkpoint(back["config"])
    m2 = build_models(cfg2)
    from fishrl.train.train_loop import _load_model_state
    _load_model_state(m2, back["models"])
    for k, v in m2.actor.state_dict().items():
        assert torch.equal(v, src_sd[k]), f"round-trip actor mismatch at {k}"
    print("[bootstrap] round-trip OK: config reconstructs, actor bit-identical to source",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
