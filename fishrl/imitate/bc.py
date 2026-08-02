"""Behaviour-clone an actor from a scripted heuristic teacher (the BC bootstrap).

Generates adapter-labelled teacher games (fishrl.imitate.datagen), trains the
actor with masked cross-entropy on every decision, and writes a RESUME-COMPATIBLE
checkpoint: the payload carries the same architecture record, fresh optimizer
states, a frozen snapshot, and RNG state the trainer's resume path expects, so
PPO fine-tuning is just pointing the trainer at the output directory.

    python -m fishrl.imitate.bc --games 800 --workers 8 --gpu --out checkpoints-bc

The belief channel is fed ZEROS during BC (a fresh clone has no meaningful
guesser; per the belief ablation the actor leans on that channel only weakly).
The guesser/critic/public heads are saved untrained -- the PPO handoff should
start with a critic/guesser warmup phase before trusting advantages.
"""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import torch

from fishrl.imitate.datagen import TeacherEnv, generate
from fishrl.obs import vocab as V
from fishrl.train import checkpoint as ckpt
from fishrl.train.config import Config, resolve_device
from fishrl.train.train_loop import build_models

_NETS = ("actor", "critic", "guesser", "public")


def _log(s: str) -> None:
    print(f"[{time.strftime('%m-%d %H:%M')}] {s}", flush=True)


def _accuracy(actor, obs, mask, acts, device, batch: int = 4096) -> tuple:
    """(overall top-1 acc, acc on non-forced decisions [mask offers a choice])."""
    hits = n = hits_nf = n_nf = 0
    actor.eval()
    with torch.no_grad():
        for i in range(0, len(acts), batch):
            xb = obs[i:i + batch].to(device).float()
            xb = torch.cat([xb, torch.zeros(xb.shape[0], V.N_NAMES, device=device)], 1)
            mb = mask[i:i + batch].to(device).float()
            ab = acts[i:i + batch].to(device)
            pred = actor.log_probs(xb, mb).argmax(-1)
            ok = pred == ab
            nf = mb.sum(-1) > 1
            hits += int(ok.sum()); n += len(ab)
            hits_nf += int(ok[nf].sum()); n_nf += int(nf.sum())
    actor.train()
    return hits / max(n, 1), hits_nf / max(n_nf, 1)


def winrate_vs(actor, opponent: str, n_games: int, seed0: int = 900_000,
               max_decisions: int = 2000) -> float:
    """Clone vs an engine-internal profile (belief channel zeroed, matching BC)."""
    from fishrl.train.belief_env import BeliefAugmentedEnv
    from fishrl.train.collector import actor_act_fn
    act = actor_act_fn(actor)
    wins = 0
    for i in range(n_games):
        env = BeliefAugmentedEnv(None, belief=False,
                                 env=TeacherEnv(opponent, max_decisions=max_decisions))
        env.reset(seed=seed0 + i)
        for agent in env.agent_iter(max_iter=max_decisions * 8):
            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
                continue
            a, _ = act(env.observe(agent))
            env.step(a)
        wins += int(env.winner == "p1")
    return wins / max(n_games, 1)


def _payload(cfg: Config, m, meta: dict) -> dict:
    """A resume-compatible checkpoint (mirrors train_loop._payload minus league)."""
    frozen = copy.deepcopy(m)
    for net in (frozen.actor, frozen.critic, frozen.guesser, frozen.public):
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
    # fresh optimizers, constructed exactly as train() builds them so the saved
    # (empty) state_dicts load onto identical param groups
    opt_ppo = torch.optim.Adam(list(m.actor.parameters()) + list(m.critic.parameters()),
                               lr=cfg.lr_ppo)
    opt_g = torch.optim.Adam(m.guesser.parameters(), lr=cfg.lr_guesser)
    opt_p = torch.optim.Adam(m.public.parameters(), lr=cfg.lr_public)
    rng = {"torch": torch.get_rng_state(), "numpy": np.random.get_state()}
    if str(cfg.device).startswith("cuda") and torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    return {
        "format": ckpt.FORMAT,
        "config": {"seed": cfg.seed, "use_belief": cfg.use_belief,
                   "encoders": {n: cfg.enc_for(n) for n in _NETS},
                   "hidden": tuple(cfg.hidden),
                   "actor_hidden": tuple(cfg.actor_hidden) if cfg.actor_hidden else None,
                   "critic_hidden": tuple(cfg.critic_hidden),
                   "card_dim": cfg.card_dim},
        "done": 0, "elapsed": 0.0, "frozen_it": 0, "warmup_done": True,
        "models": {n: getattr(m, n).state_dict() for n in _NETS},
        "frozen": {n: getattr(frozen, n).state_dict() for n in _NETS},
        "optim": {"ppo": opt_ppo.state_dict(), "g": opt_g.state_dict(),
                  "p": opt_p.state_dict()},
        "rng": rng,
        "bc": meta,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher", default="heuristic_1_2")
    ap.add_argument("--opponents", default="heuristic,heuristic_1_1,heuristic_1_2")
    ap.add_argument("--mirror-frac", type=float, default=0.5)
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--max-decisions", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--holdout", type=float, default=0.05, help="holdout fraction, BY GAME")
    ap.add_argument("--actor-hidden", default="768,768,384")
    ap.add_argument("--card-dim", type=int, default=128)
    ap.add_argument("--actor-encoder", default="entity")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--out", default="checkpoints-bc")
    ap.add_argument("--eval-games", type=int, default=40,
                    help="post-train winrate games vs the teacher's profile (0 = skip)")
    args = ap.parse_args()
    device = resolve_device(args.gpu)

    data = generate(args.games, teacher=args.teacher,
                    opponents=tuple(args.opponents.split(",")),
                    mirror_frac=args.mirror_frac, seed=args.seed,
                    workers=args.workers, max_decisions=args.max_decisions, log=_log)

    cfg = Config(device=device, seed=args.seed,
                 actor_hidden=tuple(int(x) for x in args.actor_hidden.split(",")),
                 card_dim=args.card_dim, actor_encoder=args.actor_encoder)
    m = build_models(cfg)
    actor = m.actor
    opt = torch.optim.Adam(actor.parameters(), lr=args.lr)

    obs = torch.from_numpy(data["obs"])
    mask = torch.from_numpy(data["mask"])
    acts = torch.from_numpy(data["act"]).long()
    gid = data["game"]
    games = np.unique(gid)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(games)
    ho_games = set(games[:max(1, int(len(games) * args.holdout))].tolist())
    ho = np.isin(gid, list(ho_games))
    tr_idx = np.flatnonzero(~ho)
    ho_idx = torch.from_numpy(np.flatnonzero(ho))
    _log(f"[bc] {len(acts)} samples ({len(tr_idx)} train / {len(ho_idx)} holdout, "
         f"split by game) on {device}; actor "
         f"{sum(p.numel() for p in actor.parameters()) / 1e6:.2f}M params")

    for epoch in range(1, args.epochs + 1):
        perm = rng.permutation(tr_idx)
        tot = nb = 0.0
        for i in range(0, len(perm), args.batch):
            bi = torch.from_numpy(perm[i:i + args.batch])
            xb = obs[bi].to(device).float()
            xb = torch.cat([xb, torch.zeros(xb.shape[0], V.N_NAMES, device=device)], 1)
            mb = mask[bi].to(device).float()
            ab = acts[bi].to(device)
            logp = actor.log_probs(xb, mb)
            loss = -logp.gather(1, ab.unsqueeze(1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(bi); nb += len(bi)
        acc, acc_nf = _accuracy(actor, obs[ho_idx], mask[ho_idx], acts[ho_idx], device)
        _log(f"[bc] epoch {epoch}/{args.epochs}  loss={tot / max(nb, 1):.4f}  "
             f"holdout acc={acc:.3f} (non-forced {acc_nf:.3f})")

    meta = {"teacher": args.teacher, "games": args.games,
            "samples": int(len(acts)), "holdout_acc": acc, "holdout_acc_nonforced": acc_nf,
            "fallbacks": data["fallbacks"], "forced_targets": data["forced_targets"]}
    path = ckpt.latest_path(args.out)
    ckpt.save_checkpoint(path, _payload(cfg, m, meta))
    _log(f"[bc] checkpoint -> {path}")

    if args.eval_games > 0:
        wr = winrate_vs(actor, args.teacher, args.eval_games,
                        max_decisions=args.max_decisions)
        _log(f"[bc] clone vs internal {args.teacher}: {wr:.3f} over {args.eval_games} games")


if __name__ == "__main__":
    main()
