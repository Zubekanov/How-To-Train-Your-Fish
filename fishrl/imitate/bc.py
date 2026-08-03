"""Behaviour-clone an actor from a scripted heuristic teacher (the BC bootstrap).

Generates adapter-labelled teacher games (fishrl.imitate.datagen), trains the
actor with masked cross-entropy on every decision, and writes a RESUME-COMPATIBLE
checkpoint: the payload carries the same architecture record, fresh optimizer
states, a frozen snapshot, and RNG state the trainer's resume path expects, so
PPO fine-tuning is just pointing the trainer at the output directory.

    python -m fishrl.imitate.bc --games 800 --workers 8 --gpu --out checkpoints-bc

Plain BC saturates well below the teacher's playing strength: ~88% per-decision
agreement still compounds into off-distribution states over a ~200-decision game
(the clone measured ~0.08 vs the teacher's own profile). ``--dagger-rounds``
closes that covariate-shift gap the standard way: each round the STUDENT drives
the games (sampling, so it visits its own mistakes) while the teacher labels
every visited state; the new data is aggregated and training continues. Rounds
stop early once the student's sampled winrate vs the teacher's profile reaches
``--dagger-target-wr`` -- the "release point" for handing the clone to PPO.

    python -m fishrl.imitate.bc --init-ckpt checkpoints-bc-rl0/best.pt \
        --games 2000 --dagger-rounds 4 --dagger-games 1500 --workers 10 --gpu

The belief channel is fed ZEROS throughout (a fresh clone has no meaningful
guesser; per the belief ablation the actor leans on that channel only weakly),
and use_belief=False rides into the checkpoint so the fine-tune keeps feeding
zeros. The guesser/critic/public heads are saved untrained -- the PPO handoff
should start with a critic warmup phase before trusting advantages.
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
from fishrl.train.train_loop import build_models, config_from_checkpoint

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


def _train_epochs(actor, opt, data: dict, device, epochs: int, batch: int,
                  holdout: float, seed: int, log=_log, tag: str = "bc") -> tuple:
    """Masked cross-entropy over `data` with a BY-GAME holdout split; returns the
    final (holdout acc, non-forced acc)."""
    obs = torch.from_numpy(data["obs"])
    mask = torch.from_numpy(data["mask"])
    acts = torch.from_numpy(data["act"]).long()
    gid = data["game"]
    games = np.unique(gid)
    rng = np.random.default_rng(seed)
    rng.shuffle(games)
    ho_games = set(games[:max(1, int(len(games) * holdout))].tolist())
    ho = np.isin(gid, list(ho_games))
    tr_idx = np.flatnonzero(~ho)
    ho_idx = torch.from_numpy(np.flatnonzero(ho))
    log(f"[{tag}] {len(acts)} samples ({len(tr_idx)} train / {len(ho_idx)} holdout, "
        f"split by game) on {device}")
    acc = acc_nf = float("nan")
    for epoch in range(1, epochs + 1):
        perm = rng.permutation(tr_idx)
        tot = nb = 0.0
        for i in range(0, len(perm), batch):
            bi = torch.from_numpy(perm[i:i + batch])
            xb = obs[bi].to(device).float()
            xb = torch.cat([xb, torch.zeros(xb.shape[0], V.N_NAMES, device=device)], 1)
            mb = mask[bi].to(device).float()
            ab = acts[bi].to(device)
            logp = actor.log_probs(xb, mb)
            loss = -logp.gather(1, ab.unsqueeze(1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(bi); nb += len(bi)
        acc, acc_nf = _accuracy(actor, obs[ho_idx], mask[ho_idx], acts[ho_idx], device)
        log(f"[{tag}] epoch {epoch}/{epochs}  loss={tot / max(nb, 1):.4f}  "
            f"holdout acc={acc:.3f} (non-forced {acc_nf:.3f})")
    return acc, acc_nf


def _aggregate(a: dict, b: dict) -> dict:
    """DAgger dataset aggregation; game ids offset so split-by-game stays sound."""
    off = int(a["game"].max()) + 1 if len(a["game"]) else 0
    return {
        "obs": np.concatenate([a["obs"], b["obs"]]),
        "mask": np.concatenate([a["mask"], b["mask"]]),
        "act": np.concatenate([a["act"], b["act"]]),
        "seat": np.concatenate([a["seat"], b["seat"]]),
        "game": np.concatenate([a["game"], b["game"] + off]),
        "winners": a["winners"] + b["winners"],
        "fallbacks": a["fallbacks"] + b["fallbacks"],
        "forced_targets": a["forced_targets"] + b["forced_targets"],
    }


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
    ap.add_argument("--init-ckpt", default=None,
                    help="start from an existing checkpoint's actor (architecture rides "
                         "in it; the --actor-* flags are ignored). Skips the initial "
                         "training pass -- the base data still generates for aggregation.")
    ap.add_argument("--dagger-rounds", type=int, default=0,
                    help="on-policy relabeling rounds: student drives, teacher labels, "
                         "aggregate, continue training")
    ap.add_argument("--dagger-games", type=int, default=1000, help="games per DAgger round")
    ap.add_argument("--dagger-target-wr", type=float, default=0.5,
                    help="stop rounds once sampled winrate vs the teacher's own profile "
                         "reaches this (the PPO release point)")
    ap.add_argument("--dagger-eval-games", type=int, default=100,
                    help="games for the per-round winrate gate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--out", default="checkpoints-bc")
    ap.add_argument("--eval-games", type=int, default=40,
                    help="post-train winrate games vs the teacher's profile (0 = skip)")
    args = ap.parse_args()
    device = resolve_device(args.gpu)

    if args.init_ckpt:
        pl = ckpt.load_checkpoint(args.init_ckpt, map_location="cpu")
        # use_belief=False regardless of the source: BC/DAgger always trains on zeros
        # (set in the saved dict -- config_from_checkpoint reads it explicitly, so an
        # override kwarg would collide)
        cd = dict(pl["config"])
        cd["use_belief"] = False
        cfg = config_from_checkpoint(cd, device=device)
        m = build_models(cfg)
        m.actor.load_state_dict(pl["models"]["actor"])
        _log(f"[bc] actor initialised from {args.init_ckpt} "
             f"(bc meta: {pl.get('bc', 'n/a')})")
    else:
        # use_belief=False rides into the checkpoint config: the clone is trained on
        # ZEROED belief dims, so the fine-tune must keep feeding zeros -- resuming
        # with live guesser output would shift the input on exactly those dims.
        cfg = Config(device=device, seed=args.seed, use_belief=False,
                     actor_hidden=tuple(int(x) for x in args.actor_hidden.split(",")),
                     card_dim=args.card_dim, actor_encoder=args.actor_encoder)
        m = build_models(cfg)
    actor = m.actor
    opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    _log(f"[bc] actor {sum(p.numel() for p in actor.parameters()) / 1e6:.2f}M params "
         f"on {device}")

    data = generate(args.games, teacher=args.teacher,
                    opponents=tuple(args.opponents.split(",")),
                    mirror_frac=args.mirror_frac, seed=args.seed,
                    workers=args.workers, max_decisions=args.max_decisions, log=_log)
    acc = acc_nf = float("nan")
    if not args.init_ckpt:
        acc, acc_nf = _train_epochs(actor, opt, data, device, args.epochs, args.batch,
                                    args.holdout, args.seed, _log, tag="bc")

    def _save(round_no: int, wr=None) -> str:
        meta = {"teacher": args.teacher, "games": int(len(data["winners"])),
                "samples": int(len(data["act"])), "holdout_acc": acc,
                "holdout_acc_nonforced": acc_nf, "dagger_round": round_no,
                **({"wr_vs_teacher": wr} if wr is not None else {}),
                "fallbacks": data["fallbacks"], "forced_targets": data["forced_targets"]}
        path = ckpt.latest_path(args.out)
        ckpt.save_checkpoint(path, _payload(cfg, m, meta))
        _log(f"[bc] checkpoint -> {path}")
        return path

    path = _save(0)

    for r in range(1, args.dagger_rounds + 1):
        wr = winrate_vs(actor, args.teacher, args.dagger_eval_games,
                        max_decisions=args.max_decisions)
        _log(f"[dagger] round {r}: sampled wr vs {args.teacher} = {wr:.3f} "
             f"(target {args.dagger_target_wr})")
        if wr >= args.dagger_target_wr:
            _log(f"[dagger] target reached -- releasing at round {r}")
            _save(r, wr)
            break
        d = generate(args.dagger_games, teacher=args.teacher,
                     opponents=tuple(args.opponents.split(",")),
                     mirror_frac=args.mirror_frac, seed=args.seed + 100_000 * r,
                     workers=args.workers, max_decisions=args.max_decisions,
                     log=_log, student_ckpt=path)
        data = _aggregate(data, d)
        acc, acc_nf = _train_epochs(actor, opt, data, device, args.epochs, args.batch,
                                    args.holdout, args.seed + r, _log, tag=f"dagger{r}")
        path = _save(r)

    if args.eval_games > 0:
        wr = winrate_vs(actor, args.teacher, args.eval_games,
                        max_decisions=args.max_decisions)
        _log(f"[bc] clone vs internal {args.teacher}: {wr:.3f} over {args.eval_games} games")
        _save(args.dagger_rounds, wr)


if __name__ == "__main__":
    main()
