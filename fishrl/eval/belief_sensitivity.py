"""Is the guesser's belief channel actually used by the actor?

The actor's input is `persp (OBS_DIM) + guess (N_NAMES)`, where `guess` is the
guesser's per-name expected opponent-hand counts. The guesser forward is ~7.4% of
collection wall-clock; this probe asks whether that buys anything -- does the actor's
policy MOVE when the belief channel is ablated/perturbed, relative to a
dimension-matched perturbation of the (clearly-used) perspective channel?

Measured on ON-POLICY states from a quickly-trained actor (an untrained actor would
only reveal random architectural sensitivity, not learned dependence):

  * belief-zero / belief-mean / belief-shuffle: replace the N_NAMES belief dims with
    zeros / their dataset column-mean / a row-permutation (a wrong-but-realistic
    belief). KL(pi0 || pi_alt), total-variation, and argmax-flip rate.
  * CONTROL: the same mean-ablation applied to a RANDOM N_NAMES-sized slice of the
    perspective block. If belief KL << perspective KL, the belief is under-used for
    its width.
  * gradient saliency: mean |d logpi(a0) / d x| per belief dim vs per perspective dim.

    python -m fishrl.eval.belief_sensitivity --gpu --iters 25 --games 60
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from fishrl.obs.encoder import OBS_DIM
from fishrl.obs import vocab as V
from fishrl.train.belief_env import BeliefAugmentedEnv
from fishrl.train.collector import actor_act_fn, collect_games
from fishrl.train.config import Config, resolve_device

NB = V.N_NAMES                        # belief-channel width (last NB dims of x_act)


def _probs(actor, x, mask):
    with torch.no_grad():
        return actor.log_probs(x, mask).exp()


def _kl(p, q, eps=1e-9):              # KL(p||q) per row
    return (p * ((p + eps).log() - (q + eps).log())).sum(-1)


def _tv(p, q):                        # total variation per row
    return 0.5 * (p - q).abs().sum(-1)


def main():
    ap = argparse.ArgumentParser(description="Guesser belief-sensitivity probe.")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--iters", type=int, default=25, help="reference-agent training iters")
    ap.add_argument("--games", type=int, default=60, help="on-policy probe games")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = resolve_device(args.gpu)

    # 1. quick-train a reference agent (flat actor + guesser)
    from fishrl.train.train_loop import build_models, train
    print(f"[train] reference agent (flat, {args.iters} iters) on {device}...", flush=True)
    cfg = Config(device=device, iters=args.iters, encoder="flat", seed=args.seed,
                 games_per_iter=8, warmup_games=32)
    m = train(cfg, build_models(cfg), log=lambda s: print(f"  {s}", flush=True))
    actor, guesser = m.actor, m.guesser
    actor.eval()

    # 2. collect on-policy states the trained actor actually visits
    print(f"[collect] {args.games} on-policy games...", flush=True)
    benv = BeliefAugmentedEnv(guesser, max_decisions=cfg.max_decisions)
    buf = collect_games(benv, actor_act_fn(actor), args.games, 7000 + args.seed,
                        critic=None, max_decisions=cfg.max_decisions)
    b = buf.compute(cfg.gamma, cfg.lam)
    x = b["x_act"].to(device)
    mask = b["mask"].to(device)
    n = x.shape[0]
    print(f"  {n} decisions; belief width NB={NB} of ACTOR_IN={x.shape[1]} "
          f"(OBS_DIM={OBS_DIM})", flush=True)

    belief = x[:, OBS_DIM:]
    print(f"  belief channel: mean L2/row {belief.norm(dim=1).mean():.3f}, "
          f"per-dim std {belief.std(0).mean():.4f} "
          f"(near 0 -> guesser output barely varies across states)", flush=True)

    p0 = _probs(actor, x, mask)

    # 3. ablations of the belief channel
    rng = np.random.default_rng(args.seed)
    belief_mean = belief.mean(0, keepdim=True)
    variants = {}
    xz = x.clone(); xz[:, OBS_DIM:] = 0.0;              variants["belief-zero"] = xz
    xm = x.clone(); xm[:, OBS_DIM:] = belief_mean;      variants["belief-mean"] = xm
    perm = torch.as_tensor(rng.permutation(n), device=device)
    xs = x.clone(); xs[:, OBS_DIM:] = belief[perm];     variants["belief-shuffle"] = xs
    # CONTROL: mean-ablate a random NB-sized slice of the perspective block
    cols = torch.as_tensor(rng.choice(OBS_DIM, size=NB, replace=False), device=device)
    xp = x.clone(); xp[:, cols] = x[:, cols].mean(0, keepdim=True)
    variants["persp-mean(control)"] = xp

    print("\n  variant                 mean_KL    mean_TV    argmax-flip%", flush=True)
    a0 = p0.argmax(-1)
    for name, xv in variants.items():
        pv = _probs(actor, xv, mask)
        flip = (pv.argmax(-1) != a0).float().mean().item()
        print(f"  {name:22s} {_kl(p0, pv).mean():.5f}   {_tv(p0, pv).mean():.5f}   "
              f"{100*flip:6.2f}", flush=True)

    # 4. per-dim gradient saliency: |d logpi(a0)/d x| on belief vs perspective dims
    sub = min(2048, n)
    xs2 = x[:sub].clone().requires_grad_(True)
    lp = actor.log_probs(xs2, mask[:sub])
    sel = lp.gather(1, a0[:sub, None]).sum()
    sel.backward()
    g = xs2.grad.abs().mean(0)                      # mean |grad| per input dim
    belief_sal = g[OBS_DIM:].mean().item()
    persp_sal = g[:OBS_DIM].mean().item()
    print(f"\n  per-dim saliency |d logpi(a0)/dx|: belief {belief_sal:.3e}  "
          f"persp {persp_sal:.3e}  ratio {belief_sal/max(persp_sal,1e-12):.3f}", flush=True)
    print("  (ratio >> 1: belief dims individually more influential than perspective "
          "dims; << 1: under-used)", flush=True)


if __name__ == "__main__":
    main()
