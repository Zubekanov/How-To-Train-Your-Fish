"""Encoder A/B on the privileged critic: flat vs entity vs attention.

    python -m fishrl.eval.ab_encoder --gpu --seeds 3 --games 120
    python -m fishrl.eval.ab_encoder --data onpolicy --onpolicy-iters 20   # on-policy states
    python -m fishrl.eval.ab_encoder --learning-curve                      # Brier vs data budget

WHAT THIS MEASURES (and the traps it is built to avoid)
-------------------------------------------------------
The critic predicts P(p1 wins) from god-state; lower held-out Brier = better-
calibrated value head = lower-variance advantages. But a naive single-batch Brier
ranking is *not* a result, for three reasons this harness addresses:

1. Effective sample size is the number of GAMES, not transitions. Transitions
   within a game are heavily autocorrelated (one shuffle, one trajectory, a few
   swing moments decide the label), so 2k transitions from 6 games carry ~6 games'
   worth of independent signal. We therefore collect many games per split and
   report Brier/accuracy as **mean ± std across independent seeds** (each seed =
   fresh data draw + fresh critic init). A ranking without that error bar is noise.

2. CPU throughput is the co-equal axis, and it is the one that actually decides the
   encoder. Rollout collection is CPU-bound on the (pure-Python) engine and the
   actors run on a CPU server, so attention over ~118 entities can win on Brier yet
   lose the run by halving samples/sec. We report **CPU milliseconds per decision**
   (guesser + actor + critic, batch=1, on CPU) alongside Brier. Parameter count is
   reported but de-emphasised — VRAM is cheap; CPU latency is not.

3. Random self-play is the WRONG distribution. The critic ultimately sees on-policy
   states from a partially-trained agent, not random-vs-random games. ``--data
   random`` (default, cheap) only probes capacity/calibration on an off-distribution
   set; ``--data onpolicy`` trains a shared reference agent for ``--onpolicy-iters``
   and collects from it, which is the distribution the decision should rest on.

The decision metric is Brier-per-CPU-ms with an error bar, on on-policy data — not
any single number below.
"""
from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import torch

from fishrl.data.features import GOD_DIM
from fishrl.models.estimators import PrivilegedCritic
from fishrl.models.guesser import HandGuesser
from fishrl.models.policy import ACTOR_IN, MaskedActor
from fishrl.obs import vocab as V
from fishrl.obs.encoder import OBS_DIM
from fishrl.spaces import action_space as A
from fishrl.train.belief_env import BeliefAugmentedEnv
from fishrl.train.collector import actor_act_fn, collect_games, random_act_fn
from fishrl.train.config import Config, resolve_device
from fishrl.train.losses import outcome_bce

ENCODERS = ("flat", "entity", "attention")


def _log(msg: str) -> None:
    print(msg, flush=True)          # flush so progress shows live even when piped


def _free(model) -> None:
    """Drop a critic and release its CUDA blocks so 3 encoders x N seeds of fits
    don't accumulate / fragment the allocator across the run."""
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── data collection ───────────────────────────────────────────────────────────
def _collect(label, seed, n_games, act_fn, guesser, max_decisions=2000):
    """Collect ``n_games`` and return the decided (god, y) plus per-decision inputs
    for the CPU-latency probe (x_act, persp, prev_guess)."""
    benv = BeliefAugmentedEnv(guesser, max_decisions=max_decisions)
    buf = collect_games(benv, act_fn, n_games, seed, critic=None, max_decisions=max_decisions)
    b = buf.compute(0.997, 0.95)
    keep = b["valid"] > 0
    out = {k: b[k][keep] for k in ("god", "y_p1", "x_act", "persp", "prev_guess")}
    _log(f"  [collect] {label}: {n_games} games -> {int(keep.sum())} decided "
         f"transitions (p1 win rate {float(out['y_p1'].mean()):.2f})")
    return out


def _reference_agent(iters, seed, device):
    """Train a shared, encoder-agnostic reference agent so on-policy data reflects a
    partially-trained policy's state distribution (not random self-play)."""
    from fishrl.train.train_loop import build_models, train
    _log(f"[onpolicy] training reference agent (flat, {iters} iters, seed {seed})...")
    cfg = Config(device=device, iters=iters, encoder="flat", seed=seed,
                 games_per_iter=8, warmup_games=32)
    m = train(cfg, build_models(cfg), log=lambda s: _log(f"  {s}"))
    return actor_act_fn(m.actor), m.guesser


# ── fit / eval ────────────────────────────────────────────────────────────────
# Data (god, y) stays on CPU; minibatches are moved to the device. On-policy splits
# are ~40x larger than random ones (>100k transitions), and a full-batch forward
# through the attention encoder materializes an O(B * tokens^2) attention tensor —
# minibatching is what keeps that off the 12 GiB ceiling.
def _fit(encoder, gx, gy, device, steps, batch=4096):
    c = PrivilegedCritic(encoder=encoder).to(device)
    opt = torch.optim.Adam(c.parameters(), lr=2e-3)
    M = gx.shape[0]
    rng = np.random.default_rng(0)
    for _ in range(steps):
        idx = torch.from_numpy(rng.integers(0, M, size=min(batch, M)))
        xb, yb = gx[idx].to(device), gy[idx].to(device)
        loss = outcome_bce(c(xb), yb, torch.ones_like(yb))
        opt.zero_grad(); loss.backward(); opt.step()
    return c


def _brier_acc(c, hx, hy, device, batch=4096):
    se = correct = 0.0
    n = hx.shape[0]
    with torch.no_grad():
        for s in range(0, n, batch):
            p = torch.sigmoid(c(hx[s:s + batch].to(device))).cpu()
            yb = hy[s:s + batch]
            se += float(((p - yb) ** 2).sum())
            correct += float(((p > 0.5).float() == yb).float().sum())
    return se / n, correct / n


def _cpu_ms_per_decision(encoder, sample, reps=300):
    """Wall-clock ms for one decision's forward passes (guesser + actor + critic,
    batch=1) on CPU — the deciding cost axis, since collection/actors run on CPU."""
    torch.manual_seed(0)
    actor = MaskedActor(encoder=encoder).eval()
    critic = PrivilegedCritic(encoder=encoder).eval()
    guesser = HandGuesser(encoder=encoder).eval()
    xs = sample["x_act"].cpu(); gs = sample["god"].cpu()
    ps = sample["persp"].cpu(); pg = sample["prev_guess"].cpu()
    n = xs.shape[0]
    mask = torch.ones(1, A.N)

    def one(i):
        guesser(ps[i:i + 1], pg[i:i + 1])         # belief (runs first, in env.observe)
        actor.log_probs(xs[i:i + 1], mask)        # policy
        critic.p1_winprob(gs[i:i + 1])            # value

    with torch.no_grad():
        for i in range(min(5, n)):                # warm caches / lazy init
            one(i)
        t0 = time.perf_counter()
        for r in range(reps):
            one(r % n)
        dt = time.perf_counter() - t0
    return 1000.0 * dt / reps


# ── learning curve (replaces the meaningless "can it overfit 3k points" test) ──
def _learning_curve(train, holdout, device, steps, fractions=(0.1, 0.25, 0.5, 1.0)):
    gx, gy = train["god"], train["y_p1"]                 # CPU; _fit moves minibatches
    hx, hy = holdout["god"], holdout["y_p1"]
    M = gx.shape[0]
    _log("\n=== generalization vs data budget (held-out Brier; lower better) ===")
    header = "frac     n_train  " + "  ".join(f"{e:>9s}" for e in ENCODERS)
    _log(header)
    for f in fractions:
        n = max(int(M * f), 1)
        row = []
        for enc in ENCODERS:
            c = _fit(enc, gx[:n], gy[:n], device, steps)
            brier, _ = _brier_acc(c, hx, hy, device)
            row.append(brier)
            _free(c)
        _log(f"{f:<8.2f} {n:<8d} " + "  ".join(f"{b:9.4f}" for b in row))


# ── main ──────────────────────────────────────────────────────────────────────
def _agg(xs):
    m = statistics.mean(xs)
    s = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return m, s


def main():
    ap = argparse.ArgumentParser(description="Encoder A/B on the privileged critic.")
    ap.add_argument("--gpu", action="store_true", help="fit critics on CUDA if available")
    ap.add_argument("--encoders", nargs="+", default=list(ENCODERS))
    ap.add_argument("--seeds", type=int, default=3, help="independent data+init draws (error bar)")
    ap.add_argument("--games", type=int, default=120, help="games per train split")
    ap.add_argument("--holdout-games", type=int, default=120, help="games per holdout split")
    ap.add_argument("--steps", type=int, default=300, help="critic fit steps")
    ap.add_argument("--reps", type=int, default=300, help="timing reps for CPU ms/decision")
    ap.add_argument("--data", choices=["random", "onpolicy"], default="random",
                    help="random self-play (cheap, off-distribution) or on-policy from a "
                         "partially-trained reference agent (the distribution that matters)")
    ap.add_argument("--onpolicy-iters", type=int, default=20, help="reference-agent train iters")
    ap.add_argument("--learning-curve", action="store_true",
                    help="also report held-out Brier vs train-data budget (seed 0)")
    args = ap.parse_args()
    device = resolve_device(args.gpu)
    encoders = [e for e in args.encoders if e in ENCODERS]

    _log(f"=== encoder A/B {encoders} | device {device} | data={args.data} ===")
    _log(f"    {args.seeds} seeds x ({args.games} train / {args.holdout_games} holdout) games"
         f"; fit {args.steps} steps")
    if args.data == "random":
        _log("    NOTE: random self-play is OFF the on-policy distribution -- these numbers")
        _log("          bound capacity/calibration only. Use --data onpolicy to decide.")

    # Shared reference agent for on-policy data (one agent; seeds vary the trajectories).
    ref_act, ref_guesser = (None, None)
    if args.data == "onpolicy":
        ref_act, ref_guesser = _reference_agent(args.onpolicy_iters, seed=0, device=device)

    def make_act_and_guesser(seed):
        if args.data == "onpolicy":
            return ref_act, ref_guesser
        return random_act_fn(np.random.default_rng(seed)), HandGuesser()

    briers = {e: [] for e in encoders}
    accs = {e: [] for e in encoders}
    last_holdout = None
    for si in range(args.seeds):
        _log(f"\n[seed {si}] collecting...")
        act, guesser = make_act_and_guesser(seed=si)
        train = _collect("train", seed=si * 7919, n_games=args.games, act_fn=act, guesser=guesser)
        hold = _collect("holdout", seed=si * 7919 + 104729, n_games=args.holdout_games,
                        act_fn=act, guesser=guesser)
        last_holdout = hold
        gx, gy = train["god"], train["y_p1"]             # CPU; _fit moves minibatches
        hx, hy = hold["god"], hold["y_p1"]
        for enc in encoders:
            c = _fit(enc, gx, gy, device, args.steps)
            brier, acc = _brier_acc(c, hx, hy, device)
            briers[enc].append(brier); accs[enc].append(acc)
            _log(f"  [seed {si}] {enc:9s} holdout_brier {brier:.4f}  acc {acc:.4f}")
            _free(c)

    # CPU latency is seed-independent (architecture, not data) — measure once.
    _log("\n[timing] CPU ms/decision (guesser+actor+critic, batch=1)...")
    ms = {e: _cpu_ms_per_decision(e, last_holdout, reps=args.reps) for e in encoders}
    params = {e: sum(p.numel() for p in PrivilegedCritic(encoder=e).parameters()) for e in encoders}

    _log("\n=== summary (mean +/- std across seeds) ===")
    _log(f"{'encoder':9s} {'holdout_brier':>16s} {'holdout_acc':>16s} "
         f"{'cpu_ms/dec':>11s} {'brier*ms':>9s} {'critic_params':>14s}")
    for enc in encoders:
        bm, bs = _agg(briers[enc]); am, asd = _agg(accs[enc])
        _log(f"{enc:9s} {bm:7.4f} +/-{bs:6.4f} {am:7.4f} +/-{asd:6.4f} "
             f"{ms[enc]:11.3f} {bm * ms[enc]:9.4f} {params[enc]:14,d}")
    _log("\nDecision metric = Brier*ms (lower better): calibration weighted by CPU cost.")
    _log("With overlapping +/-std the encoders are statistically tied -- add seeds/games.")

    if args.learning_curve:
        _learning_curve(train, hold, device, args.steps)


if __name__ == "__main__":
    main()
