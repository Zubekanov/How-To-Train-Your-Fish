"""Fixed-wall-clock actor head-to-head: the experiment that actually decides the
actor's encoder (and, with the belief-off arm, whether the guesser earns its keep).

Three arms, each trained for the SAME wall-clock budget per seed (fair comparison
given entity is slightly slower per iter), all with the entity critic (free now that
the value pass is batched) and a flat guesser:

    flat        : flat actor,   belief on
    entity      : entity actor, belief on
    flat-nobel  : flat actor,   belief OFF (controlled guesser ablation)

Decided on win-rate two independent ways:
  * vs the fixed heuristic AI (absolute skill anchor), and
  * direct round-robin matches between the trained arms (relative), seat-balanced.

    python -m fishrl.eval.actor_headtohead --gpu --minutes 60 --seeds 2
    python -m fishrl.eval.actor_headtohead --gpu --minutes 1 --seeds 1   # smoke
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from fishrl.env.aec_env import FishAEC
from fishrl.eval.metrics import (
    act_from_actor, winrate_vs_attacker, winrate_vs_heuristic, winrate_vs_random)
from fishrl.models import device_of
from fishrl.obs import vocab as V
from fishrl.train.config import Config, resolve_device
from fishrl.train.train_loop import build_models, train

ARMS = [
    {"name": "flat", "actor_encoder": "flat", "use_belief": True},
    {"name": "entity", "actor_encoder": "entity", "use_belief": True},
    {"name": "flat-nobel", "actor_encoder": "flat", "use_belief": False},
]
_Z = np.zeros(V.N_NAMES, dtype=np.float32)


def _log(msg):
    print(msg, flush=True)


# ── direct A-vs-B match (both seats are trained policies, each with its OWN belief) ──
def _aug_obs(models, use_belief, base, prev):
    """Build a seat's belief-augmented obs from ITS OWN guesser + carried prev guess."""
    persp = base["observation"]
    if use_belief:
        gdev = device_of(models.guesser)
        with torch.no_grad():
            guess = models.guesser(
                torch.as_tensor(persp, dtype=torch.float32).unsqueeze(0).to(gdev),
                torch.as_tensor(prev, dtype=torch.float32).unsqueeze(0).to(gdev),
            ).squeeze(0).cpu().numpy().astype(np.float32)
    else:
        guess = _Z
    return ({"observation": np.concatenate([persp, guess]).astype(np.float32),
             "action_mask": base["action_mask"]}, guess)


def _match(mA, ubA, mB, ubB, n_games, base_seed, max_decisions=2000):
    """A as p1, B as p2. Returns (A_wins, B_wins, draws)."""
    aw = bw = draw = 0
    for gi in range(n_games):
        env = FishAEC(max_decisions=max_decisions)
        env.reset(seed=base_seed + gi)
        prev = {"p1": _Z.copy(), "p2": _Z.copy()}
        i = 0
        while env.agents and i < max_decisions * 6:
            i += 1
            seat = env.agent_selection
            if env.terminations[seat] or env.truncations[seat]:
                env.step(None)
                continue
            mm, ub = (mA, ubA) if seat == "p1" else (mB, ubB)
            aug, g = _aug_obs(mm, ub, env.observe(seat), prev[seat])
            prev[seat] = g
            env.step(act_from_actor(mm.actor, aug))
        w = env.g.result.get("winner")
        aw, bw, draw = aw + (w == "p1"), bw + (w == "p2"), draw + (w not in ("p1", "p2"))
    return aw, bw, draw


def _head2head(X, Y, n_games, seed):
    """Seat-balanced: half the games with X as p1, half with Y as p1. Returns X's
    win-rate over decided games and the raw (Xw, Yw, draws)."""
    half = n_games // 2
    xw1, yw1, d1 = _match(X[0], X[1], Y[0], Y[1], half, seed)
    yw2, xw2, d2 = _match(Y[0], Y[1], X[0], X[1], half, seed + 10_000)
    xw, yw, dr = xw1 + xw2, yw1 + yw2, d1 + d2
    dec = xw + yw
    return (xw / dec if dec else 0.5), (xw, yw, dr)


def _train_arm(arm, seed, device, minutes, ckpt_dir):
    cfg = Config(device=device, encoder="flat", actor_encoder=arm["actor_encoder"],
                 use_belief=arm["use_belief"], critic_encoder="entity",
                 iters=1_000_000, seed=seed, games_per_iter=8, warmup_games=32)
    _log(f"  [train] arm={arm['name']} seed={seed} budget={minutes}min ...")
    m = train(cfg, build_models(cfg), log=lambda s: _log(f"    {s}"), max_seconds=minutes * 60)
    m.actor.eval()
    path = os.path.join(ckpt_dir, f"h2h_{arm['name']}_s{seed}.pt")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({"arm": arm, "seed": seed, "actor": m.actor.state_dict(),
                "guesser": m.guesser.state_dict()}, path)
    return m


def main():
    ap = argparse.ArgumentParser(description="Fixed-wall-clock actor head-to-head.")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--minutes", type=float, default=60.0, help="training budget per arm")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--heuristic-games", type=int, default=100)
    ap.add_argument("--match-games", type=int, default=100, help="per arm-pair, seat-balanced")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--out", default="h2h_results.json")
    args = ap.parse_args()
    device = resolve_device(args.gpu)

    _log(f"=== actor head-to-head | {device} | {args.minutes}min/arm | {args.seeds} seeds ===")
    _log(f"    arms: {[a['name'] for a in ARMS]}; entity critic; flat guesser")
    names = [a["name"] for a in ARMS]
    vs_attk = {n: [] for n in names}     # PRIMARY transitive anchor (in the agents' band)
    vs_heur = {n: [] for n in names}     # ceiling milestone (heuristic saturates ~0 early)
    vs_rand = {n: [] for n in names}     # floor sanity ("did it learn at all")
    pairs = [("flat", "entity"), ("flat", "flat-nobel"), ("entity", "flat-nobel")]
    h2h = {f"{x} vs {y}": [] for x, y in pairs}      # list of X-winrate per seed
    h2h_raw = {f"{x} vs {y}": [] for x, y in pairs}

    for seed in range(args.seeds):
        _log(f"\n[seed {seed}] training {len(ARMS)} arms ...")
        trained = {}
        for arm in ARMS:
            try:
                m = _train_arm(arm, seed, device, args.minutes, args.ckpt_dir)
            except Exception as e:                    # don't lose other arms on one failure
                _log(f"  [ERROR] arm {arm['name']} seed {seed}: {e!r}")
                continue
            trained[arm["name"]] = (m, arm["use_belief"])
            wa = winrate_vs_attacker(m, n_games=args.heuristic_games, seed=700_000,
                                     use_belief=arm["use_belief"])
            wh = winrate_vs_heuristic(m, n_games=args.heuristic_games, seed=900_000,
                                      use_belief=arm["use_belief"])
            wr = winrate_vs_random(m, n_games=args.heuristic_games, seed=800_000,
                                   use_belief=arm["use_belief"])
            vs_attk[arm["name"]].append(wa)
            vs_heur[arm["name"]].append(wh)
            vs_rand[arm["name"]].append(wr)
            _log(f"  [eval] {arm['name']:11s} vs_attacker {wa:.3f}  "
                 f"vs_heuristic {wh:.3f}  vs_random {wr:.3f}")

        for x, y in pairs:                            # seed-matched round-robin
            if x in trained and y in trained:
                wrx, raw = _head2head(trained[x], trained[y], args.match_games, seed=600_000 + seed)
                h2h[f"{x} vs {y}"].append(wrx)
                h2h_raw[f"{x} vs {y}"].append(raw)
                _log(f"  [match] {x} vs {y}: {x} win-rate {wrx:.3f}  raw(Xw,Yw,D)={raw}")
        del trained
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def ms(xs):
        a = np.array(xs, dtype=np.float64)
        return (float(a.mean()), float(a.std())) if len(a) else (float("nan"), float("nan"))

    _log("\n=== SUMMARY (win-rate mean +/- std over seeds) ===")
    _log("  vs_attacker is the PRIMARY decider (transitive, in-band); vs_heuristic is a")
    _log("  ceiling milestone; vs_random a floor check; head-to-head is a cycle-detector only.")
    _log(f"  {'arm':11s} {'vs_attacker':>15s} {'vs_heuristic':>15s} {'vs_random':>15s}")
    for n in names:
        ma, sa = ms(vs_attk[n])
        mh, sh = ms(vs_heur[n])
        mr, sr = ms(vs_rand[n])
        _log(f"  {n:11s} {ma:.3f}+/-{sa:.3f}  {mh:.3f}+/-{sh:.3f}  {mr:.3f}+/-{sr:.3f}")
    _log("\n  direct head-to-head (cycle-detector / tiebreaker, NOT the decider):")
    for k, xs in h2h.items():
        mh, sh = ms(xs)
        _log(f"  {k:24s} {mh:.3f}+/-{sh:.3f}   raw_per_seed={h2h_raw[k]}")

    out = {"minutes": args.minutes, "seeds": args.seeds, "device": device,
           "vs_attacker": vs_attk, "vs_heuristic": vs_heur, "vs_random": vs_rand,
           "head2head": h2h, "head2head_raw": h2h_raw}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    _log(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
