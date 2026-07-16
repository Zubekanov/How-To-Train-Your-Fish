"""Frozen-baseline A/B for the scenario curriculum — the experiment all of this is for.

From ONE frozen checkpoint, run two training branches for an equal budget that
differ ONLY in `scenario_frac` (control 0.0 vs treatment >0), then score both on
the gate metric: full-game win-rate vs the heuristic. This is the disciplined test
— a fixed baseline (the live trainer keeps drifting), one variable, the win-rate as
the judge (never the scenario win-rate).

    python -m fishrl.eval.scenario_ab --base-ckpt checkpoints/frozen.pt \
        --out-dir /tmp/scn_ab --iters 400 --scenario-frac 0.3 --eval-games 100

Freeze the base first (the service rewrites latest.pt): cp checkpoints/latest.pt
checkpoints/frozen.pt, then point --base-ckpt at the copy.
"""
from __future__ import annotations

import argparse
import os
import shutil

from fishrl.eval.metrics import winrate_vs_heuristic
from fishrl.train import checkpoint as ckpt
from fishrl.train.config import Config, resolve_device
from fishrl.train.train_loop import Models, build_models, config_from_checkpoint, train

NETS = ("actor", "critic", "guesser", "public")


def _cfg_from_base(base: str, device: str, **over) -> Config:
    """Build a Config matching the base checkpoint's FULL architecture (encoders/seed/
    head widths/card_dim), overriding the experiment knobs. Mirrors the resume path."""
    saved = ckpt.load_checkpoint(base, map_location="cpu")["config"]
    return config_from_checkpoint(saved, device=device, **over)


def _run_branch(name, base, out_dir, device, iters, scenario_frac,
                report_every_seconds) -> Models:
    bdir = os.path.join(out_dir, name)
    os.makedirs(bdir, exist_ok=True)
    latest = ckpt.latest_path(bdir)
    shutil.copy2(base, latest)                 # each branch resumes from the SAME frozen base
    cfg = _cfg_from_base(base, device, iters=iters, scenario_frac=scenario_frac,
                         ckpt_dir=bdir, report_winrate_games=0,
                         report_every_seconds=report_every_seconds,
                         checkpoint_every_seconds=report_every_seconds)
    models = build_models(cfg)
    print(f"[{name}] training {iters} iters from {base} (scenario_frac={scenario_frac})", flush=True)
    train(cfg, models, log=lambda s: print(f"[{name}] {s}", flush=True),
          resume_path=latest, checkpoint_path=latest)
    return models


def main():
    ap = argparse.ArgumentParser(description="Frozen-baseline A/B for the scenario curriculum.")
    ap.add_argument("--base-ckpt", required=True, help="frozen checkpoint both branches resume from")
    ap.add_argument("--out-dir", required=True, help="dir for control/ and treatment/ branches")
    ap.add_argument("--iters", type=int, default=400, help="training iters per branch")
    ap.add_argument("--scenario-frac", type=float, default=0.3, help="treatment scenario fraction")
    ap.add_argument("--eval-games", type=int, default=100, help="vs-heuristic games per branch")
    ap.add_argument("--report-every-seconds", type=float, default=1800.0)
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.base_ckpt):
        ap.error(f"base checkpoint not found: {args.base_ckpt}")
    device = resolve_device(args.gpu)
    os.makedirs(args.out_dir, exist_ok=True)
    eseed = 900_000                              # fixed eval seed -> comparable across branches
    # Evaluate with the settings the base checkpoint resolves to (a belief-off base must
    # be scored belief-off — defaulting use_belief=True would feed a stale guesser).
    ecfg = _cfg_from_base(args.base_ckpt, device)
    ekw = dict(n_games=args.eval_games, seed=eseed,
               max_decisions=ecfg.max_decisions, use_belief=ecfg.use_belief)

    control = _run_branch("control", args.base_ckpt, args.out_dir, device,
                          args.iters, 0.0, args.report_every_seconds)
    wr_ctrl = winrate_vs_heuristic(control, **ekw)
    print(f"[control] vs-heuristic WR = {wr_ctrl:.3f} (n={args.eval_games})", flush=True)

    treatment = _run_branch("treatment", args.base_ckpt, args.out_dir, device,
                            args.iters, args.scenario_frac, args.report_every_seconds)
    wr_treat = winrate_vs_heuristic(treatment, **ekw)
    print(f"[treatment] vs-heuristic WR = {wr_treat:.3f} (n={args.eval_games})", flush=True)

    print("\n==== SCENARIO A/B (vs-heuristic win-rate is the gate) ====")
    print(f"  base:           {args.base_ckpt}")
    print(f"  iters/branch:   {args.iters}    eval games: {args.eval_games}")
    print(f"  control   (scenario_frac=0.0): {wr_ctrl:.3f}")
    print(f"  treatment (scenario_frac={args.scenario_frac}): {wr_treat:.3f}")
    print(f"  delta:          {wr_treat - wr_ctrl:+.3f}")
    print("  (note: one seed/run; repeat or raise eval-games + iters before trusting a small delta)")


if __name__ == "__main__":
    main()
