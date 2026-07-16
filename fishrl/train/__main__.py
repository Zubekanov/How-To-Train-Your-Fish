"""Self-play training entry point.

    python -m fishrl.train --iters 200 --encoder entity            # bounded run
    python -m fishrl.train --resume --iters 0                       # resume + run forever
    python -m fishrl.train --gpu --iters 200                        # on CUDA

Runs warmup + PPO self-play with crash-safe checkpointing to `<ckpt-dir>/latest.pt`. On
`--resume` it restores the latest checkpoint (models + optimizers + RNG + iteration counter)
and continues; `--iters 0` runs indefinitely until SIGTERM/SIGINT (graceful checkpoint).
"""
from __future__ import annotations

import argparse
import os

from fishrl.train.threads import DEFAULT_RESERVE, preconfigure

# BEFORE the imports below: BLAS/OpenMP read the *_NUM_THREADS env vars once,
# when torch loads its native libraries. fishrl.train.threads is torch-free.
_N_THREADS = preconfigure()

from fishrl.eval.metrics import collect_eval_batch, estimator_metrics, panel_winrates  # noqa: E402
from fishrl.train import checkpoint as ckpt                                            # noqa: E402
from fishrl.train.config import Config, resolve_device                                 # noqa: E402
from fishrl.train.train_loop import build_models, config_from_checkpoint, train         # noqa: E402

NETS = ("actor", "critic", "guesser", "public")


def _parse_hidden(spec: str | None) -> tuple | None:
    """'768,768,384' -> (768, 768, 384); None/'' -> None (fall back to Config.hidden)."""
    if not spec:
        return None
    return tuple(int(x) for x in spec.split(","))


def main():
    ap = argparse.ArgumentParser(description="Self-play training for fishrl.")
    ap.add_argument("--gpu", action="store_true", help="train on CUDA if available")
    ap.add_argument("--iters", type=int, default=100, help="iteration cap; <=0 runs unbounded")
    ap.add_argument("--games-per-iter", type=int, default=8)
    ap.add_argument("--collect-workers", type=int, default=Config.collect_workers,
                    help="collector worker processes per iteration (0 = serial, the "
                         "historic path; parallelism is across processes, torch pinned "
                         "to 1 thread each -- see fishrl.train.pcollect)")
    ap.add_argument("--collect-affinity", default=Config.collect_affinity,
                    help="comma-separated logical-processor indices the collector "
                         "workers are restricted to (machine flag; keeps them off "
                         "slow E-cores on hybrid CPUs -- see Config.collect_affinity)")
    ap.add_argument("--pipeline-collect", action="store_true",
                    default=Config.pipeline_collect,
                    help="REGIME: collect iteration N+1's games while updating on N's "
                         "batch (one-update-stale behavior policy; PPO's ratio absorbs "
                         "it). Requires --collect-workers > 0.")
    ap.add_argument("--warmup-games", type=int, default=64)
    ap.add_argument("--pool-frac", type=float, default=Config.pool_frac,
                    help="fraction of each iteration's games played vs a PFSP league "
                         "opponent (sampled by difficulty); 0 = pure self-play")
    ap.add_argument("--pfsp-mode", choices=["hard", "var"], default=Config.pfsp_mode,
                    help="opponent priority: 'hard' favours opponents you lose to, "
                         "'var' favours even matchups")
    ap.add_argument("--league-size", type=int, default=Config.league_size,
                    help="ring length of frozen past-self league members (0 = anchors only)")
    ap.add_argument("--scenario-pool", action="store_true", default=Config.scenarios_in_pool,
                    help="fold the scenarios into the main PFSP league (their play rate "
                         "floats with difficulty within --pool-frac) instead of the fixed "
                         "--scenario-frac carve-out; overrides --scenario-frac")
    ap.add_argument("--scenario-frac", type=float, default=Config.scenario_frac,
                    help="fraction of each iteration's games seeded from a curriculum "
                         "scenario start-state (0 = off). Mixed INTO self-play; reward stays "
                         "terminal. Judge progress on vs-heuristic WR, NOT scenario win-rate.")
    ap.add_argument("--enforce-free-attack", default=Config.enforce_free_attack,
                    action=argparse.BooleanOptionalAction,
                    help="hard rule across all training: declaring fewer than all eligible "
                         "attackers into an empty opposing board is an instant loss")
    ap.add_argument("--p1-adv-weight", type=float, default=Config.p1_adv_weight,
                    help="policy-gradient weight on p1-seat advantages (>1 steers the shared "
                         "net toward p1, the seat the vs-heuristic metric always measures; "
                         "1.0 = symmetric). The game is seat-symmetric but self-play drifts "
                         "p2-favouring -- see fishrl.eval.seat_report")
    ap.add_argument("--ent-end", type=float, default=Config.ent_end,
                    help="entropy-coefficient FLOOR held after the anneal horizon (the "
                         "sustained exploration level for an unbounded run)")
    ap.add_argument("--ent-anneal-iters", type=int, default=Config.ent_anneal_iters,
                    help="iters over which entropy anneals ent_start->ent_end when --iters<=0. "
                         "Scale to the run: 3000 (~2h) collapses exploration almost immediately "
                         "on a multi-100k-iter run.")
    ap.add_argument("--ent-reheat-period", type=int, default=Config.ent_reheat_period,
                    help="cyclical entropy RE-HEAT period in iters (0 = off). After the anneal, "
                         "the coefficient sawtooths ent_reheat_peak->ent_end every period, to "
                         "escape self-play local optima on a very long run.")
    ap.add_argument("--ent-reheat-peak", type=float, default=Config.ent_reheat_peak,
                    help="peak entropy coefficient at the start of each re-heat cycle")
    ap.add_argument("--scenario-weight", action="append", default=[], metavar="NAME=W",
                    help="override a scenario's PFSP prior weight (repeatable); e.g. "
                         "deckout=0.5 to stop a floored scenario from soaking the pool")
    ENC = ["flat", "entity", "attention"]
    ap.add_argument("--encoder", choices=ENC, default="flat",
                    help="base encoder for any net without a per-net override")
    # per-net overrides (None -> fall back to --encoder). The critic is off the
    # deployment path and batchable, so e.g. `--encoder flat --critic-encoder entity`
    # buys entity's calibration win at near-zero collection cost.
    ap.add_argument("--actor-encoder", choices=ENC, default=None)
    ap.add_argument("--critic-encoder", choices=ENC, default=None)
    ap.add_argument("--guesser-encoder", choices=ENC, default=None)
    ap.add_argument("--public-encoder", choices=ENC, default=None)
    # Model-size knobs. FRESH-only: on --resume the architecture is read from the
    # checkpoint (a resized model can't load old weights), so these are ignored there.
    ap.add_argument("--actor-hidden", default=None, metavar="A,B,C",
                    help="actor head widths, comma-separated (e.g. 768,768,384); "
                         "default = the shared `hidden` (256,256). Actor-only, so the "
                         "guesser stays small on the collection hot path.")
    ap.add_argument("--card-dim", type=int, default=Config.card_dim,
                    help="entity/attention per-card embedding width d (default 64); "
                         "sets enc_dim, so the head adapts. Ignored by flat nets.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--resume", action="store_true",
                    help="resume from <ckpt-dir>/latest.pt if it exists")
    ap.add_argument("--fresh", action="store_true",
                    help="start fresh, overwriting an existing checkpoint")
    ap.add_argument("--report-every-seconds", type=float, default=3600.0)
    ap.add_argument("--report-winrate-games", type=int, default=30,
                    help="games per opponent in each status-line win-rate panel")
    ap.add_argument("--checkpoint-every-seconds", type=float, default=900.0)
    ap.add_argument("--tick-every-seconds", type=float, default=Config.tick_every_seconds,
                    help="flush per-iteration telemetry ticks to <ckpt-dir>/ticks.json at "
                         "most this often (fishrl.serve streams them; <=0 disables)")
    ap.add_argument("--archive-every-iters", type=int, default=Config.archive_every_iters,
                    help="write a permanent archive_{it}.pt every N iterations "
                         "(never pruned, unlike step_*.pt milestones); <=0 disables")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="optional wall-clock cap (across restarts); omit for indefinite")
    ap.add_argument("--reserve-cores", type=int, default=None,
                    help="keep this many PHYSICAL cores free of training threads "
                         f"(default {DEFAULT_RESERVE} unless *_NUM_THREADS is already set "
                         "in the environment). Applied before torch loads; parsed early.")
    ap.add_argument("--claim", action="store_true",
                    help="take relay ownership of --ckpt-dir even if owner.json says "
                         "another host owns it or the lineage was exported (see "
                         "fishrl.train.ownership; normal handoffs never need this)")
    args = ap.parse_args()

    if _N_THREADS is not None:
        import torch
        torch.set_num_threads(_N_THREADS)             # env caps BLAS; this caps torch's own pool

    # Relay guards, both before any state is touched:
    #  * ownership stamp -- is it this host's turn to train this lineage?
    #  * trainer.lock -- is another trainer process live on this dir right now?
    # The lock handle is held (not closed) for the rest of the process; the OS
    # releases it on ANY exit, so `transfer export` can trust a held lock as
    # "trainer is running" and a free one as "safe to cut the zip".
    from fishrl.train import ownership
    from fishrl.train.locks import hold_lockfile
    if args.claim:
        ownership.claim(args.ckpt_dir)
        print(f"[owner] claimed {args.ckpt_dir} for '{ownership.this_host()}'", flush=True)
    refusal = ownership.check(args.ckpt_dir)
    if refusal is not None:
        raise SystemExit(f"[owner] refusing to train: {refusal}")
    _trainer_lock = hold_lockfile(ownership.trainer_lock_path(args.ckpt_dir))  # noqa: F841
    if _trainer_lock is None:
        raise SystemExit(f"[owner] refusing to train: another trainer is live on "
                         f"{args.ckpt_dir} (trainer.lock is held)")

    latest = ckpt.latest_path(args.ckpt_dir)
    have_ckpt = os.path.exists(latest)
    if have_ckpt and not args.resume and not args.fresh:
        ap.error(f"{latest} exists; pass --resume to continue or --fresh to overwrite it")
    resume = args.resume and have_ckpt
    if args.resume and not have_ckpt:
        print(f"[resume] no checkpoint at {latest}; starting fresh", flush=True)

    # --scenario-weight NAME=W overrides, merged over the Config default weights.
    scen_w = dict(Config.__dataclass_fields__["scenario_weights"].default_factory())
    for spec in args.scenario_weight:
        name, _, val = spec.partition("=")
        if not _:
            ap.error(f"--scenario-weight expects NAME=W, got {spec!r}")
        scen_w[name.strip()] = float(val)

    common = dict(device=resolve_device(args.gpu), iters=args.iters,
                  games_per_iter=args.games_per_iter, collect_workers=args.collect_workers,
                  collect_affinity=args.collect_affinity,
                  pipeline_collect=args.pipeline_collect,
                  warmup_games=args.warmup_games,
                  pool_frac=args.pool_frac, pfsp_mode=args.pfsp_mode,
                  league_size=args.league_size, scenario_frac=args.scenario_frac,
                  scenarios_in_pool=args.scenario_pool, scenario_weights=scen_w,
                  enforce_free_attack=args.enforce_free_attack,
                  p1_adv_weight=args.p1_adv_weight,
                  ent_end=args.ent_end, ent_anneal_iters=args.ent_anneal_iters,
                  ent_reheat_period=args.ent_reheat_period, ent_reheat_peak=args.ent_reheat_peak,
                  ckpt_dir=args.ckpt_dir, report_every_seconds=args.report_every_seconds,
                  report_winrate_games=args.report_winrate_games,
                  checkpoint_every_seconds=args.checkpoint_every_seconds,
                  tick_every_seconds=args.tick_every_seconds,
                  archive_every_iters=args.archive_every_iters)
    if resume:
        # architecture + seed MUST match the saved weights -> take them from the checkpoint
        # (the --encoder* / --actor-hidden / --card-dim flags are FRESH-only, ignored here).
        saved = ckpt.load_checkpoint(latest, map_location="cpu")["config"]
        cfg = config_from_checkpoint(saved, **common)
    else:
        per_net = {f"{n}_encoder": getattr(args, f"{n}_encoder")
                   for n in NETS if getattr(args, f"{n}_encoder") is not None}
        cfg = Config(encoder=args.encoder, seed=args.seed,
                     actor_hidden=_parse_hidden(args.actor_hidden),
                     card_dim=args.card_dim, **per_net, **common)

    encs = {n: cfg.enc_for(n) for n in NETS}
    cap = "unbounded" if cfg.iters <= 0 else cfg.iters
    print(f"device: {cfg.device} | encoders: {encs} | iters: {cap} | "
          f"actor_hidden={cfg.head_hidden('actor')} card_dim={cfg.card_dim} | "
          f"pool: {cfg.pool_frac:.2f} (pfsp={cfg.pfsp_mode}, league={cfg.league_size}) | "
          f"p1_adv={cfg.p1_adv_weight:.2f} ent_end={cfg.ent_end:.3f} | "
          f"resume: {resume} | ckpt: {latest}", flush=True)

    from fishrl.train.keepawake import keep_awake
    keep_awake("training")                            # Windows: no idle-sleep mid-run

    max_seconds = args.max_hours * 3600.0 if args.max_hours else None
    models = train(cfg, build_models(cfg), log=lambda s: print(s, flush=True),
                   max_seconds=max_seconds,
                   resume_path=latest if resume else None, checkpoint_path=latest)
    print(f"saved checkpoint -> {latest}", flush=True)

    # higher-game-count final readout for bounded runs (the in-loop status lines use fewer
    # games each). Skipped for the unbounded service, which exits via signal and shouldn't
    # spend tens of seconds on eval during shutdown.
    if cfg.iters > 0:
        # thread the run's belief/decision settings through — a belief-off run
        # evaluated with belief ON feeds live guesser output to an actor trained on zeros
        print("final win-rates:",
              panel_winrates(models, n_games=100, use_belief=cfg.use_belief,
                             max_decisions=cfg.max_decisions), flush=True)
        print("estimators:", estimator_metrics(models, collect_eval_batch(models, n_games=8)), flush=True)


if __name__ == "__main__":
    main()
