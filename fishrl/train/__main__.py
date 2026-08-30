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
import time

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
    ap.add_argument("--minibatch", type=int, default=Config.minibatch,
                    help="PPO minibatch size (default 256). On a small GPU the update is "
                         "launch-bound, so a larger minibatch cuts kernel-launch overhead -- "
                         "pair it with a larger --games-per-iter so the step count stays sane. "
                         "Changes gradient noise: a training-dynamics knob, not free.")
    ap.add_argument("--collect-workers", type=int, default=Config.collect_workers,
                    help="collector worker processes per iteration (0 = serial, the "
                         "historic path; parallelism is across processes, torch pinned "
                         "to 1 thread each -- see fishrl.train.pcollect)")
    ap.add_argument("--collect-affinity", default=Config.collect_affinity,
                    help="comma-separated logical-processor indices the collector "
                         "workers are restricted to (machine flag; keeps them off "
                         "slow E-cores on hybrid CPUs -- see Config.collect_affinity)")
    ap.add_argument("--infer-server", action="store_true",
                    default=Config.infer_server,
                    help="MACHINE: batch all collector workers' learner forwards on "
                         "one server process on the training device instead of "
                         "batch-1 per worker (DRAM-bound at 8 workers). Wall-clock "
                         "only -- same weights per iteration, worker-side sampling, "
                         "local fallback on failure. Requires --collect-workers > 0.")
    ap.add_argument("--pipeline-collect", action="store_true",
                    default=Config.pipeline_collect,
                    help="REGIME: collect iteration N+1's games while updating on N's "
                         "batch (one-update-stale behavior policy; PPO's ratio absorbs "
                         "it). Requires --collect-workers > 0.")
    ap.add_argument("--collect-stream", action="store_true", default=Config.collect_stream,
                    help="REGIME: one task per game, games consumed in completion order "
                         "(no per-iteration straggler wait); implies --pipeline-collect.")
    ap.add_argument("--stream-depth", type=int, default=Config.stream_depth,
                    help="iteration-sized game sets kept in flight under --collect-stream.")
    ap.add_argument("--warmup-games", type=int, default=64)
    ap.add_argument("--pool-frac", type=float, default=Config.pool_frac,
                    help="fraction of each iteration's games played vs a PFSP league "
                         "opponent (sampled by difficulty); 0 = pure self-play")
    ap.add_argument("--pfsp-mode", choices=["hard", "var"], default=Config.pfsp_mode,
                    help="opponent priority: 'hard' favours opponents you lose to, "
                         "'var' favours even matchups")
    ap.add_argument("--league-size", type=int, default=Config.league_size,
                    help="ring length of frozen past-self league members (0 = anchors only)")
    ap.add_argument("--league-every", type=int, default=Config.league_every,
                    help="take a past-self snapshot every Nth status report (default 1 = "
                         "every report). At 1 the ring spans league_size reports (~2h), so "
                         "past selves are near-clones; N>1 widens the population's time "
                         "horizon so PFSP hard-mode can punish cycling. Resume-tunable.")
    ap.add_argument("--scenario-pool", action="store_true", default=Config.scenarios_in_pool,
                    help="fold the scenarios into the main PFSP league (their play rate "
                         "floats with difficulty within --pool-frac) instead of the fixed "
                         "--scenario-frac carve-out; overrides --scenario-frac")
    ap.add_argument("--scenario-frac", type=float, default=Config.scenario_frac,
                    help="fraction of each iteration's games seeded from a curriculum "
                         "scenario start-state (0 = off). Mixed INTO self-play; reward stays "
                         "terminal. Judge progress on vs-heuristic WR, NOT scenario win-rate.")
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
    ap.add_argument("--freeze-actor-iters", type=int, default=Config.freeze_actor_iters,
                    help="BC handoff: while it < N the PPO update trains the CRITIC only, "
                         "calibrating advantages on-policy before the cloned actor moves "
                         "(0 = off; compared against the absolute iteration counter)")
    ap.add_argument("--kl-teacher-coef", type=float, default=Config.kl_teacher_coef,
                    help="BC handoff: KL(teacher||pi) penalty at it=0, teacher = the actor "
                         "as loaded at run start (the BC clone on a resume); 0 = off")
    ap.add_argument("--kl-teacher-iters", type=int, default=Config.kl_teacher_iters,
                    help="BC handoff: anneal the KL-to-teacher coefficient linearly to zero "
                         "over this many iterations")
    ap.add_argument("--scenario-weight", action="append", default=[], metavar="NAME=W",
                    help="override a scenario's PFSP prior weight (repeatable); e.g. "
                         "deckout=0.5 to stop a floored scenario from soaking the pool")
    ap.add_argument("--scenario-selfplay", type=float, default=Config.scenario_selfplay_frac,
                    help="probability a scenario game plays BOTH seats with the current "
                         "policy instead of the v1.3 engine seat (both seats' transitions "
                         "train; keep <1 — the script seat is the curriculum's external "
                         "pressure). Runtime knob, resume-tunable.")
    ap.add_argument("--scenario-boost", type=float, default=Config.scenario_boost,
                    help="scenario-pool mode: flat multiplier on every scenario member's "
                         "PFSP weight (scenario episodes are short, so a >1 boost raises "
                         "their game count at sub-proportional wall-clock cost; 1 = off)")
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
    ap.add_argument("--train-public", default=Config.train_public,
                    action=argparse.BooleanOptionalAction,
                    help="train the DIAGNOSTIC public estimator (default on). --no-train-public "
                         "skips its per-decision encode + aux training for a small collection "
                         "speedup, dropping the pub calibration telemetry.")
    # v3 architecture knobs. FRESH-only like --encoder*: on --resume they come
    # exclusively from the checkpoint, so a shared train.args can't corrupt a
    # legacy resume (the ODROID/v2 safety rule).
    ap.add_argument("--belief", choices=["guesser", "bookkeeper", "none"], default=None,
                    help="what fills the actor's 20-dim belief slot: the learned guesser "
                         "(legacy), the zero-parameter analytic bookkeeper (v3), or zeros. "
                         "FRESH-only; a resume reads the checkpoint's mode.")
    ap.add_argument("--critic-view", choices=["god", "public", "hands"], default=None,
                    help="the PPO critic's input: privileged god features (legacy) or the "
                         "public mutual-knowledge features (v3; drops the separate public "
                         "estimator). FRESH-only; a resume reads the checkpoint's view.")
    ap.add_argument("--text-change", choices=["full", "guided", "auto"], default=None,
                    help="choose_text_change action space: full 25-way (legacy), guided "
                         "{EFFECT, NO-OP}, or auto (EFFECT forced). FRESH-only.")
    ap.add_argument("--obs-tgt", action="store_true", default=None,
                    help="append the choose_targets candidate pack (per legal-list index: class/"
                         "controller/tapped/combat/targeted-by-stack/text-altered features of the "
                         "card that PICK index resolves to) + stack-target sight to the top FOUR "
                         "objects (features.TGT_DIM). FRESH-only; a live run gets it via "
                         "python -m fishrl.train.widen_tgt. Needs --obs-ctx.")
    ap.add_argument("--obs-ctx", action="store_true", default=None,
                    help="append the decision-context pack (stack targets / search eligibility / "
                         "builder arrangements / blocker focus + critic step/pending/combat/pay) "
                         "to the actor input + hands critic (features.CTX_DIM). Also gates the "
                         "name-sorted search/graveyard PICK remap. FRESH-only; a live run gets it "
                         "via python -m fishrl.train.widen_ctx. Needs --obs-split.")
    ap.add_argument("--obs-split", action="store_true", default=None,
                    help="append the FoF split-context block to the actor input + hands critic "
                         "(features.SPLIT_DIM). FRESH-only; a live run gets it via "
                         "python -m fishrl.train.widen_split. Needs --obs-counts.")
    ap.add_argument("--obs-counts", action="store_true", default=None,
                    help="append the per-name count block to the actor input + hands critic "
                         "(features.COUNT_DIM). FRESH-only; a live run gets it via "
                         "fishrl.train.widen_counts (in place).")
    # runtime v3 knobs (resume-tunable)
    ap.add_argument("--critic-deckout-aux", type=float, default=None,
                    help="weight of the public critic's deckout-winner auxiliary loss "
                         "(parity credit; 0 disables the gradient, the head remains). "
                         "Resume-tunable; default = the checkpoint's value (fresh: 0).")
    ap.add_argument("--critic-epochs", type=int, default=Config.critic_epochs,
                    help="PPO epochs on which the critic gets gradient (actor: all). 0 = all "
                         "(legacy); 1 = fit each batch once (curbs per-batch memorisation). "
                         "Resume-tunable.")
    ap.add_argument("--critic-consistency", type=float, default=Config.critic_consistency,
                    help="critic temporal-consistency penalty weight on det_next pairs "
                         "(same seat, same turn, deterministic transition). Resume-tunable.")
    ap.add_argument("--critic-td-mix", type=float, default=Config.critic_td_mix,
                    help="critic target blend b: (1-b)*terminal outcome + b*lambda-return "
                         "(p1 frame). 0 = legacy outcome-only BCE. Resume-tunable.")
    ap.add_argument("--ent-start", type=float, default=Config.ent_start,
                    help="entropy coefficient at the START of the anneal (default 0.02). "
                         "Lower it for a warm-started policy that must not be re-inflated.")
    ap.add_argument("--lr-ppo", type=float, default=Config.lr_ppo,
                    help="Adam LR for the shared actor+critic PPO optimizer (default 3e-4). "
                         "Resume-tunable: on --resume this value is re-asserted onto the "
                         "loaded optimizer state, which otherwise pins the launch-time LR "
                         "forever. The late-run plateau lever: with the GNS estimator "
                         "saturated (gns_b=nan, B_crit >> batch) updates are noise-"
                         "dominated and halving the LR ~doubles the effective batch.")
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

    if args.gpu:
        import torch
        # TF32 tensor-core matmul path on Ampere+ (the 3060): ~free for RL and faster for
        # the head GEMMs. Harmless on CPU / older GPUs (the flags are simply ignored).
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

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
    # Retry briefly before refusing: is_locked() probes (the dashboard's 1 s
    # status poll, the eval panel's --follow wait) ACQUIRE the lock for a few
    # microseconds each -- a single attempt can land inside one and refuse with
    # no trainer anywhere (observed 2026-08-06: launcher starts dashboard+panel
    # first, trainer lost the race). A real trainer holds the lock CONTINUOUSLY
    # for its whole life, so retrying cannot false-pass; it only rides out probes.
    _trainer_lock = None                                  # noqa: F841 -- held for process life
    for _ in range(12):
        _trainer_lock = hold_lockfile(ownership.trainer_lock_path(args.ckpt_dir))
        if _trainer_lock is not None:
            break
        time.sleep(0.25)
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
                  games_per_iter=args.games_per_iter, minibatch=args.minibatch,
                  collect_workers=args.collect_workers,
                  collect_affinity=args.collect_affinity,
                  pipeline_collect=args.pipeline_collect or args.collect_stream,
                  collect_stream=args.collect_stream, stream_depth=args.stream_depth,
                  infer_server=args.infer_server,
                  warmup_games=args.warmup_games,
                  pool_frac=args.pool_frac, pfsp_mode=args.pfsp_mode,
                  league_size=args.league_size, league_every=args.league_every,
                  lr_ppo=args.lr_ppo, scenario_frac=args.scenario_frac,
                  scenarios_in_pool=args.scenario_pool, scenario_weights=scen_w,
                  scenario_boost=args.scenario_boost,
                  scenario_selfplay_frac=args.scenario_selfplay,
                  p1_adv_weight=args.p1_adv_weight,
                  freeze_actor_iters=args.freeze_actor_iters,
                  kl_teacher_coef=args.kl_teacher_coef,
                  kl_teacher_iters=args.kl_teacher_iters,
                  critic_epochs=args.critic_epochs,
                  critic_td_mix=args.critic_td_mix,
                  critic_consistency=args.critic_consistency,
                  ent_start=args.ent_start,
                  ent_end=args.ent_end, ent_anneal_iters=args.ent_anneal_iters,
                  ent_reheat_period=args.ent_reheat_period, ent_reheat_peak=args.ent_reheat_peak,
                  train_public=args.train_public,
                  # resume-tunable v3 runtime knob; None = keep the checkpoint's value
                  **({"critic_deckout_aux": args.critic_deckout_aux}
                     if args.critic_deckout_aux is not None else {}),
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
        # FRESH-only v3 architecture flags (a resume takes these from the checkpoint)
        v3 = {}
        if args.belief is not None:
            v3["belief_mode"] = args.belief
        if args.critic_view is not None:
            v3["critic_view"] = args.critic_view
        if args.text_change is not None:
            v3["text_change_mode"] = args.text_change
        if args.obs_counts:
            v3["obs_counts"] = True
        if args.obs_split:
            v3["obs_split"] = True
        if args.obs_ctx:
            v3["obs_ctx"] = True
        if args.obs_tgt:
            v3["obs_tgt"] = True
        cfg = Config(encoder=args.encoder, seed=args.seed,
                     actor_hidden=_parse_hidden(args.actor_hidden),
                     card_dim=args.card_dim, **per_net, **v3, **common)

    encs = {n: cfg.enc_for(n) for n in NETS}
    cap = "unbounded" if cfg.iters <= 0 else cfg.iters
    print(f"device: {cfg.device} | encoders: {encs} | iters: {cap} | "
          f"actor_hidden={cfg.head_hidden('actor')} card_dim={cfg.card_dim} | "
          f"belief={cfg.belief_mode} critic_view={cfg.critic_view} "
          f"deckout_aux={cfg.critic_deckout_aux} critic_epochs={cfg.critic_epochs or cfg.ppo_epochs}/{cfg.ppo_epochs} td_mix={cfg.critic_td_mix} consist={cfg.critic_consistency} text_change={cfg.text_change_mode} | "
          f"pool: {cfg.pool_frac:.2f} (pfsp={cfg.pfsp_mode}, league={cfg.league_size}"
          f"x{cfg.league_every}) | "
          f"p1_adv={cfg.p1_adv_weight:.2f} ent_end={cfg.ent_end:.3f} lr={cfg.lr_ppo:g} | "
          f"resume: {resume} | ckpt: {latest}", flush=True)

    from fishrl.train.keepawake import keep_awake
    keep_awake("training")                            # Windows: no idle-sleep mid-run

    # Stamp each console line with the local time to the minute, so a long console
    # (or a scrollback after a resource-storm hiccup) can be read against the clock.
    # journald adds its own timestamp on the Linux service, so this is redundant-but-
    # harmless there and the [status]/[league] parsers match anywhere in the line.
    def _log(s):
        print(f"[{time.strftime('%m-%d %H:%M')}] {s}", flush=True)

    max_seconds = args.max_hours * 3600.0 if args.max_hours else None
    models = train(cfg, build_models(cfg), log=_log,
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
        print("estimators:", {k: v for k, v in estimator_metrics(models, collect_eval_batch(models, n_games=8)).items() if not k.startswith("_")}, flush=True)


if __name__ == "__main__":
    main()
