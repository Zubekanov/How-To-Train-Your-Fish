"""Hyperparameters for the self-play training stack."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


def resolve_device(gpu: bool) -> str:
    """Map a --gpu flag to a torch device string, falling back to CPU (with a
    warning) when CUDA was requested but isn't available."""
    if gpu and torch.cuda.is_available():
        return "cuda"
    if gpu:
        print("[warn] --gpu requested but CUDA is not available; running on CPU", flush=True)
    return "cpu"


@dataclass
class Config:
    # rollout / credit
    gamma: float = 0.997
    lam: float = 0.95
    max_decisions: int = 2000
    games_per_iter: int = 8
    # Per-seat policy-gradient weight. The game is seat-SYMMETRIC (random & attacker
    # mirror both sit at ~0.50), but long self-play drifts into a seat-asymmetric
    # equilibrium -- at it~484k the shared policy won 0.625 as p2 but only 0.375 as p1.
    # That matters because the objective (beat the engine heuristic) is only ever measured
    # from p1: the engine resolves the heuristic on p2, so both training-vs-heuristic and
    # the vs-heuristic eval put the learner in p1. `p1_adv_weight` > 1 scales p1-seat
    # advantages up before normalization, steering the shared net's capacity toward the
    # seat that the metric actually reads (and counteracting the p2 drift). 1.0 = the
    # historic symmetric behaviour. See fishrl.eval.seat_report / selfplay_seat_diagnostics.
    p1_adv_weight: float = 1.0
    # Parallel local collection: fan each iteration's games across this many
    # persistent collector worker processes (fishrl.train.pcollect), shipping the
    # CURRENT weights every iteration -- wall-clock only, training semantics
    # unchanged (collection was 62% of iteration time at 20 cores). 0 = the
    # serial path, byte-identical: the ODROID service default.
    collect_workers: int = 0
    # CPU affinity for the collector workers: comma-separated LOGICAL-processor
    # indices every worker is restricted to (the scheduler distributes them
    # within the set). "" = unpinned, the default everywhere. Machine flag, not
    # regime: on hybrid Intel parts under Windows 10 (which is not
    # hybrid-scheduler-aware) unpinned workers drift onto E-cores measured
    # 2.3x slower per decision -- pin one LP per physical P-core (HT siblings
    # enumerate adjacently, so e.g. "0,2,4,6,8,10,12,14" on an 8P-core part).
    # Give at least as many LPs as workers or they queue inside the mask.
    collect_affinity: str = ""
    # Central batched inference for the collectors (fishrl.train.inference): one
    # server process on `device` answers all workers' learner actor/guesser
    # forwards in pending-batch order, replacing 8 processes each streaming the
    # full weights per batch-1 forward (measured DRAM-bound: 3.7 ms/decision
    # contended vs 0.8 ms batched 8-wide). Machine flag, wall-clock only: same
    # per-iteration weights (pushed + ACKed at submit), sampling RNG stays in
    # the workers, and any server failure falls back to the local nets.
    # Requires collect_workers > 0; default off (serial/ODROID byte-identical).
    infer_server: bool = False
    # Pipelined collection (REGIME, not machine: it changes training semantics).
    # When on, the workers play iteration N+1's games WHILE the GPU updates on
    # iteration N's batch, hiding the update under collection. The behavior
    # policy is then one update STALE relative to the policy being optimized --
    # PPO's importance ratio + clip absorb exactly this kind of lag (ratios
    # simply don't start at 1), visible as a higher approx_kl floor. Off by
    # default; requires collect_workers > 0 (a serial trainer cannot overlap).
    pipeline_collect: bool = False

    # ── Opponent pool / PFSP (prioritized fictitious self-play) ──────────────
    # Fraction of each iteration's games the learner plays against a POOL opponent
    # (sampled from the league below) instead of mirror self-play. 0.0 -> pure
    # self-play (original behaviour). Only the learner seat's transitions are
    # trained; the opponent's are off-policy and never buffered.
    pool_frac: float = 0.25
    # League = scripted anchors (also the eval anchors) + a ring of frozen past-self
    # snapshots (true fictitious self-play). league_size is the past-self ring length
    # (0 -> anchors only). A snapshot is appended every status report.
    # "heuristic" is v1.0 — the eval anchor and the run's long-standing opponent;
    # "heuristic_1_1"/"heuristic_1_2" (frozen at their releases) and
    # "heuristic_1_3" (current testbench mainline, evaluator off) are POOL
    # opponents only (never the eval anchor, so the vs-heuristic baseline
    # stays comparable).
    pfsp_anchors: tuple = ("random", "attacker", "heuristic", "heuristic_1_1",
                           "heuristic_1_2", "heuristic_1_3")
    league_size: int = 8
    # Opponent sampling priority over the learner's per-opponent win-rate `x`:
    #   "hard" -> (1-x)^pfsp_p  : focus on opponents you LOSE to (default)
    #   "var"  -> x*(1-x)       : focus on EVEN matchups (AlphaStar main-agent style)
    # pfsp_eps floors every member's weight so nothing starves; pfsp_wr_ema is the
    # EMA rate for updating a member's win-rate from pool-game outcomes.
    pfsp_mode: str = "hard"
    pfsp_p: float = 2.0
    pfsp_eps: float = 0.05
    pfsp_wr_ema: float = 0.1

    # ── Scenario-based curriculum (opt-in; 0.0 -> no scenarios, behaviour unchanged) ──
    # Fraction of each iteration's games seeded from a short, targeted SCENARIO start-
    # state instead of a full game. Scenarios shape only the initial-state distribution
    # + termination; reward stays terminal ±1. The remaining (1 - scenario_frac) games
    # split into self-play / PFSP pool exactly as before. Judge progress on the full-game
    # vs-heuristic eval ONLY — scenario win-rates are debug, never a success metric.
    scenario_frac: float = 0.0
    # Fold the scenarios INTO the main PFSP opponent league instead of the fixed
    # scenario_frac carve-out: each registered scenario (positive weight) becomes a
    # league member competing with the anchors/past-selves for the pool_frac budget,
    # so its play rate floats with the learner's difficulty on it rather than being
    # pinned to a fixed combined share. Takes precedence over scenario_frac (which
    # is forced to 0 when this is set). Telemetry is unchanged: scenario games are
    # still reported on the [scenario] line / opp_scenario, never in the league's
    # paren counts.
    scenarios_in_pool: bool = False
    # relative sampling weights over registered scenarios (see fishrl.train.scenarios).
    # NOTE: every registered scenario must appear here — the legacy carve-out's
    # sample_scenario_name treats a MISSING name as weight 0 (excluded), while the
    # pool path defaults it to 1.0; listing them all keeps the two modes agreeing.
    scenario_weights: dict = field(
        default_factory=lambda: {"known_threat": 1.0, "known_threat_random": 1.0,
                                 "board_presence": 1.0, "deckout": 1.0,
                                 "survive_lethal": 1.0, "survive_lethal_vision": 1.0,
                                 "survive_lethal_single": 1.0,
                                 # envelope-constructed set (2026-08-21, scenarios/constructed.py)
                                 "fish_war": 1.0, "response_window": 1.0,
                                 "response_window_bend": 1.0, "protect_the_fish": 1.0,
                                 "removal_in_hand": 1.0, "deckout_short": 1.0,
                                 "deckout_with_fish": 1.0, "lethal_on_board": 1.0,
                                 "steer_the_top": 1.0, "undoing_call": 1.0})
    # scenarios_in_pool only: a flat multiplier on every scenario member's PFSP
    # sampling weight (composes with the per-scenario scenario_weights prior).
    # Scenario episodes are far shorter than full games (~1/4 the decisions), so
    # boosting their PLAY-COUNT share costs sub-proportional wall-clock — and the
    # pool_frac cap still bounds the whole pool slice, mirror self-play keeps the
    # rest. 1.0 restores the old behaviour.
    scenario_boost: float = 3.0

    # PPO
    clip: float = 0.2
    ppo_epochs: int = 4
    minibatch: int = 256
    lr_ppo: float = 3e-4
    grad_clip: float = 1.0
    ent_start: float = 0.02
    ent_end: float = 0.005
    critic_coef: float = 0.5

    # auxiliary supervised heads
    lr_guesser: float = 1e-3
    lr_public: float = 1e-3
    aux_steps: int = 2          # SGD steps on guesser/public per iteration (between PPO updates)
    # The public estimator is DIAGNOSTIC-ONLY (calibration telemetry; never feeds the policy).
    # False skips its per-decision encode (encode_public -> zeros) AND its aux training -- a
    # small collection speedup that costs the pub_acc/pub_brier/brier_gap telemetry. Default
    # True = unchanged behaviour.
    train_public: bool = True

    # schedule
    warmup_games: int = 64
    warmup_epochs: int = 3
    iters: int = 100            # <= 0 means UNBOUNDED (run until signal / max_seconds)
    ent_anneal_iters: int = 3000  # entropy-anneal horizon used when iters <= 0 (unbounded)
    # Cyclical entropy RE-HEATING for very long unbounded runs. The plain anneal collapses
    # to ent_end and holds there forever, which on a multi-100k-iter run means exploration
    # is pinned at the floor for ~the whole run -- fine for sharpening, bad for escaping a
    # self-play local optimum (the plateau failure mode). When ent_reheat_period > 0, AFTER
    # the initial anneal the coefficient follows a cosine sawtooth: it jumps to
    # ent_reheat_peak at the start of each period and decays back to ent_end over the period,
    # re-injecting exploration periodically (LR-warm-restarts, for entropy). 0 = disabled
    # (a flat ent_end floor, the historic behaviour). Cost: a mild win-rate wobble each
    # re-heat, so enable it when a run stalls, not preemptively-aggressively.
    ent_reheat_period: int = 0
    ent_reheat_peak: float = 0.02
    # ── PPO handoff after a behaviour-cloning bootstrap (fishrl.imitate) ─────────
    # Both default OFF: the normal trainer is bit-identical with them at 0. Compared
    # against the ABSOLUTE iteration counter (a BC checkpoint starts at done=0), so a
    # crash-restart of the fine-tune re-enters the same phase it left.
    freeze_actor_iters: int = 0    # while done < N, the PPO update trains the CRITIC only
                                   # (the actor takes no gradient at all), so advantages
                                   # are calibrated on-policy before the clone moves
    kl_teacher_coef: float = 0.0   # KL(teacher || pi) penalty coefficient at it=0; the
                                   # teacher is the actor snapshotted at run start (the BC
                                   # clone on a resume). Keeps early PPO from destroying
                                   # the cloned prior on garbage advantages ...
    kl_teacher_iters: int = 0      # ... annealed linearly to zero over this many iters
    # Head-MLP widths. `hidden` is the default for the actor / guesser / public heads;
    # `actor_hidden` (when set) overrides it for the ACTOR only, so the policy can be
    # deepened/widened without also growing the guesser (which runs per-decision on the
    # collection hot path). `critic_hidden` is separate (critic is off the deploy path).
    hidden: tuple = (256, 256)
    actor_hidden: tuple | None = None            # None -> fall back to `hidden`
    # Entity/attention card-embedding width `d` (shared by every entity encoder built
    # for this run — actor and critic if they use one). Bigger d = more per-card capacity
    # and a wider pooled block feeding the head. Ignored by flat nets.
    card_dim: int = 64

    def __post_init__(self):
        # Reconcile the legacy bool with belief_mode. Rule: an explicit
        # use_belief=False with belief_mode left at its default means the caller is
        # using the old API -> "none". Afterwards use_belief mirrors belief_mode
        # (guesser OR bookkeeper both feed the channel), so old call sites that read
        # cfg.use_belief keep meaning "the belief slot carries information".
        if not self.use_belief and self.belief_mode == "guesser":
            self.belief_mode = "none"
        self.use_belief = self.belief_mode != "none"

    @property
    def has_guesser(self) -> bool:
        # The guesser NET is absent only in bookkeeper mode. Legacy "none"
        # (use_belief=False) keeps the net — historically it was still built and
        # aux-trained with the channel zeroed (the ablation arms and BC checkpoints
        # depend on that), and dropping it would orphan their saved weights.
        return self.belief_mode != "bookkeeper"

    @property
    def has_public(self) -> bool:
        # Legacy runs ALWAYS build the public net (train_public only gates its
        # encode + training, matching the historic behaviour so v2 resumes are
        # unchanged); in critic_view="public" the critic IS the public head.
        return self.critic_view == "god"

    def head_hidden(self, net: str) -> tuple:
        """Resolve a net's head-MLP widths: the actor honours `actor_hidden`; the
        critic uses `critic_hidden`; everything else uses `hidden`."""
        if net == "actor" and self.actor_hidden is not None:
            return tuple(self.actor_hidden)
        if net == "critic":
            return tuple(self.critic_hidden)
        return tuple(self.hidden)

    # status reporting: train() consolidates per-iter logs into ONE status line emitted
    # every `report_every_seconds` of wall-clock (default hourly), plus a final line.
    # Each line carries window-mean losses, estimator calibration + guesser MAE on the
    # latest batch, and current win-rates vs the frozen-self / random / attacker /
    # heuristic anchors (`report_winrate_games` games each, fixed eval seeds so the
    # trend is comparable across reports). Lower the interval for finer-grained logs.
    # report_winrate_games <= 0 SKIPS the inline panel entirely (it is single-threaded and
    # blocks the loop) -- the long-running service does this and runs the parallel,
    # out-of-band evaluator (fishrl.eval.parallel_panel) on a systemd timer instead.
    report_every_seconds: float = 3600.0
    report_winrate_games: int = 30
    # crash-recovery checkpoint cadence -- decoupled from the (expensive) report cadence so
    # state is saved often (cheap, no eval) without running the win-rate panel each time.
    checkpoint_every_seconds: float = 900.0
    # near-live telemetry ticks: one lightweight row per ITERATION (losses, transitions,
    # collect/update seconds) flushed to <ckpt-dir>/ticks.json at most this often, ring-
    # capped (fishrl.serve streams them over SSE -- reports stay the hourly durable record).
    # <= 0 disables. Trainer-only, single writer, atomic replace: readers never lock.
    tick_every_seconds: float = 60.0
    keep_last_checkpoints: int = 3    # numbered step_*.pt milestones to retain
    # Permanent archive cadence: every N iterations write archive_{it}.pt, which is
    # NEVER pruned (unlike the rolling step_*.pt milestones) — the run's long-term
    # history for later comparison/rollback. <= 0 disables.
    archive_every_iters: int = 10_000
    # Front-end encoder: "flat" (MLP over the raw observation), "entity" (shared
    # card-embedding encoder, masked mean/max pool; far fewer params), or "attention"
    # (the entity front-end + cross-zone self-attention and a learned per-zone
    # attention pool — relational, less lossy than mean/max). Default "flat" so
    # behaviour is unchanged until the A/B (eval/ab_encoder) backs flipping it.
    #
    # `encoder` is the BASE applied to any net without an explicit override below.
    # The four nets have different throughput exposure, so the encoder is decoupled
    # per-net (resolve via `enc_for`):
    #   * actor / guesser run per-decision DURING collection -> on the throughput path.
    #   * critic runs per-decision too (value baseline) but its god_feat is buffered,
    #     so it is batchable post-collection; and it is gone at deployment entirely.
    #     -> an entity critic is a near-free calibration win once values are batched.
    #   * public is diagnostic-only.
    # On-policy A/B (eval/ab_encoder): entity Brier 0.261 vs flat 0.342 -> entity is
    # the calibration winner, so it's the natural critic override.
    encoder: str = "flat"
    # when False, the actor's belief channel is fed zeros (guesser skipped) -- the
    # controlled "is the belief worth anything?" ablation. Architecture is unchanged.
    # LEGACY BOOL: `belief_mode` below is the source of truth; this field survives for
    # constructor/call-site compat and is reconciled in __post_init__.
    use_belief: bool = True
    # ── v3 knobs (all persisted in the checkpoint config; legacy checkpoints default
    #    to the pre-v3 behaviour via config_from_checkpoint) ────────────────────────
    # What fills the actor's 20-dim belief slot:
    #   "guesser"    -- the learned HandGuesser (legacy; net built + trained)
    #   "bookkeeper" -- the analytic hand-bookkeeper vector (zero-parameter, computed
    #                   from the viewer's own observation; no guesser net exists).
    #                   Measured (Guesser Deposition, 2026-08-17): captures 77% of the
    #                   channel's information at none of its ~32%-of-collection cost.
    #   "none"       -- zeros (the ablation; equivalent to legacy use_belief=False)
    belief_mode: str = "guesser"
    # Which encoding feeds the PPO critic:
    #   "god"    -- privileged full-hidden-state features (legacy)
    #   "public" -- mutual-knowledge features; the PublicEstimator diagnostic is NOT
    #               built (it would duplicate the critic), and encode_public is forced
    #               on regardless of train_public. Measured basis: the privileged
    #               head's Brier edge over public has depreciated to ~0 (audits
    #               2026-08-17).
    critic_view: str = "god"
    # Weight of the critic's deckout-winner auxiliary loss (the parity-credit lever —
    # both audits' #1 recommendation). The aux head EXISTS whenever critic_view ==
    # "public" (so this knob is resume-tunable without state_dict surgery); the weight
    # only scales the loss term. 0.0 = head present, no gradient contribution.
    critic_deckout_aux: float = 0.0
    # choose_text_change action-space treatment (fishrl.spaces.masking):
    #   "full"   -- the legacy 25-way from->to block
    #   "guided" -- mask down to {EFFECT, NO-OP}: the type written on the targeted
    #               card -> a type absent from the targeted player's permanents, plus
    #               one provably-inert pair (decline)
    #   "auto"   -- mask to {EFFECT} alone; the collector's single-legal-action fast
    #               path plays it with no policy forward (auto-resolve)
    text_change_mode: str = "full"
    actor_encoder: str | None = None
    # critic defaults to entity: it's the measured on-policy calibration winner
    # (Brier 0.261 vs flat 0.342), off the deployment path, and ~free now that the
    # value pass is batched post-collection. The actor/guesser stay "flat" until a
    # win-rate head-to-head backs flipping them.
    critic_encoder: str | None = "entity"
    guesser_encoder: str | None = None
    public_encoder: str | None = None
    critic_hidden: tuple = (512, 512, 256)
    device: str = "cpu"          # "cpu" | "cuda" (set via resolve_device / --gpu)
    seed: int = 0
    ckpt_dir: str = "checkpoints"

    def enc_for(self, net: str) -> str:
        """Resolve the encoder for one net ('actor'|'critic'|'guesser'|'public'),
        falling back to the base `encoder` when no per-net override is set."""
        return getattr(self, f"{net}_encoder") or self.encoder

    def ent_at(self, it: int, horizon: int) -> float:
        """Linearly anneal entropy from ent_start -> ent_end over `horizon` iters, then
        hold at ent_end. `horizon` lets the unbounded loop (iters <= 0) anneal over a finite
        window (ent_anneal_iters) instead of dividing by a non-positive iter count."""
        if horizon <= 1:
            return self.ent_end
        frac = min(it / (horizon - 1), 1.0)
        return self.ent_start + frac * (self.ent_end - self.ent_start)

    def ent_coef(self, it: int) -> float:
        horizon = self.iters if self.iters > 0 else self.ent_anneal_iters
        if self.ent_reheat_period > 0 and it >= horizon:
            # past the initial anneal: cosine sawtooth from ent_reheat_peak down to ent_end,
            # restarting every ent_reheat_period iters. phase 0 -> peak, phase 1 -> floor.
            import math
            phase = ((it - horizon) % self.ent_reheat_period) / self.ent_reheat_period
            decay = 0.5 * (1.0 + math.cos(math.pi * phase))          # 1 -> 0 across a period
            return self.ent_end + decay * (self.ent_reheat_peak - self.ent_end)
        return self.ent_at(it, horizon)
