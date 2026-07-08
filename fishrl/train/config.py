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
    # "heuristic_1_1" (frozen at release) and "heuristic_1_2" (current testbench)
    # are POOL opponents only (never the eval anchor, so the vs-heuristic baseline
    # stays comparable).
    pfsp_anchors: tuple = ("random", "attacker", "heuristic", "heuristic_1_1",
                           "heuristic_1_2")
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
    # relative sampling weights over registered scenarios (see fishrl.train.scenarios)
    scenario_weights: dict = field(
        default_factory=lambda: {"known_threat": 1.0, "known_threat_random": 1.0,
                                 "board_presence": 1.0, "deckout": 1.0,
                                 "survive_lethal": 1.0, "survive_lethal_vision": 1.0})
    # Degenerate-correct hard rule wired into ALL training games (not a scenario):
    # declaring fewer than all eligible attackers into an empty opposing board = instant
    # loss. Attacking into an empty board is 100% correct in this format. Default on;
    # disable with --no-enforce-free-attack.
    enforce_free_attack: bool = True

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

    # schedule
    warmup_games: int = 64
    warmup_epochs: int = 3
    iters: int = 100            # <= 0 means UNBOUNDED (run until signal / max_seconds)
    ent_anneal_iters: int = 3000  # entropy-anneal horizon used when iters <= 0 (unbounded)
    hidden: tuple = (256, 256)

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
    use_belief: bool = True
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
        return self.ent_at(it, horizon)
