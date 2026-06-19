"""Orchestration: warmup, then iterate {collect self-play, PPO, aux updates}.

The guesser is frozen during each iteration's collection + PPO update (stable actor
input distribution) and slow-refreshed by `aux_update` between iterations.

Logging is consolidated: rather than a line per iteration, `train` accumulates the
per-iter losses and emits ONE status line every `cfg.report_every_seconds` (default
hourly) plus a final line. Each status line carries the window-mean losses, estimator
calibration + guesser MAE on the latest batch, and current win-rates vs the
frozen-self / random / attacker / heuristic anchors.
"""
from __future__ import annotations

import copy
import os
import signal
import time
from dataclasses import dataclass

import numpy as np
import torch

from fishrl.models import device_of
from fishrl.models.estimators import PrivilegedCritic, PublicEstimator
from fishrl.models.guesser import HandGuesser
from fishrl.models.policy import MaskedActor
from fishrl.train import checkpoint as ckpt
from fishrl.train.belief_env import BeliefAugmentedEnv
from fishrl.train.collector import actor_act_fn, collect_games
from fishrl.train.config import Config
from fishrl.train.ppo import aux_update, ppo_update


@dataclass
class Models:
    actor: MaskedActor
    critic: PrivilegedCritic
    guesser: HandGuesser
    public: PublicEstimator


def build_models(cfg: Config) -> Models:
    torch.manual_seed(cfg.seed)
    dev = cfg.device
    return Models(
        MaskedActor(cfg.hidden, cfg.enc_for("actor")).to(dev),
        PrivilegedCritic(cfg.critic_hidden, cfg.enc_for("critic")).to(dev),
        HandGuesser(cfg.hidden, cfg.enc_for("guesser")).to(dev),
        PublicEstimator(cfg.hidden, cfg.enc_for("public")).to(dev),
    )


def _snapshot(m: Models) -> Models:
    """Frozen (eval-mode, grad-free) deep copy of the policy + guesser used as the
    'vs frozen self' anchor. Critic/public are copied along but unused by the panel."""
    s = copy.deepcopy(m)
    for net in (s.actor, s.critic, s.guesser, s.public):
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
    return s


def _encoders(cfg: Config) -> dict:
    return {n: cfg.enc_for(n) for n in ("actor", "critic", "guesser", "public")}


def _model_state(m: Models) -> dict:
    return {n: getattr(m, n).state_dict() for n in ("actor", "critic", "guesser", "public")}


def _load_model_state(m: Models, state: dict) -> None:
    for n in ("actor", "critic", "guesser", "public"):
        getattr(m, n).load_state_dict(state[n])


def _rng_state(device: str) -> dict:
    rng = {"torch": torch.get_rng_state(), "numpy": np.random.get_state()}
    if str(device).startswith("cuda") and torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    return rng


def _set_rng_state(rng: dict) -> None:
    torch.set_rng_state(rng["torch"])
    np.random.set_state(rng["numpy"])
    if "cuda" in rng and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])


def train(cfg: Config, models: Models | None = None, log=print,
          max_seconds: float | None = None, resume_path: str | None = None,
          checkpoint_path: str | None = None) -> Models:
    """Run warmup + PPO self-play. If `max_seconds` is set, stop once that wall-clock
    budget is spent (cfg.iters is then just an upper cap) -- this is how fixed-wall-
    clock head-to-heads are run. The entropy schedule anneals over the budget (by
    elapsed fraction) rather than over iters so it doesn't depend on the iter count.

    Durability (opt-in via the two path kwargs; both default None so existing callers are
    unaffected):
      * `resume_path` -- if the file exists, restore models + frozen anchor + optimizers +
        RNG + the iteration counter and skip warmup, so the run continues seamlessly.
      * `checkpoint_path` -- enable periodic `latest.pt` saves (every
        cfg.checkpoint_every_seconds, cheap, no eval) + a milestone at each status report,
        and install SIGTERM/SIGINT handlers that checkpoint-then-exit gracefully. This is
        what makes the systemd service survivable; `cfg.iters <= 0` runs it indefinitely.

    Logging is consolidated into a status line emitted every cfg.report_every_seconds
    (plus a final line) -- see module docstring."""
    from fishrl.eval.metrics import estimator_metrics, guesser_mae, panel_winrates
    from fishrl.train.warmup import warmup
    m = models or build_models(cfg)

    opt_ppo = torch.optim.Adam(list(m.actor.parameters()) + list(m.critic.parameters()), lr=cfg.lr_ppo)
    opt_g = torch.optim.Adam(m.guesser.parameters(), lr=cfg.lr_guesser)
    opt_p = torch.optim.Adam(m.public.parameters(), lr=cfg.lr_public)

    # Mutable run state (the resume-vs-warmup branch below sets the initial values). The
    # 'frozen self' anchor is the post-warmup policy, re-snapshot at every status report, so
    # 'vs frozen' reads as improvement over the previous report's self (>0.5 still improving).
    done = 0
    frozen_it = 0
    elapsed_offset = 0.0           # cumulative training seconds carried across restarts
    frozen: Models | None = None
    KEYS = ("policy_loss", "critic_loss", "entropy", "approx_kl", "guesser_loss", "public_loss")
    acc = {k: 0.0 for k in KEYS}
    win_iters = win_T = 0
    last_batch = None
    run_start = last_report = last_ckpt = time.perf_counter()

    def total_elapsed() -> float:
        return elapsed_offset + (time.perf_counter() - run_start)

    def _payload() -> dict:
        return {
            "format": ckpt.FORMAT,
            "config": {"seed": cfg.seed, "encoders": _encoders(cfg),
                       "use_belief": cfg.use_belief, "critic_hidden": cfg.critic_hidden},
            "done": done, "elapsed": total_elapsed(), "frozen_it": frozen_it,
            "warmup_done": True,
            "models": _model_state(m), "frozen": _model_state(frozen),
            "optim": {"ppo": opt_ppo.state_dict(), "g": opt_g.state_dict(),
                      "p": opt_p.state_dict()},
            "rng": _rng_state(cfg.device),
        }

    def _checkpoint(milestone: bool = False) -> None:
        if checkpoint_path is None:
            return
        payload = _payload()
        ckpt.save_checkpoint(checkpoint_path, payload)
        if milestone:
            ckpt.save_milestone(os.path.dirname(os.path.abspath(checkpoint_path)),
                                done, payload, cfg.keep_last_checkpoints)

    def emit(final: bool = False) -> None:
        nonlocal frozen, frozen_it, win_iters, win_T, last_report, last_ckpt
        now = time.perf_counter()
        n = max(win_iters, 1)
        mean = {k: acc[k] / n for k in KEYS}
        t0 = time.perf_counter()
        # Win-rate panel: single-threaded and slow, so it BLOCKS this loop. With
        # report_winrate_games<=0 it is skipped entirely -- the long-running service runs
        # the parallel, out-of-band evaluator (fishrl.eval.parallel_panel) on a timer
        # instead, so training never stalls for win rates.
        wr = (panel_winrates(m, frozen=frozen, n_games=cfg.report_winrate_games,
                             max_decisions=cfg.max_decisions, use_belief=cfg.use_belief)
              if cfg.report_winrate_games > 0 else None)
        est = estimator_metrics(m, last_batch) if last_batch is not None else {}
        gmae = guesser_mae(m, last_batch) if last_batch is not None else float("nan")
        eval_s = time.perf_counter() - t0
        dt_h = max(now - last_report, 1e-9) / 3600.0
        nan = float("nan")
        tag = "final" if final else f"{total_elapsed() / 3600.0:.2f}h"
        wr_str = (
            f" | WR frozen@{frozen_it}={wr.get('frozen', nan):.2f} "
            f"random={wr['random']:.2f} attacker={wr['attacker']:.2f} "
            f"heuristic={wr['heuristic']:.2f} (n={cfg.report_winrate_games}, eval {eval_s:.1f}s)"
            if wr is not None else " | WR via eval timer"
        )
        log(
            f"[status {tag} it={done} (+{win_iters}, {win_iters / dt_h:.1f}/h) T={win_T}] "
            f"pi={mean['policy_loss']:.3f} V={mean['critic_loss']:.3f} "
            f"H={mean['entropy']:.3f} kl={mean['approx_kl']:.4f} "
            f"guess={mean['guesser_loss']:.3f} pub={mean['public_loss']:.3f} | "
            f"calib priv(acc={est.get('priv_acc', nan):.2f},brier={est.get('priv_brier', nan):.2f}) "
            f"pub(acc={est.get('pub_acc', nan):.2f},brier={est.get('pub_brier', nan):.2f}) "
            f"gmae={gmae:.2f}" + wr_str
        )
        for k in KEYS:
            acc[k] = 0.0
        win_iters = win_T = 0
        frozen = _snapshot(m)            # roll the anchor forward to the current policy
        frozen_it = done
        last_report = time.perf_counter()
        _checkpoint(milestone=True)      # report-time save carries a numbered milestone
        last_ckpt = time.perf_counter()

    # graceful shutdown: only when checkpointing is active (so eval/test callers, which set
    # no checkpoint_path, never have their signal handlers touched). Installed BEFORE warmup
    # so a stop during the one-time fresh-start warmup is honoured once warmup returns. The
    # handler just flips a flag, checked at the iteration boundary.
    stop = {"v": False}
    prev_handlers: dict = {}
    if checkpoint_path is not None:
        def _on_signal(signum, frame):
            stop["v"] = True
        for sig in (signal.SIGTERM, signal.SIGINT):
            prev_handlers[sig] = signal.signal(sig, _on_signal)

    resuming = resume_path is not None and os.path.exists(resume_path)
    try:
        if resuming:
            payload = ckpt.load_checkpoint(resume_path, map_location=cfg.device)
            saved_enc = payload.get("config", {}).get("encoders")
            if saved_enc is not None and saved_enc != _encoders(cfg):
                raise ValueError(
                    f"checkpoint encoder mismatch: saved {saved_enc} != cfg {_encoders(cfg)}; "
                    "rebuild models with the saved encoders before resuming")
            _load_model_state(m, payload["models"])
            opt_ppo.load_state_dict(payload["optim"]["ppo"])
            opt_g.load_state_dict(payload["optim"]["g"])
            opt_p.load_state_dict(payload["optim"]["p"])
            frozen = copy.deepcopy(m)
            _load_model_state(frozen, payload["frozen"])
            for net in (frozen.actor, frozen.critic, frozen.guesser, frozen.public):
                net.eval()
                for p in net.parameters():
                    p.requires_grad_(False)
            done = int(payload["done"])
            frozen_it = int(payload.get("frozen_it", done))
            elapsed_offset = float(payload.get("elapsed", 0.0))
            _set_rng_state(payload["rng"])
            log(f"[resume] from {resume_path} at it={done} (elapsed {elapsed_offset / 3600.0:.2f}h)")
        else:
            w = warmup(m.guesser, m.critic, m.public, cfg, seed=cfg.seed)
            log(f"[warmup] {w}")
            frozen = _snapshot(m)

        benv = BeliefAugmentedEnv(m.guesser, belief=cfg.use_belief, max_decisions=cfg.max_decisions)
        run_start = time.perf_counter()         # exclude warmup/resume setup from elapsed
        last_report = last_ckpt = run_start

        while cfg.iters <= 0 or done < cfg.iters:
            if max_seconds is not None and total_elapsed() > max_seconds:
                log(f"[stop] wall-clock budget {max_seconds:.0f}s reached at it={done}")
                break
            if stop["v"]:
                break
            seed = cfg.seed + 1000 + done * cfg.games_per_iter
            buf = collect_games(benv, actor_act_fn(m.actor), cfg.games_per_iter, seed,
                                critic=m.critic, max_decisions=cfg.max_decisions)
            batch = buf.compute(cfg.gamma, cfg.lam)
            if max_seconds is not None:                  # anneal entropy over the budget
                frac = min(total_elapsed() / max_seconds, 1.0)
                ent = cfg.ent_start + frac * (cfg.ent_end - cfg.ent_start)
            else:
                ent = cfg.ent_coef(done)
            ppo_stats = ppo_update(batch, m.actor, m.critic, opt_ppo, cfg, ent)
            aux_stats = aux_update(batch, m.guesser, m.public, opt_g, opt_p, cfg.aux_steps)
            for k in ("policy_loss", "critic_loss", "entropy", "approx_kl"):
                acc[k] += ppo_stats[k]
            acc["guesser_loss"] += aux_stats["guesser_loss"]
            acc["public_loss"] += aux_stats["public_loss"]
            win_iters += 1
            win_T += len(buf)
            last_batch = batch
            done += 1
            if checkpoint_path is not None and \
                    time.perf_counter() - last_ckpt >= cfg.checkpoint_every_seconds:
                _checkpoint()                            # cheap state save, no eval
                last_ckpt = time.perf_counter()
            if time.perf_counter() - last_report >= cfg.report_every_seconds:
                emit()
        if stop["v"]:                                    # signal: cheap save, skip eval
            _checkpoint()
            log(f"[stop] signal received; checkpointed at it={done}")
        elif win_iters > 0:                              # cap/budget reached: final report
            emit(final=True)
        elif checkpoint_path is not None:
            _checkpoint()
    finally:
        for sig, h in prev_handlers.items():
            signal.signal(sig, h)
    return m


def policy_callable(models: Models, guesser=None):
    """Greedy/sampled masked policy for evaluation. The returned env factory wraps
    FishAEC with the (trained) guesser so eval observations match training."""
    g = guesser or models.guesser

    dev = device_of(models.actor)

    def act(obs):
        x = torch.as_tensor(obs["observation"], dtype=torch.float32).unsqueeze(0).to(dev)
        m = torch.as_tensor(obs["action_mask"], dtype=torch.float32).unsqueeze(0).to(dev)
        with torch.no_grad():
            logp = models.actor.log_probs(x, m)[0]
        return int(torch.multinomial(logp.exp(), 1))     # sample (argmax is degenerate)
    return act, g
