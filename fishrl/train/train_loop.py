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
import json
import math
import os
import platform
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
from fishrl.train import stats as stats_io
from fishrl.train.belief_env import BeliefAugmentedEnv
from fishrl.train.collector import (
    actor_act_fn,
    collect_games,
    collect_heuristic_games,
    collect_vs_opponent,
    fill_critic_values,
)
from fishrl.train.config import Config
from fishrl.train.pfsp import LeagueMember, PFSPLeague
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


def _harvest_count(entry: list, game_winners: list, lseat: str) -> None:
    """Fold one pool game's outcome(s) into a wr_train [wins, games] entry under
    the EVAL convention (metrics.py): every game counts in the denominator --
    draws and decision-cap truncations included (winner None) -- and only a
    strict decision for the learner's seat counts as a win. This is what lets
    the eval service pool these with its own top-up games into one estimate."""
    entry[1] += len(game_winners)
    entry[0] += sum(1 for w in game_winners if w == lseat)


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
    KEYS = ("policy_loss", "critic_loss", "entropy", "approx_kl", "clip_frac",
            "guesser_loss", "public_loss")
    acc = {k: 0.0 for k in KEYS}
    win_iters = win_T = 0
    # Per-window game/health telemetry (reset each report alongside the loss means):
    # game endings (truncation/draw/free-attack), mirror seat balance, scenario vs
    # full-game episode lengths, forced-decision dilution, and the collect/update
    # wall-clock split.
    GW_KEYS = ("games", "trunc", "draw", "freeatk_p1", "freeatk_p2",
               "mirror_dec", "mirror_p1", "scen_games", "scen_T",
               "forced_steps", "collect_s", "update_s")
    gwin = {k: 0.0 for k in GW_KEYS}
    # Per-report opponent composition: games played vs each opponent category, so the
    # status line can report the proportion of TRAINED (neural) opponents -- mirror
    # self-play + frozen past-selves -- vs the scripted/engine bots (random/attacker/
    # heuristic). Reset each report alongside win_iters.
    OPP_CATS = ("self", "pastself", "heuristic", "heuristic_1_1", "heuristic_1_2",
                "attacker", "random")
    opp_mix = {k: 0 for k in OPP_CATS}
    # Harvested anchor win-rates: [wins, games] vs each SCRIPTED anchor this window,
    # from the pool games training plays anyway. Counted under the EVAL convention
    # (metrics.py docstring): denominator = ALL games incl. draws/truncations, wins =
    # strictly decided for the learner -- so the eval service can pool these with its
    # own top-up games into one estimate. Distinct from league.update (EMA,
    # decided-only) and from opp_mix (game counts, no outcomes). NOTE these games run
    # under cfg.enforce_free_attack (eval games don't), so a forfeit counts as a real
    # loss here -- a deliberate, conservative-only bias (it can under-rate, never
    # inflate, the combined estimate the best.pt gate sees).
    AWIN_KINDS = ("heuristic", "heuristic_1_1", "heuristic_1_2", "attacker", "random")
    awin = {k: [0, 0] for k in AWIN_KINDS}
    # Near-live per-iteration ticks (fishrl.serve's SSE feed; reports stay the hourly
    # durable record). Buffered in memory, flushed to ticks.json at most every
    # cfg.tick_every_seconds; the file keeps only the newest TICK_KEEP rows. Single
    # writer + atomic replace -> readers (serve/monitor) need no lock. Existing rows
    # are re-read at the first flush so a resume continues the ring, not resets it.
    TICK_KEEP = 2000
    ticks: list = []             # the ring, seeded from disk so resume continues it
    if checkpoint_path is not None and cfg.tick_every_seconds > 0:
        try:
            with open(os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)),
                                   "ticks.json")) as f:
                ticks = list(json.load(f).get("ticks", []))
        except (OSError, ValueError):
            ticks = []
    tick_pending = False         # rows appended since the last flush?
    last_tick_flush = time.perf_counter()

    def _flush_ticks() -> None:
        nonlocal tick_pending, last_tick_flush
        if checkpoint_path is None or not tick_pending:
            return
        ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
        path = os.path.join(ckpt_dir, "ticks.json")
        del ticks[:-TICK_KEEP]
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"schema": 1, "ticks": ticks}, f)
        ckpt.replace_with_retry(tmp, path)
        tick_pending = False
        last_tick_flush = time.perf_counter()
    scen_mix: dict = {}            # scenario games played this window (DEBUG logging only)
    last_batch = None
    # PFSP opponent league: scripted anchors + a ring of frozen past selves, appended
    # at each report. In-memory only (never serialized), so resume refills it over the
    # first few reports. Unused when cfg.pool_frac <= 0.
    league = PFSPLeague.from_config(cfg)
    # Scenario curriculum, one of two modes:
    #  * scenarios_in_pool — scenarios are MEMBERS of the main league, competing with
    #    the anchors/past-selves for the pool_frac budget by PFSP difficulty; their
    #    combined share floats instead of being pinned to scenario_frac.
    #  * scenario_frac carve-out (legacy) — a fixed slice of each iteration sampled
    #    from a dedicated scenario league.
    scen_league = None
    if cfg.scenarios_in_pool:
        from fishrl.train.scenarios import scenario_names
        for sname in scenario_names():
            w = float(cfg.scenario_weights.get(sname, 1.0))
            if w > 0:
                league.anchors.append(LeagueMember(name=sname, kind="scenario", weight=w))
    elif cfg.scenario_frac > 0:
        from fishrl.train.scenarios import scenario_names
        scen_league = PFSPLeague.scenario_league(cfg, scenario_names())
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
            # League continuity across restarts: member EMAs + the past-self ring.
            # Absent in pre-persistence checkpoints (resume tolerates that).
            "league": league.state_dict(),
            "scen_league": scen_league.state_dict() if scen_league is not None else None,
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
            f"heuristic={wr['heuristic']:.2f} heuristic11={wr.get('heuristic11', float('nan')):.2f} "
            f"(n={cfg.report_winrate_games}, eval {eval_s:.1f}s)"
            if wr is not None else " | WR via eval timer"
        )
        # Proportion of this window's games played vs a TRAINED (neural) opponent --
        # the mirror self-play policy + frozen past-selves -- vs scripted/engine bots.
        mix_total = sum(opp_mix.values()) or 1
        scen_total = sum(scen_mix.values())
        grand_total = mix_total + scen_total             # all games this window, incl. scenarios
        trained_frac = (opp_mix["self"] + opp_mix["pastself"]) / mix_total
        opp_str = f" | opp trained={trained_frac:.2f}"
        # Window game telemetry (see GW_KEYS): episode lengths split full-game vs
        # scenario, ending mix, mirror seat balance, forced-decision dilution, and
        # the collect share of wall-clock.
        games = int(gwin["games"])
        scen_games = int(gwin["scen_games"])
        full_games = games - scen_games
        full_T = win_T - int(gwin["scen_T"])
        len_full = (full_T / full_games) if full_games else nan
        len_scen = (gwin["scen_T"] / scen_games) if scen_games else nan
        trunc_rate = (gwin["trunc"] / games) if games else nan
        draw_rate = (gwin["draw"] / games) if games else nan
        seat_p1 = (gwin["mirror_p1"] / gwin["mirror_dec"]) if gwin["mirror_dec"] else nan
        fdec = (gwin["forced_steps"] / win_T) if win_T else nan
        wall = gwin["collect_s"] + gwin["update_s"]
        collect_frac = (gwin["collect_s"] / wall) if wall > 0 else nan
        brier_gap = (est["pub_brier"] - est["priv_brier"]) if "priv_brier" in est else nan
        game_str = (
            f" | games={games} len={len_full:.1f} slen={len_scen:.1f} "
            f"trunc={trunc_rate:.2f} draw={draw_rate:.2f} "
            f"fatk_p1={int(gwin['freeatk_p1'])} fatk_p2={int(gwin['freeatk_p2'])} "
            f"seat_p1={seat_p1:.2f} fdec={fdec:.2f} | wall collect={collect_frac:.2f}"
        )
        log(
            f"[status {tag} it={done} (+{win_iters}, {win_iters / dt_h:.1f}/h) T={win_T}] "
            f"pi={mean['policy_loss']:.3f} V={mean['critic_loss']:.3f} "
            f"H={mean['entropy']:.3f} kl={mean['approx_kl']:.4f} "
            f"clip={mean['clip_frac']:.2f} "
            f"guess={mean['guesser_loss']:.3f} pub={mean['public_loss']:.3f} | "
            f"calib priv(acc={est.get('priv_acc', nan):.2f},brier={est.get('priv_brier', nan):.2f}) "
            f"pub(acc={est.get('pub_acc', nan):.2f},brier={est.get('pub_brier', nan):.2f}) "
            f"gap={brier_gap:.2f} gmae={gmae:.2f}" + opp_str + game_str + wr_str
        )
        # Per-scenario curriculum members (carve-out league or the main league's
        # scenario anchors) — feeds both the stats.json report and the [scenario]
        # log line below.
        scen_members = (scen_league.members() if scen_league is not None
                        else [mm for mm in league.anchors if mm.kind == "scenario"])
        if checkpoint_path is not None:                  # dump this datapoint to stats.json
            ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
            # Reports are PURE trainer metrics (no win-rates) so every row has one schema;
            # win-rates -- whether from the inline panel here or the out-of-band eval service --
            # live only in the "evals" array.
            rec = {
                "it": done, "elapsed_h": total_elapsed() / 3600.0, "wall_time": time.time(),
                # Which machine trained this window: after a relay handoff the merged
                # stats interleave hosts, and per-device throughput (the dashboard's
                # last-1k it/h) needs to segment on this rather than guess.
                "host": platform.node(),
                "iters": win_iters, "iters_per_h": win_iters / dt_h, "transitions": win_T,
                "policy_loss": mean["policy_loss"], "critic_loss": mean["critic_loss"],
                "entropy": mean["entropy"], "approx_kl": mean["approx_kl"],
                "guesser_loss": mean["guesser_loss"], "public_loss": mean["public_loss"],
                "priv_acc": est.get("priv_acc"), "pub_acc": est.get("pub_acc"),
                "priv_brier": est.get("priv_brier"), "pub_brier": est.get("pub_brier"),
                "brier_gap": brier_gap, "gmae": gmae,
                "clip_frac": mean["clip_frac"],
                # window game telemetry (NaN -> null via the sanitizer below)
                "games": games, "dec_per_game": len_full, "scen_dec_per_game": len_scen,
                "trunc_rate": trunc_rate, "draw_rate": draw_rate,
                "freeatk_p1": int(gwin["freeatk_p1"]), "freeatk_p2": int(gwin["freeatk_p2"]),
                "mirror_p1_wr": seat_p1, "forced_dec_frac": fdec,
                "collect_s": gwin["collect_s"], "update_s": gwin["update_s"],
                "collect_frac": collect_frac,
                # Window composition as shares of ALL games incl. scenario-seeded ones, so the
                # league slices + opp_scenario partition the window (~sum to 1) and the website's
                # stacked mix shows scenarios. opp_self+opp_past==opp_trained. NOTE: these use the
                # grand-total denominator (mix_total+scen_total); the human [status]/[league] log
                # lines stay league-normalized (share of self-play/PFSP games only). scenario_mix
                # breaks the opp_scenario slice down per manufactured scenario (same denominator).
                "opp_trained": (opp_mix["self"] + opp_mix["pastself"]) / grand_total,
                "opp_self": opp_mix["self"] / grand_total,
                "opp_past": opp_mix["pastself"] / grand_total,
                "opp_heuristic": opp_mix["heuristic"] / grand_total,        # v1.0 only
                "opp_heuristic11": opp_mix["heuristic_1_1"] / grand_total,  # v1.1 (frozen)
                "opp_heuristic12": opp_mix["heuristic_1_2"] / grand_total,  # testbench v1.2
                "opp_attacker": opp_mix["attacker"] / grand_total,
                "opp_random": opp_mix["random"] / grand_total,
                "opp_scenario": scen_total / grand_total,
                "scenario_mix": {k: v / grand_total for k, v in sorted(scen_mix.items())},
                # PFSP curriculum win-rate EMA per scenario (the sampler's difficulty
                # signal, NOT a skill measure) — mirrors the [scenario] line's wr table
                # so the website replica gets it over HTTP instead of journald.
                "scenario_wr": {mm.name: mm.wr for mm in scen_members},
                # Harvested anchor outcomes this window as [wins, games] under the eval
                # convention -- the eval service tops each anchor up to its target and
                # publishes the combined estimate. Keyed with the EVAL names.
                "wr_train": {"heuristic": list(awin["heuristic"]),
                             "heuristic11": list(awin["heuristic_1_1"]),
                             "heuristic12": list(awin["heuristic_1_2"]),
                             "attacker": list(awin["attacker"]),
                             "random": list(awin["random"])},
                "source": "live",
            }
            rec = {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                   for k, v in rec.items()}             # NaN/inf -> null (valid JSON)
            stats_io.append_report(ckpt_dir, rec)
            if wr is not None:                           # inline panel on -> an eval-array row
                stats_io.append_eval(ckpt_dir, {
                    "it": done, "frozen_at": frozen_it, "elapsed_h": total_elapsed() / 3600.0,
                    "wall_time": time.time(), "n": cfg.report_winrate_games, "workers": 1,
                    "took_s": eval_s, "frozen": wr.get("frozen"), "random": wr["random"],
                    "attacker": wr["attacker"], "heuristic": wr["heuristic"],
                    "heuristic11": wr.get("heuristic11"),
                    "new_best": False, "source": "inline",
                })
        if cfg.pool_frac > 0 and league.members():        # PFSP composition + win-rate table
            # The paren-count format is parsed by the website collector and by
            # backfill_stats with a STRICT 5-token sequence, so `heuristic=` prints the
            # MERGED v1.0+v1.1+v1.2 count to keep the format stable. The per-version
            # split rides AFTER the closing paren as `h11=`/`h12=` tokens (strict
            # parsers stop at the paren and ignore them; split-aware ones subtract them
            # out), and also lives in stats.json (opp_heuristic / opp_heuristic11 /
            # opp_heuristic12) and the wr table after the bar, which names each
            # versioned profile separately.
            log(f"[league it={done}] games={mix_total} trained={trained_frac:.2f} "
                f"(self={opp_mix['self']} past={opp_mix['pastself']} "
                f"heuristic={opp_mix['heuristic'] + opp_mix['heuristic_1_1'] + opp_mix['heuristic_1_2']} "
                f"attacker={opp_mix['attacker']} "
                f"random={opp_mix['random']}) "
                f"h11={opp_mix['heuristic_1_1']} h12={opp_mix['heuristic_1_2']} | "
                f"{league.summary(exclude_kinds=('scenario',))}")
        # Per-scenario games this window + the PFSP win-rate driving selection
        # (scen_members computed above, before the stats.json report). Every
        # registered scenario is listed (0 games shown too); win-rates are a
        # CURRICULUM signal (what to practise), not a success metric — judge real
        # progress on the vs-heuristic eval.
        if scen_members and (cfg.scenario_frac > 0 or cfg.scenarios_in_pool):
            counts = " ".join(f"{mm.name}={scen_mix.get(mm.name, 0)}"
                              for mm in scen_members)
            wrs = " ".join(f"{mm.name}={mm.wr:.2f}({mm.games})"
                           for mm in sorted(scen_members, key=lambda mm: mm.wr))
            log(f"[scenario it={done}] games={scen_total} ({counts}) | wr {wrs}")
        for k in KEYS:
            acc[k] = 0.0
        win_iters = win_T = 0
        for k in OPP_CATS:
            opp_mix[k] = 0
        for k in AWIN_KINDS:
            awin[k] = [0, 0]
        for k in GW_KEYS:
            gwin[k] = 0.0
        scen_mix.clear()
        frozen = _snapshot(m)            # roll the anchor forward to the current policy
        league.add_snapshot(m, done)     # add this report's policy as a past-self member
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
    # STOP-file protocol: dropping a file named STOP next to the checkpoint asks the
    # trainer to checkpoint-and-exit at the next iteration boundary -- the graceful
    # stop for processes you can't signal (another console on Windows; the fishrl.serve
    # "End session" button). The file is CONSUMED on trigger, so under systemd
    # (Restart=always) the restarted trainer doesn't immediately re-stop -- which is
    # also why the ODROID dashboard stops via systemctl, not this file.
    stop_file = (os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "STOP")
                 if checkpoint_path is not None else None)
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
            # Restore league continuity (EMA win-rates + past-self ring). Older
            # checkpoints carry no league keys -> fresh leagues, the old behaviour.
            def _opponent_nets():
                actor = MaskedActor(cfg.hidden, cfg.enc_for("actor")).to(cfg.device)
                guesser = HandGuesser(cfg.hidden, cfg.enc_for("guesser")).to(cfg.device)
                return actor, guesser
            if payload.get("league"):
                league.load_state_dict(payload["league"], make_nets=_opponent_nets)
            if scen_league is not None and payload.get("scen_league"):
                scen_league.load_state_dict(payload["scen_league"])
            elif cfg.scenarios_in_pool and payload.get("scen_league"):
                # A checkpoint from the carve-out era: its per-scenario EMAs seed the
                # matching scenario members of the MAIN league (matched by name), so
                # switching modes doesn't reset curriculum difficulty.
                league.load_state_dict(payload["scen_league"])
            n_selves = len(league.selves)
            log(f"[resume] from {resume_path} at it={done} (elapsed {elapsed_offset / 3600.0:.2f}h, "
                f"league selves={n_selves})")
        else:
            w = warmup(m.guesser, m.critic, m.public, cfg, seed=cfg.seed)
            log(f"[warmup] {w}")
            frozen = _snapshot(m)

        benv = BeliefAugmentedEnv(m.guesser, belief=cfg.use_belief, max_decisions=cfg.max_decisions,
                                  enforce_free_attack=cfg.enforce_free_attack)
        run_start = time.perf_counter()         # exclude warmup/resume setup from elapsed
        last_report = last_ckpt = run_start

        while cfg.iters <= 0 or done < cfg.iters:
            if max_seconds is not None and total_elapsed() > max_seconds:
                log(f"[stop] wall-clock budget {max_seconds:.0f}s reached at it={done}")
                break
            if stop_file is not None and os.path.exists(stop_file):
                try:
                    os.remove(stop_file)                 # consume: one stop per drop
                except OSError:
                    pass
                stop["v"] = True
                log(f"[stop] STOP file consumed at it={done}")
            if stop["v"]:
                break
            # Split the iteration into mirror self-play + PFSP pool games. Pool games
            # draw an opponent from the league (sampled by difficulty) and record only
            # the learner's transitions. Critic values are filled ONCE on the merged
            # buffer (each sub-collect runs critic=None). pool_frac<=0 -> pure self-play.
            # Carve scenario-seeded games out first; the rest splits self-play / PFSP pool
            # exactly as before. scenario_frac <= 0 -> n_scen 0 -> unchanged behaviour.
            n_scen = int(round(cfg.games_per_iter * cfg.scenario_frac)) if cfg.scenario_frac > 0 else 0
            n_scen = max(0, min(n_scen, cfg.games_per_iter))
            if cfg.scenarios_in_pool:
                n_scen = 0                              # scenarios come out of the pool budget
            rest = cfg.games_per_iter - n_scen
            n_pool = int(round(rest * cfg.pool_frac))
            n_pool = max(0, min(n_pool, rest))
            n_self = rest - n_pool
            seed = cfg.seed + 1000 + done * cfg.games_per_iter
            t_collect = time.perf_counter()
            buf = collect_games(benv, actor_act_fn(m.actor), n_self, seed,
                                critic=None, max_decisions=cfg.max_decisions)
            opp_mix["self"] += n_self                       # mirror self-play games this iter
            # Mirror seat balance: decided mirror games only (buf holds ONLY the
            # self-play games at this point). Drift from 0.5 = seat exploitation.
            gwin["mirror_dec"] += sum(1 for w in buf.games if w in ("p1", "p2"))
            gwin["mirror_p1"] += sum(1 for w in buf.games if w == "p1")
            pool_rng = np.random.default_rng(cfg.seed + 900_000 + done)
            for pidx in range(n_pool):
                oseed = cfg.seed + 500_000 + done * cfg.games_per_iter + pidx * 17
                member = league.sample(pool_rng)
                if member is None:                          # empty league -> mirror fallback
                    opp_mix["self"] += 1
                    gbuf = collect_games(benv, actor_act_fn(m.actor), 1, oseed,
                                         critic=None, max_decisions=cfg.max_decisions)
                    gwin["mirror_dec"] += sum(1 for w in gbuf.games if w in ("p1", "p2"))
                    gwin["mirror_p1"] += sum(1 for w in gbuf.games if w == "p1")
                    buf.merge(gbuf)
                    continue
                if member.kind == "scenario":               # pool-mode curriculum game:
                    from fishrl.train.scenarios import ScenarioEnv, get_scenario
                    lseat = "p1"                            # learner is p1, p2 the engine bot
                    senv = BeliefAugmentedEnv(
                        m.guesser, belief=cfg.use_belief,
                        env=ScenarioEnv(get_scenario(member.name),
                                        max_decisions=cfg.max_decisions,
                                        enforce_free_attack=cfg.enforce_free_attack))
                    gbuf = collect_games(senv, actor_act_fn(m.actor), 1, oseed,
                                         critic=None, max_decisions=cfg.max_decisions)
                    # scenario accounting, NOT opp_mix: the [league] paren counts and
                    # mix_total stay scenario-free (same telemetry as carve-out mode)
                    scen_mix[member.name] = scen_mix.get(member.name, 0) + 1
                    gwin["scen_games"] += 1
                    gwin["scen_T"] += len(gbuf.steps)
                    if gbuf.games and gbuf.games[-1] in ("p1", "p2"):
                        league.update(member, gbuf.games[-1] == "p1")
                    buf.merge(gbuf)
                    continue
                opp_mix["pastself" if member.kind == "self" else member.kind] += 1
                if member.kind.startswith("heuristic"):     # engine-driven -> learner is p1
                    lseat = "p1"                            # (kind doubles as the ai_profile)
                    gbuf = collect_heuristic_games(m.guesser, m.actor, 1, oseed,
                                                   critic=None, use_belief=cfg.use_belief,
                                                   max_decisions=cfg.max_decisions,
                                                   profile=member.kind)
                else:                                       # scripted / past-self, seat-balanced
                    lseat = "p1" if pidx % 2 == 0 else "p2"
                    gbuf = collect_vs_opponent(m, member, 1, oseed, critic=None,
                                               use_belief=cfg.use_belief,
                                               max_decisions=cfg.max_decisions,
                                               learner_seat=lseat,
                                               enforce_free_attack=cfg.enforce_free_attack)
                # Update the member's learner win-rate from the recorded game result —
                # buf.games counts a game even when the learner never got a decision
                # (losing before your first priority is still a loss; skipping those
                # biased the EMA upward exactly against fast-killing opponents).
                if gbuf.games and gbuf.games[-1] in ("p1", "p2"):
                    league.update(member, gbuf.games[-1] == lseat)
                if member.kind in AWIN_KINDS:            # harvest: all games / strict wins
                    _harvest_count(awin[member.kind], gbuf.games, lseat)
                buf.merge(gbuf)
            # Scenario-seeded games: short, targeted start-states (terminal ±1 reward).
            # Reuses the self-play collector via a belief-wrapped ScenarioEnv, so the
            # transitions are identical in shape and merge into the same PPO buffer.
            if n_scen > 0 and scen_league is not None and scen_league.members():
                from fishrl.train.scenarios import ScenarioEnv, get_scenario
                scn_rng = np.random.default_rng(cfg.seed + 700_000 + done)
                for sidx in range(n_scen):
                    member = scen_league.sample(scn_rng)    # PFSP over scenarios by difficulty
                    sname = member.name
                    senv = BeliefAugmentedEnv(
                        m.guesser, belief=cfg.use_belief,
                        env=ScenarioEnv(get_scenario(sname), max_decisions=cfg.max_decisions,
                                        enforce_free_attack=cfg.enforce_free_attack))
                    sseed = cfg.seed + 300_000 + done * cfg.games_per_iter + sidx * 31
                    sbuf = collect_games(senv, actor_act_fn(m.actor), 1, sseed,
                                         critic=None, max_decisions=cfg.max_decisions)
                    gwin["scen_games"] += 1
                    gwin["scen_T"] += len(sbuf.steps)       # scenario episode lengths
                    buf.merge(sbuf)
                    scen_mix[sname] = scen_mix.get(sname, 0) + 1
                    # the learner is p1 (p2 is the engine bot); count the game even if
                    # the learner never got a decision before it ended
                    if sbuf.games and sbuf.games[-1] in ("p1", "p2"):
                        scen_league.update(member, sbuf.games[-1] == "p1")
            fill_critic_values(buf, m.critic)
            # Window game telemetry from the merged buffer: how games ended (buf.meta),
            # forced-decision dilution (1-legal-action steps), collect wall-clock.
            gwin["games"] += len(buf.games)
            for w, gm in zip(buf.games, buf.meta):
                if gm["truncated"]:
                    gwin["trunc"] += 1
                elif w is None:
                    gwin["draw"] += 1
                if gm["forced"] in ("p1", "p2"):
                    gwin[f"freeatk_{gm['forced']}"] += 1
            gwin["forced_steps"] += sum(1 for s in buf.steps if int(s.mask.sum()) == 1)
            iter_collect_s = time.perf_counter() - t_collect
            gwin["collect_s"] += iter_collect_s
            batch = buf.compute(cfg.gamma, cfg.lam)
            if max_seconds is not None:                  # anneal entropy over the budget
                frac = min(total_elapsed() / max_seconds, 1.0)
                ent = cfg.ent_start + frac * (cfg.ent_end - cfg.ent_start)
            else:
                ent = cfg.ent_coef(done)
            t_update = time.perf_counter()
            ppo_stats = ppo_update(batch, m.actor, m.critic, opt_ppo, cfg, ent, rng_seed=done)
            aux_stats = aux_update(batch, m.guesser, m.public, opt_g, opt_p, cfg.aux_steps)
            iter_update_s = time.perf_counter() - t_update
            gwin["update_s"] += iter_update_s
            for k in ("policy_loss", "critic_loss", "entropy", "approx_kl", "clip_frac"):
                acc[k] += ppo_stats[k]
            acc["guesser_loss"] += aux_stats["guesser_loss"]
            acc["public_loss"] += aux_stats["public_loss"]
            win_iters += 1
            win_T += len(buf)
            last_batch = batch
            done += 1
            if checkpoint_path is not None and cfg.tick_every_seconds > 0:
                ticks.append({
                    "it": done, "wall_time": time.time(), "T": len(buf),
                    "games": len(buf.games),
                    "policy_loss": ppo_stats["policy_loss"],
                    "critic_loss": ppo_stats["critic_loss"],
                    "entropy": ppo_stats["entropy"], "approx_kl": ppo_stats["approx_kl"],
                    "guesser_loss": aux_stats["guesser_loss"],
                    "public_loss": aux_stats["public_loss"],
                    "collect_s": round(iter_collect_s, 3), "update_s": round(iter_update_s, 3),
                })
                tick_pending = True
                if time.perf_counter() - last_tick_flush >= cfg.tick_every_seconds:
                    _flush_ticks()
            if checkpoint_path is not None and cfg.archive_every_iters > 0 \
                    and done % cfg.archive_every_iters == 0:
                # permanent, never-pruned archive of the run every N iterations
                # (the rolling step_*.pt milestones keep only the last few)
                apath = ckpt.save_archive(
                    os.path.dirname(os.path.abspath(checkpoint_path)), done, _payload())
                log(f"[archive] saved {apath} at it={done}")
            if checkpoint_path is not None and \
                    time.perf_counter() - last_ckpt >= cfg.checkpoint_every_seconds:
                _checkpoint()                            # cheap state save, no eval
                last_ckpt = time.perf_counter()
            if time.perf_counter() - last_report >= cfg.report_every_seconds:
                emit()
        _flush_ticks()                                   # whatever is buffered, out to disk
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
