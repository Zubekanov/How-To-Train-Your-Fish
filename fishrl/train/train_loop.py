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
import gc
import os
import platform
import signal
import time
from dataclasses import dataclass

import numpy as np
import torch

from fishrl.data.buffer import RolloutBuffer
from fishrl.models import device_of
from fishrl.models.estimators import PrivilegedCritic, PublicEstimator, make_critic
from fishrl.models.guesser import HandGuesser
from fishrl.data import features
from fishrl.train import stackprof
from fishrl.models.policy import ACTOR_IN, MaskedActor
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
from fishrl.train import sysstats
from fishrl.train.pfsp import SCRIPTED_KINDS, LeagueMember, PFSPLeague
from fishrl.train.ppo import aux_update, ppo_update


@dataclass
class Models:
    actor: MaskedActor
    critic: PrivilegedCritic          # or PublicCritic when cfg.critic_view == "public"
    guesser: HandGuesser | None = None    # None in belief_mode="bookkeeper"/"none" runs
    public: PublicEstimator | None = None  # None when the critic IS the public head


NETS = ("actor", "critic", "guesser", "public")


def _present_nets(m: Models):
    """(name, net) for the nets this run actually has — the iteration every
    state/snapshot/encoder helper uses so 4-net (legacy) and 2-net (v3) Models
    both work."""
    return [(n, getattr(m, n)) for n in NETS if getattr(m, n) is not None]


def build_models(cfg: Config) -> Models:
    torch.manual_seed(cfg.seed)
    dev = cfg.device
    guesser = public = None
    if getattr(cfg, "has_guesser", True):
        guesser = HandGuesser(cfg.head_hidden("guesser"), cfg.enc_for("guesser"),
                              cfg.card_dim).to(dev)
    if getattr(cfg, "has_public", True):
        public = PublicEstimator(cfg.head_hidden("public"), cfg.enc_for("public"),
                                 cfg.card_dim).to(dev)
    # The count block is a process-wide layout switch (features.set_count_block): every
    # loader goes through build_models, so the encoders agree with the nets built here.
    counts = bool(getattr(cfg, "obs_counts", False))
    if counts:
        assert cfg.belief_mode == "bookkeeper" and cfg.critic_view == "hands", "obs_counts rides the bookkeeper belief and the hands critic"
    features.set_count_block(counts)
    extra = features.COUNT_DIM if counts else 0
    return Models(
        MaskedActor(cfg.head_hidden("actor"), cfg.enc_for("actor"), cfg.card_dim,
                    in_dim=ACTOR_IN + extra).to(dev),
        make_critic(getattr(cfg, "critic_view", "god"), cfg.head_hidden("critic"),
                    cfg.enc_for("critic"), cfg.card_dim, extra_in=extra).to(dev),
        guesser, public,
    )


def config_from_checkpoint(cd: dict, **overrides) -> Config:
    """Rebuild a Config from a checkpoint's saved `config` dict, reading EVERY
    architecture key the model shapes depend on (encoders, head widths, card_dim)
    so a reloaded model matches the saved state_dict. This is the ONE place that
    knows the checkpoint->architecture mapping -- all loaders (train resume, eval
    panels, probes, migrations) go through it, so adding a new architecture knob
    can never silently break a subset of loaders again. Missing keys fall back to
    the historic defaults, so pre-existing checkpoints (no hidden/card_dim keys)
    reconstruct the old flat/(256,256)/d64 architecture unchanged.

    `overrides` win over the saved values -- for the RUNTIME knobs (device, iters,
    pool_frac, ...) that are not part of the saved architecture."""
    # v3 payloads carry only the nets they built in `encoders`; .get(n) leaves the
    # per-net encoder at None for absent nets (enc_for then falls back to the base,
    # which is harmless -- the net is never constructed).
    per_net = {f"{n}_encoder": cd["encoders"].get(n) for n in NETS}
    ah = cd.get("actor_hidden")
    # Architecture-defining v3 keys, with legacy defaults so every pre-v3 checkpoint
    # reconstructs unchanged: belief_mode falls back to the old use_belief bool,
    # critic_view to "god". critic_deckout_aux is a RUNTIME knob (resume-tunable),
    # carried here only as the checkpoint's last value; CLI overrides win.
    belief_mode = cd.get("belief_mode",
                         "guesser" if cd.get("use_belief", True) else "none")
    kw = dict(
        seed=cd.get("seed", 0),
        use_belief=cd.get("use_belief", True),
        belief_mode=belief_mode,
        critic_view=cd.get("critic_view", "god"),
        critic_deckout_aux=float(cd.get("critic_deckout_aux", 0.0)),
        text_change_mode=cd.get("text_change_mode", "full"),
        obs_counts=bool(cd.get("obs_counts", False)),
        hidden=tuple(cd.get("hidden", (256, 256))),
        actor_hidden=tuple(ah) if ah is not None else None,
        critic_hidden=tuple(cd.get("critic_hidden", (512, 512, 256))),
        card_dim=int(cd.get("card_dim", 64)),
        **per_net,
    )
    kw.update(overrides)          # runtime knobs (and explicit CLI values) win
    return Config(**kw)


def _snapshot(m: Models) -> Models:
    """Frozen (eval-mode, grad-free) deep copy of the policy + guesser used as the
    'vs frozen self' anchor. Critic/public are copied along but unused by the panel."""
    s = copy.deepcopy(m)
    for _n, net in _present_nets(s):
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
    return s


def _encoders(cfg: Config) -> dict:
    """The checkpoint's architecture record: encoder per BUILT net only (a v3
    payload simply has no guesser/public keys — the discriminator legacy loaders
    key on via .get())."""
    nets = ["actor", "critic"]
    if getattr(cfg, "has_guesser", True):
        nets.append("guesser")
    if getattr(cfg, "has_public", True):
        nets.append("public")
    return {n: cfg.enc_for(n) for n in nets}


def _model_state(m: Models) -> dict:
    return {n: net.state_dict() for n, net in _present_nets(m)}


def _load_model_state(m: Models, state: dict) -> None:
    # Present-net ∩ payload keys: a legacy 4-net payload loads into a legacy Models,
    # a v3 2-net payload into a v3 Models; a legacy payload loaded by a v3 Models
    # (or vice versa) is an architecture mismatch the resume guard rejects upstream.
    for n, net in _present_nets(m):
        if n in state:
            net.load_state_dict(state[n])


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
    # .cpu(): resume loads the payload with map_location=cfg.device, which on a
    # --gpu resume moves EVERY tensor to CUDA -- including these RNG states, and
    # torch requires CPU ByteTensors here ("RNG state must be a torch.ByteTensor").
    # On a CPU resume .cpu() returns the same tensor, so the ODROID path is untouched.
    torch.set_rng_state(rng["torch"].cpu())
    np.random.set_state(rng["numpy"])
    if "cuda" in rng and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])


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
    from fishrl.data import features
    from fishrl.spaces import masking
    # Public encoding is ON iff the diagnostic wants it OR the critic eats it — the
    # v3 landmine: with critic_view="public" a train_public=False gate would zero the
    # critic's own food (fill_critic_values also hard-asserts against this).
    features.set_public_encoding(cfg.train_public or cfg.critic_view in features.PUBLIC_FAMILY)
    if cfg.critic_view in features.PUBLIC_FAMILY:
        features.set_public_view(cfg.critic_view)
    masking.set_text_change_mode(getattr(cfg, "text_change_mode", "full"))
    m = models or build_models(cfg)

    opt_ppo = torch.optim.Adam(list(m.actor.parameters()) + list(m.critic.parameters()), lr=cfg.lr_ppo)
    opt_g = (torch.optim.Adam(m.guesser.parameters(), lr=cfg.lr_guesser)
             if m.guesser is not None else None)
    opt_p = (torch.optim.Adam(m.public.parameters(), lr=cfg.lr_public)
             if m.public is not None else None)

    # Mutable run state (the resume-vs-warmup branch below sets the initial values). The
    # 'frozen self' anchor is the post-warmup policy, re-snapshot at every status report, so
    # 'vs frozen' reads as improvement over the previous report's self (>0.5 still improving).
    done = 0
    frozen_it = 0
    # Handoff origin: the iteration a critic swap / bootstrap happened at. The
    # freeze-actor phase and the KL-to-teacher anneal count from HERE, so an in-place
    # critic replacement (swap_critic) gets its warmup without resetting `done`.
    # 0 for every checkpoint written before 2026-08-21 -> the absolute-iteration
    # behaviour those runs were launched with.
    handoff_start = 0
    elapsed_offset = 0.0           # cumulative training seconds carried across restarts
    frozen: Models | None = None
    KEYS = ("policy_loss", "critic_loss", "entropy", "approx_kl", "clip_frac",
            "guesser_loss", "public_loss", "deckout_aux_loss",
            "gns_b", "gns_tr_sigma", "gns_g2")
    acc = {k: 0.0 for k in KEYS}
    win_iters = win_T = 0
    # Per-window game/health telemetry (reset each report alongside the loss means):
    # game endings (truncation/draw), mirror seat balance, scenario vs full-game
    # episode lengths, forced-decision dilution, and the collect/update wall-clock split.
    GW_KEYS = ("games", "trunc", "draw",
               "mirror_dec", "mirror_p1", "scen_games", "scen_T",
               "forced_steps", "collect_s", "update_s", "book_s")
    gwin = {k: 0.0 for k in GW_KEYS}
    # Per-report opponent composition: games played vs each opponent category, so the
    # status line can report the proportion of TRAINED (neural) opponents -- mirror
    # self-play + frozen past-selves -- vs the scripted/engine bots (random/attacker/
    # heuristic). Reset each report alongside win_iters.
    OPP_CATS = ("self", "pastself", "heuristic", "heuristic_1_1", "heuristic_1_2",
                "heuristic_1_3", "attacker", "random")
    opp_mix = {k: 0 for k in OPP_CATS}
    # Harvested anchor win-rates: [wins, games] vs each SCRIPTED anchor this window,
    # from the pool games training plays anyway. Counted under the EVAL convention
    # (metrics.py docstring): denominator = ALL games incl. draws/truncations, wins =
    # strictly decided for the learner -- so the eval service can pool these with its
    # own top-up games into one estimate. Distinct from league.update (EMA,
    # decided-only) and from opp_mix (game counts, no outcomes).
    AWIN_KINDS = ("heuristic", "heuristic_1_1", "heuristic_1_2", "heuristic_1_3",
                  "attacker", "random")
    awin = {k: [0, 0] for k in AWIN_KINDS}
    # Near-live per-iteration ticks (fishrl.serve's SSE feed; reports stay the hourly
    # durable record). Buffered in memory, flushed to ticks.json at most every
    # cfg.tick_every_seconds; the file keeps only the newest TICK_KEEP rows. Single
    # writer + atomic replace -> readers (fishrl.serve) need no lock. Existing rows
    # are re-read at the first flush so a resume continues the ring, not resets it.
    TICK_KEEP = 2000
    ticks: list = []             # the ring, seeded from disk so resume continues it
    if checkpoint_path is not None and cfg.tick_every_seconds > 0:
        sysstats.start()         # background %cpu/%ram/%gpu sampler feeding the ticks
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
    # Critic calibration is scored on each batch BEFORE the update touches it (an
    # honest online held-out estimate: the critic has never seen these games). Scored
    # after the update it measured the fit to the batch just trained on, which a
    # critic that can fingerprint games (critic_view="hands") drives to ~0 Brier
    # regardless of how it generalises (2026-08-21; held-out probe showed .28 while
    # the post-update number read .03).
    last_pre_est: dict = {}
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
        league.add_scenarios(cfg, scenario_names())   # weight x scenario_boost
    elif cfg.scenario_frac > 0:
        from fishrl.train.scenarios import scenario_names
        scen_league = PFSPLeague.scenario_league(cfg, scenario_names())
    run_start = last_report = last_ckpt = time.perf_counter()

    def total_elapsed() -> float:
        return elapsed_offset + (time.perf_counter() - run_start)

    def _payload() -> dict:
        return {
            "format": ckpt.FORMAT,
            # Full architecture record so every loader (config_from_checkpoint) rebuilds
            # the EXACT model shapes -- hidden/actor_hidden/card_dim were previously implicit
            # (always the defaults), which would silently mismatch a resized model.
            "config": {"seed": cfg.seed, "encoders": _encoders(cfg),
                       "use_belief": cfg.use_belief, "critic_hidden": list(cfg.critic_hidden),
                       "hidden": list(cfg.hidden),
                       "actor_hidden": (list(cfg.actor_hidden)
                                        if cfg.actor_hidden is not None else None),
                       "card_dim": cfg.card_dim,
                       "belief_mode": cfg.belief_mode, "critic_view": cfg.critic_view,
                       "critic_deckout_aux": cfg.critic_deckout_aux,
                       "text_change_mode": cfg.text_change_mode,
                       "obs_counts": cfg.obs_counts},
            "done": done, "elapsed": total_elapsed(), "frozen_it": frozen_it,
            "handoff_start": handoff_start,
            "warmup_done": True,
            "models": _model_state(m), "frozen": _model_state(frozen),
            "optim": {"ppo": opt_ppo.state_dict(),
                      **({"g": opt_g.state_dict()} if opt_g is not None else {}),
                      **({"p": opt_p.state_dict()} if opt_p is not None else {})},
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
        est = dict(last_pre_est)                         # pre-update (held-out) calibration
        if not cfg.train_public:                         # pub head disabled -> its calib is meaningless
            est = {k: v for k, v in est.items() if not k.startswith("pub_")}
        gmae = (guesser_mae(m, last_batch)
                if (last_batch is not None and m.guesser is not None) else float("nan"))
        v3 = cfg.critic_view in features.PUBLIC_FAMILY
        eval_s = time.perf_counter() - t0
        dt_h = max(now - last_report, 1e-9) / 3600.0
        nan = float("nan")
        tag = "final" if final else f"{total_elapsed() / 3600.0:.2f}h"
        wr_str = (
            f" | WR frozen@{frozen_it}={wr.get('frozen', nan):.2f} "
            f"random={wr['random']:.2f} attacker={wr['attacker']:.2f} "
            f"heuristic={wr['heuristic']:.2f} heuristic11={wr.get('heuristic11', float('nan')):.2f} "
            f"heuristic12={wr.get('heuristic12', float('nan')):.2f} "
            f"heuristic13={wr.get('heuristic13', float('nan')):.2f} "
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
        wall = gwin["collect_s"] + gwin["update_s"] + gwin["book_s"]
        collect_frac = (gwin["collect_s"] / wall) if wall > 0 else nan
        book_frac = (gwin["book_s"] / wall) if wall > 0 else nan
        # ratio of the window MEANS (per-iteration g2 is a difference of two noisy
        # estimates and can hit zero/negative -> inf; the means are well-behaved)
        _g2 = acc["gns_g2"] / max(win_iters, 1)
        gns_b = (acc["gns_tr_sigma"] / max(win_iters, 1)) / _g2 if _g2 > 0 else nan
        wt = pcol.pop_timing() if pcol is not None else {}
        stackprof.report(log)
        brier_gap = (est["pub_brier"] - est["priv_brier"]) if "pub_brier" in est else nan
        game_str = (
            f" | games={games} len={len_full:.1f} slen={len_scen:.1f} "
            f"trunc={trunc_rate:.2f} draw={draw_rate:.2f} "
            f"seat_p1={seat_p1:.2f} fdec={fdec:.2f} | wall collect={collect_frac:.2f} book={book_frac:.2f} gns_b={gns_b:.0f}"
            + (f" worker gap/play={wt['worker_gap_s']:.2f}/{wt['worker_play_s']:.2f}s" if wt else "")
        )
        if v3:
            # v3 status format: no guesser/public tokens; one critic calib group +
            # the parity-aux loss. Parsed by backfill_stats' both-era regex.
            # per-turn Brier (t1-10/11-20/21-30/31+) rides after aux=; backfill's regex
            # stops at aux= so the token is tolerated by both eras' parsers
            by_turn = "/".join(f"{est.get(f'critic_brier_{lbl}', nan):.2f}"
                               for lbl, _lo, _hi in features.TURN_BUCKETS)
            calib_str = (
                f"calib critic(acc={est.get('critic_acc', nan):.2f},"
                f"brier={est.get('critic_brier', nan):.2f}) "
                f"aux={mean.get('deckout_aux_loss', nan):.3f} brier/t={by_turn} "
                f"jump={est.get('critic_jump_mean', nan):.3f}/{est.get('critic_jump_p90', nan):.3f}"
            )
            head_str = ""
        else:
            calib_str = (
                f"calib priv(acc={est.get('priv_acc', nan):.2f},brier={est.get('priv_brier', nan):.2f}) "
                f"pub(acc={est.get('pub_acc', nan):.2f},brier={est.get('pub_brier', nan):.2f}) "
                f"gap={brier_gap:.2f} gmae={gmae:.2f}"
            )
            head_str = f"guess={mean['guesser_loss']:.3f} pub={mean['public_loss']:.3f} "
        log(
            f"[status {tag} it={done} (+{win_iters}, {win_iters / dt_h:.1f}/h) T={win_T}] "
            f"pi={mean['policy_loss']:.3f} V={mean['critic_loss']:.3f} "
            f"H={mean['entropy']:.3f} kl={mean['approx_kl']:.4f} "
            f"clip={mean['clip_frac']:.2f} "
            + head_str + "| " + calib_str + opp_str + game_str + wr_str
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
                # ...and on what device. Three PC sessions once ran CPU updates
                # unnoticed (launchers missing --gpu); the dashboard shows this.
                "device": str(cfg.device),
                # Regime marker for A/B analysis: rows trained with pipelined
                # (one-update-stale) collection must be distinguishable later.
                # Sparse on purpose -- absent means strictly on-policy.
                **({"pipeline": True} if pipeline else {}),
                **({"stream": True} if stream else {}),
                "iters": win_iters, "iters_per_h": win_iters / dt_h, "transitions": win_T,
                "policy_loss": mean["policy_loss"], "critic_loss": mean["critic_loss"],
                "entropy": mean["entropy"], "approx_kl": mean["approx_kl"],
                # Era-keyed calibration block: v3 rows carry critic_* (+ the parity-aux
                # loss); legacy rows keep the historic guesser/priv/pub/gmae fields.
                **({"critic_acc": est.get("critic_acc"),
                    "critic_brier": est.get("critic_brier"),
                    "deckout_aux_loss": mean.get("deckout_aux_loss"),
                    # per-turn calibration buckets (pre-update, same forward)
                    **{k: est.get(k) for lbl, _lo, _hi in features.TURN_BUCKETS
                       for k in (f"critic_brier_{lbl}", f"critic_acc_{lbl}", f"critic_n_{lbl}")},
                    # turn-by-turn arrays (index t-1; last slot pools TURN_MAX+), same forward
                    "critic_turn": est.get("critic_turn"),
                    # deterministic-transition |dV| (same seat, same game, same turn, no
                    # opponent decision between): mean / p90 / pair count
                    "critic_jump_mean": est.get("critic_jump_mean"),
                    "critic_jump_p90": est.get("critic_jump_p90"),
                    "critic_jump_n": est.get("critic_jump_n")}
                   if v3 else
                   {"guesser_loss": mean["guesser_loss"], "public_loss": mean["public_loss"],
                    "priv_acc": est.get("priv_acc"), "pub_acc": est.get("pub_acc"),
                    "priv_brier": est.get("priv_brier"), "pub_brier": est.get("pub_brier"),
                    "brier_gap": brier_gap, "gmae": gmae}),
                "clip_frac": mean["clip_frac"],
                # window game telemetry (NaN -> null via the sanitizer below)
                "games": games, "dec_per_game": len_full, "scen_dec_per_game": len_scen,
                "trunc_rate": trunc_rate, "draw_rate": draw_rate,
                "mirror_p1_wr": seat_p1, "forced_dec_frac": fdec,
                "collect_s": gwin["collect_s"], "update_s": gwin["update_s"],
                "collect_frac": collect_frac, "book_s": gwin["book_s"], "book_frac": book_frac,
                # gradient noise scale (actor, epoch-0 minibatch grads): critical batch in
                # decisions; compare with T/iter to read the games/iter slack
                "gns_b": gns_b, "gns_tr_sigma": acc["gns_tr_sigma"] / max(win_iters, 1),
                "gns_g2": acc["gns_g2"] / max(win_iters, 1),
                **wt,
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
                "opp_heuristic13": opp_mix["heuristic_1_3"] / grand_total,  # testbench v1.3
                "opp_attacker": opp_mix["attacker"] / grand_total,
                "opp_random": opp_mix["random"] / grand_total,
                "opp_scenario": scen_total / grand_total,
                "scenario_mix": {k: v / grand_total for k, v in sorted(scen_mix.items())},
                # PFSP curriculum win-rate EMA per scenario (the sampler's difficulty
                # signal, NOT a skill measure) — mirrors the [scenario] line's wr table
                # so the website replica gets it over HTTP instead of journald.
                "scenario_wr": {mm.name: mm.wr for mm in scen_members},
                # Same for the scripted league anchors, keyed by profile name —
                # mirrors the [league] line's wr table. Matters most for the
                # versioned heuristics (heuristic_1_1/_1_2/_1_3), which have
                # no eval anchor, so this EMA is their only win-rate signal.
                # Members that never played stay out (their 0.5 prior isn't data).
                "league_wr": {mm.name: mm.wr for mm in league.anchors
                              if mm.kind in SCRIPTED_KINDS and mm.games > 0},
                # Harvested anchor outcomes this window as [wins, games] under the eval
                # convention -- the eval service tops each anchor up to its target and
                # publishes the combined estimate. Keyed with the EVAL names.
                "wr_train": {"heuristic": list(awin["heuristic"]),
                             "heuristic11": list(awin["heuristic_1_1"]),
                             "heuristic12": list(awin["heuristic_1_2"]),
                             "heuristic13": list(awin["heuristic_1_3"]),
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
                    "heuristic12": wr.get("heuristic12"),
                    "heuristic13": wr.get("heuristic13"),
                    "new_best": False, "source": "inline",
                })
        if cfg.pool_frac > 0 and league.members():        # PFSP composition + win-rate table
            # The paren-count format is parsed by the website collector and by
            # backfill_stats with a STRICT 5-token sequence, so `heuristic=` prints the
            # MERGED v1.0+v1.1+v1.2+v1.3 count to keep the format stable. The
            # per-version split rides AFTER the closing paren as `h11=`/`h12=`/`h13=`
            # tokens (strict parsers stop at the paren and ignore them; split-aware
            # ones subtract them out), and also lives in stats.json (opp_heuristic /
            # opp_heuristic11 / opp_heuristic12 / opp_heuristic13) and the wr table
            # after the bar, which names each versioned profile separately.
            log(f"[league it={done}] games={mix_total} trained={trained_frac:.2f} "
                f"(self={opp_mix['self']} past={opp_mix['pastself']} "
                f"heuristic={opp_mix['heuristic'] + opp_mix['heuristic_1_1'] + opp_mix['heuristic_1_2'] + opp_mix['heuristic_1_3']} "
                f"attacker={opp_mix['attacker']} "
                f"random={opp_mix['random']}) "
                f"h11={opp_mix['heuristic_1_1']} h12={opp_mix['heuristic_1_2']} "
                f"h13={opp_mix['heuristic_1_3']} | "
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
    if stop_file is not None and os.path.exists(stop_file):
        try:                                             # a STOP predating this process is not
            os.remove(stop_file)                         # ours to honor -- clearing it stops a
            log("[stop] cleared a stale STOP from a previous session")  # leftover from killing a
        except OSError:                                  # fresh --resume start dead on arrival
            pass
    if checkpoint_path is not None:
        def _on_signal(signum, frame):
            stop["v"] = True
        for sig in (signal.SIGTERM, signal.SIGINT):
            prev_handlers[sig] = signal.signal(sig, _on_signal)

    # Parallel local collection (opt-in): a persistent pool of collector worker
    # processes that plays each iteration's games with THIS iteration's weights.
    # 0 = the serial path below, unchanged (the ODROID service default).
    pcol = None
    if getattr(cfg, "collect_workers", 0) > 0:
        from fishrl.train.pcollect import CollectorStopped, ParallelCollector
        # A stop requested mid-collection (STOP file or a signal) must be seen even
        # while a recovery storm holds the collector -- not only at the loop boundary.
        _stop_now = (lambda: stop["v"] or (stop_file is not None and os.path.exists(stop_file)))
        if os.name == "posix" and int(getattr(cfg, "shm_pool_blocks", 0) or 0) == 0:
            depth = int(getattr(cfg, "stream_depth", 2)) if getattr(cfg, "collect_stream", False) else 1
            cfg.shm_pool_blocks = (depth + 1) * int(cfg.games_per_iter) + int(cfg.collect_workers)
        pcol = ParallelCollector(cfg, int(cfg.collect_workers), should_stop=_stop_now)
        log(f"[pcollect] {cfg.collect_workers} collector worker processes")
    pipeline = bool(getattr(cfg, "pipeline_collect", False))
    if pipeline and pcol is None:
        log("[pcollect] --pipeline-collect ignored: needs --collect-workers > 0 "
            "(a serial trainer has nothing to overlap the update with)")
        pipeline = False
    if pipeline:
        log("[pcollect] PIPELINED collection: iteration N+1's games play during "
            "N's update (behavior policy one update stale; see Config.pipeline_collect)")
    pipe_pending = None                # (specs, metas, futures) of the in-flight iteration
    stackprof.start()
    # A batch is ~100k Step objects; the default thresholds (700, 10, 10) made the cyclic
    # GC traverse them on every few hundred allocations (~20% of the iteration showed up
    # as dealloc/GC at the take() seam). Nothing here is cyclic: collect rarely, and
    # freeze the long-lived model/league graph out of the traversals entirely.
    gc.set_threshold(200_000, 50, 50)
    gc.freeze()
    stream = pipeline and bool(getattr(cfg, "collect_stream", False))
    if stream:
        log(f"[pcollect] STREAMED collection: one task per game, {cfg.stream_depth} "
            f"iteration-sized sets in flight, games consumed in completion order "
            f"(see Config.collect_stream)")
    stream_items: list = []            # in-flight (future, chunk, seed) single-game tasks
    stream_metas: list = []            # their bookkeeping metas, aligned with stream_items
    stream_it = 0                      # spec-set counter (seeds / league sampling streams)

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
            if opt_g is not None and "g" in payload["optim"]:
                opt_g.load_state_dict(payload["optim"]["g"])
            if opt_p is not None and "p" in payload["optim"]:
                opt_p.load_state_dict(payload["optim"]["p"])
            frozen = copy.deepcopy(m)
            _load_model_state(frozen, payload["frozen"])
            for _n, net in _present_nets(frozen):
                net.eval()
                for p in net.parameters():
                    p.requires_grad_(False)
            done = int(payload["done"])
            frozen_it = int(payload.get("frozen_it", done))
            handoff_start = int(payload.get("handoff_start", 0))
            elapsed_offset = float(payload.get("elapsed", 0.0))
            _set_rng_state(payload["rng"])
            # Restore league continuity (EMA win-rates + past-self ring). Older
            # checkpoints carry no league keys -> fresh leagues, the old behaviour.
            def _opponent_nets():
                # Past-self opponents must match the LEARNER's actor/guesser shapes
                # (head widths + card_dim), or their saved weights won't load. In
                # bookkeeper/none belief modes there is no guesser to rebuild.
                actor = MaskedActor(cfg.head_hidden("actor"), cfg.enc_for("actor"),
                                    cfg.card_dim, in_dim=m.actor.in_dim).to(cfg.device)
                guesser = None
                if cfg.has_guesser:
                    guesser = HandGuesser(cfg.head_hidden("guesser"), cfg.enc_for("guesser"),
                                          cfg.card_dim).to(cfg.device)
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

        # BC-handoff KL anchor: snapshot the actor AS LOADED (on a resume from a
        # fishrl.imitate.bc checkpoint this is the teacher clone) before any PPO
        # update moves it. Frozen for the whole run; the coefficient anneals to zero.
        teacher_ref = None
        kl_until = handoff_start + cfg.kl_teacher_iters
        freeze_until = handoff_start + cfg.freeze_actor_iters
        if cfg.kl_teacher_coef > 0.0 and cfg.kl_teacher_iters > 0 and done < kl_until:
            teacher_ref = copy.deepcopy(m.actor)
            teacher_ref.eval()
            for p in teacher_ref.parameters():
                p.requires_grad_(False)
            log(f"[handoff] KL-to-teacher anchor snapshotted at it={done} "
                f"(coef={cfg.kl_teacher_coef}, anneal to it={kl_until}"
                f"{f', handoff_start={handoff_start}' if handoff_start else ''})")
        if freeze_until > done:
            log(f"[handoff] actor FROZEN until it={freeze_until} "
                f"(critic-only warmup{f', handoff_start={handoff_start}' if handoff_start else ''})")

        benv = BeliefAugmentedEnv(m.guesser, mode=cfg.belief_mode,
                                  max_decisions=cfg.max_decisions)

        def _split_games() -> tuple:
            """(n_self, n_pool, n_scen) for one iteration -- pure cfg, shared by the
            serial branch and _pspecs so the two can never drift apart."""
            n_scen = (int(round(cfg.games_per_iter * cfg.scenario_frac))
                      if cfg.scenario_frac > 0 else 0)
            n_scen = max(0, min(n_scen, cfg.games_per_iter))
            if cfg.scenarios_in_pool:
                n_scen = 0                              # scenarios come out of the pool budget
            rest = cfg.games_per_iter - n_scen
            n_pool = int(round(rest * cfg.pool_frac))
            n_pool = max(0, min(n_pool, rest))
            return rest - n_pool, n_pool, n_scen

        def _pspecs(done_it: int) -> tuple:
            """Game specs + bookkeeping metas for PARALLEL collection of iteration
            `done_it`. Makes EXACTLY the same sampling decisions (same RNG streams,
            seeds and seat parity) as the serial branch in the loop below -- keep
            the two in lockstep when editing either. A function of the iteration
            counter so pipelined mode can build iteration N+1's specs before N's
            bookkeeping lands (league EMAs are then one iteration stale for
            sampling -- part of the disclosed staleness regime)."""
            n_self, n_pool, n_scen = _split_games()
            seed = cfg.seed + 1000 + done_it * cfg.games_per_iter
            specs, metas = [], []
            for gi in range(n_self):
                specs.append({"kind": "self", "seed": seed + gi})
                metas.append(("mirror", None, None))
            pool_rng = np.random.default_rng(cfg.seed + 900_000 + done_it)
            for pidx in range(n_pool):
                oseed = cfg.seed + 500_000 + done_it * cfg.games_per_iter + pidx * 17
                member = league.sample(pool_rng)
                if member is None:                      # empty league -> mirror fallback
                    specs.append({"kind": "self", "seed": oseed})
                    metas.append(("mirror", None, None))
                elif member.kind == "scenario":
                    specs.append({"kind": "scenario", "name": member.name, "seed": oseed})
                    metas.append(("pool_scen", member, "p1"))
                elif member.kind.startswith("heuristic"):
                    specs.append({"kind": "heuristic", "profile": member.kind, "seed": oseed})
                    metas.append(("pool", member, "p1"))
                else:
                    lseat = "p1" if pidx % 2 == 0 else "p2"
                    spec = {"kind": "opponent", "okind": member.kind,
                            "seed": oseed, "lseat": lseat}
                    if member.kind == "self":           # frozen nets ride with the spec
                        spec["opp_state"] = pcol.opp_state_for(member)
                    specs.append(spec)
                    metas.append(("pool", member, lseat))
            if n_scen > 0 and scen_league is not None and scen_league.members():
                scn_rng = np.random.default_rng(cfg.seed + 700_000 + done_it)
                for sidx in range(n_scen):
                    smember = scen_league.sample(scn_rng)
                    sseed = cfg.seed + 300_000 + done_it * cfg.games_per_iter + sidx * 31
                    specs.append({"kind": "scenario", "name": smember.name, "seed": sseed})
                    metas.append(("carve_scen", smember, "p1"))
            return specs, metas

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
            n_self, n_pool, n_scen = _split_games()
            seed = cfg.seed + 1000 + done * cfg.games_per_iter
            t_collect = time.perf_counter()
            if pcol is not None:
                # ── Parallel path. Specs come from _pspecs (kept in lockstep with the
                # serial branch in the else:); buffers come back in spec order and every
                # piece of bookkeeping is applied here on the main thread.
                try:
                    if stream:
                        # Streamed: keep stream_depth sets in flight; consume the next
                        # games_per_iter FINISHED games whatever set they came from,
                        # then top up with one set on the current (pre-update) weights.
                        def _stream_fill(n_sets: int):
                            nonlocal stream_it
                            for _ in range(n_sets):
                                sspecs, smetas = _pspecs(stream_it)
                                stream_items.extend(pcol.submit(m, sspecs, stream_it, per_game=True))
                                stream_metas.extend(smetas)
                                stream_it += 1
                        if not stream_items:
                            stream_it = done
                            _stream_fill(int(cfg.stream_depth))
                        got, rest = pcol.take(stream_items, cfg.games_per_iter)   # noqa: F841
                        taken = {i for i, _b in got}
                        bufs = [b for _i, b in got]
                        metas = [stream_metas[i] for i, _b in got]
                        stream_metas = [mt for i, mt in enumerate(stream_metas) if i not in taken]
                        stream_items = rest
                        _stream_fill(1)
                    elif pipeline:
                        # This iteration's games were submitted during the previous one
                        # (first pass: submit now). Then start N+1's games BEFORE the
                        # update below, with the current -- pre-update -- weights: that is
                        # exactly the one-update staleness the flag buys throughput with.
                        # collect_s under pipeline therefore measures the STALL waiting
                        # for workers, not the games' wall-clock.
                        if pipe_pending is None:
                            specs, metas = _pspecs(done)
                            pipe_pending = (specs, metas, pcol.submit(m, specs, done))
                        specs, metas, futs = pipe_pending
                        bufs = pcol.gather(futs, len(specs))
                        nspecs, nmetas = _pspecs(done + 1)
                        pipe_pending = (nspecs, nmetas, pcol.submit(m, nspecs, done + 1))
                    else:
                        specs, metas = _pspecs(done)
                        bufs = pcol.collect(m, specs, done)   # strictly on-policy
                except CollectorStopped:
                    # A stop landed while a resource storm held the collector in
                    # recovery. Consume the STOP (the loop-top poll won't run now) and
                    # exit gracefully -- the checkpoint below the loop still fires.
                    if stop_file is not None and os.path.exists(stop_file):
                        try:
                            os.remove(stop_file)
                        except OSError:
                            pass
                    stop["v"] = True
                    log(f"[stop] stop requested during collection at it={done}")
                    break
                buf = RolloutBuffer()
                for (tag, member, lseat), gbuf in zip(metas, bufs):
                    won = gbuf.games[-1] if gbuf.games else None
                    if tag == "mirror":
                        opp_mix["self"] += 1
                        gwin["mirror_dec"] += sum(1 for w in gbuf.games if w in ("p1", "p2"))
                        gwin["mirror_p1"] += sum(1 for w in gbuf.games if w == "p1")
                    elif tag in ("pool_scen", "carve_scen"):
                        scen_mix[member.name] = scen_mix.get(member.name, 0) + 1
                        gwin["scen_games"] += 1
                        gwin["scen_T"] += len(gbuf.steps)
                        if won in ("p1", "p2"):
                            (league if tag == "pool_scen" else scen_league).update(
                                member, won == "p1")
                    else:                                   # pool game vs a league member
                        opp_mix["pastself" if member.kind == "self" else member.kind] += 1
                        if won in ("p1", "p2"):
                            league.update(member, won == lseat)
                        if member.kind in AWIN_KINDS:       # harvest: all games / strict wins
                            _harvest_count(awin[member.kind], gbuf.games, lseat)
                    buf.merge(gbuf)
            else:
                buf = collect_games(benv, actor_act_fn(m.actor), n_self, seed,
                                    critic=None, max_decisions=cfg.max_decisions,
                                    critic_view=cfg.critic_view)
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
                                             critic=None, max_decisions=cfg.max_decisions,
                                             critic_view=cfg.critic_view)
                        gwin["mirror_dec"] += sum(1 for w in gbuf.games if w in ("p1", "p2"))
                        gwin["mirror_p1"] += sum(1 for w in gbuf.games if w == "p1")
                        buf.merge(gbuf)
                        continue
                    if member.kind == "scenario":               # pool-mode curriculum game:
                        from fishrl.train.scenarios import ScenarioEnv, get_scenario
                        lseat = "p1"                            # learner is p1, p2 the engine bot
                        senv = BeliefAugmentedEnv(
                            m.guesser, mode=cfg.belief_mode,
                            env=ScenarioEnv(get_scenario(member.name),
                                            max_decisions=cfg.max_decisions))
                        gbuf = collect_games(senv, actor_act_fn(m.actor), 1, oseed,
                                             critic=None, max_decisions=cfg.max_decisions,
                                             critic_view=cfg.critic_view)
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
                                                       critic=None,
                                                       belief_mode=cfg.belief_mode,
                                                       max_decisions=cfg.max_decisions,
                                                       profile=member.kind,
                                                       critic_view=cfg.critic_view)
                    else:                                       # scripted / past-self, seat-balanced
                        lseat = "p1" if pidx % 2 == 0 else "p2"
                        gbuf = collect_vs_opponent(m, member, 1, oseed, critic=None,
                                                   belief_mode=cfg.belief_mode,
                                                   max_decisions=cfg.max_decisions,
                                                   learner_seat=lseat,
                                                   critic_view=cfg.critic_view)
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
                            m.guesser, mode=cfg.belief_mode,
                            env=ScenarioEnv(get_scenario(sname), max_decisions=cfg.max_decisions))
                        sseed = cfg.seed + 300_000 + done * cfg.games_per_iter + sidx * 31
                        sbuf = collect_games(senv, actor_act_fn(m.actor), 1, sseed,
                                             critic=None, max_decisions=cfg.max_decisions,
                                             critic_view=cfg.critic_view)
                        gwin["scen_games"] += 1
                        gwin["scen_T"] += len(sbuf.steps)       # scenario episode lengths
                        buf.merge(sbuf)
                        scen_mix[sname] = scen_mix.get(sname, 0) + 1
                        # the learner is p1 (p2 is the engine bot); count the game even if
                        # the learner never got a decision before it ended
                        if sbuf.games and sbuf.games[-1] in ("p1", "p2"):
                            scen_league.update(member, sbuf.games[-1] == "p1")
            fill_critic_values(buf, m.critic, view=cfg.critic_view)
            # Window game telemetry from the merged buffer: how games ended (buf.meta),
            # forced-decision dilution (1-legal-action steps), collect wall-clock.
            gwin["games"] += len(buf.games)
            for w, gm in zip(buf.games, buf.meta):
                if gm["truncated"]:
                    gwin["trunc"] += 1
                elif w is None:
                    gwin["draw"] += 1
            gwin["forced_steps"] += sum(1 for s in buf.steps if int(s.mask.sum()) == 1)
            iter_collect_s = time.perf_counter() - t_collect
            gwin["collect_s"] += iter_collect_s
            t_book = time.perf_counter()
            batch = buf.compute(cfg.gamma, cfg.lam, p1_adv_weight=cfg.p1_adv_weight)
            gwin["book_s"] += time.perf_counter() - t_book   # serial GAE + stacking (neither GPU nor workers)
            n_buf = len(buf)
            blk_ids = buf.release()             # views into shared memory die here (fd per block)
            bufs = got = None                   # the per-game buffers share those Step objects
            gc.collect(0)                       # reap ~100k Steps + 360 mappings now, off the seam
            if pcol is not None and blk_ids:
                pcol.recycle(blk_ids)           # pooled blocks back to the workers
            if max_seconds is not None:                  # anneal entropy over the budget
                frac = min(total_elapsed() / max_seconds, 1.0)
                ent = cfg.ent_start + frac * (cfg.ent_end - cfg.ent_start)
            else:
                ent = cfg.ent_coef(done)
            t_update = time.perf_counter()
            freeze = done < freeze_until
            if cfg.freeze_actor_iters > 0 and done == freeze_until:
                log(f"[handoff] actor unfrozen at it={done}")
            kl_now = 0.0
            if teacher_ref is not None and done < kl_until:
                kl_now = cfg.kl_teacher_coef * (1.0 - (done - handoff_start) / cfg.kl_teacher_iters)
            last_pre_est = estimator_metrics(m, batch)   # score BEFORE fitting this batch
            ppo_stats = ppo_update(batch, m.actor, m.critic, opt_ppo, cfg, ent, rng_seed=done,
                                   ref_actor=teacher_ref, kl_ref_coef=kl_now,
                                   train_actor=not freeze)
            if m.guesser is not None or (m.public is not None and cfg.train_public):
                aux_stats = aux_update(batch, m.guesser, m.public, opt_g, opt_p,
                                       cfg.aux_steps, train_public=cfg.train_public)
            else:                          # v3: no supervised aux heads exist
                aux_stats = {"guesser_loss": 0.0, "public_loss": float("nan")}
            iter_update_s = time.perf_counter() - t_update
            gwin["update_s"] += iter_update_s
            for k in ("policy_loss", "critic_loss", "entropy", "approx_kl", "clip_frac",
                      "deckout_aux_loss", "gns_b", "gns_tr_sigma", "gns_g2"):
                acc[k] += ppo_stats[k]
            acc["guesser_loss"] += aux_stats["guesser_loss"]
            acc["public_loss"] += aux_stats["public_loss"]
            win_iters += 1
            win_T += n_buf
            last_batch = batch
            done += 1
            if checkpoint_path is not None and cfg.tick_every_seconds > 0:
                sysv = sysstats.latest()
                tick = {
                    "it": done, "wall_time": time.time(), "T": n_buf,
                    "games": len(buf.games),
                    "policy_loss": ppo_stats["policy_loss"],
                    "critic_loss": ppo_stats["critic_loss"],
                    "entropy": ppo_stats["entropy"], "approx_kl": ppo_stats["approx_kl"],
                    **({"kl_teacher": ppo_stats["kl_teacher"]} if kl_now > 0 else {}),
                    "guesser_loss": aux_stats["guesser_loss"],
                    "public_loss": aux_stats["public_loss"],
                    **({"deckout_aux_loss": ppo_stats["deckout_aux_loss"]}
                       if cfg.critic_view in features.PUBLIC_FAMILY else {}),
                    "collect_s": round(iter_collect_s, 3), "update_s": round(iter_update_s, 3),
                    # machine utilization at the last sampler beat (%; None = unknown) --
                    # the dashboard's system panel plots these over iteration
                    "cpu": None if sysv["cpu"] is None else round(sysv["cpu"], 1),
                    "ram": None if sysv["ram"] is None else round(sysv["ram"], 1),
                    "gpu": None if sysv["gpu"] is None else round(sysv["gpu"], 1),
                }
                # NaN/inf -> null like the report rows: python's json would emit
                # BARE NaN (v3's public_loss), which is not JSON — the browser's
                # JSON.parse rejects the whole payload and the dashboard header
                # freezes while python-side consumers parse it fine.
                ticks.append({k: (None if isinstance(v, float) and not math.isfinite(v)
                                  else v) for k, v in tick.items()})
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
        if pcol is not None:
            pcol.close()
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
