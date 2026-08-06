"""Out-of-band, parallel win-rate panel.

The inline panel in `train_loop` is single-threaded sequential self-play (a batch-1
forward per decision), so it scales linearly with games and *blocks* the training loop
while it runs -- ~2.6 min for n=10, so n=100 would stall training ~26 min/report. This
module instead reads a checkpoint and fans the games out across processes (each worker
pinned to one BLAS thread; parallelism is across processes, not within a forward), so a
100-game panel finishes in a few minutes on the spare cores *without* pausing the trainer.

Run it from a systemd timer (see fishrl-eval.{service,timer}) against `latest.pt`:

    python -m fishrl.eval.parallel_panel --ckpt-dir checkpoints --n-games 100

Games are split into contiguous, seed-aligned chunks, so the union of seeds per anchor is
identical to the serial panel's -- the parallel number is the same estimate, just faster.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

NETS = ("actor", "critic", "guesser", "public")

# A rolling "best so far" checkpoint, selected by win-rate vs the heuristic anchor. best.pt is a
# full, resume-able checkpoint (same schema as latest.pt); best.json records the winning panel.
BEST, BEST_META = "best.pt", "best.json"

# Per-worker globals, populated by _init in each subprocess (set once, reused across tasks).
_G: dict = {}


def _init(cfg_dict: dict, models_state: dict, frozen_state: dict, max_decisions: int,
          affinity: list | None = None) -> None:
    """Subprocess initializer: rebuild the two policies once per worker and pin to 1 thread.

    Parallelism here is across processes; letting each worker also spin up a BLAS/torch
    thread pool would oversubscribe the box, so we cap intra-op threads to 1. `affinity`
    optionally restricts the worker to given logical processors -- on the hybrid PC the
    launchers park panel workers on E-cores so a mid-session panel never time-slices
    against the trainer's P-core-pinned collectors (see Config.collect_affinity)."""
    import torch

    from fishrl.train.pcollect import _apply_affinity
    from fishrl.train.train_loop import build_models, config_from_checkpoint, _load_model_state

    torch.set_num_threads(1)
    _apply_affinity(affinity or [])
    cfg = config_from_checkpoint(cfg_dict)      # full architecture: encoders + head widths + card_dim
    m = build_models(cfg)
    _load_model_state(m, models_state)
    frozen = build_models(cfg)
    _load_model_state(frozen, frozen_state)
    _G.update(m=m, frozen=frozen, ub=cfg.use_belief, md=max_decisions)


def _seed_torch(seed: int) -> None:
    """Pin torch's RNG so the (stochastic) action sampling is reproducible. Without this the
    win rates carry action-sampling noise on top of the policy, so hour-over-hour deltas would
    be partly RNG; chunk boundaries are deterministic (n_games, workers), so a fixed checkpoint
    + worker count yields the same panel every time."""
    import torch
    torch.manual_seed(seed)


def _task_random(start: int, count: int, seed: int) -> int:
    from fishrl.eval.metrics import winrate_vs_random
    _seed_torch(seed + start)
    v = winrate_vs_random(_G["m"], n_games=count, seed=seed + start,
                          max_decisions=_G["md"], use_belief=_G["ub"])
    return round(v * count)


def _task_attacker(start: int, count: int, seed: int) -> int:
    from fishrl.eval.metrics import winrate_vs_attacker
    _seed_torch(seed + start)
    v = winrate_vs_attacker(_G["m"], n_games=count, seed=seed + start,
                            max_decisions=_G["md"], use_belief=_G["ub"])
    return round(v * count)


def _task_heuristic(start: int, count: int, seed: int, profile: str = "heuristic") -> int:
    from fishrl.eval.metrics import winrate_vs_heuristic
    _seed_torch(seed + start)
    v = winrate_vs_heuristic(_G["m"], n_games=count, seed=seed + start,
                             max_decisions=_G["md"], use_belief=_G["ub"], profile=profile)
    return round(v * count)


def _task_seat_diag(start: int, count: int, seed: int) -> dict:
    """One chunk of the seat / play-draw diagnostic: mirror self-play counters."""
    from fishrl.eval.metrics import seat_diag_counts
    _seed_torch(seed + start)
    return seat_diag_counts(_G["m"], n_games=count, seed=seed + start,
                            max_decisions=_G["md"], use_belief=_G["ub"])


def _task_match(trained_is_p1: bool, count: int, base_seed: int) -> tuple:
    """One orientation of the frozen match: returns (trained_wins, decided_games)."""
    from fishrl.eval.metrics import _match_models
    m, fz, ub = _G["m"], _G["frozen"], _G["ub"]
    _seed_torch(base_seed)
    if trained_is_p1:
        aw, bw = _match_models(m, ub, fz, ub, count, base_seed, _G["md"])
        return aw, aw + bw
    aw, bw = _match_models(fz, ub, m, ub, count, base_seed, _G["md"])
    return bw, aw + bw


def _even_chunks(n: int, k: int) -> list:
    """Split n into <=k contiguous (start, count) chunks with EVEN counts (so each chunk is
    internally seat-balanced and local i%2 == global i%2, matching the serial seat schedule).
    Requires n even; a trailing odd unit is folded into the last chunk."""
    base = (n // k) & ~1                      # even floor
    counts = [base] * k
    rem = n - base * k
    i = 0
    while rem > 0:                            # hand out the remainder 2 at a time, 1 if odd tail
        add = 2 if rem >= 2 else 1
        counts[i % k] += add
        rem -= add
        i += 1
    out, s = [], 0
    for c in counts:
        if c:
            out.append((s, c))
            s += c
    return out


def _chunks(n: int, k: int) -> list:
    """Contiguous (start, count) chunks, any size (used where seat parity is irrelevant)."""
    base, rem = divmod(n, k)
    out, s = [], 0
    for i in range(k):
        c = base + (1 if i < rem else 0)
        if c:
            out.append((s, c))
            s += c
    return out


# Per-anchor eval seeds -- identical to panel_winrates() so reports are comparable.
SEED_RANDOM, SEED_ATTACKER, SEED_HEURISTIC = 800_000, 700_000, 900_000
SEED_HEURISTIC11 = 950_000
SEED_HEURISTIC12 = 960_000
SEED_HEURISTIC13 = 970_000
SEED_FROZEN_A, SEED_FROZEN_B = 500_000, 510_000
SEED_SEAT_DIAG = 100  # matches metrics.selfplay_seat_diagnostics default seed band

# Anchors whose outcomes the trainer harvests from its own PFSP pool games (the
# report rows' "wr_train": anchor -> [wins, games], eval convention). Frozen-self
# is NOT harvestable: the league's past-selves are assorted ring members, not the
# last-report snapshot this panel matches against.
HARVEST_ANCHORS = ("heuristic", "heuristic11", "heuristic12", "heuristic13",
                   "attacker", "random")


def plan_topup(targets: dict, harvest: dict | None) -> dict:
    """Per-anchor plan: ``{anchor: (topup, wins_train, n_train)}``.

    Harvested pool games ADD to the estimate; they do not displace top-up. The panel
    always plays its full per-anchor target and the harvested games are EXTRA samples
    on top, so the combined sample is ``n_train + target``. Harvest therefore buys
    PRECISION -- a tighter win-rate curve -- at the historic panel cost.

    It previously returned ``max(0, target - n_train)``, which capped the combined
    sample at `target`: harvest bought a *cheaper* panel and the curves were no less
    noisy for it. The heuristic anchors are the ones the pool actually plays (~1k games
    per 5000-it dashboard window each), so uncapping them is what shrinks the noise
    band; the starved anchors (random/attacker, ~100-200 pool games) are carried by
    their top-up either way and stay the collapse canaries.

    `harvest` is a report row's ``wr_train`` (None, or a missing anchor -> no harvested
    games -> a plain full-target panel). The attacker top-up is rounded UP to even --
    its games alternate seats in pairs."""
    out = {}
    for k in HARVEST_ANCHORS:
        w, n = (harvest or {}).get(k) or (0, 0)
        topup = int(targets[k])
        if k == "attacker" and topup % 2:
            topup += 1
        out[k] = (topup, int(w), int(n))
    return out


def find_harvest(stats: dict, max_age_s: float, now: float) -> tuple:
    """Sum the harvest counts of EVERY report window not yet consumed by an earlier eval.

    Returns ``(summed_wr_train, high_water_it)``, or ``(None, None)`` when there is
    nothing to harvest (no row with ``wr_train`` -- a pre-harvest trainer -- or every
    fresh row is too old).

    Consumption is a HIGH-WATER MARK: an eval records the highest report ``it`` it
    consumed as ``harvest_from``, and a later eval takes every report ABOVE that. Evals
    run more often than reports, so the old "newest unconsumed row only" rule silently
    DROPPED any report that landed while another eval was in flight; summing all fresh
    rows recovers those games instead of discarding them. A window is still never
    double-counted -- the mark only moves forward.

    Rows older than `max_age_s` are skipped (a stalled/stopped trainer, or a backlog
    after an eval outage: their window no longer reflects the checkpoint being
    evaluated). Every miss degrades to a FULL panel, never a thinner one."""
    rows = [r for r in stats.get("reports", []) if r.get("wr_train")]
    if not rows:
        return None, None
    seen = [e.get("harvest_from") for e in stats.get("evals", [])]
    hwm = max([int(c) for c in seen if c is not None], default=-1)
    fresh = [r for r in rows
             if int(r.get("it", -1)) > hwm
             and now - (r.get("wall_time") or 0.0) <= max_age_s]
    if not fresh:
        return None, None
    total: dict = {}
    for r in fresh:
        for k, wn in r["wr_train"].items():
            e = total.setdefault(k, [0, 0])
            e[0] += int(wn[0])
            e[1] += int(wn[1])
    return total, max(int(r["it"]) for r in fresh)


def _maybe_save_best(ckpt_dir: str, payload: dict, r: dict) -> bool:
    """If this panel's heuristic win-rate beats the stored best, persist the *evaluated* payload
    to best.pt and record the panel in best.json. Returns True iff a new best was written.

    Saves the in-memory `payload` (the checkpoint we just evaluated) rather than re-reading
    latest.pt -- the trainer overwrites latest.pt every ~15 min and a panel takes ~90s, so a
    file copy could capture a *different* checkpoint than the one that earned this win-rate.
    best.json (atomic write) is the source of truth for the threshold; a missing/corrupt file
    means "no best yet", so the first successful panel always seeds it."""
    from fishrl.train.checkpoint import save_checkpoint

    meta_path = os.path.join(ckpt_dir, BEST_META)
    prev = None
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                prev = json.load(f)
        except (OSError, ValueError):
            prev = None
    if prev is not None and r["heuristic"] <= prev.get("heuristic", -1.0):
        return False

    save_checkpoint(os.path.join(ckpt_dir, BEST), payload)
    tmp = meta_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(r, f, indent=2)
    from fishrl.train.checkpoint import replace_with_retry
    replace_with_retry(tmp, meta_path)                # Windows: tolerate a monitor read
    return True


def parallel_panel(ckpt_path: str, n_games: int = 100, max_workers: int | None = None,
                   max_decisions: int = 2000, save_best: bool = False,
                   reserve_cores: int = 0, n_random: int | None = None,
                   n_attacker: int | None = None, n_frozen: int | None = None,
                   harvest: dict | None = None, harvest_from: int | None = None,
                   affinity: str = "", seat_diag_games: int = 0) -> dict:
    """Compute win-rates vs random / attacker / heuristic / frozen-self from a checkpoint,
    fanning the games across a process pool. Returns the rates plus eval metadata.

    `n_games` is the versioned-heuristic (v1.0/1.1/1.2/1.3) target; `n_random`/`n_attacker`/`n_frozen`
    default to it (tiered targets: random saturates early, frozen is the most
    expensive anchor). The panel always plays the full target per anchor; with
    `harvest` (summed ``wr_train`` from the report windows since the last eval) those
    pool games are ADDED, so the published estimate is over ``n_train + target`` games
    -- more samples, not a cheaper panel. Consumers see one number per anchor either
    way. No harvest -> plain full-target panel, the historic behaviour.

    When `save_best`, also roll best.pt (highest heuristic win-rate seen) next to the
    checkpoint; the returned dict carries `new_best` (bool). The gate reads the same
    combined estimate (harvested forfeit games can only bias it LOW -- see train_loop's
    wr_train comment -- so it never falsely promotes)."""
    import torch  # noqa: F401  (ensure torch import cost is paid in the parent too)

    from fishrl.train.checkpoint import load_checkpoint

    pl = load_checkpoint(ckpt_path, map_location="cpu")
    cfg_dict = pl["config"]
    # reserve_cores keeps CPUs free for the rest of the machine; default 0 so
    # the ODROID timer invocation (and its seed-affecting worker count) is
    # unchanged. Workers already run with set_num_threads(1).
    workers = max_workers or max(1, min(6, (os.cpu_count() or 2) - reserve_cores))

    targets = {"heuristic": n_games, "heuristic11": n_games, "heuristic12": n_games,
               "heuristic13": n_games,
               "attacker": n_attacker if n_attacker is not None else n_games,
               "random": n_random if n_random is not None else n_games}
    plan = plan_topup(targets, harvest)
    nf = n_frozen if n_frozen is not None else n_games

    # Even chunks for the seat-alternating attacker; plain chunks elsewhere (the
    # heuristic/random anchors are p1-only) and for the frozen halves.
    half = max(2, nf // 2)
    fch = _chunks(half, workers)

    from fishrl.train.pcollect import parse_affinity
    t0 = time.perf_counter()
    ex = ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(cfg_dict, pl["models"], pl["frozen"], max_decisions,
                                       parse_affinity(affinity)))
    try:
        futs = {
            "random":   [ex.submit(_task_random, s, c, SEED_RANDOM)
                         for s, c in _chunks(plan["random"][0], workers)],
            "attacker": [ex.submit(_task_attacker, s, c, SEED_ATTACKER)
                         for s, c in _even_chunks(plan["attacker"][0], workers)],
            "heuristic": [ex.submit(_task_heuristic, s, c, SEED_HEURISTIC)
                          for s, c in _chunks(plan["heuristic"][0], workers)],
            "heuristic11": [ex.submit(_task_heuristic, s, c, SEED_HEURISTIC11, "heuristic_1_1")
                            for s, c in _chunks(plan["heuristic11"][0], workers)],
            "heuristic12": [ex.submit(_task_heuristic, s, c, SEED_HEURISTIC12, "heuristic_1_2")
                            for s, c in _chunks(plan["heuristic12"][0], workers)],
            "heuristic13": [ex.submit(_task_heuristic, s, c, SEED_HEURISTIC13, "heuristic_1_3")
                            for s, c in _chunks(plan["heuristic13"][0], workers)],
            "frozen_a": [ex.submit(_task_match, True, c, SEED_FROZEN_A + s) for s, c in fch],
            "frozen_b": [ex.submit(_task_match, False, c, SEED_FROZEN_B + s) for s, c in fch],
        }
        # Seat / play-draw diagnostic (self-play, both seats the shared policy). Off by
        # default (seat_diag_games=0) so the ODROID timer's worker/seed layout is unchanged.
        seat_futs = [ex.submit(_task_seat_diag, s, c, SEED_SEAT_DIAG)
                     for s, c in _chunks(seat_diag_games, workers)] if seat_diag_games > 0 else []
        topup_wins = {k: sum(f.result() for f in futs[k]) for k in HARVEST_ANCHORS}
        fa = [f.result() for f in futs["frozen_a"]]
        fb = [f.result() for f in futs["frozen_b"]]
        seat_counts = [f.result() for f in seat_futs]
    finally:
        ex.shutdown(wait=True)

    mw = sum(w for w, _ in fa) + sum(w for w, _ in fb)
    dec = sum(d for _, d in fa) + sum(d for _, d in fb)
    # Combined estimate per anchor: harvested window games + the deficit just played.
    combined, anchor_n = {}, {}
    for k in HARVEST_ANCHORS:
        deficit, w_train, n_train = plan[k]
        total = n_train + deficit
        combined[k] = (w_train + topup_wins[k]) / total if total else 0.0
        anchor_n[k] = {"train": n_train, "topup": deficit}
    r = {
        "random": combined["random"],
        "attacker": combined["attacker"],
        "heuristic": combined["heuristic"],
        "heuristic11": combined["heuristic11"],   # versioned yardsticks; best.pt
        "heuristic12": combined["heuristic12"],   # stays keyed on v1.0
        "heuristic13": combined["heuristic13"],
        "frozen": (mw / dec) if dec else 0.5,
        "n": n_games, "workers": workers,
        "anchor_n": anchor_n, "harvest_from": harvest_from,
        "it": int(pl.get("done", 0)), "frozen_it": int(pl.get("frozen_it", 0)),
        "elapsed_h": float(pl.get("elapsed", 0.0)) / 3600.0,
        "took_s": time.perf_counter() - t0,
    }
    if seat_counts:
        from fishrl.eval.metrics import seat_diag_rates
        agg = {k: sum(c.get(k, 0) for c in seat_counts) for k in seat_counts[0]}
        r["seat"] = seat_diag_rates(agg)
    if save_best:
        r["new_best"] = _maybe_save_best(os.path.dirname(os.path.abspath(ckpt_path)), pl, r)
    return r


def wait_for_lock(path: str, timeout_s: float, poll_s: float = 5.0) -> bool:
    """Block until `path` (trainer.lock) is HELD, or `timeout_s` passes. The
    --follow grace period: a relay session may spend minutes on the pull leg
    before its trainer starts."""
    from fishrl.train.locks import is_locked
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if is_locked(path):
            return True
        time.sleep(poll_s)
    return False


def sleep_while_locked(path: str, interval_s: float, poll_s: float = 5.0) -> bool:
    """Sleep up to `interval_s`, waking early if the trainer releases the lock.
    Returns True while the trainer is still live (run another panel)."""
    from fishrl.train.locks import is_locked
    deadline = time.time() + interval_s
    while time.time() < deadline:
        if not is_locked(path):
            return False
        time.sleep(min(poll_s, max(0.1, deadline - time.time())))
    return is_locked(path)


def run_once(args, ckpt_path: str, allow_harvest: bool) -> None:
    """One panel + one eval row, the historic one-shot behaviour."""
    from fishrl.train import stats as stats_io

    harvest, harvest_from = None, None
    if allow_harvest and not args.no_harvest:
        harvest, harvest_from = find_harvest(stats_io.load(args.ckpt_dir),
                                             args.harvest_max_age_seconds, time.time())
    if harvest is not None:
        counts = " ".join(f"{k}={harvest.get(k, [0, 0])[0]}/{harvest.get(k, [0, 0])[1]}"
                          for k in HARVEST_ANCHORS)
        print(f"[eval] harvesting report it={harvest_from}: {counts} "
              f"(added on top of the full per-anchor target)", flush=True)

    r = parallel_panel(ckpt_path, n_games=args.n_games, max_workers=args.max_workers,
                       max_decisions=args.max_decisions, save_best=not args.no_best,
                       reserve_cores=args.reserve_cores, n_random=args.n_random,
                       n_attacker=args.n_attacker, n_frozen=args.n_frozen,
                       harvest=harvest, harvest_from=harvest_from,
                       affinity=args.affinity, seat_diag_games=args.seat_diag_games)
    row = {                                              # dump this panel to stats.json["evals"]
        "it": r["it"], "frozen_at": r["frozen_it"], "elapsed_h": r["elapsed_h"],
        "wall_time": time.time(), "n": r["n"], "workers": r["workers"], "took_s": r["took_s"],
        "frozen": r["frozen"], "random": r["random"], "attacker": r["attacker"],
        "heuristic": r["heuristic"], "heuristic11": r["heuristic11"],
        "heuristic12": r["heuristic12"], "heuristic13": r["heuristic13"],
        "anchor_n": r["anchor_n"], "harvest_from": r["harvest_from"],
        "new_best": bool(r.get("new_best")), "source": "eval",
    }
    if "seat" in r:                                      # learned seat / play-draw split
        sd = r["seat"]
        row["seat_p1_wr"] = sd["seat_p1_wr"]
        row["seat_p2_wr"] = sd["seat_p2_wr"]
        row["play_wr"] = sd["play_wr"]
        row["draw_wr"] = sd["draw_wr"]
        row["choose_first_frac"] = (None if sd["choose_first_frac"] != sd["choose_first_frac"]
                                    else sd["choose_first_frac"])   # NaN -> null
        row["seat_diag_n"] = sd["decided"]
    stats_io.append_eval(args.ckpt_dir, row)
    best = "  *** NEW BEST (heuristic) -> best.pt ***" if r.get("new_best") else ""
    seat = (f" | seat p1={r['seat']['seat_p1_wr']:.3f} p2={r['seat']['seat_p2_wr']:.3f} "
            f"play={r['seat']['play_wr']:.3f} draw={r['seat']['draw_wr']:.3f}"
            if "seat" in r else "")
    print(
        f"[eval it={r['it']} @{r['elapsed_h']:.2f}h n={r['n']} w={r['workers']} "
        f"took={r['took_s']:.1f}s] WR frozen@{r['frozen_it']}={r['frozen']:.3f} "
        f"random={r['random']:.3f} attacker={r['attacker']:.3f} heuristic={r['heuristic']:.3f} "
        f"heuristic11={r['heuristic11']:.3f} heuristic12={r['heuristic12']:.3f} "
        f"heuristic13={r['heuristic13']:.3f}"
        f"{seat}{best}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Parallel out-of-band win-rate panel.")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--ckpt", default=None,
                    help="evaluate this checkpoint file instead of <ckpt-dir>/latest.pt "
                         "(e.g. an archive_*.pt for backfill; harvest is disabled -- the "
                         "trainer's window counts describe a different policy)")
    ap.add_argument("--follow", type=float, default=0.0, metavar="SECONDS",
                    help="service mode for machines with no systemd timer (the PC): wait "
                         "for a live trainer (trainer.lock), run a panel every SECONDS "
                         "while it trains, exit when the session ends")
    ap.add_argument("--follow-grace", type=float, default=1800.0,
                    help="--follow: give the trainer this long to appear (a relay pull "
                         "leg can take minutes) before giving up")
    ap.add_argument("--n-games", type=int, default=100,
                    help="heuristic-version target games (even)")
    ap.add_argument("--n-random", type=int, default=30,
                    help="random-anchor target (saturates near 1.0 early; cheap tier)")
    ap.add_argument("--n-attacker", type=int, default=50, help="attacker-anchor target")
    ap.add_argument("--n-frozen", type=int, default=50,
                    help="frozen-self match games (panel-only: not harvestable)")
    ap.add_argument("--seat-diag-games", type=int, default=0,
                    help="self-play games for the SEAT / play-draw diagnostic (0 = off). "
                         "Reports the learned p1-vs-p2 win split -- the game is seat-"
                         "symmetric, so a gap is pure policy specialization, and the "
                         "vs-heuristic metric only ever sees the learner as p1.")
    ap.add_argument("--max-workers", type=int, default=None)
    ap.add_argument("--max-decisions", type=int, default=2000)
    ap.add_argument("--affinity", default="",
                    help="comma-separated logical-processor indices the panel workers "
                         "are restricted to (machine flag; the PC launchers park them "
                         "on E-cores away from the trainer's collectors)")
    ap.add_argument("--reserve-cores", type=int, default=0,
                    help="keep this many CPUs free of eval workers (0 = historic behaviour)")
    ap.add_argument("--no-best", action="store_true",
                    help="skip rolling best.pt (highest heuristic win-rate) for this run")
    ap.add_argument("--no-harvest", action="store_true",
                    help="ignore the trainer's wr_train counts; play full targets")
    ap.add_argument("--harvest-max-age-seconds", type=float, default=7200.0,
                    help="ignore harvest rows older than this (stalled trainer guard)")
    args = ap.parse_args()

    from fishrl.train.checkpoint import latest_path
    from fishrl.train.keepawake import keep_awake
    from fishrl.train.ownership import trainer_lock_path

    keep_awake("eval panel")                          # Windows: don't doze mid-panel

    if args.ckpt is not None:                         # arbitrary checkpoint (backfill)
        if args.follow:
            ap.error("--ckpt and --follow are mutually exclusive")
        if not os.path.exists(args.ckpt):
            print(f"[eval] no checkpoint at {args.ckpt}; nothing to evaluate", flush=True)
            return
        run_once(args, args.ckpt, allow_harvest=False)
        return

    latest = latest_path(args.ckpt_dir)
    if args.follow <= 0:                              # historic one-shot (the ODROID timer)
        if not os.path.exists(latest):
            print(f"[eval] no checkpoint at {latest}; nothing to evaluate", flush=True)
            return
        run_once(args, latest, allow_harvest=True)
        return

    # --follow: the PC's stand-in for the systemd timer. Lifetime = the training
    # session's, via trainer.lock (held for the trainer's life, OS-released).
    lock = trainer_lock_path(args.ckpt_dir)
    print(f"[eval] follow mode: waiting up to {args.follow_grace:.0f}s for a trainer "
          f"on {args.ckpt_dir}, then a panel every {args.follow:.0f}s", flush=True)
    if not wait_for_lock(lock, args.follow_grace):
        print("[eval] no trainer appeared within the grace period; exiting", flush=True)
        return
    while True:
        if os.path.exists(latest):
            run_once(args, latest, allow_harvest=True)
        if not sleep_while_locked(lock, args.follow):
            print("[eval] trainer stopped; follow mode done", flush=True)
            return


if __name__ == "__main__":
    main()
