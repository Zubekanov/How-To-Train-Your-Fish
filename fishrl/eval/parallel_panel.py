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


def _init(cfg_dict: dict, models_state: dict, frozen_state: dict, max_decisions: int) -> None:
    """Subprocess initializer: rebuild the two policies once per worker and pin to 1 thread.

    Parallelism here is across processes; letting each worker also spin up a BLAS/torch
    thread pool would oversubscribe the box, so we cap intra-op threads to 1."""
    import torch

    from fishrl.train.config import Config
    from fishrl.train.train_loop import build_models, _load_model_state

    torch.set_num_threads(1)
    per_net = {f"{n}_encoder": cfg_dict["encoders"][n] for n in NETS}
    cfg = Config(seed=cfg_dict["seed"], use_belief=cfg_dict.get("use_belief", True),
                 critic_hidden=tuple(cfg_dict.get("critic_hidden", (512, 512, 256))), **per_net)
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
SEED_FROZEN_A, SEED_FROZEN_B = 500_000, 510_000

# Anchors whose outcomes the trainer harvests from its own PFSP pool games (the
# report rows' "wr_train": anchor -> [wins, games], eval convention). Frozen-self
# is NOT harvestable: the league's past-selves are assorted ring members, not the
# last-report snapshot this panel matches against.
HARVEST_ANCHORS = ("heuristic", "heuristic11", "attacker", "random")


def plan_topup(targets: dict, harvest: dict | None) -> dict:
    """Per-anchor top-up plan: ``{anchor: (deficit, wins_train, n_train)}``.

    `targets` maps each HARVEST_ANCHORS entry to its total game target; `harvest`
    is a report row's ``wr_train`` (None or a missing anchor -> no harvested games
    -> the deficit is the full target, i.e. exactly the pre-harvest panel). The
    attacker deficit is rounded UP to even -- its games alternate seats in pairs
    -- so it may overshoot the target by one game."""
    out = {}
    for k in HARVEST_ANCHORS:
        w, n = (harvest or {}).get(k) or (0, 0)
        w, n = int(w), int(n)
        deficit = max(0, int(targets[k]) - n)
        if k == "attacker" and deficit % 2:
            deficit += 1
        out[k] = (deficit, w, n)
    return out


def find_harvest(stats: dict, max_age_s: float, now: float) -> tuple:
    """Pick the newest report row carrying usable harvest counts.

    Returns ``(wr_train, report_it)`` or ``(None, None)`` when there is nothing to
    harvest -- no row with ``wr_train`` (pre-harvest trainer), the newest one is
    older than `max_age_s` (stalled/stopped trainer: its window no longer reflects
    the checkpoint being evaluated), or it was already consumed by an earlier eval
    row (``harvest_from``) -- double-counting the same window would just replay the
    previous estimate. Every miss degrades to a FULL panel, never a thinner one."""
    rows = [r for r in stats.get("reports", []) if r.get("wr_train")]
    if not rows:
        return None, None
    row = max(rows, key=lambda r: (r.get("wall_time") or 0.0))
    if now - (row.get("wall_time") or 0.0) > max_age_s:
        return None, None
    consumed = {e.get("harvest_from") for e in stats.get("evals", [])}
    if row.get("it") in consumed:
        return None, None
    return row["wr_train"], row.get("it")


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
                   harvest: dict | None = None, harvest_from: int | None = None) -> dict:
    """Compute win-rates vs random / attacker / heuristic / frozen-self from a checkpoint,
    fanning the games across a process pool. Returns the rates plus eval metadata.

    `n_games` is the heuristic/heuristic11 target; `n_random`/`n_attacker`/`n_frozen`
    default to it (tiered targets: random saturates early, frozen is the most
    expensive anchor). With `harvest` (a report row's ``wr_train``), the panel only
    plays each anchor's DEFICIT below target and publishes the combined
    harvested+top-up estimate under the usual keys -- consumers see one number per
    anchor either way. No harvest -> full-target panel, the historic behaviour.

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

    targets = {"heuristic": n_games, "heuristic11": n_games,
               "attacker": n_attacker if n_attacker is not None else n_games,
               "random": n_random if n_random is not None else n_games}
    plan = plan_topup(targets, harvest)
    nf = n_frozen if n_frozen is not None else n_games

    # Even chunks for the seat-alternating attacker; plain chunks elsewhere (the
    # heuristic/random anchors are p1-only) and for the frozen halves.
    half = max(2, nf // 2)
    fch = _chunks(half, workers)

    t0 = time.perf_counter()
    ex = ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(cfg_dict, pl["models"], pl["frozen"], max_decisions))
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
            "frozen_a": [ex.submit(_task_match, True, c, SEED_FROZEN_A + s) for s, c in fch],
            "frozen_b": [ex.submit(_task_match, False, c, SEED_FROZEN_B + s) for s, c in fch],
        }
        topup_wins = {k: sum(f.result() for f in futs[k]) for k in HARVEST_ANCHORS}
        fa = [f.result() for f in futs["frozen_a"]]
        fb = [f.result() for f in futs["frozen_b"]]
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
        "heuristic11": combined["heuristic11"],   # v1.1 yardstick; best.pt stays keyed on v1.0
        "frozen": (mw / dec) if dec else 0.5,
        "n": n_games, "workers": workers,
        "anchor_n": anchor_n, "harvest_from": harvest_from,
        "it": int(pl.get("done", 0)), "frozen_it": int(pl.get("frozen_it", 0)),
        "elapsed_h": float(pl.get("elapsed", 0.0)) / 3600.0,
        "took_s": time.perf_counter() - t0,
    }
    if save_best:
        r["new_best"] = _maybe_save_best(os.path.dirname(os.path.abspath(ckpt_path)), pl, r)
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description="Parallel out-of-band win-rate panel.")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--n-games", type=int, default=100,
                    help="heuristic/heuristic11 target games (even)")
    ap.add_argument("--n-random", type=int, default=30,
                    help="random-anchor target (saturates near 1.0 early; cheap tier)")
    ap.add_argument("--n-attacker", type=int, default=50, help="attacker-anchor target")
    ap.add_argument("--n-frozen", type=int, default=50,
                    help="frozen-self match games (panel-only: not harvestable)")
    ap.add_argument("--max-workers", type=int, default=None)
    ap.add_argument("--max-decisions", type=int, default=2000)
    ap.add_argument("--reserve-cores", type=int, default=0,
                    help="keep this many CPUs free of eval workers (0 = historic behaviour)")
    ap.add_argument("--no-best", action="store_true",
                    help="skip rolling best.pt (highest heuristic win-rate) for this run")
    ap.add_argument("--no-harvest", action="store_true",
                    help="ignore the trainer's wr_train counts; play full targets")
    ap.add_argument("--harvest-max-age-seconds", type=float, default=7200.0,
                    help="ignore harvest rows older than this (stalled trainer guard)")
    args = ap.parse_args()

    from fishrl.train import stats as stats_io
    from fishrl.train.checkpoint import latest_path

    latest = latest_path(args.ckpt_dir)
    if not os.path.exists(latest):
        print(f"[eval] no checkpoint at {latest}; nothing to evaluate", flush=True)
        return

    harvest, harvest_from = None, None
    if not args.no_harvest:
        harvest, harvest_from = find_harvest(stats_io.load(args.ckpt_dir),
                                             args.harvest_max_age_seconds, time.time())
    if harvest is not None:
        counts = " ".join(f"{k}={harvest.get(k, [0, 0])[0]}/{harvest.get(k, [0, 0])[1]}"
                          for k in HARVEST_ANCHORS)
        print(f"[eval] harvesting report it={harvest_from}: {counts} "
              f"(top-up only plays each anchor's deficit)", flush=True)

    r = parallel_panel(latest, n_games=args.n_games, max_workers=args.max_workers,
                       max_decisions=args.max_decisions, save_best=not args.no_best,
                       reserve_cores=args.reserve_cores, n_random=args.n_random,
                       n_attacker=args.n_attacker, n_frozen=args.n_frozen,
                       harvest=harvest, harvest_from=harvest_from)
    stats_io.append_eval(args.ckpt_dir, {                # dump this panel to stats.json["evals"]
        "it": r["it"], "frozen_at": r["frozen_it"], "elapsed_h": r["elapsed_h"],
        "wall_time": time.time(), "n": r["n"], "workers": r["workers"], "took_s": r["took_s"],
        "frozen": r["frozen"], "random": r["random"], "attacker": r["attacker"],
        "heuristic": r["heuristic"], "heuristic11": r["heuristic11"],
        "anchor_n": r["anchor_n"], "harvest_from": r["harvest_from"],
        "new_best": bool(r.get("new_best")), "source": "eval",
    })
    best = "  *** NEW BEST (heuristic) -> best.pt ***" if r.get("new_best") else ""
    print(
        f"[eval it={r['it']} @{r['elapsed_h']:.2f}h n={r['n']} w={r['workers']} "
        f"took={r['took_s']:.1f}s] WR frozen@{r['frozen_it']}={r['frozen']:.3f} "
        f"random={r['random']:.3f} attacker={r['attacker']:.3f} heuristic={r['heuristic']:.3f} "
        f"heuristic11={r['heuristic11']:.3f}"
        f"{best}",
        flush=True,
    )


if __name__ == "__main__":
    main()
