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


def _task_heuristic(start: int, count: int, seed: int) -> int:
    from fishrl.eval.metrics import winrate_vs_heuristic
    _seed_torch(seed + start)
    v = winrate_vs_heuristic(_G["m"], n_games=count, seed=seed + start,
                             max_decisions=_G["md"], use_belief=_G["ub"])
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
SEED_FROZEN_A, SEED_FROZEN_B = 500_000, 510_000


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
    os.replace(tmp, meta_path)
    return True


def parallel_panel(ckpt_path: str, n_games: int = 100, max_workers: int | None = None,
                   max_decisions: int = 2000, save_best: bool = False) -> dict:
    """Compute win-rates vs random / attacker / heuristic / frozen-self from a checkpoint,
    fanning the games across a process pool. Returns the rates plus eval metadata.

    When `save_best`, also roll best.pt (highest heuristic win-rate seen) next to the checkpoint;
    the returned dict carries `new_best` (bool)."""
    import torch  # noqa: F401  (ensure torch import cost is paid in the parent too)

    from fishrl.train.checkpoint import load_checkpoint

    pl = load_checkpoint(ckpt_path, map_location="cpu")
    cfg_dict = pl["config"]
    workers = max_workers or min(6, max(1, (os.cpu_count() or 2)))

    # Even chunks for the seat-alternating anchors; plain chunks for the frozen halves.
    rch = _even_chunks(n_games, workers)
    half = max(2, n_games // 2)
    fch = _chunks(half, workers)

    t0 = time.perf_counter()
    ex = ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(cfg_dict, pl["models"], pl["frozen"], max_decisions))
    try:
        futs = {
            "random":   [ex.submit(_task_random, s, c, SEED_RANDOM) for s, c in rch],
            "attacker": [ex.submit(_task_attacker, s, c, SEED_ATTACKER) for s, c in rch],
            "heuristic": [ex.submit(_task_heuristic, s, c, SEED_HEURISTIC) for s, c in rch],
            "frozen_a": [ex.submit(_task_match, True, c, SEED_FROZEN_A + s) for s, c in fch],
            "frozen_b": [ex.submit(_task_match, False, c, SEED_FROZEN_B + s) for s, c in fch],
        }
        wins = {k: sum(f.result() for f in futs[k]) for k in ("random", "attacker", "heuristic")}
        fa = [f.result() for f in futs["frozen_a"]]
        fb = [f.result() for f in futs["frozen_b"]]
    finally:
        ex.shutdown(wait=True)

    mw = sum(w for w, _ in fa) + sum(w for w, _ in fb)
    dec = sum(d for _, d in fa) + sum(d for _, d in fb)
    r = {
        "random": wins["random"] / n_games,
        "attacker": wins["attacker"] / n_games,
        "heuristic": wins["heuristic"] / n_games,
        "frozen": (mw / dec) if dec else 0.5,
        "n": n_games, "workers": workers,
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
    ap.add_argument("--n-games", type=int, default=100, help="games per anchor (even)")
    ap.add_argument("--max-workers", type=int, default=None)
    ap.add_argument("--max-decisions", type=int, default=2000)
    ap.add_argument("--no-best", action="store_true",
                    help="skip rolling best.pt (highest heuristic win-rate) for this run")
    args = ap.parse_args()

    from fishrl.train.checkpoint import latest_path

    latest = latest_path(args.ckpt_dir)
    if not os.path.exists(latest):
        print(f"[eval] no checkpoint at {latest}; nothing to evaluate", flush=True)
        return
    r = parallel_panel(latest, n_games=args.n_games, max_workers=args.max_workers,
                       max_decisions=args.max_decisions, save_best=not args.no_best)
    from fishrl.train import stats as stats_io
    stats_io.append_eval(args.ckpt_dir, {                # dump this panel to stats.json["evals"]
        "it": r["it"], "frozen_at": r["frozen_it"], "elapsed_h": r["elapsed_h"],
        "wall_time": time.time(), "n": r["n"], "workers": r["workers"], "took_s": r["took_s"],
        "frozen": r["frozen"], "random": r["random"], "attacker": r["attacker"],
        "heuristic": r["heuristic"], "new_best": bool(r.get("new_best")), "source": "eval",
    })
    best = "  *** NEW BEST (heuristic) -> best.pt ***" if r.get("new_best") else ""
    print(
        f"[eval it={r['it']} @{r['elapsed_h']:.2f}h n={r['n']} w={r['workers']} "
        f"took={r['took_s']:.1f}s] WR frozen@{r['frozen_it']}={r['frozen']:.3f} "
        f"random={r['random']:.3f} attacker={r['attacker']:.3f} heuristic={r['heuristic']:.3f}"
        f"{best}",
        flush=True,
    )


if __name__ == "__main__":
    main()
