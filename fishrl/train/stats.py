"""Append-only JSON stats time-series shared by the trainer and the eval service.

`stats.json` (next to the checkpoints) is the machine-readable companion to the journald
status/eval lines: a dump of every datapoint so far, for offline plotting. It is written by
TWO independent processes, each owning ONE array:

  * the trainer appends a record to ``"reports"`` at each status report (losses, calibration,
    guesser MAE, and inline win-rates when those are enabled);
  * the eval service appends to ``"evals"`` at each win-rate panel.

Both do a read-modify-write of the whole file, so they coordinate with an exclusive ``flock``
on a sibling lock file -- a writer only ever rewrites with its own array appended, and the lock
serialises the two processes so neither clobbers the other's latest append. The write itself is
atomic (tmp + ``os.replace``), so a crash mid-write leaves the previous good file intact, and a
reader never sees a half-written file. Cadences are slow (hourly report / per-eval), so lock
contention is effectively nil; this is correctness insurance, not a hot path.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os

SCHEMA = 1
STATS = "stats.json"
_LOCK = "stats.json.lock"


def stats_path(ckpt_dir: str) -> str:
    return os.path.join(ckpt_dir, STATS)


@contextlib.contextmanager
def _locked(ckpt_dir: str):
    os.makedirs(ckpt_dir, exist_ok=True)
    lf = open(os.path.join(ckpt_dir, _LOCK), "w")
    try:
        fcntl.flock(lf, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lf, fcntl.LOCK_UN)
        lf.close()


def _read(ckpt_dir: str) -> dict:
    """Load the current file (or a fresh skeleton); a missing/corrupt file = start over so a
    truncated write can never wedge the writers."""
    try:
        with open(stats_path(ckpt_dir)) as f:
            d = json.load(f)
    except (FileNotFoundError, ValueError):
        d = {}
    d.setdefault("schema", SCHEMA)
    d.setdefault("reports", [])
    d.setdefault("evals", [])
    return d


def _atomic_write(ckpt_dir: str, d: dict) -> None:
    path = stats_path(ckpt_dir)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _append(ckpt_dir: str, key: str, record: dict) -> None:
    with _locked(ckpt_dir):
        d = _read(ckpt_dir)
        d[key].append(record)
        _atomic_write(ckpt_dir, d)


def append_report(ckpt_dir: str, record: dict) -> None:
    """Trainer-side datapoint (one per status report)."""
    _append(ckpt_dir, "reports", record)


def append_eval(ckpt_dir: str, record: dict) -> None:
    """Eval-service datapoint (one per win-rate panel)."""
    _append(ckpt_dir, "evals", record)


def merge(ckpt_dir: str, reports=None, evals=None, key: str = "it",
          replace_sources=(), live_source=None) -> dict:
    """Merge historic records into the file. For each array:

      * drop existing rows whose ``source`` is in ``replace_sources`` (rows the caller owns and is
        regenerating -- e.g. the backfill tool re-deriving its own rows from updated parse logic);
      * stamp a ``source`` on any surviving row that lacks one, from ``live_source[name]`` (so
        rows written before ``source`` existed get normalised);
      * add the caller's rows that are not already present (deduped by ``key``, existing wins);
      * sort by ``key``.

    Runs under the same lock + atomic write as the appenders, so it is safe to call while the
    trainer / eval service are live. Returns counts of rows added per array."""
    replace_sources = set(replace_sources)
    live_source = live_source or {}
    with _locked(ckpt_dir):
        d = _read(ckpt_dir)
        added = {}
        for name, new in (("reports", reports or []), ("evals", evals or [])):
            kept = [r for r in d[name] if r.get("source") not in replace_sources]
            if name in live_source:
                for r in kept:
                    r.setdefault("source", live_source[name])
            have = {r.get(key) for r in kept}
            fresh = [r for r in new if r.get(key) not in have]
            kept.extend(fresh)
            kept.sort(key=lambda r: (r.get(key) is None, r.get(key)))
            d[name] = kept
            added[name] = {"added": len(fresh), "total": len(kept)}
        _atomic_write(ckpt_dir, d)
        return added
