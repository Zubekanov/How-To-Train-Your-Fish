"""Relay-training ownership: which machine may train this checkpoint lineage.

The relay model is ONE model lineage alternating hosts (ODROID 24/7, the PC
when it's on), with exactly one trainer at a time. Two machines share no
filesystem, so this is a PROTOCOL, not a mutex: an ``owner.json`` stamp in
the checkpoint dir plus a generation counter carried through the transfer
zip. The failure direction is deliberate: every interrupted handoff resolves
to *at most* one owner -- possibly zero (recoverable with an explicit
``python -m fishrl.transfer claim``), never two.

  state="active"   this host may train the lineage.
  state="released" the lineage was exported; whoever imports the zip becomes
                   the owner. A released dir refuses to train until claimed.

Generation increments at each export (the handoff event). An import whose
zip carries a LOWER generation than the local stamp is a stale zip -- the
lineage has already handed off past it -- and is refused.

Complementing the stamp, the LIVE-trainer signal is ``trainer.lock`` (see
``locks.hold_lockfile``): held by the trainer process for its lifetime,
probed by ``transfer export`` and by second-trainer startups. The stamp says
"whose turn"; the lock says "running right now".
"""
from __future__ import annotations

import json
import os
import platform
import time

from fishrl.train.checkpoint import replace_with_retry

OWNER = "owner.json"
TRAINER_LOCK = "trainer.lock"


def owner_path(ckpt_dir: str) -> str:
    return os.path.join(ckpt_dir, OWNER)


def trainer_lock_path(ckpt_dir: str) -> str:
    return os.path.join(ckpt_dir, TRAINER_LOCK)


def this_host() -> str:
    return platform.node() or "unknown-host"


def read(ckpt_dir: str) -> dict | None:
    """The current stamp, or None when the dir has never been stamped (a
    pre-relay checkpoint dir -- treated as claimable by whoever trains it)."""
    try:
        with open(owner_path(ckpt_dir), encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return None


def _write(ckpt_dir: str, stamp: dict) -> dict:
    os.makedirs(ckpt_dir, exist_ok=True)
    stamp["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp = owner_path(ckpt_dir) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(stamp, f, indent=2)
    replace_with_retry(tmp, owner_path(ckpt_dir))
    return stamp


def claim(ckpt_dir: str, host: str | None = None, generation: int | None = None) -> dict:
    """Stamp `host` as the active owner. `generation` defaults to the existing
    stamp's (or 0 on a never-stamped dir) -- claiming asserts a turn, it does
    not advance the handoff counter (only `release` does)."""
    prev = read(ckpt_dir)
    if generation is None:
        generation = int(prev["generation"]) if prev else 0
    return _write(ckpt_dir, {"host": host or this_host(), "state": "active",
                             "generation": int(generation)})


def release(ckpt_dir: str) -> dict:
    """Mark the lineage exported: state -> released, generation+1. The exported
    zip carries the NEW generation, so the importer's claim matches it."""
    prev = read(ckpt_dir)
    gen = (int(prev["generation"]) if prev else 0) + 1
    host = prev["host"] if prev else this_host()
    return _write(ckpt_dir, {"host": host, "state": "released", "generation": gen})


def check(ckpt_dir: str, host: str | None = None) -> str | None:
    """May `host` train this dir? Returns None when OK (auto-claiming a
    never-stamped dir -- the pre-relay back-compat path), else a human-readable
    refusal explaining the state and the fix."""
    host = host or this_host()
    stamp = read(ckpt_dir)
    if stamp is None:
        claim(ckpt_dir, host)
        return None
    if stamp.get("state") == "active":
        if stamp.get("host") == host:
            return None
        return (f"{ckpt_dir} is owned by '{stamp.get('host')}' (active, "
                f"generation {stamp.get('generation')}). If that machine has truly "
                f"stopped training this lineage, take ownership with --claim or "
                f"'python -m fishrl.transfer claim --ckpt-dir {ckpt_dir}'.")
    return (f"{ckpt_dir} was exported (released at generation "
            f"{stamp.get('generation')} by '{stamp.get('host')}'). Import the newer "
            f"run zip to continue the lineage here, or fork it deliberately with "
            f"--claim / 'python -m fishrl.transfer claim --ckpt-dir {ckpt_dir}'.")
