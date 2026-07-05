"""Crash-safe training checkpoints for the long-running self-play service.

A checkpoint is a single torch-serialised dict (schema ``format: 1``) holding EVERYTHING
needed to resume `train()` mid-run with full continuity: all four model state_dicts, the
frozen-self anchor, the three optimizer states, RNG state, the iteration counter, and the
cumulative wall-clock. `latest.pt` is the canonical resume target; `step_*.pt` milestones are
kept for inspection/rollback only.

Writes are ATOMIC: serialise to ``<path>.tmp``, fsync, then ``os.replace`` onto the final name
(atomic on POSIX). A process killed mid-write therefore leaves the previous good checkpoint
intact — never a half-written file.
"""
from __future__ import annotations

import glob
import os
import time

import torch

LATEST = "latest.pt"
FORMAT = 1


def latest_path(ckpt_dir: str) -> str:
    return os.path.join(ckpt_dir, LATEST)


def replace_with_retry(tmp: str, dst: str, attempts: int = 10, delay: float = 0.2) -> None:
    """`os.replace`, but tolerant of Windows readers. On POSIX a rename over an
    open file is fine; on Windows it raises PermissionError while a reader (a
    dashboard fetch, the export tool) briefly holds `dst` open -- retry a few times
    before giving up. First attempt always taken, so POSIX is a passthrough."""
    for i in range(attempts):
        try:
            os.replace(tmp, dst)
            return
        except PermissionError:
            if os.name != "nt" or i == attempts - 1:
                raise
            time.sleep(delay)


def save_checkpoint(path: str, payload: dict) -> None:
    """Atomically write `payload` to `path` (tmp + fsync + os.replace)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    replace_with_retry(tmp, path)


def save_milestone(ckpt_dir: str, done: int, payload: dict, keep_last: int = 3) -> str:
    """Write a numbered milestone `step_{done:08d}.pt` atomically, then prune the oldest
    milestones beyond `keep_last`. Returns the milestone path. The prune glob matches
    only `step_*.pt`, so permanent `archive_*.pt` checkpoints are never touched."""
    path = os.path.join(ckpt_dir, f"step_{done:08d}.pt")
    save_checkpoint(path, payload)
    if keep_last > 0:
        existing = sorted(glob.glob(os.path.join(ckpt_dir, "step_*.pt")))
        for old in existing[:-keep_last]:
            try:
                os.remove(old)
            except OSError:
                pass
    return path


def save_archive(ckpt_dir: str, done: int, payload: dict) -> str:
    """Write a PERMANENT `archive_{done:08d}.pt` atomically. Archives are the run's
    long-term history (every cfg.archive_every_iters iterations) and are never
    pruned — unlike the rolling step_*.pt milestones. Returns the archive path."""
    path = os.path.join(ckpt_dir, f"archive_{done:08d}.pt")
    save_checkpoint(path, payload)
    return path


def load_checkpoint(path: str, map_location="cpu") -> dict:
    """Load a checkpoint dict. `weights_only=False` because the payload carries RNG/optim
    state (plain Python/torch containers we wrote ourselves), not just tensors."""
    return torch.load(path, map_location=map_location, weights_only=False)
