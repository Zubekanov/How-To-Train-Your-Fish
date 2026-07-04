"""Cross-process file locking, portable across POSIX and Windows.

POSIX (the ODROID service) uses ``fcntl.flock`` exactly as stats.py always
did; Windows has no fcntl, so the first byte of the file is range-locked via
``msvcrt`` instead. Semantics match: exclusive, one holder at a time,
released before close (and by the OS if the holder dies -- which is what
makes a HELD lock a trustworthy liveness signal for `hold_lockfile`).

Two consumers:
  * stats.py serialises its read-modify-write of stats.json (blocking
    `lock`/`unlock` under `_locked`);
  * the relay ownership layer marks a live trainer: the trainer holds
    `trainer.lock` for its whole lifetime (`hold_lockfile`), and
    `fishrl.transfer export` / a second trainer probe it with `try_lock`.
"""
from __future__ import annotations

import time

try:
    import fcntl

    def lock(f):
        fcntl.flock(f, fcntl.LOCK_EX)

    def try_lock(f) -> bool:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def unlock(f):
        fcntl.flock(f, fcntl.LOCK_UN)
except ImportError:                                    # Windows
    import msvcrt

    def lock(f):
        # LK_LOCK only retries ~10x over 10s then raises, unlike flock's
        # indefinite block -- loop to restore true blocking. Cadences here
        # are slow, so the loop is insurance, not a hot path.
        f.seek(0)
        while True:
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
                return
            except OSError:
                time.sleep(0.05)

    def try_lock(f) -> bool:
        f.seek(0)
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False

    def unlock(f):
        f.seek(0)                                      # range must match the lock
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def open_lockfile(path: str):
    """Open (creating if needed) a lock file ready for `lock`/`try_lock`.

    "a+" (not "w"): no truncation race between competing openers, and msvcrt
    locks a byte RANGE, so the file must be non-empty -- seed one byte on
    first use. (Two openers racing the seed append at most one byte each;
    harmless.)"""
    f = open(path, "a+")
    if f.seek(0, 2) == 0:
        f.write("\0")
        f.flush()
    return f


def hold_lockfile(path: str):
    """Acquire `path` exclusively without blocking and KEEP it: returns the open
    handle on success (caller holds it for process lifetime; the OS releases it
    on any exit, clean or not), or None if another process holds the lock."""
    f = open_lockfile(path)
    if try_lock(f):
        return f
    f.close()
    return None


def is_locked(path: str) -> bool:
    """Probe: is `path` currently held by some process? Acquires and immediately
    releases on the False path, so it never lingers as a holder itself."""
    f = open_lockfile(path)
    try:
        if try_lock(f):
            unlock(f)
            return False
        return True
    finally:
        f.close()
