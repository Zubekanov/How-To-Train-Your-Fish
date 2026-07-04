"""Portable lock shim: exclusivity, non-blocking probes, and crash release.

The relay layer trusts a HELD trainer.lock as "a trainer is running right now",
so the properties that matter are: a held lock is visible from another process
(and another handle in the same process), try_lock never blocks, and the OS
releases the lock when the holder dies -- clean exit or kill alike."""
import os
import subprocess
import sys
import time

from fishrl.train.locks import hold_lockfile, is_locked, try_lock, unlock, open_lockfile

# A child that grabs the lock, reports, and sleeps until killed.
_HOLDER = """
import sys, time
from fishrl.train.locks import hold_lockfile
h = hold_lockfile(sys.argv[1])
print("HELD" if h is not None else "MISS", flush=True)
time.sleep(60)
"""


def test_hold_and_probe_same_process(tmp_path):
    p = str(tmp_path / "t.lock")
    assert is_locked(p) is False                     # probe creates but doesn't hold
    h = hold_lockfile(p)
    assert h is not None
    assert is_locked(p) is True                      # a second handle sees the hold
    assert hold_lockfile(p) is None                  # and cannot double-acquire
    unlock(h)
    h.close()
    assert is_locked(p) is False


def test_try_lock_is_nonblocking(tmp_path):
    p = str(tmp_path / "t.lock")
    h = hold_lockfile(p)
    f = open_lockfile(p)
    t0 = time.perf_counter()
    assert try_lock(f) is False
    assert time.perf_counter() - t0 < 2.0            # no LK_LOCK-style retry stall
    f.close()
    unlock(h)
    h.close()


def test_lock_released_when_holder_dies(tmp_path):
    p = str(tmp_path / "t.lock")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    child = subprocess.Popen([sys.executable, "-c", _HOLDER, p],
                             stdout=subprocess.PIPE, text=True, env=env)
    try:
        assert child.stdout.readline().strip() == "HELD"
        assert is_locked(p) is True                  # visible across processes
        child.kill()
        child.wait(timeout=30)
        deadline = time.time() + 10                  # OS release is prompt but not instant
        while time.time() < deadline and is_locked(p):
            time.sleep(0.1)
        assert is_locked(p) is False                 # crash = release (the liveness signal)
    finally:
        if child.poll() is None:
            child.kill()
