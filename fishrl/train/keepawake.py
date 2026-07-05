"""Keep Windows from idle-sleeping while work is in flight. No-op elsewhere.

``SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)`` tells the OS
this process is doing real work, which suspends the idle-sleep timer WITHOUT
touching power settings. The display is deliberately NOT requested, so the
monitor still blanks on schedule. The flag lives with the calling thread and
is cleared by the OS the moment the process exits (crash included), so the
machine's normal sleep behaviour resumes by itself.

Held by: the trainer (an overnight session must survive a short sleep
timeout), fishrl.relay (the handback transfer runs AFTER the trainer exits --
exactly when the machine would otherwise doze off mid-scp), and the eval
panel. NOT by fishrl.serve: a passive dashboard should never keep the PC up.
"""
from __future__ import annotations

import os

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def keep_awake(reason: str = "fishrl") -> bool:
    """Suspend idle sleep for the life of this process (Windows; no-op and
    False elsewhere). Safe to call more than once."""
    if os.name != "nt":
        return False
    try:
        import ctypes
        ok = ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        if ok:
            print(f"[keepawake] {reason}: system idle-sleep suspended until this "
                  f"process exits (display may still blank)", flush=True)
        return bool(ok)
    except Exception:
        return False


def allow_sleep() -> None:
    """Explicitly restore normal sleep (process exit does this anyway)."""
    if os.name != "nt":
        return
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    except Exception:
        pass
