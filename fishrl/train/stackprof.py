"""Opt-in in-process stack sampler (FISHRL_STACKPROF=1): a daemon thread samples
sys._current_frames() every `interval` s and histograms (thread, function, file:line)
at the top of each thread's stack. Meant for ptrace-less boxes (docker without
SYS_PTRACE, where py-spy cannot attach). `report()` prints the top entries and resets."""
from __future__ import annotations

import collections
import os
import sys
import threading
import time

_LOCK = threading.Lock()
_HIST: dict = collections.Counter()
_N = 0
_ON = bool(os.environ.get("FISHRL_STACKPROF"))


def _sample(interval: float) -> None:
    global _N
    names = {}
    while True:
        time.sleep(interval)
        for t in threading.enumerate():
            names[t.ident] = t.name
        frames = sys._current_frames()
        with _LOCK:
            _N += 1
            for tid, f in frames.items():
                nm = names.get(tid, str(tid))
                if nm.startswith("stackprof"):
                    continue
                # top two frames: where it is + who called it
                top = f"{f.f_code.co_name}@{os.path.basename(f.f_code.co_filename)}:{f.f_lineno}"
                up = f.f_back
                upn = f"{up.f_code.co_name}@{os.path.basename(up.f_code.co_filename)}" if up else "-"
                _HIST[(nm, top, upn)] += 1


def start(interval: float = 0.01) -> None:
    if not _ON:
        return
    threading.Thread(target=_sample, args=(interval,), daemon=True, name="stackprof").start()


def report(log=print, top: int = 14) -> None:
    global _N
    if not _ON:
        return
    with _LOCK:
        n, items = _N, _HIST.most_common(top)
        _HIST.clear(); _N = 0
    if not n:
        return
    log(f"[stackprof] {n} samples")
    for (nm, where, up), c in items:
        log(f"[stackprof]   {100*c/n:5.1f}%  {nm:<28} {where}  <- {up}")
