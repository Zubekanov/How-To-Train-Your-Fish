"""System-utilization sampling for telemetry ticks: %CPU, %RAM, %GPU.

Stdlib-only and cross-platform (Windows via ctypes/kernel32, Linux via /proc):
one background daemon thread samples every few seconds and the per-iteration
tick append just reads the latest values -- the trainer's hot loop never pays
a sampling cost. GPU%% prefers ``torch.cuda.utilization`` (NVML-backed, needs
the pynvml package) and falls back to one ``nvidia-smi`` query per sample
interval; ``None`` where neither works (the ODROID). Every read is
best-effort: a failed probe yields None for that field, never an exception.
"""
from __future__ import annotations

import os
import subprocess
import threading
import time

_lock = threading.Lock()
_latest: dict = {"cpu": None, "ram": None, "gpu": None}
_started = False


# ── probes ────────────────────────────────────────────────────────────────────

def _cpu_times() -> tuple | None:
    """(busy, total) cumulative jiffies/ticks; %CPU comes from two readings."""
    try:
        if os.name == "nt":
            import ctypes

            class FT(ctypes.Structure):
                _fields_ = [("lo", ctypes.c_uint32), ("hi", ctypes.c_uint32)]

            idle, kern, user = FT(), FT(), FT()
            if not ctypes.windll.kernel32.GetSystemTimes(
                    ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user)):
                return None

            def v(ft: FT) -> int:
                return (ft.hi << 32) | ft.lo
            busy = (v(kern) - v(idle)) + v(user)       # kernel time INCLUDES idle
            return busy, busy + v(idle)
        with open("/proc/stat") as f:
            vals = [int(x) for x in f.readline().split()[1:]]
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)   # idle + iowait
        return sum(vals) - idle, sum(vals)
    except Exception:                                  # noqa: BLE001 -- telemetry, never fatal
        return None


def _ram_pct() -> float | None:
    try:
        if os.name == "nt":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),
                            ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                            ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                            ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                            ("ullAvailExtendedVirtual", ctypes.c_uint64)]

            ms = MS()
            ms.dwLength = ctypes.sizeof(MS)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                return None
            return float(ms.dwMemoryLoad)
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, rest = line.partition(":")
                info[k] = int(rest.split()[0])
        return 100.0 * (1.0 - info["MemAvailable"] / info["MemTotal"])
    except Exception:                                  # noqa: BLE001
        return None


def _make_gpu_probe():
    """Pick the cheapest working GPU%% source ONCE; the sampler calls it per tick."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.utilization()                   # raises if pynvml is absent

            def nvml() -> float | None:
                try:
                    return float(torch.cuda.utilization())
                except Exception:                      # noqa: BLE001
                    return None
            return nvml
    except Exception:                                  # noqa: BLE001
        pass
    cmd = ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            def smi() -> float | None:
                try:
                    q = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                    if q.returncode == 0 and q.stdout.strip():
                        return float(q.stdout.strip().splitlines()[0])
                except Exception:                      # noqa: BLE001
                    pass
                return None
            return smi
    except Exception:                                  # noqa: BLE001
        pass
    return lambda: None                                # no GPU / no tooling (the ODROID)


# ── the sampler ───────────────────────────────────────────────────────────────

def _loop(interval_s: float) -> None:
    gpu_probe = _make_gpu_probe()
    prev = _cpu_times()
    while True:
        time.sleep(interval_s)
        cur = _cpu_times()
        cpu = None
        if prev is not None and cur is not None and cur[1] > prev[1]:
            cpu = 100.0 * (cur[0] - prev[0]) / (cur[1] - prev[1])
        prev = cur
        ram, gpu = _ram_pct(), gpu_probe()
        with _lock:
            _latest.update(cpu=cpu, ram=ram, gpu=gpu)


def start(interval_s: float = 5.0) -> None:
    """Start the sampler thread (idempotent; daemon -- dies with the trainer)."""
    global _started
    with _lock:
        if _started:
            return
        _started = True
    threading.Thread(target=_loop, args=(interval_s,),
                     name="fishrl-sysstats", daemon=True).start()


def latest() -> dict:
    """{'cpu': %, 'ram': %, 'gpu': %} -- any field None when unavailable/not yet sampled."""
    with _lock:
        return dict(_latest)
