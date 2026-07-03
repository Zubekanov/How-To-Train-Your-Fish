"""CPU thread-budget control for the single-process trainer (torch-free).

The trainer's parallelism is BLAS/OpenMP/torch intra-op threads, which read the
``*_NUM_THREADS`` environment variables ONCE when the native libraries load —
i.e. at ``import torch``. So the budget must be decided before any fishrl
module that pulls in torch is imported; ``fishrl/train/__main__.py`` calls
:func:`preconfigure` between its stdlib and fishrl imports for exactly that
reason (this module must therefore never import torch).

``--reserve-cores N`` keeps N physical cores free for the rest of the machine
(the desktop use case: train without throttling everything else). Precedence:

  * flag present on argv  -> threads = max(1, physical_cores() - N), always;
  * any *_NUM_THREADS already set (the ODROID systemd unit's case) -> no-op,
    the environment wins and behaviour is unchanged;
  * neither -> DEFAULT_RESERVE applies.
"""
from __future__ import annotations

import functools
import os
import subprocess
import sys

DEFAULT_RESERVE = 2
_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


@functools.lru_cache(maxsize=1)
def physical_cores() -> int:
    """Physical (not SMT/logical) core count. ``os.cpu_count()`` is logical,
    which over-subscribes BLAS on hyper-threaded desktops (e.g. 28 logical on
    20 physical cores). Resolution order: FISHRL_PHYSICAL_CORES env override,
    OS query (wmic / PowerShell CIM on Windows, /proc/cpuinfo on Linux), then
    the logical count as a last resort."""
    env = os.environ.get("FISHRL_PHYSICAL_CORES")
    if env:
        try:
            n = int(env)
            if n > 0:
                return n
        except ValueError:
            pass
    try:
        if sys.platform == "win32":
            for cmd in (["wmic", "cpu", "get", "NumberOfCores", "/value"],
                        ["powershell", "-NoProfile", "-Command",
                         "(Get-CimInstance Win32_Processor | "
                         "Measure-Object NumberOfCores -Sum).Sum"]):
                try:
                    out = subprocess.run(cmd, capture_output=True, text=True,
                                         timeout=10).stdout
                except (OSError, subprocess.TimeoutExpired):
                    continue
                digits = [int(s.split("=")[-1]) for s in out.split()
                          if s.split("=")[-1].isdigit()]
                if digits:
                    return sum(digits) if "wmic" in cmd[0] else digits[0]
        elif sys.platform.startswith("linux"):
            cores = set()
            phys = core = None
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("physical id"):
                        phys = line.split(":")[1].strip()
                    elif line.startswith("core id"):
                        core = line.split(":")[1].strip()
                    elif not line.strip():
                        if core is not None:
                            cores.add((phys, core))
                        phys = core = None
            if core is not None:
                cores.add((phys, core))
            if cores:
                return len(cores)
    except Exception:
        pass
    return os.cpu_count() or 1


def peek_reserve_cores(argv=None):
    """Pre-argparse scan for ``--reserve-cores N`` / ``--reserve-cores=N``.
    Returns the int, or None when absent; malformed values are also None here
    and left for argparse to reject with a proper error later."""
    argv = sys.argv[1:] if argv is None else argv
    for i, a in enumerate(argv):
        val = None
        if a == "--reserve-cores" and i + 1 < len(argv):
            val = argv[i + 1]
        elif a.startswith("--reserve-cores="):
            val = a.split("=", 1)[1]
        if val is not None:
            try:
                return int(val)
            except ValueError:
                return None
    return None


def preconfigure(argv=None):
    """Apply the thread budget to the ``*_NUM_THREADS`` env vars. MUST run
    before torch (or anything importing it) loads. Returns the thread count
    applied, or None when an existing environment was left in charge (the
    caller then skips ``torch.set_num_threads`` too)."""
    reserve = peek_reserve_cores(argv)
    if reserve is None:
        if any(os.environ.get(v) for v in _ENV_VARS):
            return None                                # deployed env wins (ODROID unit)
        reserve = DEFAULT_RESERVE
    phys = physical_cores()
    threads = max(1, phys - max(0, reserve))
    for v in _ENV_VARS:
        os.environ[v] = str(threads)
    print(f"[threads] physical={phys} reserve={reserve} -> "
          f"OMP/MKL/OPENBLAS/NUMEXPR={threads}", flush=True)
    return threads
