# Running fishrl on Windows

The trainer, eval panel, monitor GUI and transfer tool all run natively on
Windows (tested: Windows 10, 20-core desktop, RTX 3060, CPython 3.13). The
systemd units in this directory are Linux-only — on Windows you run the
modules directly (foreground), and `Ctrl-C` performs the same graceful
checkpoint that SIGTERM does under systemd.

## Setup

```powershell
cd How-To-Train-Your-Fish
python -m venv .venv            # (`py -3.13 -m venv .venv` if you use the launcher)
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .[gui]           # gui extra = matplotlib for the monitor window
```

**Torch**: keep every machine that shares checkpoints inside the
`requirements.txt` pin (`torch>=2.12,<2.13`) — loading a checkpoint saved by a
newer torch into an older one is the direction that breaks.

* CPU (matches the ODROID deployment exactly):
  `pip install "torch>=2.12,<2.13"`
* CUDA (verified working: 2.12.1+cu126 wheels exist for Windows/py3.13):
  `pip install "torch>=2.12,<2.13" --index-url https://download.pytorch.org/whl/cu126`

CUDA only accelerates the batched critic/PPO update phase (game collection is
pure Python); either build round-trips checkpoints with the ODROID —
`map_location` handles the device change and CUDA RNG state is only applied
where CUDA exists.

## Training

```powershell
python -m fishrl.train --resume --iters 0 --encoder flat --critic-encoder entity `
    --scenario-frac 0.3 --report-every-seconds 3600 --checkpoint-every-seconds 900 `
    --reserve-cores 2 --gui
```

(or just run `deploy\fishrl-selfplay.ps1`). `Ctrl-C` = graceful stop with a
final checkpoint. Resume with the same command.

### `--reserve-cores N`

Keeps N **physical** cores free of training threads so the machine stays
responsive (BLAS/torch thread caps — the same mechanism the ODROID unit uses
with `*_NUM_THREADS=6` on 8 cores). Default is 2 when nothing else set the
`*_NUM_THREADS` environment variables; an environment that sets them (like the
systemd unit) always wins. Physical core detection: `FISHRL_PHYSICAL_CORES`
env override → `wmic`/PowerShell CIM → `/proc/cpuinfo` → logical count.

The eval panel takes the same flag (caps its worker processes):

```powershell
python -m fishrl.eval.parallel_panel --ckpt-dir checkpoints --n-games 100 --reserve-cores 2
```

### `--gui` / the monitor

`--gui` opens `fishrl.monitor` in a subprocess — a read-only tkinter window
over `stats.json`/`best.json` (win-rates, losses, throughput, opponent mix,
staleness badge). Closing it never affects training; it can also run
standalone, including against a checkpoint dir another machine is writing:

```powershell
python -m fishrl.monitor --ckpt-dir checkpoints --refresh 5
```

## Moving training between machines

Export on the machine that trained (trainer ideally stopped for a clean cut):

```powershell
python -m fishrl.transfer export --ckpt-dir checkpoints          # -> fishrl_run_<it>_<ts>.zip
python -m fishrl.transfer export --ckpt-dir checkpoints --with-archives   # + archive_*.pt history
```

Import on the other machine (e.g. back on the ODROID):

```bash
python -m fishrl.transfer import --zip fishrl_run_*.zip --ckpt-dir checkpoints --force --merge-stats
python -m fishrl.train --resume --iters 0 --ckpt-dir checkpoints
```

The zip carries `latest.pt` (models, optimizers, RNG, counters, the full PFSP
league — everything resume needs), `stats.json`, `best.pt`/`best.json`, and a
`manifest.json` (git commit, torch version, iteration, sha256 per file).
`--merge-stats` unions the two machines' stats histories instead of replacing
(existing rows win, sorted by iteration); `--force` is required to overwrite
an existing `latest.pt`. Import warns on a torch minor-version mismatch.

## Notes / limitations

* `fishrl.eval.backfill_stats` reads journald and is inert on Windows (live
  stats.json writing does not depend on it).
* The `stats.json.lock` coordination uses `msvcrt` byte-range locking on
  Windows and `flock` on POSIX — behaviour is identical.
* No Windows service wrapper is provided; run in a terminal (or wire
  `fishrl-selfplay.ps1` into Task Scheduler if you want it unattended).
