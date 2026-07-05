# Running fishrl on Windows

The trainer, eval panel, telemetry dashboard and transfer tool all run natively on
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
pip install -e .
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
    --collect-workers 8 --reserve-cores 2
```

(or just run `deploy\fishrl-selfplay.ps1`). `Ctrl-C` = graceful stop with a
final checkpoint. Resume with the same command.

While the trainer, a relay session, or the eval panel is running, the PC's
**idle-sleep is suspended** (`SetThreadExecutionState`; the display still blanks
normally) — leave it training overnight with your usual short sleep timeout, and
normal sleep behaviour returns the moment the process exits. The relay holds the
flag through the handback, so the machine can't doze off mid-transfer after a
remote "End session" click. The dashboard (`fishrl.serve`) deliberately does NOT
keep the machine awake.

### `--collect-workers N`

Game collection is per-decision, single-threaded Python — the serial trainer
uses one core while the rest idle (62% of iteration wall-clock on this box).
`--collect-workers 8` fans each iteration's games across 8 persistent worker
processes, shipping the **current** weights every iteration, so training
semantics are unchanged (strictly on-policy, identical sampling decisions,
all league/telemetry bookkeeping on the main thread) — wall-clock only.
Measured on this machine: **2.7×** (528 → 1440 it/h in the bench; the serial
CUDA update is the remaining bottleneck, so more than ~8 workers adds
nothing at 8 games/iteration). Scenario snapshot pools pre-build in the
workers at startup (one-time, parallel — expect a ~30s first iteration when
scenarios are enabled). `0` (default) is the serial path, byte-identical:
what the ODROID service runs.

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

### Win-rates on the PC: `--follow`

There is no systemd timer here, so the launchers start the panel in **follow
mode** alongside the trainer: it waits for `trainer.lock`, runs a panel every
15 min while the session trains (harvest keeps each one cheap), and exits when
the session ends — eval rows and best.pt now accrue during PC sessions exactly
like on the ODROID. One-shot runs still work, the dashboard has a *Run eval
panel now* button, and `--ckpt archive_00080000.pt` evaluates an arbitrary
checkpoint (backfill; harvest disabled since the trainer's window counts
describe a different policy).

## Telemetry: `fishrl.serve` and near-live ticks

The trainer writes a lightweight per-iteration tick row to `<ckpt-dir>/ticks.json`
(flushed at most every `--tick-every-seconds`, default 60; reports stay the hourly
durable record). `python -m fishrl.serve` (or `deploy\fishrl-serve.ps1`) is a read-only
sibling process exposing it all on the LAN:

```powershell
deploy\fishrl-serve.ps1                  # binds the primary LAN IPv4 on :8765
```

* `/` — the **dashboard**: live SVG charts (win-rates with new-best stars, losses from
  per-iteration ticks, throughput, opponent mix), owner/turn + trainer + staleness
  chips, updating over SSE,
* `/api/reports|evals|ticks?since_it=N` — idempotent range queries (the website DB
  pulls these; omit `since_it` to rebuild from scratch),
* `/api/stream` — server-sent events for the "watch it now" view,
* `/api/summary`, `/api/actions`.

**Actions** (the wrapper passes `--allow-actions`; omit it for strictly read-only):
on this PC the dashboard gets one button — *End session & hand back* — which drops a
`STOP` file the trainer consumes at the next iteration boundary (graceful checkpoint);
if the session was started by `fishrl-relay.ps1`, the relay then exports and restarts
the ODROID automatically. On the ODROID the buttons are *Stop/Start trainer* and *Run
eval panel now* (`sudo -n systemctl`; needs the NOPASSWD rule the relay already uses).
All actions are POST-only and re-validated server-side; nothing destructive is exposed.

It never binds 0.0.0.0 and (without `--allow-actions`) never touches the training
files. While actively tweaking on this machine you can also drop
`--report-every-seconds` (e.g. 600) for denser report rows — the hourly default is an
ODROID log-volume choice, not a requirement.

## Relay training (this PC ↔ the ODROID)

**The one click:** double-click `deploy\fishrl-pc.bat` (shortcut-friendly) — it starts
the dashboard, opens it in your browser, and runs the relay session below in the
console. **The one command** (same thing minus dashboard/browser):

```powershell
deploy\fishrl-relay.ps1
```

It stops the ODROID service (graceful checkpoint), exports/fetches/imports the lineage
here, trains in the foreground (`--collect-workers 8 --reserve-cores 2`), and when you press
**Ctrl-C** it checkpoints, exports back, imports on the ODROID and restarts the service.
`python -m fishrl.relay status` shows both sides; `pull` / `handback` run either leg
alone (`handback` is also the recovery if a session ends without returning — as is just
re-running `train`, which skips the pull when the lineage is already here).

Under the hood it's the ownership protocol (`owner.json` = whose turn, `trainer.lock` =
live right now): one lineage, one trainer, every interruption resolves to at most one
owner, and `python -m fishrl.transfer claim` un-sticks an interrupted handoff. The
manual `transfer export/import` runbook in `deploy/README.md` remains the fallback.
`ssh`/`scp` use the `odroid-lan` host config; `sudo systemctl` on the ODROID prompts
unless a NOPASSWD rule covers those two commands.

## Moving training between machines

Export on the machine that trained (the trainer MUST be stopped — export refuses while
`trainer.lock` is held; `--allow-live` makes a snapshot copy that does NOT hand off
ownership):

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
