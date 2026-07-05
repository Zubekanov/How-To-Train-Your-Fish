# Deployment (systemd)

The production trainer and the hourly win-rate panel run as **system-level** systemd
units (`/etc/systemd/system/`, running as the user via `User=` — NOT `systemctl --user`
units). This directory is the committed source of truth for them.

| File                     | Installs to                                | Role |
|--------------------------|--------------------------------------------|------|
| `fishrl-selfplay.service`| `/etc/systemd/system/fishrl-selfplay.service` | long-running self-play trainer (auto-resume, `Restart=always`) |
| `fishrl-selfplay.sh`     | `/home/zubekanov/bin/fishrl-selfplay.sh`      | trainer launcher (venv + `python -m fishrl.train --resume --iters 0 ...`) |
| `fishrl-eval.service`    | `/etc/systemd/system/fishrl-eval.service`     | oneshot out-of-band win-rate panel (100 games/anchor) |
| `fishrl-eval.timer`      | `/etc/systemd/system/fishrl-eval.timer`       | drives the panel hourly |
| `fishrl-eval.sh`         | `/home/zubekanov/bin/fishrl-eval.sh`          | panel launcher (`python -m fishrl.eval.parallel_panel`) |
| `fishrl-serve.service`   | `/etc/systemd/system/fishrl-serve.service`    | LAN telemetry server (range API + SSE; read-only) |

Both launchers hardcode the repo path and the shared venv
(`/home/zubekanov/Repositories/Website_Dev/.venv`); edit those two lines when
deploying elsewhere.

## Install / update

```bash
sudo cp deploy/fishrl-selfplay.service deploy/fishrl-eval.service deploy/fishrl-eval.timer deploy/fishrl-serve.service /etc/systemd/system/
install -m 755 deploy/fishrl-selfplay.sh deploy/fishrl-eval.sh ~/bin/
sudo systemctl daemon-reload && sudo systemctl enable --now fishrl-selfplay
sudo systemctl enable --now fishrl-eval.timer
sudo systemctl enable --now fishrl-serve                # telemetry API/SSE on :8765
journalctl -u fishrl-selfplay -f        # hourly status lines
journalctl -u fishrl-eval.service -f    # [eval ...] panel lines
```

A restart is also how the trainer picks up new code (e.g. newly registered
scenarios): `sudo systemctl restart fishrl-selfplay` — `--resume` continues from
`checkpoints/latest.pt`, and `TimeoutStopSec=150` gives the graceful final
checkpoint time to land first.

## Relay training (ODROID ↔ PC)

**Automated:** on the PC, `deploy\fishrl-relay.ps1` (i.e. `python -m fishrl.relay
train`) runs this whole section as one command — remote stop → export → import here →
train → Ctrl-C → export → import there → restart. The steps below are what it does,
kept as the manual fallback and the recovery reference.

One model lineage, exactly one trainer at a time (see `fishrl.train.ownership`).
`owner.json` in the checkpoint dir tracks whose turn it is; `trainer.lock` is held by a
live trainer (`transfer export` refuses while it's held). Every interrupted handoff
resolves to at most ONE owner — possibly zero, recovered with `transfer claim`.

**PC takes over** (run on the ODROID unless marked):

```bash
sudo systemctl stop fishrl-selfplay             # graceful checkpoint (TimeoutStopSec=150)
python -m fishrl.transfer export --ckpt-dir checkpoints    # cuts zip, RELEASES ownership
# copy the zip to the PC (scp/share), then ON THE PC:
#   python -m fishrl.transfer import --zip <zip> --ckpt-dir checkpoints --force --merge-stats
#   python -m fishrl.train --resume --iters 0 ... (or deploy\fishrl-selfplay.ps1)
```

**Handing back**: Ctrl-C the PC trainer, `transfer export` there, copy the zip over, then
here: `transfer import --zip <zip> --ckpt-dir checkpoints --force --merge-stats` and
`sudo systemctl start fishrl-selfplay`. `--merge-stats` keeps the dashboard history
continuous across hosts. The hourly eval timer can stay enabled throughout — with no
fresh checkpoint or harvest rows it just re-scores/idles cheaply.

| owner.json state            | trainer here | meaning / fix |
|-----------------------------|--------------|---------------|
| `active`, this host         | runs         | normal |
| `active`, other host        | refuses      | other machine's turn; `--claim` only if it truly stopped |
| `released`                  | refuses      | lineage was exported; import the newer zip, or `transfer claim` to fork deliberately |
| missing                     | auto-claims  | pre-relay dir (first run after this update) |

Failure rules: a failed **export** leaves this dir active (keep training). An export
whose zip never got imported leaves NOBODY active — `python -m fishrl.transfer claim`
on whichever machine should resume. A stale zip (older generation than the local
stamp) is refused on import. Worst case is never corruption, only a deliberate choice.

## Note: stale `--user` unit

An older `fishrl-selfplay` **user** unit (`systemctl --user`) exists on this box from
before the move to a system unit. It is NOT the production service — ignore it, or
remove it (`systemctl --user disable --now fishrl-selfplay; rm ~/.config/systemd/user/fishrl-selfplay.service; systemctl --user daemon-reload`).
The real service is the system one above (`systemctl status fishrl-selfplay`, no `--user`).
