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

Both launchers hardcode the repo path and the shared venv
(`/home/zubekanov/Repositories/Website_Dev/.venv`); edit those two lines when
deploying elsewhere.

## Install / update

```bash
sudo cp deploy/fishrl-selfplay.service deploy/fishrl-eval.service deploy/fishrl-eval.timer /etc/systemd/system/
install -m 755 deploy/fishrl-selfplay.sh deploy/fishrl-eval.sh ~/bin/
sudo systemctl daemon-reload && sudo systemctl enable --now fishrl-selfplay
sudo systemctl enable --now fishrl-eval.timer
journalctl -u fishrl-selfplay -f        # hourly status lines
journalctl -u fishrl-eval.service -f    # [eval ...] panel lines
```

A restart is also how the trainer picks up new code (e.g. newly registered
scenarios): `sudo systemctl restart fishrl-selfplay` — `--resume` continues from
`checkpoints/latest.pt`, and `TimeoutStopSec=150` gives the graceful final
checkpoint time to land first.

## Note: stale `--user` unit

An older `fishrl-selfplay` **user** unit (`systemctl --user`) exists on this box from
before the move to a system unit. It is NOT the production service — ignore it, or
remove it (`systemctl --user disable --now fishrl-selfplay; rm ~/.config/systemd/user/fishrl-selfplay.service; systemctl --user daemon-reload`).
The real service is the system one above (`systemctl status fishrl-selfplay`, no `--user`).
