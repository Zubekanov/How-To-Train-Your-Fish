# Windows analogue of fishrl-serve.service: the LAN telemetry dashboard (a
# sibling of the trainer; safe to start/stop any time). Binds the primary LAN
# IPv4 on :8765 -- pass --bind/--port through to override. --allow-actions is
# the deliberate opt-in for the dashboard's "End session & hand back" button
# (drops the STOP file; a fishrl-relay session then syncs back by itself).
$Repo = Split-Path -Parent $PSScriptRoot
& "$Repo\.venv\Scripts\Activate.ps1"
Set-Location $Repo
python -m fishrl.serve --ckpt-dir "$Repo\checkpoints" --allow-actions @args
