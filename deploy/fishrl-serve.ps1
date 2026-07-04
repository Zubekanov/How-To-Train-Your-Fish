# Windows analogue of fishrl-serve.service: the LAN telemetry server (read-only
# sibling of the trainer; safe to start/stop any time). Binds the primary LAN
# IPv4 on :8765 -- pass --bind/--port through to override.
$Repo = Split-Path -Parent $PSScriptRoot
& "$Repo\.venv\Scripts\Activate.ps1"
Set-Location $Repo
python -m fishrl.serve --ckpt-dir "$Repo\checkpoints" @args
