# Windows analogue of fishrl-selfplay.sh: resume-and-run-forever self-play
# with the monitor window. Ctrl-C = graceful checkpoint (same as SIGTERM under
# systemd). Core reservation replaces the systemd unit's *_NUM_THREADS env.
$Repo = Split-Path -Parent $PSScriptRoot
& "$Repo\.venv\Scripts\Activate.ps1"
Set-Location $Repo
python -m fishrl.train --resume --iters 0 `
    --encoder flat --critic-encoder entity `
    --ckpt-dir "$Repo\checkpoints" --scenario-frac 0.3 --warmup-games 16 `
    --report-every-seconds 3600 --report-winrate-games 0 `
    --checkpoint-every-seconds 900 `
    --reserve-cores 2 --gui
