# Windows analogue of fishrl-selfplay.sh: resume-and-run-forever self-play.
# Ctrl-C = graceful checkpoint (same as SIGTERM under systemd). Core reservation
# replaces the systemd unit's *_NUM_THREADS env. Watch it on the fishrl-serve
# dashboard (deploy\fishrl-serve.ps1).
$Repo = Split-Path -Parent $PSScriptRoot
& "$Repo\.venv\Scripts\Activate.ps1"
Set-Location $Repo
# Win-rate panels while the session trains (stand-in for the ODROID's eval
# timer): waits for the trainer, evals every 15 min, exits with the session.
Start-Process -WindowStyle Minimized -FilePath "$Repo\.venv\Scripts\python.exe" `
    -ArgumentList "-m","fishrl.eval.parallel_panel","--ckpt-dir","$Repo\checkpoints", `
                  "--follow","900","--reserve-cores","12", `
                  "--affinity","16,17,18,19,20,21,22,23,24,25,26,27"
# Training-regime flags come from deploy\train.args (shared with the ODROID's
# fishrl-selfplay.sh) so the lineage trains identically on both hosts; only
# machine flags (cadences, workers) are set here.
$Regime = ((Get-Content "$Repo\deploy\train.args" -Raw).Trim() -split '\s+')
python -m fishrl.train --resume --iters 0 `
    --ckpt-dir "$Repo\checkpoints" $Regime `
    --report-every-seconds 900 --report-winrate-games 0 `
    --checkpoint-every-seconds 900 `
    --gpu --collect-workers 8 --reserve-cores 2 --collect-affinity 0,2,4,6,8,10,12,14 --pipeline-collect
