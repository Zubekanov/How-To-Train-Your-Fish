# THE one command for training on this PC: pulls the lineage off the ODROID
# (stops its service, exports, imports here), trains in the foreground, and on
# Ctrl-C checkpoints + hands the result back and restarts the ODROID service.
# Watch it on the dashboard (deploy\fishrl-serve.ps1). Ownership is enforced end-to-end; if anything is
# interrupted, re-run this (it skips the pull leg if the lineage is already
# here) or run `python -m fishrl.relay handback`.
#
# Trainer settings mirror the ODROID unit, with PC niceties (reserve + GUI +
# denser reports). Anything you pass to this script is forwarded to the relay,
# e.g.:  deploy\fishrl-relay.ps1 --no-return
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
python -m fishrl.relay train --ckpt-dir "$Repo\checkpoints" @args -- `
    $Regime `
    --report-every-seconds 900 --report-winrate-games 0 `
    --checkpoint-every-seconds 900 `
    --gpu --collect-workers 16 --reserve-cores 2 --collect-affinity 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --pipeline-collect --infer-server
