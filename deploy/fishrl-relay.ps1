# THE one command for training on this PC: pulls the lineage off the ODROID
# (stops its service, exports, imports here), trains in the foreground with the
# monitor window, and on Ctrl-C checkpoints + hands the result back and restarts
# the ODROID service. Ownership is enforced end-to-end; if anything is
# interrupted, re-run this (it skips the pull leg if the lineage is already
# here) or run `python -m fishrl.relay handback`.
#
# Trainer settings mirror the ODROID unit, with PC niceties (reserve + GUI +
# denser reports). Anything you pass to this script is forwarded to the relay,
# e.g.:  deploy\fishrl-relay.ps1 --no-return
$Repo = Split-Path -Parent $PSScriptRoot
& "$Repo\.venv\Scripts\Activate.ps1"
Set-Location $Repo
python -m fishrl.relay train --ckpt-dir "$Repo\checkpoints" @args -- `
    --scenario-frac 0.3 --warmup-games 16 `
    --report-every-seconds 900 --report-winrate-games 0 `
    --checkpoint-every-seconds 900 `
    --collect-workers 8 --reserve-cores 2 --gui
