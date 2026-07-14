# Branch the 592h run onto the deckout-clock observation.
#
# The clock adds 3 scalars (library parity, next-drawer, who-decks-first) that the nets
# provably could NOT compute from len(library)/80.0 -- see fishrl/eval/probe_deckout_clock.py.
# `migrate_clock` warm-started latest.pt onto the new observation with the new input columns
# zeroed, so the branch STARTS functionally identical to the 592h checkpoint and only then
# learns to use the channel. Nothing here touches the source run.
#
# The trainer always resumes from <ckpt-dir>\latest.pt, so the branch is a NEW ckpt-dir with
# the migrated checkpoint installed as its latest.pt. The original checkpoints\ is left intact
# and you can always go back to deploy\fishrl-selfplay.ps1.
#
# Watch it with:  deploy\fishrl-serve.ps1 -CkptDir checkpoints-clock   (or edit that script)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
& "$Repo\.venv\Scripts\Activate.ps1"
Set-Location $Repo

$Src = "$Repo\checkpoints"
$Dst = "$Repo\checkpoints-clock"

# ── one-time branch setup (idempotent: a later run just resumes) ──────────────
if (-not (Test-Path "$Dst\latest.pt")) {
    if (-not (Test-Path "$Src\clock.pt")) {
        Write-Host "[clock] $Src\clock.pt is missing. Create it with:" -ForegroundColor Yellow
        Write-Host "  python -m fishrl.train.migrate_clock --in checkpoints\latest.pt --out checkpoints\clock.pt"
        exit 1
    }
    New-Item -ItemType Directory -Force $Dst | Out-Null
    Copy-Item "$Src\clock.pt" "$Dst\latest.pt"

    # Carry the dashboard history across so the branch point is visible IN CONTEXT -- the whole
    # question is whether deckout / board_presence move, and you want the flat 592h baseline on
    # the same chart. The source stats.json is copied, never moved.
    foreach ($f in @("stats.json", "ticks.json")) {
        if (Test-Path "$Src\$f") { Copy-Item "$Src\$f" "$Dst\$f" }
    }

    # Deliberately NOT copied:
    #   owner.json    -- it says state="released"; the trainer would REFUSE to start ("import the
    #                    newer run zip..."). An absent stamp auto-claims the dir instead.
    #   best.pt/.json -- best.pt is the pre-clock obs dim, and best.json carries a stale win-rate
    #                    threshold. The branch seeds its own best on its first eval panel.
    #   trainer.lock  -- a stale lock file would block the launch.
    Write-Host "[clock] branched: $Dst\latest.pt  (source run untouched)" -ForegroundColor Green
}

# ── win-rate panels alongside the session (same as fishrl-selfplay.ps1) ───────
Start-Process -WindowStyle Minimized -FilePath "$Repo\.venv\Scripts\python.exe" `
    -ArgumentList "-m","fishrl.eval.parallel_panel","--ckpt-dir","$Dst", `
                  "--follow","900","--reserve-cores","12", `
                  "--affinity","16,17,18,19,20,21,22,23,24,25,26,27"

# Identical training regime to the source run (deploy\train.args) -- the ONLY difference between
# this branch and the 592h lineage is the deckout clock. Anything else would confound the read.
$Regime = ((Get-Content "$Repo\deploy\train.args" -Raw).Trim() -split '\s+')
python -m fishrl.train --resume --iters 0 `
    --ckpt-dir "$Dst" $Regime `
    --report-every-seconds 900 --report-winrate-games 0 `
    --checkpoint-every-seconds 900 `
    --gpu --collect-workers 8 --reserve-cores 2 --collect-affinity 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --pipeline-collect
