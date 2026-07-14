# Windows analogue of fishrl-serve.service: the LAN telemetry dashboard (a
# sibling of the trainer; safe to start/stop any time). Binds the primary LAN
# IPv4 on :8765 -- pass --bind/--port through to override. --allow-actions is
# the deliberate opt-in for the dashboard's "End session & hand back" button
# (drops the STOP file; a fishrl-relay session then syncs back by itself).
#
# -CkptDir picks the lineage to watch. It used to be hard-coded to "checkpoints",
# which silently showed the WRONG run once a branch existed:
#     .\fishrl-serve.ps1                              # the main lineage
#     .\fishrl-serve.ps1 -CkptDir checkpoints-clock   # the deckout-clock branch
param([string]$CkptDir = "checkpoints")
$Repo = Split-Path -Parent $PSScriptRoot
$Dir = if ([System.IO.Path]::IsPathRooted($CkptDir)) { $CkptDir } else { Join-Path $Repo $CkptDir }
if (-not (Test-Path $Dir)) { Write-Error "no such ckpt-dir: $Dir"; exit 1 }
& "$Repo\.venv\Scripts\Activate.ps1"
Set-Location $Repo
# Replace any already-running instance: the dashboard page is baked in at
# startup, so a survivor from an old session serves STALE code forever --
# and Windows' SO_REUSEADDR semantics let both bind :8765 at once, with the
# oldest winning the connections (observed: a June instance shadowing every
# UI change since). Stop-Process on ourselves is impossible (new PID).
Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
    Where-Object { $_.CommandLine -match "fishrl\.serve" } |
    ForEach-Object {
        Write-Host "[fishrl-serve] stopping stale instance pid=$($_.ProcessId)"
        try { Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop } catch {}
    }
Write-Host "[fishrl-serve] watching $Dir"
python -m fishrl.serve --ckpt-dir "$Dir" --allow-actions @args
