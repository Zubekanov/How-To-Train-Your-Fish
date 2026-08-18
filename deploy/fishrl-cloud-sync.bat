@echo off
rem ============================================================================
rem  fishrl-cloud-sync.bat -- pull the CLOUD mainline (Vast.ai box) back onto
rem  this machine: model checkpoints + telemetry, staged then swapped, so the
rem  local checkpoints-v3 inherits the cloud run and is --resume'able from the
rem  sync point (fishrl-pc.bat just works afterwards).
rem
rem    fishrl-cloud-sync.bat          sync only; the cloud trainer keeps running
rem    fishrl-cloud-sync.bat /stop    gracefully STOP the cloud trainer first
rem                                   (drops the STOP marker; it checkpoints at
rem                                   the next iteration boundary), then sync.
rem
rem  USE /stop WHEN BRINGING THE MAINLINE HOME: resuming locally while the cloud
rem  box still trains would fork the lineage again (two active trainers). A
rem  plain sync is for peeking at progress / keeping a warm local copy -- fine
rem  while the cloud runs, but do NOT start the local trainer from it.
rem
rem  Pulled: latest.pt best.pt best.json stats.json ticks.json
rem  (step_/archive_ snapshots stay on the box; add them to the scp line below
rem  if you want them). The pre-sync local files are kept in
rem  checkpoints-v3\presync-<stamp>\ -- delete those folders when satisfied.
rem
rem  Safety rails:
rem    * refuses to run while a LOCAL trainer holds the lineage (process check);
rem    * downloads into a temp dir and swaps only after latest.pt passes a size
rem      sanity check, so an interrupted transfer never corrupts the lineage
rem      (the remote writer is atomic-rename, so reads are never torn);
rem    * owner.json is NOT pulled -- the local stamp stays "DESKTOP-TBTMS9B
rem      active", which is exactly what a later local --resume needs.
rem
rem  When the Vast instance changes, update the three VAST_* lines (the ssh
rem  command on the instance card has the port/host; proxy form works too).
rem ============================================================================
setlocal EnableDelayedExpansion
for %%i in ("%~dp0..") do set "REPO=%%~fi"
set "PY=%REPO%\.venv\Scripts\python.exe"
set "CKPT=%REPO%\checkpoints-v3"
if defined FISHRL_SYNC_DEST set "CKPT=%FISHRL_SYNC_DEST%"

set "VAST_HOST=root@175.28.230.22"
set "VAST_PORT=50380"
rem proxy alternative:  set "VAST_HOST=root@ssh9.vast.ai"  set "VAST_PORT=21177"
set "RCKPT=/workspace/fishrl-repo/checkpoints-v3"
set "SSHOPTS=-o BatchMode=yes -o ConnectTimeout=15"

rem -- refuse while a local trainer is live on this lineage ---------------------
for /f %%c in ('powershell -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*fishrl.train*' -and $_.CommandLine -like '*checkpoints-v3*' }).Count"') do set "NLOCAL=%%c"
if not "%NLOCAL%"=="0" if not "%NLOCAL%"=="" (
    echo [sync] REFUSING: a local trainer is running on checkpoints-v3 ^(%NLOCAL% process^).
    echo        End the local session first -- syncing under it would fight its
    echo        checkpoint writes and fork the lineage.
    exit /b 1
)

rem -- optional graceful remote stop -------------------------------------------
if /i "%~1"=="/stop" (
    echo [sync] stopping the cloud trainer gracefully ^(STOP marker^)...
    ssh -p %VAST_PORT% %SSHOPTS% %VAST_HOST% "touch %RCKPT%/STOP; for i in $(seq 1 60); do pgrep -f 'python -m fishrl.train' >/dev/null || { echo '[remote] trainer exited cleanly'; exit 0; }; sleep 5; done; echo '[remote] WARNING: trainer still running after 5 min'; exit 2"
    if errorlevel 3 (
        echo [sync] ERROR: could not reach the cloud box; aborting.
        exit /b 1
    ) else if errorlevel 2 (
        echo [sync] WARNING: cloud trainer did not exit in 5 min -- syncing anyway,
        echo        but do NOT resume locally until it is confirmed stopped.
    ) else if errorlevel 1 (
        echo [sync] ERROR: remote stop failed; aborting.
        exit /b 1
    )
)

rem -- pull into a temp dir, then swap ------------------------------------------
for /f %%t in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss"') do set "STAMP=%%t"
set "TMPD=%CKPT%\.sync-tmp"
if exist "%TMPD%" rmdir /s /q "%TMPD%"
mkdir "%TMPD%" 2>nul
if not exist "%CKPT%" mkdir "%CKPT%"

echo [sync] pulling from %VAST_HOST%:%RCKPT% ...
rem one connection, one stream: remote tar -> local tar (Windows 10 ships tar.exe)
ssh -p %VAST_PORT% %SSHOPTS% %VAST_HOST% "cd %RCKPT% && tar cf - latest.pt best.pt best.json stats.json ticks.json" | tar xf - -C "%TMPD%"
if errorlevel 1 (
    echo [sync] ERROR: transfer failed; local lineage untouched.
    exit /b 1
)

rem latest.pt sanity: a real v3 checkpoint is ~150 MB; refuse a stub/torn file.
for %%f in ("%TMPD%\latest.pt") do set "LSIZE=%%~zf"
if not defined LSIZE set "LSIZE=0"
if %LSIZE% LSS 50000000 (
    echo [sync] ERROR: pulled latest.pt is only %LSIZE% bytes -- refusing to swap.
    exit /b 1
)

set "BAK=%CKPT%\presync-%STAMP%"
mkdir "%BAK%" 2>nul
for %%f in (latest.pt best.pt best.json stats.json ticks.json) do (
    if exist "%CKPT%\%%f" move /y "%CKPT%\%%f" "%BAK%\%%f" >nul
)
for %%f in (latest.pt best.pt best.json stats.json ticks.json) do (
    if exist "%TMPD%\%%f" move /y "%TMPD%\%%f" "%CKPT%\%%f" >nul
)
rmdir /s /q "%TMPD%" 2>nul

rem -- summary -------------------------------------------------------------------
"%PY%" -c "import torch; pl = torch.load(r'%CKPT%\latest.pt', map_location='cpu', weights_only=False); print('[sync] synced at iteration', pl.get('done'), '| elapsed', round(pl.get('elapsed', 0) / 3600.0, 2), 'h')" 2>nul
echo [sync] done. Pre-sync files kept in %BAK%
if /i "%~1"=="/stop" (
    echo [sync] cloud trainer stopped; this lineage is now resumable locally:
    echo        double-click deploy\fishrl-pc.bat
) else (
    echo [sync] NOTE: the cloud trainer is still the mainline. Do NOT start the
    echo        local trainer from this copy -- that would fork the lineage.
    echo        Bring it home for real with:  fishrl-cloud-sync.bat /stop
)
exit /b 0
