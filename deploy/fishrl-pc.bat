@echo off
rem ============================================================================
rem  One-click PC training session for fishrl (double-click me, or make a
rem  desktop shortcut).
rem
rem    1. starts the LAN dashboard in a minimized window and opens it in the
rem       default browser (http://<this-pc>:8765/)
rem    2. pulls the lineage off the ODROID: stops its service (graceful
rem       checkpoint), exports, imports here
rem    3. trains in THIS window. End the session with Ctrl-C here or the
rem       dashboard's "End session & hand back" button -- either way the
rem       trainer checkpoints, the result is exported back, imported on the
rem       ODROID, and its service is restarted.
rem
rem  Extra RELAY flags pass through, e.g.:   fishrl-pc.bat --no-return
rem  If a handback ever fails (ODROID unreachable), the lineage stays safely
rem  here -- re-run this script (it skips the pull) or run the handback line
rem  printed at the end.
rem ============================================================================
setlocal
for %%i in ("%~dp0..") do set "REPO=%%~fi"
set "PY=%REPO%\.venv\Scripts\python.exe"

start "fishrl dashboard" /min powershell -NoProfile -ExecutionPolicy Bypass -File "%REPO%\deploy\fishrl-serve.ps1"
rem Win-rate panels while the session trains (the PC's stand-in for the ODROID's
rem hourly eval timer): waits for the trainer, evals every 15 min, exits with it.
start "fishrl eval" /min "%PY%" -m fishrl.eval.parallel_panel --ckpt-dir "%REPO%\checkpoints" --follow 900 --reserve-cores 12
ping -n 4 127.0.0.1 >nul
start "" "http://%COMPUTERNAME%:8765/"

rem Training-regime flags come from deploy\train.args (single line; shared with
rem the ODROID's fishrl-selfplay.sh) so the lineage trains identically on both
rem hosts; only machine flags (cadences, workers) are set here.
set /p REGIME=<"%REPO%\deploy\train.args"
"%PY%" -m fishrl.relay train --ckpt-dir "%REPO%\checkpoints" %* -- ^
    %REGIME% ^
    --report-every-seconds 900 --report-winrate-games 0 ^
    --checkpoint-every-seconds 900 ^
    --collect-workers 8 --reserve-cores 2

echo.
echo Session ended. If the handback failed above, the lineage is still on this
echo PC -- when the ODROID is reachable again, re-run this script (it skips the
echo pull leg) or run the handback alone (note the leading ^& in PowerShell):
echo   cmd:        "%PY%" -m fishrl.relay handback --ckpt-dir "%REPO%\checkpoints"
echo   PowerShell: ^& "%PY%" -m fishrl.relay handback --ckpt-dir "%REPO%\checkpoints"
pause
