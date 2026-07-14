@echo off
rem ============================================================================
rem  One-click PC training session for fishrl (double-click me, or make a
rem  desktop shortcut).
rem
rem    1. starts the LAN dashboard in a minimized window and opens it in the
rem       default browser (http://<this-pc>:8765/)
rem    2. runs win-rate panels alongside the session, every 15 min
rem    3. trains in THIS window, resuming checkpoints\latest.pt
rem
rem  End with Ctrl-C here or the dashboard's "End session" button: either way
rem  the trainer checkpoints gracefully at the next iteration boundary.
rem
rem  THIS IS LOCAL-ONLY, deliberately. It used to be a RELAY launcher: it pulled
rem  the lineage off the ODROID, trained here, and handed it back. That is gone.
rem  The PC does ~1800 it/h against the ODROID's ~205, so the relay bought ~10%%
rem  of the throughput for 100%% of the operational risk -- and every model change
rem  (a new observation feature, say) became a two-machine migration, because a
rem  checkpoint is only loadable by a checkout that encodes the game the same way.
rem  This script does not touch the ODROID.
rem
rem  The relay still exists (python -m fishrl.relay) and its preflight now REFUSES
rem  to move a checkpoint between checkouts whose feature dims differ, so it can no
rem  longer silently break the far end. It is just not the default any more.
rem
rem  Extra training flags pass through, e.g.:   fishrl-pc.bat --max-hours 12
rem ============================================================================
setlocal
for %%i in ("%~dp0..") do set "REPO=%%~fi"
set "PY=%REPO%\.venv\Scripts\python.exe"
set "CKPT=%REPO%\checkpoints"

if not exist "%CKPT%\latest.pt" (
    echo [fishrl-pc] no checkpoint at %CKPT%\latest.pt -- nothing to resume.
    echo             For a fresh run:  python -m fishrl.train --fresh --iters 0
    pause
    exit /b 1
)

start "fishrl dashboard" /min powershell -NoProfile -ExecutionPolicy Bypass -File "%REPO%\deploy\fishrl-serve.ps1"
rem Win-rate panels while the session trains: waits for the trainer (trainer.lock),
rem evals every 15 min, exits with the session.
start "fishrl eval" /min "%PY%" -m fishrl.eval.parallel_panel --ckpt-dir "%CKPT%" --follow 900 --reserve-cores 12 --affinity 16,17,18,19,20,21,22,23,24,25,26,27
ping -n 4 127.0.0.1 >nul

rem Open the address the dashboard ACTUALLY binds -- lan_ip(), the same helper
rem fishrl.serve binds with. It used to open http://%COMPUTERNAME%:8765/, but the
rem hostname resolves to a VIRTUAL host-only adapter here (192.168.56.1, VirtualBox/
rem Hyper-V) with nothing listening on it, while serve binds the real LAN IPv4
rem (192.168.4.25). curl retries the other resolved addresses and succeeds; a browser
rem stops at the first and just shows "unavailable" -- with the server running fine.
for /f "usebackq tokens=*" %%u in (`"%PY%" -c "from fishrl.serve.__main__ import lan_ip; print(lan_ip())"`) do set "LANIP=%%u"
if not defined LANIP set "LANIP=127.0.0.1"
echo [fishrl-pc] dashboard: http://%LANIP%:8765/
start "" "http://%LANIP%:8765/"

rem Training-regime flags come from deploy\train.args (single line) so the regime
rem is declared in one place and cannot drift between launchers.
set /p REGIME=<"%REPO%\deploy\train.args"
"%PY%" -m fishrl.train --resume --iters 0 --ckpt-dir "%CKPT%" %REGIME% ^
    --report-every-seconds 900 --report-winrate-games 0 ^
    --checkpoint-every-seconds 900 ^
    --gpu --collect-workers 8 --reserve-cores 2 --collect-affinity 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --pipeline-collect %*

echo.
echo Session ended; the trainer checkpointed to %CKPT%\latest.pt.
pause
