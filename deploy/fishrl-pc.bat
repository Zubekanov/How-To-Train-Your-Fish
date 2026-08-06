@echo off
rem ============================================================================
rem  One-click PC training session for fishrl (double-click me, or make a
rem  desktop shortcut).
rem
rem    1. starts the LAN dashboard in a minimized window and opens it in the
rem       default browser (http://<this-pc>:8765/)
rem    2. runs win-rate panels alongside the session, every 15 min
rem    3. trains in THIS window: resumes %CKPT%\latest.pt, or FRESH-starts (with the
rem       architecture in deploy\train.args) when the dir is empty
rem
rem  The active lineage is checkpoints-v2 (the entity / bigger-actor run, ~1700 it/h on
rem  this box). The original 622h flat run is preserved untouched in checkpoints\.
rem
rem  End with Ctrl-C here or the dashboard's "End training session" button:
rem  either way the trainer checkpoints gracefully at the next iteration
rem  boundary. The BUTTON additionally tears the whole session down (it drops a
rem  TEARDOWN marker this script consumes): eval panel, dashboard + its window,
rem  and this console all close by themselves. Ctrl-C keeps the dashboard up and
rem  this window open so you can read the run out.
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
rem The ACTIVE run lives here. checkpoints-v2 is the entity/bigger-actor lineage; the
rem original 622h flat run is preserved untouched in %REPO%\checkpoints. To go back to
rem it, point CKPT there. (--resume auto-starts fresh when the dir has no checkpoint, so
rem the first launch of a new dir bootstraps the architecture from deploy\train.args.)
set "CKPT=%REPO%\checkpoints-v2"

rem Create the lineage dir up front: the dashboard (started below, before the trainer)
rem refuses a non-existent --ckpt-dir, and on a fresh run the trainer hasn't made it yet.
if not exist "%CKPT%" mkdir "%CKPT%"

rem A TEARDOWN leftover from a session that died before consuming it is not ours
rem to honor -- same hygiene the trainer applies to a stale STOP.
if exist "%CKPT%\TEARDOWN" del "%CKPT%\TEARDOWN"

if not exist "%CKPT%\latest.pt" (
    echo [fishrl-pc] no checkpoint in %CKPT% -- FRESH start with the deploy\train.args
    echo             architecture ^(entity actor, 768/768/384, card_dim 128^).
    echo             The original flat run is untouched in %REPO%\checkpoints.
)

start "fishrl dashboard" /min powershell -NoProfile -ExecutionPolicy Bypass -File "%REPO%\deploy\fishrl-serve.ps1" -CkptDir "%CKPT%"
rem Win-rate panels while the session trains: waits for the trainer (trainer.lock),
rem evals every 15 min, exits with the session.
start "fishrl eval" /min "%PY%" -m fishrl.eval.parallel_panel --ckpt-dir "%CKPT%" --follow 900 --reserve-cores 12 --affinity 16,17,18,19,20,21,22,23,24,25,26,27 --seat-diag-games 48
ping -n 4 127.0.0.1 >nul

rem Open the address the dashboard ACTUALLY binds -- lan_ip(), the same helper
rem fishrl.serve binds with. It used to open http://%COMPUTERNAME%:8765/, but the
rem hostname resolves to a VIRTUAL host-only adapter here (192.168.56.1, VirtualBox/
rem Hyper-V) with nothing listening on it, while serve binds the real LAN IPv4
rem (192.168.4.25). curl retries the other resolved addresses and succeeds; a browser
rem stops at the first and just shows "unavailable" -- with the server running fine.
rem Temp-file capture, NOT for /f: for /f re-runs its command via `cmd /c`,
rem which strips the first+last quote of a command that STARTS with one and
rem mangled this into 'python.exe" -c "from' -- the probe errored on every
rem launch and LANIP silently fell back to 127.0.0.1.
"%PY%" -c "from fishrl.serve.__main__ import lan_ip; print(lan_ip())" > "%TEMP%\fishrl_lanip.txt" 2>nul
set /p LANIP=<"%TEMP%\fishrl_lanip.txt"
del "%TEMP%\fishrl_lanip.txt" 2>nul
if not defined LANIP set "LANIP=127.0.0.1"
echo [fishrl-pc] dashboard: http://%LANIP%:8765/
start "" "http://%LANIP%:8765/"

rem Training-regime flags come from deploy\train.args (single line) so the regime
rem is declared in one place and cannot drift between launchers.
set /p REGIME=<"%REPO%\deploy\train.args"
"%PY%" -m fishrl.train --resume --iters 0 --ckpt-dir "%CKPT%" %REGIME% ^
    --report-every-seconds 900 --report-winrate-games 0 ^
    --checkpoint-every-seconds 900 ^
    --gpu --collect-workers 8 --reserve-cores 2 --collect-affinity 0,2,4,6,8,10,12,14 --pipeline-collect --infer-server %*

rem ?????? the trainer has exited (Ctrl-C, or the dashboard's "End training session") ??????
rem The eval panel follows trainer.lock and DOES exit on its own -- but it only checks
rem between panels, so an in-flight one keeps its minimized window grinding for another
rem minute or two after the session is over. That reads as a hang. Nothing it produces
rem now is useful (the checkpoint is final; the next session evaluates it anyway), so
rem end it with the session.
echo.
echo [fishrl-pc] trainer exited; stopping the eval panel...
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*parallel_panel*' -and $_.CommandLine -like '*checkpoints-v2*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"

rem The dashboard's "End training session" drops a TEARDOWN marker next to the
rem STOP file: it asks for the WHOLE session to close, not just the trainer.
rem Consume it and take everything down -- the dashboard (killing its python
rem ends fishrl-serve.ps1, which closes its minimized window) and this console
rem (script end closes a double-clicked window; from a terminal it just
rem returns). The browser tab can't be closed from here; the page shows
rem "session ended" once the dashboard is gone.
if exist "%CKPT%\TEARDOWN" goto :teardown

echo.
echo Session ended; the trainer checkpointed to %CKPT%\latest.pt.
echo The dashboard is still running at http://%LANIP%:8765/ -- close its window when done.
pause
exit /b 0

:teardown
del "%CKPT%\TEARDOWN"
echo [fishrl-pc] full teardown requested from the dashboard; closing it too...
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -like '*fishrl.serve*' -and $_.CommandLine -like '*checkpoints-v2*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
echo [fishrl-pc] session ended; the trainer checkpointed to %CKPT%\latest.pt.
exit /b 0
