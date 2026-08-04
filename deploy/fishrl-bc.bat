@echo off
rem ============================================================================
rem  One-click BC-LINEAGE training session (double-click me) -- the sibling of
rem  fishrl-pc.bat for the behaviour-cloning bootstrap experiment.
rem
rem    1. starts the LAN dashboard for checkpoints-bc on :8766 (the v2 dashboard
rem       keeps :8765 -- both can run side by side) and opens it in the browser
rem    2. runs win-rate panels alongside the session, every 15 min
rem    3. trains in THIS window: resumes checkpoints-bc\latest.pt with the
rem       regime in deploy\train-bc.args
rem
rem  The lineage is CREATED by the BC trainer, not by this script:
rem      python -m fishrl.imitate.bc --games 4000 --workers 10 --epochs 24 ^
rem          --gpu --out checkpoints-bc
rem  clones the scripted teacher into checkpoints-bc\latest.pt (architecture
rem  rides in the checkpoint; docs/BC_ADAPTER.md). This script only CONTINUES
rem  that checkpoint with PPO, so it refuses to run without one -- a fresh start
rem  here would silently train a default-architecture model from scratch.
rem
rem  The BC-handoff phases (critic-only warmup, KL-to-teacher anneal) are
rem  one-shot flags for the FIRST fine-tune launch, already consumed by this
rem  lineage -- they are deliberately NOT in train-bc.args. A future fresh clone
rem  wants them back:  fishrl-bc.bat --freeze-actor-iters 500 ^
rem                        --kl-teacher-coef 0.3 --kl-teacher-iters 5000
rem
rem  End with Ctrl-C here or the dashboard's "End session" button (STOP file;
rem  the trainer checkpoints gracefully at the next iteration boundary).
rem  Extra training flags pass through, e.g.:   fishrl-bc.bat --max-hours 12
rem ============================================================================
setlocal
for %%i in ("%~dp0..") do set "REPO=%%~fi"
set "PY=%REPO%\.venv\Scripts\python.exe"
set "CKPT=%REPO%\checkpoints-bc"
set "PORT=8766"

if not exist "%CKPT%\latest.pt" (
    echo [fishrl-bc] no checkpoint in %CKPT%.
    echo             This lineage starts from a behaviour-cloned actor; create it with
    echo               %PY% -m fishrl.imitate.bc --games 4000 --workers 10 --epochs 24 --gpu --out checkpoints-bc
    echo             then run this script again.
    pause
    exit /b 1
)

start "fishrl-bc dashboard" /min powershell -NoProfile -ExecutionPolicy Bypass -File "%REPO%\deploy\fishrl-serve.ps1" -CkptDir "%CKPT%" --port %PORT%
rem Win-rate panels while the session trains: waits for the trainer (trainer.lock),
rem evals every 15 min, exits with the session.
start "fishrl-bc eval" /min "%PY%" -m fishrl.eval.parallel_panel --ckpt-dir "%CKPT%" --follow 900 --reserve-cores 12 --affinity 16,17,18,19,20,21,22,23,24,25,26,27 --seat-diag-games 48
ping -n 4 127.0.0.1 >nul

rem Open the address the dashboard ACTUALLY binds (lan_ip(), same helper serve
rem binds with) -- see fishrl-pc.bat for why not %COMPUTERNAME%.
for /f "usebackq tokens=*" %%u in (`"%PY%" -c "from fishrl.serve.__main__ import lan_ip; print(lan_ip())"`) do set "LANIP=%%u"
if not defined LANIP set "LANIP=127.0.0.1"
echo [fishrl-bc] dashboard: http://%LANIP%:%PORT%/
start "" "http://%LANIP%:%PORT%/"

rem Training-regime flags come from deploy\train-bc.args (single line), the
rem BC-lineage sibling of deploy\train.args. Architecture comes from the
rem checkpoint (always a resume here), so the regime line is runtime-only.
set /p REGIME=<"%REPO%\deploy\train-bc.args"
"%PY%" -m fishrl.train --resume --iters 0 --ckpt-dir "%CKPT%" %REGIME% ^
    --report-every-seconds 900 --report-winrate-games 0 ^
    --checkpoint-every-seconds 900 ^
    --gpu --collect-workers 8 --reserve-cores 2 --collect-affinity 0,2,4,6,8,10,12,14 --pipeline-collect --infer-server %*

rem ?????? the trainer has exited (Ctrl-C, or the dashboard's "End training session") ??????
rem End THIS lineage's eval panel with the session (matched on the ckpt dir so a
rem v2 panel running alongside is untouched); an in-flight panel would otherwise
rem grind for another minute or two, which reads as a hang.
echo.
echo [fishrl-bc] trainer exited; stopping the eval panel...
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*parallel_panel*' -and $_.CommandLine -like '*checkpoints-bc*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"

echo.
echo Session ended; the trainer checkpointed to %CKPT%\latest.pt.
echo The dashboard is still running at http://%LANIP%:%PORT%/ -- close its window when done.
pause
