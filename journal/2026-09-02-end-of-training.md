# 2026-09-02 — end of training: lineage home, box decommissioned, final report

Joseph: "I only have a day of cloud cpu left anyways and with no movement i'm
happy to call 0.75 vs 1.3 the end of training." Decisions (AskUserQuestion):
stop and sync NOW (no consolidation restart — latest.pt is mid-reheat-wobble,
accepted; best.pt is the release agent), full final report as repo markdown
AND an artifact (explicitly requested).

## What was done

1. **Stop + sync**: `deploy\fishrl-cloud-sync.bat /stop` — trainer exited
   cleanly, synced at **it=204,225 / 346.95h**; pre-sync copies in
   `checkpoints-v3\presync-20260902-085001\`. Ownership stays local; the
   lineage is resumable via `fishrl-pc.bat`.
2. **Salvage before box death** → `checkpoints-v3\cloud-final\`: all 19
   `archive_*.pt` (20k…200k), 6 `latest.pt.pre-*.pt` swap backups,
   train.log / eval-panel.log / sysmon.log (chunked tar-over-ssh; the reheat
   experiment's step_*.pt milestones were superseded by the synced latest).
3. **train.args hygiene** (5b2866c): reheat flags removed → constant 0.008
   entropy floor; final regime keeps `--lr-ppo 1.5e-4 --league-every 8
   --critic-consistency 25`. A future local resume consolidates instead of
   continuing a reheat mid-cycle.
4. **Resumability verified**: scratch-copy resume smoke on this PC — resumed
   at it=204,225 (8 league selves restored), 3 iterations, clean checkpoint
   save. The real `checkpoints-v3\` was never advanced.
5. **Final benchmark** on best.pt (it=196,263): large-sample local panel +
   the capstone mine (2000 vs 1.3 + 1600 mirror + passcalib) — numbers in
   `docs/FINAL-REPORT.md` §2.
6. **Docs**: `docs/FINAL-REPORT.md` (the end-of-run record), DESIGN.md
   status addendum, README status note, this entry; final-report artifact
   published for Joseph.

## The run's terminal state

Reheat cycle 2 (peak 0.016, escalated yesterday at it=203,867) ran ~half a
cycle and was never read — the kill criterion (trough vs the 0.757 baseline)
is recorded in 2026-09-02-reheat-dose-escalation.md should training resume.
Final v3 totals: 54.6M games, 12.6B decisions. Last new_best: it=196,263.

## Remaining manual steps (Joseph)

- Destroy the Vast.ai instance from the console (nothing sensitive on it;
  training artifacts fully salvaged).
- Optionally point the website agent at the local final stats.json for a last
  import (the box ingest source is gone).
- Delete `presync-*\` folders once satisfied with the sync.
