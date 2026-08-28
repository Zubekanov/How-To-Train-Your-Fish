# 2026-08-29 — window-pooled critic calibration headline (e37f82e)

Joseph's question: why do critic acc/brier flicker when a window holds ~300
decisions × ~46k games? Traced the telemetry source:

## Where it came from

`estimator_metrics(m, batch)` runs EVERY iteration (free — it reuses
fill_critic_values' forward), but only the LAST batch's result survived into
the report (`last_pre_est`). One batch ≈ 360 games ≈ 90k decisions — and
decisions within a game share the outcome (ICC ≈ 0.96, the by-turn-streaks
finding), so the effective sample is ~the game count, not the decision
count: SE(acc) ≈ sqrt(.72·.28/400) ≈ 2.3pp. That is exactly the observed
0.70-0.74 / 0.17-0.19 flicker. The "300× more datapoints" don't exist
statistically within one batch — but they DO exist across the window's ~130
batches, and the accumulator was already there: `_calib_sums` (per-turn
n/brier-sum/hit-sum) folds into `calib_turn`/`calib_bucket` every iteration;
only the by-turn array (`critic_turn_win`) was ever exported from it.

## The fix (one fold, no new collection)

`calib_from_sums` now also emits the overall totals (`n`, `critic_acc`,
`critic_brier`) — the turn cells partition every scored decision (turn-0
pools into slot 1, 40+ into the last slot), so the fold is exact. The
status line and stats.json HEADLINE `critic_acc`/`critic_brier` are now the
window fold over EVERY batch scored this report (~47k games → SE ~0.2pp,
~10× less noise); the last-batch values ride alongside as
`critic_acc_batch`/`critic_brier_batch`, and `critic_calib_n` records the
pooled decision count. Unchanged by design: `critic_turn` + the per-bucket
keys stay last-batch (Joseph's recorded preference for the streaky read on
the by-turn graph) with `critic_turn_win` as the rigorous companion.

Test: extended `test_calib_from_sums_matches_per_batch` — a single batch's
fold reproduces the per-batch acc/brier exactly; doubling the sums doubles n
and leaves both invariant. Full suite 343 passed / 7 skipped.

## Deploy lesson (second self-match bite)

The relaunch one-liner died at `pkill -f '[p]arallel_panel'` with exit 255:
the same compound ssh command ALSO contained the plain string
`parallel_panel` in its tmux send-keys text, so pkill matched the remote
shell's own command line and killed it (the bracket trick protects the
pattern token, not other occurrences later in the same command line). Rule
extended: the bracket-pattern kill/check must never share a command line
with ANY plain occurrence of the target string — run kills and relaunches
as separate ssh invocations. Recovered by splitting; box resumed at
it=162,601, both tmux windows live.
