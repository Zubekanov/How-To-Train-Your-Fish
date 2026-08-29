# 2026-08-29 — 163k mine: scenario lessons learnt; boost 5→2 (Joseph's call)

Trigger: Joseph proposed unwinding the scenario boost ("usefulness mined
out... 0.7 against 1.3 and no more growth indicates the lessons are probably
learnt") and chose to mine first. 2,000 games vs h1.3 (seeds 920000+) +
1,600 mirror (1020000+) on the it≈162.9k checkpoint, full analyzer battery
side-by-side against the 135k (pre-selfplay) mine files.

## Scenario lessons: learned (the cut is supported)

- wr vs 1.3: 0.643 → **0.708** (mine == telemetry).
- removal_in_hand: waste-on-land-with-fish-available 3.4% → **1.4%** (mirror
  6.7% → 1.5%).
- undoing_call: Day's Undoing blowups 19.1% → **12.7%**, mean value now
  POSITIVE (+0.016 from −0.018).
- deckout_short/deckout_stack: **deckout losses vs 1.3 halved** (127 → 69 /
  2000) — the punisher role the 1.3 engine seat used to script is now
  executed by the agent (the seat-change's #1 watch item: PASSED).
- fof_split: degenerate 0-5 splits 0.3% → 0.1%; endgame (lib≤14) 1-4 splits
  1.0% → 7.6% = CONTEXTUAL asymmetric splitting emerging, not degeneracy.
- Self-play-blind canaries vs 1.3 all improved: telegraphing catastrophic
  2.0% → 1.4% (mean err now negative), respond-rate conditioning up
  (land 0.08→0.13, spell 0.49→0.51). Scenario wr_self flat 0.46-0.47 → no
  mirror-collusion drift.
- Critic sharper: decision Brier 0.165 → 0.151.

## The "passivity blunders" — ADJUDICATED: mostly judge bias, not policy

First read: confirmed blunders 0.50 → 0.92/game, almost entirely
pass/end-turn holding castable Dandâns. Joseph challenged the
interpretation ("0.7 vs 1.3 is very strong — passivity looks intentional
and learned"), and the outcome-calibration test he prompted settles it:

- "Confirmed" means confirmed BY THE CRITIC's one-resolution counterfactual
  (replay alternatives on the copied state, same judge) — not ground truth.
- At the flagged pass/end-turn states, the critic's post-drop read is
  P(win)=0.477 but the ACTUAL win rate from those states is **0.619**
  (+14.2pp above the judge); at 135k the overshoot was +8.2pp. The
  win-rate penalty of a flagged game SHRANK (−15.7pp → −8.9pp) while flags
  doubled.
- Verdict: the sharper hands-view critic over-penalizes holding patterns
  (the known visible-hands cast-step bias — it prices "castable Dandân ⇒
  should cast" into V), and it got MORE biased there, manufacturing most of
  the doubling. The passes are largely fine → intentional, learned holding.
  Reclassified from "policy hole" to CRITIC calibration issue at hold/cast
  seams (same family as cast-seam |dV| 0.039→0.050).
- Residual smaller flags: "never reached v≥0.5" losses 2.8% → 8.0%; still
  unlearned: Mystical Tutor value-negative (−0.042), predict-name top
  deckout endgame error (0.188, 38% err).

## Decision (Joseph, after AskUserQuestion)

`--scenario-boost → 2`, `--scenario-selfplay` STAYS 0.6. Scenario share
~43% → ~30% of games (full end-to-end games ~57% → ~70%); vs-1.3 exposure
drops ~17% → ~12%. The recommended selfplay 0.4 pairing was declined (and
subsequently vindicated as unnecessary — see the adjudication above).

## Deploy gotcha: launch.sh silently overrode train.args

The first restart with `--scenario-boost 2` in train.args changed NOTHING
(scenario share stayed 43.8%): the box-local `launch.sh` (NOT in the repo)
appends its own `--scenario-boost 3` AFTER `$(cat deploy/train.args)`, and
argparse takes the last occurrence. Consequences:
- train.args' `--scenario-boost 5` was NEVER live — the run's entire
  history has been boost 3. (The 43% scenario share is the boost-3
  equilibrium; the mix math in this journal is stated against that.)
- launch.sh also pins `--kl-teacher-iters 10000` (train.args says 30000)
  and `--minibatch 1024` after train.args — audit launch.sh's trailing
  block whenever a train.args knob "doesn't take".
Fixed by deleting the token from launch.sh (sed on the box; only
scenario-boost removed, expandable_segments intact) + restart. Verified:
live cmdline now carries a single `scenario-boost 2`.

## Second cut same day: boost 0.5 (Joseph: "further shrinkage, 0.5 or 0.33")

Deployed 77ddca5, box resumed it≈164,3xx. First window (it=164,425):
scenario share **17.7%** (6,884/38,880; was 43.8% at boost 3, ~38% at
boost 2) and past-self pool games more than DOUBLED (past n 5,968→13,766/w)
— the full-game mix Joseph asked for. Share came in above the ~11-12%
linear-C projection (PFSP priorities re-equilibrate as scenarios get
sampled less; may settle lower). If still too high, boost 0.33 is one
train.args edit. Watch per the mine checklist: h1.3 slope, critic
hold/cast overshoot, deckout losses — vs-1.3 exposure now ~7% of games.

## Next-mine checklist (~175-180k)

- CRITIC hold/cast bias: rerun the outcome-calibration test on flagged
  pass/end-turn states (baseline: judge 0.477 vs actual 0.619, +14.2pp
  overshoot). If it keeps widening, the fix is critic-side (the
  visible-hands cast-step bias / critic_consistency PAY_KINDS shelf tool),
  not curriculum.
- h1.3 slope at the boost-2 mix (trend-fit, not window means).
- Cast-seam |dV| (0.050 baseline).
- Deckout losses vs 1.3 (69/2000 baseline), Day's blowups (12.7%),
  never-ahead losses (8.0%), fizzle-line occurrence (still ~0, watch only).
