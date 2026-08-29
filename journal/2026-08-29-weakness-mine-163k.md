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

## The warning sign (not scenario-shaped)

- Confirmed blunders 0.50 → **0.92/game**, almost entirely the passivity
  family (pass/end-turn holding castable Dandâns; 670+591 cases, regret
  ~0.22, turn ~16-17). Partly critic-sharpness inflation (both the drop and
  the counterfactual are judged by a sharper, jumpier critic — cast-seam
  |dV| 0.039→0.050), but directionally consistent with mirror-heavy
  training: passivity is cheap vs yourself, punished by the script.
- "Never reached v≥0.5" losses 2.8% → 8.0%.
- Still unlearned: Mystical Tutor value-negative (−0.042, unchanged);
  predict-name remains the top deckout endgame error (0.188, 38% err).

## Decision (Joseph, after AskUserQuestion)

`--scenario-boost 5 → 2`, `--scenario-selfplay` STAYS 0.6. Scenario share
~43% → ~30% of games (full end-to-end games ~57% → ~70%); vs-1.3 exposure
drops ~17% → ~12%. The recommended selfplay 0.4 pairing (holds vs-1.3
exposure ≈ flat to keep punishing the passivity hole) was declined — the
anti-passivity question moves to the next mine's checklist.

## Next-mine checklist (~175-180k)

- Blunders/omissions per game: did the passivity family grow further at
  reduced script exposure? (0.92/g baseline, critic-sharpness caveat — also
  record the per-era critic Brier next to it.)
- h1.3 slope at the new mix (trend-fit, not window means).
- Cast-seam |dV| (0.050 and rising → critic_consistency PAY_KINDS/lam shelf
  tool is the designated fix).
- Deckout losses vs 1.3 (69/2000 baseline), Day's blowups (12.7%),
  fizzle-line occurrence (still ~0, watch only).
