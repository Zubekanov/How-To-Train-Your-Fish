# 2026-08-30 — plateau diagnosis: LR halved, league spaced, consistency re-enabled

Trigger: Joseph — "WR vs heuristics has stagnated and WR vs frozen has fallen
to slightly below 0.5, can you view telemetry, diagnose, and recommend
training changes to reintroduce growth?" Then "Go for all" on the three-part
recommendation.

## Diagnosis (from box stats.json, block-aggregated + trend-fit)

- h1.3 pooled train wr (5-10k games/block): 0.649 @130k → 0.71 @150k →
  **flat 0.71-0.72 from ~152k to 183.7k** (30k its, ~2 days). All four
  heuristic anchors flattened TOGETHER (h1.0 0.91, h1.1 0.84, h1.2 0.79,
  h1.3 0.72) → global stall, not an anchor ceiling.
- The plateau PREDATES the scenario-mix changes (boost cuts at 164k). The
  boost-0.5 cut produced a brief real frozen bump (0.52-0.59 pooled for ~5k
  its) then re-flattened — mix exonerated.
- frozen_agg 0.49-0.50 (n=1600), past_wr dead 0.500 (170k games/block) —
  while kl=0.013/it and clip 0.067 keep the policy moving. Motion without
  progress = churn, and two mechanisms explain it:
  1. **Noise-limited updates**: gns_b=nan in every window (the GNS estimator
     saturated: B_crit >> batch, g^2 estimate <= 0), at an LR of 3e-4 that
     had NEVER decayed in 300h (no CLI flag existed; the optimizer state
     re-pinned the launch LR on every resume).
  2. **Population too shallow**: past-self ring = 8 snapshots x every report
     (~109 its) ≈ 2h span; mirror(40%) + past(36%) = 76% of games vs ≤2h-old
     near-clones. Cycling satisfies that pressure; scripted anchors (~5%)
     can't hold the line.
- Critic flat for 50k its: acc 0.710, Brier 0.183, jump 0.0205, aux 0.617.
- NOT the problem: entropy (H 0.43→0.41, anneal on schedule).

## 184k mine (2000 vs h1.3 seeds 1120000+, 1600 mirror 1220000+)

- wr vs 1.3: 0.724 (== telemetry; flat vs 163k's 0.708). Mirror seat 0.52 /
  play 0.51 / deckout share 0.225.
- **Outcome-calibration test (passcalib.py, now a saved script): the hold/cast
  overshoot widened AGAIN — judge 0.447 vs actual 0.632 at flagged
  pass/end-turn states = +18.5pp** (135k +8.2, 163k +14.2); flags 1.15/g,
  flagged-game penalty shrank to −7.1pp. The pre-registered critic-side
  trigger fired.
- Corroborating: cast-seam |dV| vs 1.3 widened 0.050 → 0.061 (mirror 0.032);
  telegraphing catastrophic 1.4→2.7% (mean err still negative).
- Improved/held: never-ahead losses 8.0% → 0.4% (opening v also shifted up
  0.48→0.53, so partly critic recalibration), Day's Undoing mean +0.041,
  degenerate splits ≤0.2%, removal waste 1.4-1.6%, fizzle ~0. Still
  unlearned: Mystical Tutor value-negative (−0.052). Deckout losses 80/2000
  (69 baseline, noise range).
- Gotcha for the tooling: mine.py's profile arg must be `heuristic_1_3` —
  `1.3` silently falls back to v1.0 (the accidental run read 0.908, matching
  the eval panel's v1.0 anchor 0.911).

## Deployed (5bfa3aa + 5331796, box restart at it=184,058)

1. **--lr-ppo 1.5e-4** (new, resume-tunable; the resume path re-asserts
   cfg.lr_ppo onto the loaded optimizer param_groups — without that the
   launch-time LR is pinned for the life of the lineage).
2. **--league-every 8** (new): past-self snapshot every 8th report → ring
   spans ~16h instead of ~2h; PFSP hard-mode upweights any older self that
   beats the current policy (anti-cycling). Ring re-spaces gradually over
   ~64 reports.
3. **--critic-consistency 10** (re-enable of the shelved PAY_KINDS-only
   det_next tool; the lam=50 h1.3 stall was the old pairing that spanned
   spell resolution). launch.sh's trailing `--critic-consistency 0` token
   removed on the box so train.args governs — same class as the
   scenario-boost gotcha.

Verified live: banner `lr=0.00015`, `league=8x8`, `consist=10.0`; eval panel
relaunched with --n-frozen 200 (it had exited with the trainer — a pgrep race
first reported it still up; trust the log's "follow mode done" line).

## Success signals (check ~5-8k its, then the next mine ~195-200k)

- LR: kl and clip should drop by roughly half; critic_jump down; then
  frozen_agg drifting >0.5 and h1.3 blocks resuming a climb.
- League: past_wr should rise ABOVE 0.5 (older selves are now beatable) and
  become a genuine growth trend; watch the [league] line for spaced self@its.
- Consistency: critic_consist_loss nonzero and falling; next mine reruns
  passcalib.py — the overshoot (+18.5pp baseline) should stop widening.
- Rollbacks are one flag each (lr back to 3e-4, league-every 1,
  critic-consistency 0 — all resume-tunable).
