# 2026-08-26 — obs_split + two scenarios deployed (box it=123,353)

Everything from the plateau-mine proposals shipped in one restart (bfba149):

## The split-context block (obs_split)

The mine's root cause held up all the way down: during a FoF resolution the five
revealed cards live only in `pending.context` and the arrangement lived only in
the env-side CompoundBuilder — **both the actor and the critic were blind to the
split being made**. The fix:

- `CompoundBuilder` now mirrors its live arrangement into the pending context
  after every pick (the engine's own `update_fof_split` + an env-owned
  `assigned` key) — so the game state itself carries the forming piles, and no
  builder-passing plumbing exists anywhere.
- `features.split_context_block(g, viewer)`: SPLIT_DIM=64 = 4 who-acts flags +
  per-name counts of pile1 / pile2 / unassigned. Appended after the count block
  on the actor's belief tail (inside `bookkeeper_counts`, viewer-oriented) and
  the hands critic's features (inside `encode_hands`, p1-oriented) — every
  assembly path (collectors, eval, panel, mine tools) picks it up for free.
- The collectors' compound-substep encode dedupe re-encodes while
  `split_live(g)` (the one pending whose context mutates within a decision_id).
- `widen_split.py` (clone of widen_counts): in-place zero-column widening of
  actor + critic first Linear + aux head + Adam moments + 8 league selves.
  The block is all-zero outside FoF resolutions, so the seam is
  function-identical everywhere else.

Verified before deploy: full suite green (326 tests); collection-path probe on a
widened copy of the real it=122.7k checkpoint showed **all 7 split sequences
with distinct per-toggle rows for both nets** and V exactly constant at the seam
(zero-init columns, as designed); 2 real PPO updates on CPU with sane losses;
parallel-worker + pipeline smoke clean.

Deployed: STOP → `widen_split` on the box checkpoint (it=123,353, +64 inputs,
backup `latest.pt.pre-split.pt`) → relaunch. First status healthy: kl 0.0135
(seam invisible to PPO, as intended), 529 it/h and climbing, calib normal.
No KL re-arm — nothing about the function changed at the seam.

## Scenarios + weights

- `hold_the_answer` (0.5): text-change instant in hand at own main, empty stack,
  no opposing fish — the sorcery-speed telegraphing class (err +0.05..+0.08 vs
  1.3, ~0 in mirror: self-play-blind). First window: 963 games, wr 0.79.
- `deckout_stack` (0.5): lib 4-14 with a draw spell in hand, v1.3's
  aggro-deckout punish live. First window: 863 games, wr 0.67.
- `fof_pick` 0.3 → 0.6 (the mirror-split punishment route): share doubled
  (1,102 games/window), EMA 0.65.
- Housekeeping in the same push: `deploy/train.args` now carries
  `--obs-counts --obs-split` (fresh-start faithfulness; resume-inert) and the
  earlier `--critic-view hands` fix; duplicate `_raise_fd_limit` removed.

## Success checks (next mines)

1. **Per-toggle V movement**: re-run the toggle probe in ~10k its — mean |dV|
   between toggles should leave 0.0000 as the critic learns the block.
2. **0-5 split rate**: the number that has refused to move (17% vs 1.3 / 44%
   mirror). If per-toggle credit works, this finally bends.
3. hold_the_answer: main-phase VC/Spray/Bend err (+0.083/+0.054/+0.053 at
   122.7k) via seq_probe.py on the next mine.
4. deckout_stack: deckout share of losses (15% vs-1.3 / 18.7% mirror).

## Lessons

- The engine already had `update_fof_split` (built for the UI) — mirroring
  env-side builder state through an existing engine API turned a ten-file
  plumbing job into a three-line builder change. Look for the existing
  in-progress API before threading new state through call sites.
- `--iters` is a TOTAL target, not an increment: a resume smoke with
  `--iters 2` on a 122k-iteration checkpoint runs zero updates and still prints
  the full final readout — looks like a passed smoke, tests nothing.
