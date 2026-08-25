# 2026-08-26 — Weakness/blunder/mis-sequencing mine at it≈122.7k (plateau)

Panel h1.3 0.547-0.559; the mine's 2,000 fresh-seed games vs h1.3 came in at
**0.596** (bands .600/.592 — the panel's fixed band reads a few points low).
1,600 mirror self-play games alongside. Same tooling as the 08-25 mine
(mine.py / mine_self.py / analyze* / deckout_probe) plus a new `seq_probe.py`
for within-turn ordering classes. Seeds 700000-701999 / 800000-803999.

## What moved since it=107k

- **opening_race worked**: losses decided before v ever reached 0.5 collapsed
  from 16.9% → 3.6% of losses.
- **Omissions keep shrinking under general training**: confirmed blunders
  0.51 → 0.41/game vs h1.3 (0.85 → 0.68 mirror), still the #1 class
  (pass/end-turn holding castables, turns ~20-28).
- Decision Brier 0.177 → 0.172; loss profile shifted toward thrown wins:
  38% of losses now peak at v ≥ 0.8 (was 26%), "opp resolves a fish" up to
  24% of losses.

## What did NOT move (the plateau's composition)

1. **FoF 0-5 splits: 17.1% vs h1.3** (17.4% at 107k, 19.8% at 98.5k) and
   **44.8% in mirror** (44.0%) — frozen across three checkpoints and 330k
   fof_split scenario games (scenario EMA down to 0.19 = PFSP hammering it).
   Meanwhile the CRITIC's pricing sharpened hard: a 0-5 split now costs
   −0.171 (was −0.086). Value learned, policy stuck.
   **Root cause measured**: across the five PICK toggles of a split the critic's
   V is EXACTLY flat (mean |dV| 0.0000, n=1,587 sequences) and the whole
   ~0.14 swing lands at commit + the opponent's pick — the piles-in-progress
   live in the env-side CompoundBuilder, which the critic's game-state encoding
   cannot see. GAE therefore hands every toggle the same advantage; the
   gradient cannot localize which pick was wrong. Slow statistical learning is
   the expected outcome, and it matches.
2. **Deckout endgame** unchanged: 18.7% of mirror games; predict-name suicides
   err 0.24 / 40% catastrophic, endgame AK 18.8%, pass-drift; 13% of vs-1.3
   losses are deckouts.
3. **Mis-sequencing (new probe)**: text-change instants cast from OWN MAIN with
   an empty stack — Vision Charm err +0.083 (22.5% catastrophic, n=728),
   Crystal Spray +0.054, Mind Bend +0.053 — stable since 107k. Every other
   main-phase instant is fine (AK/Brainstorm/Metamorphose ≤ 0). Crucially, in
   MIRROR games the same casts show ~zero err: the agent's own copy doesn't
   punish sorcery-speed telegraphing, only v1.3 does — self-play supplies no
   pressure on this class (same mechanism as the split hole).
   Null results from the same probe: fish/attack ordering fine both ways,
   Brainstorm-put-back losses not detectable at scale, land-drop timing fine,
   FoF cast timing fine.
4. Mulligan games 0.459 vs 0.614 kept (unmineable by scenario; monitor).
5. Critic mirror calibration drifted overconfident in the high cells
   (pred 0.7 → actual 0.62, 0.8 → 0.75); Brier 0.199 mirror vs 0.172 vs-1.3.

## Proposals (pending Joseph)

- **hold_the_answer** (~0.5): p1 main, empty stack, text-changer(s) in hand,
  opponent fishless with fish likely imminent — the correct line is usually
  HOLD for the instant window; v1.3 punishes the telegraph. Attacks finding 3
  where self-play can't.
- **deckout_stack** (~0.5, carried from 08-25): lib ~4-14, draw spell seeded in
  hand, v1.3's aggro-deckout punish live — draw-race stack timing.
- **fof_pick 0.3 → 0.6** (carried): the agent picks well (EMA 0.63); a stronger
  picker makes league self-play punish the 44.8% mirror degenerate splits.
- **Architectural (the real split fix): expose the forming piles to the critic.**
  The revealed five and the current pile assignment are public information;
  encoding them (critic-view widening, in-place per the widen_counts pattern)
  would give per-toggle TD signal and let PPO localize split credit. Scenario
  weight alone has empirically not been enough.

## Lessons

- A scenario can be "working" (EMA driven to 0.19, critic pricing sharpened)
  while the policy stays frozen — check WHERE the value signal lands before
  concluding more reps will fix a compound decision.
- Self-play blind spots are a class: any error the copy doesn't punish
  (degenerate splits, sorcery-speed telegraphing) shows ~zero err in mirror
  mines and only surfaces vs the heuristic. Mine both pools, always.
- The panel's fixed seed band read ~4pp below 2,000 fresh seeds at this
  checkpoint — treat small fixed-band deltas near a plateau with suspicion.
