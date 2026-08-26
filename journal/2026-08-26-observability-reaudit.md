# 2026-08-26 — Observability re-audit: fixes confirmed landed, remaining blinds quantified

Re-ran the aliasing-probe audit (journal 2026-08-26-observability-audit.md) with
the deployed flag set (obs_counts + obs_split + obs_ctx, hands critic). Method
unchanged: paired states differing only in decision-relevant truth; identical
encodings = blind.

## Resolution probes: 14/14 RESOLVED

Every confirmed-blind case from the original audit now separates, on BOTH nets:

1. Stack target fish-vs-land (actor + critic) — the widest hole, closed.
2. search_library: PICK index→NAME invariant under deep library swaps
   (name-sorted pick_list), eligible name-counts visible to the searcher,
   zeros for the opponent.
3. Blocker focus 0 vs 1 (actor + critic).
4. Scry/reorder arrangement: first placement visible to the scryer, private
   from the opponent.
5. Critic: attackers-declared vs not, step visible (the hands-view aliasing).
6. choose_graveyard: name-sorted stable mapping.
7. fof_split: pile-1 vs pile-2 assignment (actor + critic).

## Landed on the box, and the columns are alive

`checkpoints-v3/latest.pt` at it=126,512: obs_counts/obs_split/obs_ctx all True,
critic_view=hands. The zero-initialised context columns have moved off zero in
~2,700 iterations since the widen: actor net.0 ctx-column mean |w| = 0.285
(pre-existing columns 3.19), critic 0.292 (4.86). Gradients are flowing into
the new inputs — early movement, not yet proof of behavioural use (that is the
next mine's six success checks).

## Remaining blind spots (all documented leftovers, now with frequencies)

- **S1 — WHICH of two same-name own fish is targeted** (aliasing-confirmed
  blind). The target slot carries name one-hot + class flags, and card rows
  still have no targeted flag, so two Dandâns are indistinguishable as targets.
  Materiality: fish are identical 4/1s — it only matters when they differ in
  status (attacking/blocking/tapped). Upper bound: stack≥1 with own fish≥2 is
  2.1% of decisions vs 1.3 / 2.8% mirror; the meaningful subset is smaller.
- **S2 — stack depth ≥3: the buried object's target is invisible** (only the
  top two are encoded). 1.3% of decisions vs 1.3 / 1.8% mirror sit at depth ≥3
  (62% / 80% of games touch it at least once). The buried target becomes
  visible again as the stack unwinds to ≤2, so the exposure is transient.
- **S4 — choose_targets ordinal mapping** unchanged by design: PICK index into
  the caster-first legal list, deterministic and statistically learnable, never
  encoded (7.2/g). Deliberately not fixed in the ctx pack.
- **S3 — placement order within a pile** resolves via the last-placed one-hot
  when names differ; same-name permutations are semantically identical in this
  game, so no real gap.
- Code-trace note: `_target_slot` encodes `targets[0]` only — fine for this
  pool (every targeted spell is single-target), would need revisiting if a
  multi-target card ever entered the deck.

Verdict: no new blind spots found; the three residuals are small, known, and
none is in the mechanism class that froze the FoF splits (state an acting
player must condition on repeatedly with zero encoding). No further encoding
work proposed before the next mine reads out the behavioural checks.

Probe script: scratchpad audit2.py (session-local; reconstructable from
test_obs_split.py / test_obs_ctx.py plus the frequency counts above).

## Correction + depth benchmark (same day, after Joseph's review)

**S1 materiality was WRONG.** Same-name fish targeting matters: the fizzle
line — in response to opponent's Crystal Spray at my fish, kill THAT fish so
the Spray's targets are all illegal on resolution (CR 608.2b, engine.py:1064)
and the spell is countered, denying the cantrip draw. Executing it requires
knowing WHICH fish is targeted; currently aliased. Needs a fix, not a shrug.

**Stack-depth benchmark** (122.7k mines; "choice" = nlegal>=2):

| depth d | vs 1.3 choices | mirror choices |
|---|---|---|
| 0 | 229,988 | 373,313 |
| 1 | 55,840 | 85,233 |
| 2 | 10,573 | 24,735 |
| 3 | 2,779 | 6,067 |
| 4 | 327 | 1,044 |
| 5 | 79 | 135 |
| 6-7 | 2 | 17 |

Choices at depth >=3 (buried targets under top-2 sight): 1.06% vs 1.3 / 1.48%
mirror. Extending target sight to top-4 leaves only depth >=5 blind = 0.027% /
0.031%. Per-game max depth: 19% of vs-1.3 games and 40% of mirror games reach
depth 4+; ~5-7% reach 5+. The base obs already carries 6 stack SOURCE rows +
a depth scalar — only the target channel truncates at 2. **Recommendation:
top-4 target slots.**

**choose_targets ordinal, restated:** not aliased (the caster-first eligible
order is fully determined by visible state — unlike pre-fix search_library
where hidden library order made it provably unlearnable), but the index→card
resolution is an implicit program the net must learn: "count legal candidates
in caster-first zone order". Candidates are mostly same-name (Dandâns,
Islands), so name-sorting would NOT help here, and it would be an
action-semantics change on a 7.2/g decision a trained policy relies on.
The encoding-only fix is a pointer block: for eligible index i, the
(side, battlefield-slot) it resolves to — lookup instead of derivation, no
semantics change.

**Proposed obs_tgt pack (not yet approved), all in-place widenable:**
- A. Target slots 2 -> 4 (+2×25 dims) — from the benchmark.
- B. Per-target-slot row pointer: side bit + bf-slot one-hot (34) — resolves
  same-name fish; makes the Spray-fizzle line visible (+4×35).
- C. choose_targets index->row pointer block, first 8 eligible indices ×
  (side + bf-slot one-hot) (+8×35 = 280).
~470 dims total on both nets, same widen pattern as counts/split/ctx.
