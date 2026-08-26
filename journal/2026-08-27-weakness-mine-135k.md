# 2026-08-27 — 135k mine: plateau broken (0.64 vs 1.3), all eight obs checks read out

Mine at box it=135,492 (the flagged 135-140k window): 2,000 traced games vs
h1.3 (seeds 900000-901999) + 1,600 mirror (1000000-1001599), critic-judged,
with the tracer extended to record stack-target class, choose_targets pick
resolution (class/controller/tapped/text-altered/targeted-by-top + candidate
composition), and search fetches.

## Headline

**wr vs 1.3 = 0.643** (0.647/0.640 across independent 1,000-game chunks) — up
from 0.56 at the 122.7k plateau on the identical mine setup. The obs packs +
warm-up scenarios broke the plateau. Mirror: seat 0.474/0.526, play-first
0.504, deckout endings 20.8%. Loss reasons vs 1.3: fish war (life) 586,
deckout 127.

## The eight success checks

1. **Per-toggle |dV| — LEFT ZERO** (was exactly 0.0000 across 1,587 sequences).
   One-card pile toggle mean |dV| 0.0039 (zero-frac 8%), 2-3-vs-5-0 0.0043
   (zero-frac 0), scry top-vs-bottom 0.0095. Small vs the ~0.14 commit swing,
   but the channel is read (toggle135.py).
2. **0-5 splits: FIXED.** Degenerate 0-N splits 0.3% BOTH distributions (were
   17% vs-1.3 / 44% mirror). 92-97% of splits are 2-3, the rest almost all 1-4.
3. **Tutor fetches: state-conditioned.** Vision Charm is the top fetch (29%/25%)
   — the critic ledger's best card; at lib<=12 the distribution flips to Memory
   Lapse / NONE / Ponder (deckout-aware). Cast delta -0.038/cast (mana+tempo
   included), self-harm err 0.028 — Tutor is no longer the value-negative
   outlier, just a mediocre card.
4. **Main-phase text-changer telegraphing: FIXED.** Mean err +0.0071 vs-1.3 /
   -0.0061 mirror, catastrophic 2.0%/0.7% (was +0.083 class with 22.5%
   catastrophic vs 1.3).
5. **Response conditioned on stack target — newly differentiated.** Respond
   rate when the opponent's spell targets MY fish: 0.626 vs-1.3 / 0.606
   mirror; MY land: 0.083 / 0.265. The agent answers fish-hits and ignores
   land-hits, exactly the read_the_target intent (the 0.74 aggregate wr was
   not hiding land-answering).
6. **Cast-seam |dV|: still elevated.** cast 0.0394/0.0289 vs baseline
   0.0221/0.0222 (combat is fine: 0.0168/0.0119 — the critic combat block
   worked). The remaining critic-side item; the boxed critic_consistency
   (PAY_KINDS-only pairs, lam=0) is the tool on the shelf if it matters.
7. **Removal targeting precision.** Removal-class picks: opp fish 78%, opp
   land 18%; "picked an opp LAND while an opp fish was available" 3.4% vs-1.3
   / 6.7% mirror (the field-guide era wasted ~25% of removal on lands).
   own-land picks 4% (mostly Bend/Spray at own non-Island lands — plausibly
   Sandbar->Island conversions; mean delta mildly negative, not a crisis).
8. **Fizzle line: appearing, not yet learned.** 5 occurrences of picking OWN
   stuff that the top stack object targets (2 vs-1.3, 3 mirror), including
   genuine Crystal-Spray-on-my-targeted-fish picks. Nonzero = the line is in
   the exploration set; a warm-up scenario would concentrate reps
   (proposed: deny_the_draw below).

## Blunder ledger (residual)

Omission regrets collapsed (pass with regret>=.15: 0.22/g vs-1.3, was 0.51/g
at 107k). Worst self-harm classes are all small now (Mind Bend->land 0.038,
Sandbar cycling 0.034, Tutor 0.028). Day's Undoing keeps its 19% p(d<-.15)
vs 1.3 — the known high-variance card, unchanged. Critic decision-Brier 0.165
vs-1.3 / 0.192 mirror.

## Verdict on the restart decision rule

NO escalation. This is the anti-precedent of the deckout clock: critic metrics
moved AND actor behaviors moved (splits, targeting, response conditioning,
fetch selection) within ~12k iterations of the widens. Ride the lineage;
--rearm-kl / entropy re-heat / v4 stay shelved.

## Proposal (not yet approved)

- `deny_the_draw` scenario: opponent Crystal Spray on the stack targeting one
  of my TWO fish, answer spell (Bend/Spray/Metamorphose... engine-legal
  killer) + mana in hand; the winning line kills the TARGETED fish so the
  Spray fizzles (CR 608.2b) and the draw is denied. Concentrates reps on the
  5-occurrence fizzle line. Weight ~0.3-0.4.

Artifacts: mine/{h13_135k_all,self_135k_all}.jsonl, checks135.py, toggle135.py
(scratchpad); tracer extensions in mine.py/mine_self.py.
