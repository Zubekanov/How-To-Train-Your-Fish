# 2026-08-27 — Action-space audit: where a human/bot can act and the model cannot

Complement to the observability audits: instead of "what can the nets SEE",
this asks "what can they DO". Method: (a) empirical scan — at sampled live
mirror states, try every MASKED-OUT action id whose block apply.py routes for
the pending type against a deep-copied engine, record what the engine would
have accepted (divergence.py, 24 games / 2,940 scanned decisions after
excluding two classified-in-smoke candidates); (b) code-trace of engine
capabilities with NO apply route at all (unreachable regardless of mask);
(c) engine-API checks for the hypotheses the scan cannot reach.

## Mask contract: clean

Zero violations — every masked-in action across all pending types was accepted
by the engine. And the never-accepted-when-tried classes (ALLOC_MANA, pay
ACTIVATE, out-of-range PICKs) confirm the mask is exactly as tight as the
engine on those.

## Deliberate withholdings, now quantified (all working as designed)

- **Unaffordable/untargetable casts** (`PLAY_HAND` 1,462 accepts / 18,728
  tries): engine `play()` checks timing but not affordability; the mask's
  colour-aware gate stops cast-then-strand. A human "does" this only by
  casting and cancelling — the stall the design removed. Same class:
  CYCLE_HAND (43), non-mana ACTIVATE (21), strandable pay TAP_LAND (22).
- **Undo affordances**: CANCEL_PAY accepted at 766/766 pay rows,
  TARGET_CANCEL at 133/133 choose_targets rows — the documented anti-stall
  removals. Outcome-equivalent play is always available (casts only offered
  when completable).
- **Guided text-change**: 719 accepted withheld pairs across 43 decisions
  (~17/decision, engine even accepts the frm==to no-op diagonal). The one
  semantically-lost niche: the PROTECTIVE rewrite — changing your own
  permanent's basic-type word to a type you robustly CONTROL (e.g. immunising
  a Dandân's "Islands" clause against later island-conversion). Guided's
  EFFECT is always the severing direction. Low value in this pool; the
  self-kill/fizzle direction IS expressible (to = absent-from-own).
- **Artifacts, no capability content**: PLAY_HAND_ALT on a non-modal card is
  a duplicate PLAY_HAND (mode=None); END_TURN where masked out is accepted by
  the engine everywhere (260/260 in the smoke) but is outcome-equivalent to a
  PASS chain — an auto-yield convenience, not a capability. (Trying it on
  copies fast-forwards a full turn per try — it stalled the first 60-game
  scan; excluded and classified by code-trace.)

## REAL capability gaps (engine-legal, human/bot-available, NO model route)

1. **Tap-to-float at priority.** `E.tap` outside a payment floats the mana
   ("otherwise it floats in the mana pool" — engine-verified: accepted at
   priority, pool {'U': 1}). Neither the priority mask nor apply.py routes
   TAP_LAND at priority — the action_space.py layout comment ("float at
   priority / pay into cost") planned it, the wiring never existed.
   Inconsistently, ability mana (Svyelunite Temple) IS offered at priority,
   so the agent can float from the Temple but not from a plain Island.
   THE LINE IT BLOCKS: respond to land-neutralisation by floating first —
   above all opponent **Vision Charm [land phase-out] on the stack with my
   untapped lands: 1.25 rows/game vs 1.3** (0.66% of decisions; 0.84%
   mirror), plus opp spells targeting my lands (0.02% vs 1.3, 0.38% mirror).
   Pools empty at step end (CR 500.4), not at resolution — floating survives
   the Charm resolving and pays for an instant afterwards in the same step.
   Anti-stall safe: tapping is monotone (a land taps once per untap cycle),
   bounded by land count; the cost of a wasted float is real and learnable.
2. **Hold-priority after casting** (`play(..., hold=True)`, the human Ctrl
   affordance): the apply route always auto-passes after a cast, so the agent
   can never stack a second spell on its own first cast unless the opponent
   responds in between. Near-zero strategic value here: with public casts and
   no split-second effects, cast-order rearrangement reaches the same
   outcomes (cast B first, then A). Documented, NOT proposed for wiring.
3. **Concede** exists for humans and not the model — never optimal in a
   zero-sum ±1 game; listed for completeness.

## Proposal (not yet approved): wire TAP_LAND at priority

Mask: offer TAP_LAND i at your own priority for untapped addressable lands
(mirrors engine legality exactly — no context gating, per the no-guided-masks
rule). Apply: route priority|TAP_LAND -> E.tap (floats). No new ids, no
observation change needed (each seat's floating pool is already in the
perspective globals and the critic's per-player scalars). Trade-off to weigh:
~+N_lands legal actions on the most common decision type — exploration noise
on a settled policy vs. unlocking the anti-land-wipe line. If approved, pair
with a float_the_response scenario (opponent Charm [land] on the stack,
response instant in hand, float-then-cast wins the exchange).

Scanner: scratchpad mine/divergence.py (reusable; re-run after any
action-space or mask change).
