# 2026-08-26 — Observability audit: where the actor/critic decide blind

Follow-up to the FoF-split discovery: a systematic sweep of every pending type
(20 in the engine) + compound builder against what the perspective encoding
(actor) and hands encoding (critic) actually carry. Method: code trace of each
pending's context/mask construction, plus ALIASING PROBES — construct two states
that differ in decision-relevant context, assert the encodings are identical
(identical obs + different truth = provably blind). Decision frequencies from
the 2,000-game it=122.7k mine.

## Confirmed blind (aliasing-proved)

1. **Stack-spell targets — actor, critic, AND mask all aliased.** An opponent
   Mind Bend/Crystal Spray on the stack targeting MY FISH vs MY LAND produces
   bit-identical actor obs, hands features, and action mask (probed across
   seeds). Card rows carry no targeted flag; stack rows encode the source card
   only; targets live in `StackObject.targets`, read by nothing. 28.9
   opponent-spell-on-stack response rows/game. Mitigation vs h1.3: it targets
   fish 99.3% (5,728/5,771 targeted casts), so the prior is learnable — but 93%
   of training games are league/self games where the copy's targeting is
   erratic, i.e. the aliasing injects noise exactly where the training data
   comes from. Also: the agent cannot see WHICH of its own on-stack spells
   targets what (memoryless policy = own-target amnesia), nor which fish among
   several is targeted.
2. **search_library (Mystical Tutor) fetch is blind.** The engine reveals the
   whole library to the searcher, but the perspective encoding shows only the
   top-8 library rows; `eligible` is in library order and the PICK_SINGLE
   index→card mapping is encoded nowhere. Probe: swapping two deep library
   slots changes what index 4 fetches (Tutor vs Lapse) with identical obs.
   0.59 decisions/game — and a clean explanation for the long-standing
   "Mystical Tutor value-negative" and "tutor exemplar agreement 0.00" results.
3. **Blocker focus (sel_attacker) is invisible.** The two-step blocker
   assignment (PICK_B select attacker → PICK_A assign blocker) keeps the
   selection env-side; obs and builder-progress are identical between focus 0
   and focus 1 (only the mask's forward-only start shifts, and logits cannot
   condition on the mask). 2.07 declare_blockers rows/game. Blunted by the
   all-4/1-Dandân pool (attackers interchangeable).
4. *(Fixed this morning: fof_split/fof_choose pile blindness — obs_split.)*

## Blind by construction (identities visible, arrangement not)

5. **Compound arrangement state for scry / reorder / putback / discard /
   bottom.** The revealed cards ARE visible (they stay in the library and are
   marked known → top-8 rows), but the partial arrangement lives only in the
   builder; the sole signal is one progress scalar. A Ponder reorder is a
   3-card permutation where only the FIRST pick is informed — later picks
   can't see what was already placed. reorder 5.66/g + putback 2.91/g +
   scry 0.55/g + discard/bottom 0.35/g ≈ 9.5 decisions/game, the same
   mechanism class as the fof_split hole (smaller per-decision stakes).

## Unencoded index mappings (deterministic order, statistically learnable)

6. **choose_targets** (7.17/g): PICK_SINGLE indexes the caster-first legal list;
   the mapping is deterministic but never encoded — the net must re-derive
   "kth legal candidate" from zone rows. Homogeneous pool blunts it; plausibly
   feeds the removal-at-lands habit (which land gets sprayed is ordinal, not
   seen).
7. **choose_graveyard on overflow**: eligible is engine-order, but at gy>32 the
   ENCODED rows are value-first reordered (2026-08-22 change) — index↔row
   correspondence breaks exactly in the deckout window (13% of decisions).
8. Fine: name_card (canonical sorted names — stable), put_from_hand (hand
   order), order_triggers (trivial), pay/mana (globals carry the sub-state).

## Critic-only gaps (hands view)

9. The hands critic has NO step one-hot, NO combat block, NO pending one-hot,
   NO pay state, NO builder progress (the actor has all five). Mid-combat
   states with attackers declared are aliased against pre-combat states for
   the value function — a structural contributor to the V-jump/cast-step
   flicker class (the thing critic_consistency tried to patch at the loss
   level).

## Proposed fixes (not yet approved), ranked by value/effort

- **A. stack_ctx block** (widenable tail, actor + critic): for the top 1-2
  stack objects, target name one-hot + target-controller-is-viewer +
  target-is-fish/land flags (~2×24 dims). Attacks #1, the widest hole.
- **B. critic context mini-block** (~40 dims on the hands tail): step one-hot +
  combat(4) + pending one-hot + pay(4) + builder progress. Attacks #9.
- **C. builder-arrangement mirror v2**: extend the obs_split ctx-mirror pattern
  to scry/reorder/putback (placed-so-far counts + last-placed). Attacks #5.
- **D. search_library**: needs a design call — either canonically name-sorted
  eligible (an action-SEMANTICS change: mask+apply remap, disruptive to a
  trained policy mid-run) or an eligible-name-counts tail block (informs WHAT
  is fetchable, leaves ordinal fuzz). Attacks #2.
- **E. gy-overflow eligible remap** (env-side, tiny): make choose_graveyard's
  mask order match `graveyard_order`'s encoded order. Attacks #7.
- **F. blocker-focus scalar** (tiny). Attacks #3.

All of A-C/F are zero-column widenable in place (the obs_split/widen_counts
pattern); D's sort variant is the only action-space semantics change.

## Lesson

The FoF hole generalizes: ANY state that lives in `pending.context`, an
env-side builder, or an object field the 44-float card row doesn't carry is
invisible — and the mask being correct hides it (the policy acts legally, so
nothing crashes; it just can't condition on what it's deciding about). Aliasing
probes (two truths, one encoding) are cheap and decisive; run them whenever a
decision type underperforms despite training pressure.
