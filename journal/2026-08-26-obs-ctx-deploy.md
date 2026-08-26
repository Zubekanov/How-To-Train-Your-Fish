# 2026-08-26 — obs_ctx deployed: the audit's full fix pack (box it=123,821)

All six audit fixes (A-F, journal 2026-08-26-observability-audit.md) shipped as
one architecture flag + one in-place widening (c0589c9):

## What the nets now see (CTX_DIM=133 actor / +CRITIC_CTX_EXTRA=41 critic)

- **Stack-spell targets** (2×25): for the top-two stack objects, target name
  one-hot + is-player + controller-is-viewer + is-bf-creature + is-bf-land +
  is-on-stack. The audit's widest hole (Spray at my fish vs my land was
  bit-identical, incl. the mask; 28.9 response rows/game).
- **search_library eligibility** (20): per-name counts of what a search can
  fetch — searcher's eyes only (the library is hidden from the other seat);
  the critic sees it.
- **Builder arrangements** (61): the CompoundBuilder now mirrors scry (top/
  bottom), reorder/putback/bottom/discard (placed order) and declare_attackers
  into `pending.context`; pile counts + last-placed one-hot. A Ponder reorder
  previously had only its first pick informed.
- **Blocker focus** (2): which attacker blockers are being assigned to.
- **Critic extra** (41): step + pending one-hots, p1-oriented combat, pay —
  context the actor always had in its globals but the hands view lacked
  (mid-combat states were aliased for V).

Plus the **name-sorted PICK remap** (`masking.pick_list`, same flag): the
search_library / choose_graveyard PICK_SINGLE order becomes name-sorted, giving
indices stable semantics. An action-semantics change with zero cost — the
replaced picks were provably uninformed (the "Tutor value-negative" mechanism).
The collectors' encode dedupe re-encodes while any mirrored-builder pending is
live (`ctx_live`, extending `split_live`).

## Verification chain

Full suite green (331 tests, incl. new test_obs_ctx.py where each audit
aliasing probe must now RESOLVE: fish-vs-land target distinguishable, mask↔apply
share one ordering, scry arrangement visible to the scryer and hidden from the
opponent, blocker focus distinguishable). widen_ctx seam-checked on the real
checkpoint (+133 actor / +174 critic zero-init columns, 8 league selves).
Collection-path probe on the widened checkpoint: 1,714 buffered steps → 151
carried stack-target context, 72 carried builder arrangements (all-distinct
across consecutive picks), critic extras on every row. Two real PPO updates
clean (kl 0.0039). Box: STOP → widen_ctx at it=123,821 (backup
latest.pt.pre-ctx.pt; the widen survived an ssh disconnect mid-verify —
checkpoint writes are atomic, no harm) → relaunched, resume clean.

## In-place vs restart (Joseph's question — the decision rule)

Ride the widened lineage first. The critic side will learn in place
(supervised; the hands-critic swap precedent), and the two key actor behaviors
have live exploration (2-3 splits sampled half the time; Tutor picks scattered)
— the condition under which policy gradient can attach new inputs. The warning
precedent is the deckout clock: added in-place to the settled flat actor it
stayed provably ignored; only the v2 warm restart (trained with it from it=0,
higher entropy) used it. So: mine again at ~+10-15k its. If CRITIC metrics move
(per-toggle |dV| leaves 0.0000) but ACTOR behaviors stay frozen (0-5 splits
17%/44%, Tutor fetches uncorrelated), that is the clock pattern → escalate:
first `--rearm-kl` + the never-yet-used entropy re-heat on this lineage, and if
still frozen, a warm v4 restart (bootstrap from the current actor, all blocks
on from iteration 0, freeze/KL handoff; ~2-3 days of budget to re-climb per the
v3 experience). Deferring the restart costs nothing — the run keeps improving.

## Warm-up scenario batch (same day, 3a573f5, live from it=123,912)

Three scenarios where the newly visible channels are the DECISIVE information,
concentrating reps on the zero-initialised context columns (Joseph's ask):

- `read_the_target` (0.6): p2 text-changer on the stack with the TARGET
  randomised fish 55% / land 45% (new envelope `stack_target` override; p1
  always has a fish so both classes are live), instant + mana up — whether to
  spend the answer depends exactly on the stack-target channel.
- `tutor_fetch` (0.4): Mystical Tutor + mana, timing free — the fetch choice
  only became learnable with the name-sorted PICK remap + eligible counts.
- `arrange_the_top` (0.4): Ponder/Brainstorm/Predict in the mid-late game —
  reorder/putback reps under the builder-arrangement mirror.

First window (it=124,050): read_the_target=828 games (wr .74), tutor_fetch=1,123
(.56), arrange_the_top=814 (.46); all 18 scenario members drawing; trainer
healthy (kl 0.0136, 548 it/h). The read_the_target starting wr will be worth
watching per TARGET class once mined — the .74 aggregate can hide answering
land-hits it should ignore.

## Success checks for the next mine (superset of the obs_split list)

1. Per-toggle |dV| across split toggles (must leave 0.0000) — now also across
   scry/reorder picks.
2. 0-5 split rate below 17% vs-1.3 / 44% mirror.
3. Tutor: fetch-name distribution vs game state (should stop being uniform);
   "cast Mystical Tutor" delta sign.
4. Main-phase text-changer err (hold_the_answer + stack-target visibility
   should both push it down).
5. Response-window behavior conditioned on target (respond more when the
   target is a fish) — newly measurable, newly learnable.
6. critic_jump on combat/cast seams (the critic extra should shrink it).
