# 2026-08-26 — obs_tgt: choose_targets candidate features + stack sight to 4

Joseph's call on the re-audit residuals: the ordinal choose_targets mapping is
"probably learnable as 'opponent controls fish -> target removal at bottom of
list', but it loses a lot of nuance" — so expose the FEATURES of what each
index resolves to, rather than a bare pointer. Plus the two other residuals:
same-name-fish targeting (the Spray-fizzle line: kill your own targeted fish ->
CR 608.2b counters the Spray -> the draw rider dies with it; engine.py:1064)
and stack sight past the top two (depth benchmark).

## The pack (TGT_DIM = 2×25 + 36×12 = 482, one flag `obs_tgt`, needs obs_ctx)

- **[2×25] stack objects 3-4 from the top**: same `_target_slot` layout as the
  ctx pack's top-two. Sizing from the depth benchmark: real-choice decisions at
  depth >=3 were 1.06% (vs 1.3) / 1.48% (mirror); top-4 sight leaves 0.03%.
- **[36×12] per choose_targets legal-list index**, the card that PICK_SINGLE k
  resolves to: valid + controller-is-viewer + is-bf-creature + is-bf-land +
  is-on-stack + tapped + attacking + blocking + **targeted-by-top-stack** +
  targeted-by-2nd-stack + text-altered + name-idx/21. TGT_SLOTS=36 covers every
  observed list (max 35 across 3,600 mined games' 36,302 choose_targets
  decisions; 37% are forced single-candidate picks). The legal lists are pure
  instance ids (no player targets in this pool); Spray's list prepends stack
  spells — the on-stack flag covers Lapse/Spray spell targets.
- **No action-semantics change** (unlike the search remap): indices keep their
  engine meaning; the net just gets to SEE them. Name one-hots were dropped for
  a name scalar + class flags to keep 36 slots affordable — the nuance lives in
  the status flags, and same-name same-status candidates are genuinely
  interchangeable.
- The targeted-by flags are what make the fizzle line playable: during MY
  choose_targets (my answer spell — targets are chosen before paying/casting,
  so my spell is not on the stack yet), the opponent's Spray IS the top of
  stack, and the index holding its target lights up.
- Gating: candidate section fills for the choosing seat and the critic
  (viewer=None, p1-oriented). Public info, but only decision-relevant to the
  pending's holder. Deep-stack slots always fill (public). All zeros outside
  a choose_targets pending / stack depth >=3 → zero-column widening is
  function-identical at the seam.
- No collector dedupe extension needed (choose_targets is atomic — one action
  per pending, unlike the mirrored builders that mutate within a decision id).

## Wiring (the obs_ctx pattern, mechanical)

features.py (`set_tgt_block/tgt_block/tgt_block_on`, dims, bookkeeper tail,
encode_hands tail — appended at the ABSOLUTE tail after critic_ctx_extra so the
widen stays a zero-column append), config `obs_tgt`, `--obs-tgt` FRESH-only,
build_models (assert rides obs_ctx; extra_a/extra_c + TGT_DIM),
config_from_checkpoint + _payload, pcollect/inference lite, transplant arch
keys, `widen_tgt.py` (K=482 both nets, backup .pre-tgt.pt).

## Verification

- test_obs_tgt.py (7 tests): the fizzle aliasing probe RESOLVED (same-name fish
  0-vs-1 distinguishable via the targeted-by flag, riding into both nets),
  candidate feature correctness (on-stack/tapped/mine/creature/land + name
  scalars), depth-3 AND depth-4 targets visible, chooser/other-seat gating +
  critic orientation, dims/config, build_models widths + requires-ctx,
  widen_tgt function-identity + optimizer/league round-trip.
- Full suite 337 passed, 7 skipped.
- Rehearsal on the real-checkpoint copy (it=122716, split+ctx): widen_tgt
  +482/+482, 8 league selves, seam OK; 2-update resume smoke clean.
- Box: STOP (trainer checkpointed within seconds) → widen_tgt at **it=127,116**
  (+482/+482, 8 league selves, seam OK, backup latest.pt.pre-tgt.pt) →
  relaunch. First post-widen window at it=127,245: kl=0.0134 (baseline),
  critic acc .69 / brier .20, aux 0.612, 513.7 it/h, no errors.

**Deploy-protocol lesson:** never chain the wait-for-exit loop and the widen in
ONE ssh command line. The loop's `pgrep -f 'python -m [f]ishrl.train'` matched
its own shell because the same command line contained the literal
`python -m fishrl.train.widen_tgt` from the chained widen — the `[f]` bracket
trick protects the pattern from itself, not from a sibling command. The trainer
had stopped in seconds; the "slow stop" was the loop spinning on itself.
Ship/stop, wait, and widen/relaunch stay separate calls.

## Success checks (append to the next mine's list)

7. choose_targets precision: land-vs-fish removal targeting split (the
   removal-at-lands habit was ~25% of removal in the field guide era), and
   targeting conditioned on tapped/attacking/text-altered status.
8. The fizzle line: response casts that target OWN fish while an opponent
   Spray/Bend targets it (currently ~never; any nonzero rate with the draw
   denied is the behavior appearing).
