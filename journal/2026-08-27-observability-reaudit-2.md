# 2026-08-27 — Observability re-audit #2 (post obs_tgt): 19/21, residuals negligible

Full aliasing-probe sweep re-run under the complete deployed flag set
(obs_counts + obs_split + obs_ctx + obs_tgt, hands critic). Same method as the
two prior audits: paired states differing only in decision-relevant truth;
identical encodings = provably blind.

## Results

- **All 14 prior resolutions HOLD** (stack targets, search sorted mapping +
  counts + privacy, blocker focus, scry arrangement + privacy, critic
  attackers/step, graveyard mapping, fof_split piles — actor and critic).
- **All three former residuals now RESOLVE** (the obs_tgt pack):
  - R8: WHICH of two same-name fish the on-stack spell targets, at the
    responder's choose_targets (the Spray-fizzle line) — actor and critic.
  - R9: 3rd- and 4th-from-top stack-object targets (depth extension).
  - R10: candidate STATUS at a fixed index (tapped-vs-untapped separates; the
    first run of this probe reported blind because the scenario land was
    ALREADY tapped — probe bug, not encoding; explicit both-ways toggle
    resolves and the flag bits verify) and eligible-list order encoded.

## Remaining blind (both accepted, both micro)

1. **5th-from-top stack target** — sight stops at 4. Depth >=5 holds 0.03% of
   real choices (79+135 across 966k mined decisions); transient as the stack
   unwinds.
2. **WHICH same-name fish the 3rd-stack-object targets during choose_targets**
   — the per-candidate targeted-by flags reference the top TWO stack objects;
   the deep slots show WHAT objects 3-4 target (name/class/controller) but not
   which same-name instance. The intersection (choose_targets AND stack >=3) is
   0.11% of decisions, and the same-name-ambiguity-with-status-difference
   subset is smaller still.

No new blind spots. Every decision type in the engine now has its
decision-relevant context encoded for the acting net, except the two
intersections above.

## Box confirmation

it=135,492: all four obs flags True; tgt columns strongly alive — actor
net.0 tgt-column mean |w| = 0.400 (~8.4k its after the widen; the ctx columns
were at 0.285 after 2.7k), critic 0.143. choose_targets' 7.2/g frequency is
feeding the new columns fast.

Note: the box is now inside the ~135-140k window flagged for the next weakness
mine — the eight success checks (six from obs_ctx + removal-targeting split +
fizzle-line appearance) are ready to read out.

Probe script: scratchpad audit2.py (extended; 21 probes).
