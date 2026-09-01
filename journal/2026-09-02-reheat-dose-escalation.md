# 2026-09-02 — reheat cycle 1 was sub-therapeutic; dose escalated to peak 0.016

Trigger: Joseph — "No winrate improvements as of now" after the first reheat
cycle (196.7k → ~204k).

## Cycle-1 readout

- The sawtooth ran exactly as designed and did exactly too little: coef
  0.0084→0.0117, but **H lifted only 0.366→0.388** at the 198k peak and was
  back to 0.362 by 203k. The growth-era reference is H≈0.43 — a +0.02-nat
  lift re-opens essentially nothing. The post-184k policy is sharp enough
  (and the LR halved) that a 0.012 peak cannot re-inflate it.
- The predicted wobble happened on schedule and healed: h13 0.757→~0.745
  through the high-entropy phase; past_wr sagged 0.493-0.499, back to 0.505.
  No damage, no discovery — the signature of an underpowered perturbation.
- lam=25 continues to pay quietly: jump 0.0117→0.0094, consist loss
  0.00059→0.00040, acc/brier at their best (0.728/0.173).

Conclusion: the exploration hypothesis was NOT tested by cycle 1; the dose
was too small to move the policy off its optimum.

## Deployed (a78142a; box restart at it=203,867)

`--ent-anneal-iters 204000 --ent-reheat-period 15000 --ent-reheat-peak
0.016`. Rationale: crude cycle-1 sensitivity (ΔH ≈ +0.007 per +0.001 coef)
puts the H≈0.42-0.43 target at a ~0.016 peak; the longer period gives the
halved-LR policy time to loosen AND re-sharpen; the horizon re-phase makes
the strong peak start the moment it crosses 204,000 (peaks at 204k / 219k /
234k) instead of waiting out the old phase. Note 0.016 exceeds any
coefficient this run has ever trained at (ent_start was 0.010) — that is
the point, and the spaced league + PFSP hard-mode are the safety net.

## Read + kill criterion

- Within a few k its of the 204k peak: H should climb to ≥0.42. If it does
  not, the next notch is peak 0.020 (the config default).
- Expect a LARGER h13 wobble than cycle 1 — read the cycle as peak→floor
  (204k→219k), judge the trough and the post-cycle level, never mid-cycle.
- **Kill criterion**: if the full 0.016 cycle ends with the h13 trough at or
  below the 0.757 baseline and no post-cycle step, the exploration route is
  disconfirmed at a real dose → move to the reserves: the
  `--anchor-weight` knob for direct h1.3 gradient first, the
  function-preserving in-place actor widen second.
