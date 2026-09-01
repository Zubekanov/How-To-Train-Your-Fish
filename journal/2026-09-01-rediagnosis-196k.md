# 2026-09-01 — rediagnosis at 196k: the levers worked, the plateau changed nature

Trigger: Joseph — the 184k changes "only gave a short-lived winrate increase
and the trainer is back to flat. Can you rediagnose and propose further
improvements".

## What the 184k levers actually did

- h1.3 pooled: 0.723 (pre-restart) → **0.757** — a real **+3.4pp step** over
  ~4k its, then flat again: weighted slope over the 14 post-step blocks
  (188k-196k) = **−0.03 ± 0.11 pp/1000it**. Joseph's read is exact.
- past_wr told the same story in miniature: 0.500 → 0.515-0.527 (beating the
  pre-step ring) → back to 0.502 as the ring refilled with post-step selves.
- Mechanically everything landed: kl 0.013→0.006 (creeping back to 0.0075),
  clip 0.067→0.041, jump 0.0205→0.0117 and still falling, gns_b UNSATURATED
  (finite 2-40M vs ~100k batch), consist loss 0.00082→0.00059, critic acc
  0.712→0.727, window Brier 0.183→0.174, it/h ~420. League spacing works:
  ring selves at 189.7k…195.4k (every ~800 its), and the agent beats the
  OLDEST members 0.62-0.69 while struggling with recent ones — self-play-
  space improvement continues; it no longer converts to h1.3 wins.

## 196k mine (2000 vs h1.3 seeds 1320000+, 1600 mirror 1420000+)

- wr 0.745 (== telemetry). Decision Brier **0.132** (0.154 → 0.132, best
  ever); mirror Brier 0.196→0.176 with near-perfect mid-range calibration.
- **Outcome-calibration: overshoot NARROWED +18.5pp → +13.6pp** (judge 0.500
  vs actual 0.636); flags 1.24→0.77/g. Consistency lam=10 is repairing the
  hold/cast judge bias, as designed.
- Cast-seam |dV| vs 1.3 0.061→**0.047** (mirror 0.032→0.024); telegraphing
  catastrophic 2.7%→**0.8%**; removal waste 1.1%; fizzle ~0; degenerate
  splits 0.2%/0.6%.
- **Per-decision error mass is essentially mined out**: worst action class
  mean delta is −0.002 (pass opp turn); Mystical Tutor −0.052→−0.023;
  never-ahead losses 7/2000 (0.35%); deckout losses 71/2000 (stable). The
  residual ~25% of losses are games without large flagged errors —
  increasingly parity/variance-shaped.

## The rediagnosis

The 152-184k plateau was an OPTIMIZATION pathology (noise-dominated churn) —
fixing it bought a step. The current flat is different: optimization is
healthy (low KL, falling jump, finite gns, critic at its best), decision-
level holes vs 1.3 are gone, self-play skill still inches forward, and the
policy is SHARPENING (H 0.408→0.365, sagging toward the 0.008 floor with 54k
its of anneal left). That is the self-play local-optimum profile: the run
exploits its current strategy ever more cleanly and searches nothing new.
Each intervention now yields a one-time step, not a slope.

## Proposals (presented to Joseph)

1. **Entropy re-heat cycles** — the purpose-built, never-used stall lever
   (config comment: "enable it when a run stalls"). Machinery verified
   implemented (cosine sawtooth after the anneal horizon). Deploy
   `--ent-anneal-iters 195000 --ent-reheat-period 10000 --ent-reheat-peak
   0.012`: coef jumps ~0.0084→~0.012 and decays to 0.008 every 10k its
   (~25h/cycle). NOTE the horizon gotcha: reheat only activates past
   ent_anneal_iters, so the 250k horizon must be pulled back to ~195k or the
   sawtooth would not start until 250k. The spaced league + PFSP hard-mode
   (deployed last restart) are the ratchet that locks in what each reheat
   finds. Expect a mild wr wobble at each peak.
2. **critic-consistency 10 → 25**: every notch of judge quality has paid
   (the +3.4pp step, overshoot −5pp, Brier −0.02); jump 0.0117 and overshoot
   +13.6pp say it is not exhausted. Resume-tunable.
3. (Reserve) Direct h1.3 pressure via a small `--anchor-weight NAME=W` knob
   (exposure is 1.7% and PFSP has decayed the mastered anchor). Risks
   overfitting to 1.3's quirks — hold unless 1+2 don't move h13 in ~15k its.
4. (Escalation) If steps-then-flat repeats: capacity. Actor is 2.68M params;
   a function-preserving in-place widen (net2net duplicate-and-halve, same
   swap-in-place discipline as the critic) is the structural lever.

Recommended now: 1+2 (two train.args edits, one restart). Read at ~210k,
mine at ~215k with passcalib.py (+13.6pp baseline).

## Deployed (Joseph: "Go for it")

3f84286; box restarted at it=196,735. Verified live: banner consist=25.0
(lr/league carried), cmdline carries --ent-anneal-iters 195000
--ent-reheat-period 10000 --ent-reheat-peak 0.012; eval panel relaunched
(--n-frozen 200). Reheat coef at restart ≈ 0.0118, decaying to 0.008 by
~205k, then sawtoothing every 10k its. Read the h1.3 trend across WHOLE
reheat cycles (a mild wobble at each peak is expected, not a regression).
