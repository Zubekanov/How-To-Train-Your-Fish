# 2026-08-25 — Recovered history: architecture, design decisions, lessons

Synthesized from four parallel sweeps: all 212 commits (2026-06-17 → 2026-08-25),
docs/DESIGN.md + README in full, code-embedded docstrings/comments, and the
deploy/eval/ops surface. This is the compressed reference; the commit bodies and
DESIGN.md carry the full detail. Where this doc and the code disagree, the code wins.

## 1. Era timeline

| Era | Dates | What happened | Markers |
|---|---|---|---|
| 1. Foundation | 06-17→06-19 | Vendored engine, AEC env, flat Discrete(285)+mask, compound builders, 4-net stack (actor/priv-critic/guesser/public), first A/Bs | e373737, 1a31b1d |
| 2. Durable service | 06-20→07-04 | Atomic checkpoints/resume, out-of-band panel, stats.json+backfill, PFSP league, ODROID systemd | e1d13fa, f01ab5c |
| 3. Scenario curriculum v1 | 06-28→06-30 | Manufactured scenarios; rule set: start-state+termination only, terminal ±1 reward, constructed not harvested | 27ef043, 613f769 |
| 4. Signal correctness | 07-02 | GAE per (game,seat) segment fix, truncation bootstrap, league state into checkpoint, heuristic v1.1 split (anchor stays v1.0) | 747fe50, f332447 |
| 5. Windows + throughput | 07-04→07-13 | Windows port, relay, parallel workers (2.3-2.7x), pipeline-collect (+51%), affinity, serve dashboard | ab9707e, 1dafde1 |
| 6. Probes + clock + seat bug | 07-14→07-16 | Deckout-clock features, library-knowledge null, seat asymmetry = engine bugs, entropy horizon fix, enforce_free_attack removed | eb4d683, 243325a, 51e69da, 02ac079 |
| 7. v2 entity restart | 07-16→08-16 | checkpoints-v2 (entity 768/768/384 d=128), inference server (+26%), scale-up ladder, BC sidetrack, engine re-sync | 9bf5766, d7a2c83, fbca2cd |
| 8. Audits → v3 restart | 08-17 | Two independent audits kill guesser + god critic; bookkeeper belief, public critic, deckout aux, guided text-change; warm start from v2-best | 6013fd3 |
| 9. Cloud + hands critic | 08-19→08-21 | Vast.ai mainline, C fastenc, envelope curriculum (ten scenarios), hands critic swapped in place at it=47538, telemetry honesty passes | 9f77b89, 08df290, d18ecf0 |
| 10. Obs gaps + mining | 08-22→now | Graveyard overflow + count block, TD experiments (reverted), 93k→181k games/h, weakness mines → fof/opening scenarios | 018d833, 8a96f2e |

Checkpoint dirs are era fossils: `checkpoints/` = flat 622h lineage,
`checkpoints-pre-clock/` = pre-clock branch point, `checkpoints-v2/` = entity
restart (382h, preserved; its best.pt seeded v3), `checkpoints-v3/` = ACTIVE
mainline (cloud), `checkpoints-bc*/` = imitation sidetrack.

## 2. Current architecture (v3, as deployed)

- **Game**: Forgetful Fish / Dandân — shared ordered hidden 80-card library, text-change
  IS removal (Dandân's no-Islands sacrifice), deckout race is the real game (parity).
- **Env**: PettingZoo AEC over the vendored engine run as a two-human game;
  `g.pending.player` is the sole turn authority (turn order not alternating). Obs
  strictly from `current_view`. Reward: terminal ±1 zero-sum, NO shaping anywhere.
- **Action space**: one flat Discrete(285) + legality mask; zone-slot indexing (obs
  row k ≡ action slot k); compound decisions via env-side monotone builders
  (PICK_A/PICK_B/COMMIT/SHUFFLE). Mask contract: every unmasked action is accepted.
  Human undo affordances (CANCEL_PAY, TARGET_CANCEL) deliberately withheld — the
  anti-stall design. `text_change_mode=guided` collapses the 25-way from→to block
  to {EFFECT, NO-OP}.
- **Models (v3)**: entity actor 768/768/384 d=128 (~2.68M), no value head; belief =
  20-dim analytic `bookkeeper_counts` (guesser deleted — its learned edge was 0.026
  nats at 32% of collection cost); critic = HandsCritic (public + both hands view;
  Brier .183 vs public .190 vs god .188 on an 88.7k-state bench — god is WORSE with
  strictly more info) + deckout-winner aux head (critic underweights parity;
  |ΔP|≈0.02 on a near-deciding flip). BCE not MSE (value fn is a win probability).
  V_p1 ≡ −V_p2 by sign convention (one critic, guarded by test_perspective).
- **Training**: PPO clip 0.2, gamma .997, GAE .95 per (game,seat) segment,
  critic_epochs=1 (held-out benches peak on epoch 1; later epochs memorize),
  calibration scored PRE-update (hands critic fingerprints its own batch to .03).
  PFSP league: scripted anchors (random / attacker / heuristic v1.0=eval anchor /
  v1.1-1.3 pool-only) + past-self ring, `hard` = (1−wr)², league state checkpointed.
  Scenarios are league members (pool-frac 0.6, boost, per-name weights in
  deploy/train.args).
- **Curriculum**: envelope-constructed scenarios — randomised inside p10-p90 bands
  measured from thousands of traced real games, relocate-never-fabricate state
  surgery, natural terminators only, v1.3 engine seat. Thirteen live members; the
  legacy seven manufactures sit at weight 0 for telemetry continuity.
- **Throughput shape**: collection is CPU-bound pure Python; GPU does the update +
  central batched inference server (CUDA graphs). pipeline-collect + collect-stream
  + shm pool transport + columnar Step-free buffer. gns_b >> batch ⇒ games/HOUR
  moves the slope, not it/h. Cloud box: 120 workers, 360 games/iter, mb 1024,
  ~180k games/h.
- **Ops**: STOP file = graceful checkpoint-and-exit (consumed on trigger); all
  restarts resume in place; architecture changes are in-place with
  function-identity gates (swap_critic / transplant_critic / migrate_clock /
  widen_counts / bootstrap seeds + handoff_start for freeze/KL phases) — NEVER a
  lineage reset. best.pt = maximin over scripted anchors. Winrates come from the
  out-of-band panel + harvested pool games; the trainer never blocks for evals.

## 3. Design-decision ledger (the load-bearing ones, with the numbers)

1. **Terminal ±1 only, no shaping** — scenarios move the start-state distribution,
   never the reward; cannot change the optimal policy, only what is practised.
   (Deckout potential-based shaping designed, repeatedly proposed, never approved.)
2. **Privileged→public→hands critic**: entity critic won its A/B (Brier .261 vs
   .342); by 08-17 god's edge over public had depreciated to ~0 (.202 vs .197);
   hands won the 3-seed bench (.183). Swapped in place at it=47538 — the
   checkpoints-v4 lineage idea was discarded within a day.
3. **Guesser → bookkeeper**: two independent audits; 77% of the channel is
   arithmetic; edge 0.026 nats decaying to ~0.005, negative with ≥3 known cards;
   gmae was never anchored (all-zeros beats it). Deleted at v3 (+36% it/h).
4. **Entity encoder for the actor only at the v2 restart** — flat actor spent 92%
   of params on the input projection and plateaued; entity ≈ flat end-to-end cost
   once values were batched (the "entity is 6x slower" assumption measured at 1.4x).
   Attention: implemented, shelved, no measured win at 8.6x CPU.
5. **Deckout clock**: parity is unlearnable from lib/80 (ReLU square-wave failure;
   supervised probe 0.508 vs oracle 1.000). 3-bit clock added 07-14, ablation
   Δ0.000 at the time ("shipped, provably ignored") — FLIPPED by 08-17: v2 actor
   uses it (deckout wr 0.380 on / 0.164 zeroed / 0.080 inverted). Re-measure,
   don't cite.
6. **Anchor discipline**: v1.0 is the frozen eval anchor forever; new heuristic
   versions land as new modules + profiles (never edits), pool-only. Wholesale
   replacement (v1.1, 07-02) silently moved anchor+opponent+scenario bot at once —
   never again. Scenario-bot upgrades step scenario_wr down (07-10, 08-06 seams).
7. **Curriculum v2 (08-21)**: manufactured corners → measured envelope; every
   scenario ends by natural game result. Scenario wr = curriculum signal, NEVER a
   success metric.
8. **critic_epochs=1 + pre-update scoring + window-pooled by-turn calibration**
   (08-21/24): three telemetry-honesty passes; per-batch by-turn Brier streaks were
   ICC~0.96 sampling (~250 effective samples/cell), not critic behaviour.
9. **Throughput doctrine**: measure before assuming (env.step = 1.9% of wall;
   feature encode 41% → row caches, C fastenc, batched values); every performance
   knob must be byte-identical when off (standing equivalence test suite, re-run on
   every re-vendor); games/h is the metric, it/h is not.
10. **BC ceiling**: clone saturates ~88% agreement but ~0.08-0.1 wr vs its own
    teacher; agreement and winrate anti-correlate under DAgger. Release-at-parity
    needs RL + freeze/KL handoff, not more imitation.

## 4. Lessons (cross-cutting, deduped — the expensive ones)

- **torch 2.12.1 plain-unpickle CPU-tensor storage leak** — bit THREE times
  (worker "allocator creep"; 155 GB inference-server OOM killing a session;
  0.9 GB/h panel follow mode). Standing rule: ALL weight transport is numpy.
  Panel fix is a bare one-shot mp.Process — an executor promotes torch's harmless
  teardown access-violation into a fatal BrokenProcessPool (observed live).
- **Reversible actions + terminal-only reward = stalling**, three separate ways
  (CANCEL_PAY/TARGET_CANCEL oscillation, attacker-toggle 87% of a 6.7k-decision
  game, pay strands). Fix pattern: monotone builders, mask-level withholding,
  affordability that can never strand a payment.
- **Seat asymmetry (0.375/0.625) was ENGINE bugs, not policy** — p1-first shared
  resolution (APNAP fix) and, dominantly, p1-first target lists under PICK_SINGLE
  indexing: mask BITS were equivariant while index SEMANTICS weren't. List order is
  part of the action-space contract. p1_adv_weight was a knob built on a disproven
  mechanism (reverted). And the seat bug was NOT the plateau — that was strategic
  (deckout).
- **Probe methodology** (the library-knowledge null, bf6474d): split by GAME not
  transition (transition split inflated 0.587→0.671); baseline = the public
  estimator; control game stage; don't refit big heads on probe-sized samples.
  Negative results are cheap and worth running.
- **Telemetry lies until proven honest**: choose_first_frac structurally 0.0 for
  ~1,404 rows (flat-id-vs-0 bug — "policy picks draw" was the bug); harvested
  winrates inflated by enforce_free_attack firing on scripted opponents (random
  went 20/20); post-update critic calib = fit-to-batch; gmae unanchored; bare NaN
  freezes browser dashboards; count-based SSE cursors freeze on full rings; device
  not in telemetry let three sessions train on CPU silently.
- **Engine sync discipline**: vendored engine is read-only; fixes mirror upstream;
  standing bit-equivalence suites re-run on every re-vendor; `assign_damage` must
  never reach the trainer; every battlefield exit goes through
  `_detach_from_battlefield` (marked-damage-into-library class).
- **Windows platform tax**: E-cores 2.3x slower + non-hybrid-aware scheduler ⇒
  affinity is not cosmetic; ~1.6 GB commit per torch process (30 GB before work);
  SO_REUSEADDR allows duplicate binds (weeks-stale dashboard shadow); timed sleeps
  round to the 1ms tick; deep all-AI games need 128 MB thread stacks.
- **Cloud/many-core tax (08-22)**: image-exported OMP_NUM_THREADS=62 × 120 workers
  = futex-barrier collapse (pin BLAS to 1 BEFORE the pool); per-task 10 MB weight
  blobs through one feeder pipe; shm leaks ~14 GB per mid-flight STOP without
  pid-stamped sweep; fd limits at the take() seam.
- **In-place migration pattern**: zero-init new columns appended to the tail +
  old-weights-remap + old(x)==new(x') gate + KL re-arm. Used four times
  (clock, counts, swap_critic, transplant). 592h/382h runs are the thing at risk.
- **Mid-batch value pathologies**: outcome-only BCE leaves deterministic action
  chains untied (±8% V flicker pocketed as advantage, invisible to Brier);
  td_mix reverted; consistency at λ=50 fixed the jump but STALLED h1.3 because
  det_next pairs spanned spell resolution via the opponent's unrecorded forced
  pass — cast/pass excluded, boxed at λ=0, transplant_critic recovered the damage.
- **Pooling hides counts**: mean/max pooling gives fractions and presence, never
  absolute counts — same failure class as parity-from-len/80. Graveyard overflow
  kept the OLDEST 32 (dropped the newest Dandâns exactly in the deckout window).
- **Sampling beats argmax in eval** — greedy decode of a high-entropy masked
  policy degenerately passes and loses.
- **Fresh-short-run benchmarks lie** (game length grows 134→210 decisions);
  A/B throughput flags on settled policies at fixed wall-clock.

## 5. Reading historical data (era seams)

- `it` resets across restarts — segment by `run`/`host`; `elapsed_h` is cumulative.
- Pre-02ac079 (07-16) random/attacker numbers inflated (enforce_free_attack).
- scenario_wr steps down at scenario-bot upgrades (a8f7151 07-10, 88d6a83 08-06).
- critic_acc/brier seam at it≈48.6k (pre-update scoring) and critic_loss seam at
  it≈52.6k (critic_epochs=1) — steps are honesty appearing, not regressions.
- choose_first_frac before 2026-08-17: ignore entirely.
- v3 rows carry critic_* keys; legacy rows guesser/priv/pub/brier_gap/gmae —
  backfill parses both eras with one regex.
- `heuristic=` in [league] lines = merged v1.0-1.3 count; per-version tokens ride
  after the paren. wr_train denominators exceed panel n (harvest); draws count as
  losses everywhere.
- `pipeline`/`stream` row keys are sparse: absent = that regime was off.

## 6. Dead ends (do not retry without new evidence)

stats annotations (ee34fbb) · establish_clock scenario · free_attack scenario →
rule → removed entirely · p1_adv_weight · relay-as-default (10% throughput for
100% of the risk) · tkinter monitor · checkpoints-v4 lineage for a critic change ·
critic_td_mix · critic_consistency λ>0 with cast/pass pairs · gpi-16 · jit.script
encoders · attention encoder (shelved) · guesser / PublicEstimator / god critic ·
enforce_free_attack · deficit-only harvest top-up · fixed-anchor best.pt ·
BC-to-parity without RL.

## 7. Open questions on the books

- Deckout PBRS shaping: designed, policy-invariant, never approved.
- Belief on/off winrate ablation: never run (bookkeeper made it moot-ish).
- critic_consistency: boxed at λ=0 with PAY_KINDS-only pairs — "re-enable later".
- Hands-critic cast-step systematic: open.
- obs_counts actor-side value: unmeasured offline (critic bench null).
- Cyclical entropy re-heat: implemented, off — enable when stalled, not before.
- Tier-3 throughput flags need a settled-policy A/B.
- fof_split degeneracy (17% vs-1.3 / 44% mirror at it=107k) — scenarios live,
  outcome pending; see 2026-08-25-weakness-mine-107k.md.
