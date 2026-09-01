# fishrl — Final Report

**Training concluded 2026-09-02** at iteration 204,225 (346.95 training hours
on the v3 lineage). The final release agent is **`checkpoints-v3/best.pt`**
(iteration 196,263, the last checkpoint to pass the maximin best-gate), holding
**~0.75 win-rate against heuristic v1.3**, the strongest scripted Dandân player
in the project. The lineage remains resumable on the PC via
`deploy\fishrl-pc.bat`.

This document is the end-of-run record: what was trained, how it went, what the
final agent can and cannot do, and what was learned that outlives the run.
Design rationale and telemetry contracts live in [`DESIGN.md`](DESIGN.md); the
dated decision trail lives in [`../journal/`](../journal/).

---

## 1. The agent, final state

| Component | Final state |
|---|---|
| Actor | Entity encoder, heads 768/768/384, card_dim 128 (~2.68M params), masked flat action space (N=285) |
| Critic | Entity encoder, **hands view** (public + both hands), deckout-parity aux head (0.1), `critic_epochs=1`, temporal-consistency penalty on PAY_KINDS det-next pairs (lam 25) |
| Belief | Analytic **bookkeeper** (zero parameters) filling the 20-dim slot — the learned guesser was deleted at the v3 restart after two audits showed it inference-null and 77% arithmetic-redundant |
| Action space | `text_change` guided ({EFFECT, NO-OP}); CANCEL/strand-pay masked; affordability-gated casts |
| Opponents | PFSP hard-mode league: 6 scripted anchors + 8 past-self snapshots spaced every 8th report (~16h span) + 18 constructed scenarios (boost 0.5) |
| Optimizer | Adam 1.5e-4 (halved from 3e-4 at it=184k), minibatch 1024, entropy floor 0.008 |

Scale of the run (v3 lineage, live reports): **54.6M games, 12.6B decisions**,
346.8h wall — roughly 13 days of continuous training, the last 14 of them on a
rented Vast.ai RTX 4090 at 103–181k games/hour.

## 2. Final benchmarks

All numbers are fresh (2026-09-02, this machine) on the release agent
`best.pt` (it=196,263), not training-stream telemetry.

**Win-rates** (large-sample panel, n=400 per heuristic anchor, + the
2,000-game mine):

| Opponent | Win-rate | n |
|---|---|---|
| random | 1.000 | 50 |
| attacker | 0.980 | 100 |
| heuristic v1.0 (the long-standing eval anchor) | 0.932 | 400 |
| heuristic v1.1 | 0.850 | 400 |
| heuristic v1.2 | 0.777 | 400 |
| **heuristic v1.3 (testbench mainline)** | **0.764 ± 0.009** | **2,400** (mine 0.762/2,000 + panel 0.775/400) |
| frozen self (~110 its older) | 0.495 | 400 |

Mirror self-play (1,600 games): seat split 0.501/0.499 (the seat-equivariance
work holding), on-the-play 0.519, 30.9 mean turns, 20.6% of decided games end
by deckout. Panel seat-diagnostic (200 games): play 0.575.

**Decision quality** (critic-judged, final mine):

| Metric | Final | Context |
|---|---|---|
| Decision-level Brier vs 1.3 | 0.139 | 0.165 at it=135k |
| Degenerate 0-5 FoF splits | 0.1% | ~20% a month earlier |
| Removal wasted on lands (fish available) | 1.0% | 3.4% at 135k |
| Catastrophic main-phase telegraphing | 1.0% | 2.0% at 135k |
| Fizzle-line picks | ~0 (1/2000 games) | watch-only from day one |
| Critic-flagged "blunders" | 0.85/game | 0.84/g of it is pass/end-turn holding the outcome-calibration test says is CORRECT (judge 0.503 vs actual 0.649 at those states, +14.7pp critic bias) |

The resume point `latest.pt` (it=204,225) was checkpointed **mid-reheat-peak**
(deliberately loosened policy, entropy coefficient ~0.016) and benches h1.3
**0.665** (n=200) — a real ~10pp wobble below best.pt, which is the escalated
reheat dose visibly working on the policy at the moment the run stopped. It is
the right *training* resume point (a resume under the final train.args
consolidates it back at the 0.008 floor); **best.pt is the release agent**.

## 3. How it went — the timeline

Three lineages, each preserved on disk:

- **v1 “flat”** (`checkpoints/`, 622h): flat 1.81M actor + learned hand-guesser
  + privileged god-view critic. Plateaued; provably ignored the deckout clock.
- **v2** (`checkpoints-v2/`, 382h): bigger entity actor (768/768/384, d128).
  Fixed the seat asymmetry era (which turned out to be engine bugs, not policy).
- **v3** (`checkpoints-v3/`, 347h): bookkeeper belief + public-then-hands
  critic + parity aux, actor warm-started from v2-best. All numbers below are
  v3.

The v3 cloud era, milestone by milestone (each has a dated journal entry):

| It | Date | Event | h1.3 |
|---|---|---|---|
| 0 | 08-14 | v3 launch: bookkeeper + public critic + deckout aux + guided text-change, warm start from v2 | — |
| ~15k | 08-19 | Moved to the Vast.ai box (103k games/h) | 0.21 |
| 46,343 | 08-21 | Envelope curriculum: ten constructed scenarios, natural terminators | ~0.3 |
| 47,538 | 08-21 | **Hands critic swapped in place** (no lineage reset); critic_epochs=1 | |
| 64,218 | 08-22 | obs_counts widened in place; throughput push → 181k games/h | |
| 100k | 08-24 | FoF-split/pick + opening_race scenarios | ~0.55 |
| 123.4k–123.8k | 08-26 | Observability audit fixes: obs_split, obs_ctx (+stack targets, search, builder, critic step/pay), obs_tgt | 0.56 |
| 135k | 08-27 | **Plateau broken by the obs packs**: 0.56 → 0.643 | 0.643 |
| 155,430 | 08-28 | Scenario self-play 0.6 (both seats the policy) | ~0.70 |
| 162,402 | 08-29 | Full-population telemetry (window-pooled everything) | |
| 164,3xx | 08-29 | Scenario boost → 0.5 (lessons mined out; share 44%→18%) | 0.708 |
| 184,058 | 08-30 | Plateau levers: **LR 3e-4→1.5e-4, spaced league, consistency lam 10** → +3.4pp step | 0.723→0.757 |
| 196,263 | 09-01 | **Last new_best — the final agent** | 0.75–0.78 |
| 196,735 | 09-01 | Entropy reheat cycles + lam 25 (cycle 1 sub-therapeutic) | |
| 203,867 | 09-02 | Reheat dose escalation (peak 0.016 — cycle unread; run ended) | |
| **204,225** | **09-02** | **End of training; lineage synced home** | ~0.75 |

## 4. Why the run ended where it did

The last week was a systematic plateau hunt, and its endpoint is well
characterised rather than mysterious:

1. **152k–184k**: a genuine optimization pathology — gradient-noise-limited
   churn (GNS estimator saturated at a never-decayed 3e-4, past-self ring
   spanning only ~2h). Fixing it (LR cut, spaced league, consistency) bought a
   real **+3.4pp step** … and then flat again (slope −0.03 ± 0.11 pp/1000 it).
2. **196k rediagnosis**: optimization now healthy (KL 0.006, jump falling,
   critic at its best Brier ever), and the mines show **per-decision error
   mass vs 1.3 essentially exhausted** — worst action class −0.002 mean value
   delta, never-ahead losses 0.35%, all degeneracy canaries clean. The
   residual ~25% of losses are parity/variance-shaped games with no flagged
   errors.
3. **The exploration lever** (entropy reheat, the purpose-built stall escape)
   was tried at two doses; the first was demonstrably sub-therapeutic (+0.02
   nats), the second (peak 0.016) was cut short by the end of cloud credit
   with its cycle unread.

In short: the agent converged to a sharp, clean exploitation of its strategy
space; each further intervention bought a one-time step, not a slope. The
untested escalations, if training ever resumes, are (a) direct h1.3 gradient
via an anchor-weight knob (~1.7% exposure at the end), and (b) capacity — a
function-preserving in-place actor widen (the actor is only 2.68M params).

## 5. What the agent learned (mine-verified)

The weakness-mine series (2,000 games vs 1.3 + 1,600 mirror, critic-judged
counterfactuals + analyzer battery, ~every 15k its) tracked skills, not just
win-rate. By the final mines:

- **Removal economy**: waste-on-lands-with-fish-available 3.4% → ~1%.
- **Day's Undoing**: from a 19% blowup rate and negative mean value to
  clearly value-positive (+0.04).
- **Deckout play**: losses-by-deckout vs 1.3 halved (127 → ~70/2000); the
  punisher role the scripted seat used to play is executed by the agent.
- **FoF splits**: degenerate 0-5 splits extinct (≤0.2%); contextual
  asymmetric endgame splits emerged instead.
- **Telegraphing**: catastrophic main-phase text-changer casts 2.0% → 0.8%.
- **Holding discipline**: the mines' biggest "blunder" class — passing with
  castable spells — was adjudicated **intentional and correct** by outcome
  calibration: at flagged states the critic read ~0.45–0.50 but the agent
  actually won 0.62–0.64 of them (the critic over-prices "castable ⇒ cast").
- Still imperfect at the end: Mystical Tutor slightly value-negative
  (−0.023), predict-the-name remains the top deckout-endgame error.

## 6. Lessons ledger (what outlives the run)

Methodology:
- **Trend-fit block aggregates; never window means.** Every "regression" that
  survived block aggregation was real; every one that didn't was sampling.
  ~10k games per point minimum before judging a trend.
- **State each metric's sampling basis at the emitting line.** Three separate
  "noise mysteries" (calibration variance, frozen noise, scenario wr swings)
  were all sampling artifacts — last-batch snapshots, 10-game EMAs printed
  beside cumulative counts, n=50 side samples. Window-pooling at the source
  ended all three.
- **Run the outcome-calibration test before declaring a policy hole from
  critic-judged blunders.** The judge is the agent's own critic; measure its
  bias at the flagged states first ("passivity" was judge bias, twice).
- **A launcher that appends flags after `$(cat train.args)` silently wins**
  (argparse last-occurrence). The cloud launch.sh pinned `--scenario-boost 3`
  for the run's whole history while train.args said 5. Audit the trailing
  block whenever a knob "doesn't take"; verify via `/proc/<pid>/cmdline`.
- **An LR set at launch is pinned forever unless the resume path re-asserts
  it** — the optimizer state_dict restores the old param_group LR. The run
  trained 300h at 3e-4 before anyone noticed. `--lr-ppo` now re-asserts.
- **gns_b = nan is a message, not a bug**: the critical batch is far above the
  actual batch — the signature of noise-limited training and the cue to cut LR.

Training design:
- The **observability packs** (stack targets, search counts, builder mirrors,
  split piles, critic step/pay context) were the single biggest win of the
  run: +9pp in one stretch after months of curriculum tuning. When a skill is
  frozen, check whether the *credit path* can see the decision before
  building scenarios for it.
- **Constructed scenarios with natural terminators** taught real, transferable
  skills (mine-verified above) and were then correctly *shrunk* once mined
  out — curriculum is scaffolding, not architecture.
- **critic_consistency on deterministic PAY_KINDS pairs** (after fixing the
  pairing to not span spell resolution) was the best late-run lever: jump
  halved, decision Brier 0.154 → 0.132, judge bias narrowing, and it rode the
  +3.4pp step. The naive pairing (spanning resolutions) had stalled learning.
- **Space the past-self league.** Snapshots every report = a 2h ring = 76% of
  games vs near-clones and no anti-cycling pressure.
- Failed/neutral levers, so they are not retried: critic TD-mix (neutral at
  best), imitation beyond bootstrap (0.1 wr ceiling vs own teacher),
  privileged critics (hidden info ≈ worthless in this game), a learned hand
  guesser (deleted), entropy reheat at timid doses.

Operations:
- ALL cross-process weight transport is numpy (torch 2.12 unpickle leak).
- `expandable_segments:True` on any long CUDA run (allocator fragmentation).
- Never `pkill -f` a pattern that also appears in the same compound command's
  text; never `tmux send-keys` a launch while the old process lives.
- Ship source to credential-less boxes as `git archive | ssh tar -xf -`.

## 7. Operations after the cloud

- **Resume locally**: `deploy\fishrl-pc.bat` — resumes
  `checkpoints-v3\latest.pt` (it=204,225) with the final regime in
  `deploy/train.args` (lr 1.5e-4, league 8×8, consistency 25, entropy at the
  0.008 floor — the reheat experiment's flags were removed so a resume
  consolidates). Verified: a scratch-copy resume ran clean on this machine.
- **File map**: `checkpoints-v3\latest.pt` (resume point), `best.pt` (the
  final agent, it=196,263), `stats.json`/`ticks.json` (full telemetry
  history), `cloud-final\` (salvaged from the box before decommission: the 19
  permanent `archive_*.pt` snapshots at 20k…200k — an elo-ladder in waiting —
  the six in-place-widening swap backups, and the box's train/eval/sysmon
  logs), `presync-*\` (pre-sync local copies, deletable once satisfied).
  v2 and v1 lineages preserved untouched in `checkpoints-v2\` and
  `checkpoints\`.
- **The website**: its collector ingested the box's stats.json, which no
  longer exists; the final stats.json lives in this repo if a last import is
  wanted. The 2026-08-29 telemetry fields (frozen_agg, past_wr, …) were never
  mapped website-side.
- **The box**: nothing sensitive was ever on it (no git credentials — source
  went over as tar streams). It can be destroyed from the vast.ai console.

## 8. Open threads (if training ever resumes)

1. The unread reheat cycle: peak 0.016 was live for ~½ cycle; the kill
   criterion (trough vs the 0.757 baseline) was never evaluated.
2. The two reserves: `--anchor-weight` (direct h1.3 gradient; overfit risk) and
   the function-preserving actor widen (capacity).
3. Critic hold/cast bias: narrowing (+18.5 → +13.6pp) but not gone; lam
   escalation or a targeted cast-step feature would continue the repair.
4. Mystical Tutor and predict-the-name remain the last identified skill gaps.
