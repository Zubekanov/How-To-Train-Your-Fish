# 2026-08-29 — telemetry sampling audit: what each number is computed from

Joseph's ask (after the calib-headline fix showed the class exists): go
through EVERY emitted metric and classify its sampling basis — full window,
last batch, EMA, or a small specially-run game sample. The complete table,
with what was fixed this pass (7ffa3f2) and what is small/last-batch on
purpose.

## Status line

| token | basis | verdict |
|---|---|---|
| pi/V/H/kl/clip/aux/consist | window mean over all ~130 iters | ✓ full |
| calib critic(acc,brier) | window fold, ~11M decisions (fixed e37f82e) | ✓ full |
| brier/t (4 buckets) | LAST BATCH — deliberate (recorded preference for the streaky by-turn read); `critic_turn_win` + window bucket keys ride in stats | by design |
| jump mean/p90 | was LAST BATCH (~10-25k pairs, game-correlated) → **FIXED**: window fold via n/sum/1000-bin histogram (`jump_from_sums`, p90 exact to 0.001); last batch rides as `critic_jump_*_batch` | fixed |
| opp trained / past / cf | window counts | ✓ full |
| games/len/slen/trunc/draw/deckout/turns/seat_p1/play/fdec | window counts | ✓ full |
| wall collect/book, worker gap/play | window sums (pop_timing) | ✓ full |
| gns_b | window mean of per-iter estimates, but prints nan: g² = |ĝ|² − σ²/M goes ≤0 when B_crit ≫ M (noise dominates the gradient estimate) — an estimator SATURATION reading "critical batch far above current batch", consistent with [[throughput-ceiling-2026-08-22]]'s gns_b ≫ batch. Not fixable by pooling; treat nan as "≫ M". | documented |

## [scenario] line

- counts: window ✓.
- **wr table: was the PFSP EMA (alpha 0.1 ⇒ ~10-game effective memory)
  printed beside a CUMULATIVE game count** — a sampler signal formatted like
  a winrate, which is why per-scenario wr looked violently noisy
  (lethal_on_board 0.34→0.20→0.39 across windows that each held ~2,400
  games). **FIXED**: the table now prints true window rates `name=w/g(n)`
  with the window's decided-game n; `scenario_wr_win` {name: [wins, games]}
  added to stats. The EMA stays what it always was — the PFSP sampler's
  difficulty input — and still rides in stats `scenario_wr`.
- selfplay split: window counts ✓ (b6d00ca).

## [league] line

- counts: window ✓. wr table: PFSP EMA (same 10-game memory) with
  cumulative n — left as-is, clearly the sampler's own state; the REAL
  per-anchor window rates already exist (`wr_train` [wins,games] per
  scripted anchor, eval-service combined estimates, and `past=` for the
  pooled past-self population). Documented rather than duplicated.

## stats.json rows

- Everything above, plus: `critic_turn` last-batch (by design) with
  `critic_turn_win` companion; bucket scalar keys last-batch (by design,
  window versions derivable from critic_turn_win); `league_wr` EMA
  (sampler state — window truth is `wr_train`); `scenario_wr` EMA (window
  truth now `scenario_wr_win`).

## Eval panel

- heuristic/attacker/random anchors: harvest (~300-500 training games) +
  topup per cycle, block-aggregatable ✓.
- frozen: n=50 fresh per cycle → `frozen8` rolling n≈400 (this session).
  Cannot be harvested from training: past-self pool games are PFSP-sampled
  across many snapshots with difficulty-biased weights.
- seat diag (seat p1/p2, play/draw, choose_first): n=48 fresh games — now
  REDUNDANT for play (training-side `play=` n≈17k) and choose_first
  (training-side `cf=`, full population); kept as an independent
  fresh-sample cross-check. Small by design.

## Ticks

Per-iteration values by design (the tick stream is the high-frequency view).

## The general lesson

Three telemetry classes hid tiny effective samples behind big-looking data:
(1) last-batch snapshots where decisions-per-game ≠ information (ICC 0.96);
(2) EMAs with short memory printed beside cumulative counts; (3) small
fresh eval samples duplicating measurements the training stream already
makes at 100-1000× the n. When adding any metric: state its sampling basis
in the emitting code, and if a window accumulator can carry it, pool it.
