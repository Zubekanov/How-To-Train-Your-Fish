# 2026-08-29 — full-population game telemetry (775204d)

Joseph's observation: frozen winrate and on-the-play winrate are far noisier
than the training population justifies (~6k past-self games and ~20k mirror
games per window). Root cause: both numbers came from tiny dedicated fresh
samples — `--n-frozen 50` per panel cycle (help text: "panel-only: not
harvestable") and `--seat-diag-games 48` — ±7pp SE each, while the training
games carried the same information uncollected.

## The sweep: what the game sample offered vs what was collected

Per-game facts the collector already touches at stamp time but dropped:
`g.first_player`, `g.turn_number`, and the deckout flag (stamped per-STEP for
the aux head, never aggregated per game). Plus two populations whose outcomes
were counted only into opaque EMAs: pool games vs past selves (league EMA,
not exported for kind="self") and the policy's own PLAY_ORDER decisions
(recorded in the action column, never counted — the eval-side
choose_first_frac reads fresh diag games only).

## Wired (all live in the box's next window)

- `_stamp_game` meta now carries `first`/`turns`/`deckout` per game (rides
  the existing pcollect meta transport unchanged).
- `play=` — on-the-play winrate over decided MIRROR games (~19k/window,
  SE ~0.4pp; was n=48). Mirror-only by design: pool/scenario games confound
  first-mover edge with opponent strength / constructed advantage.
- `past=X(n)` — learner wr in pool games vs past selves (~6k/window).
  PFSP-WEIGHTED: the sampler overweights opponents the learner loses to, so
  this reads training pressure, NOT an anchor. The panel's frozen number
  stays the unbiased canary.
- `deckout=` — share of decided games ending by empty-library draw (full
  population; the field guide's mode split, now a live series).
- `turns=` — mean end turn_number (game pace).
- `cf=` — play-first choice rate over every PLAY_ORDER decision.
- Panel: `frozen8=X(n)` — rolling 8-cycle block aggregate of the frozen
  match (counts `frozen_w`/`frozen_n` now ride each eval row; each cycle is
  a one-shot child, so the window is rebuilt from stats.json). n≈400 →
  SE 2.5pp, was 7pp. Same measurand ("vs recent selves") since the anchor
  refreshes to near-current every cycle.
- stats.json rows carry all of it with counts (`play_n`, `past_n`,
  `choose_n`) for cross-window block-aggregation; backfill_stats regexes
  extended with optional groups (all four earlier eras still parse —
  fixture `_STATUS_ERA4` added).

## Not wired (considered, rejected)

- Per-past-self league member wr export: composition drifts with the league;
  the pooled `past=` series carries the usable signal.
- Per-scenario length/deckout splits: available via per-gbuf metas but the
  [scenario] line is already the widest in the log; revisit if a mine needs
  it.
- First-blood / fish-war exchange stats: needs engine tracing, not stamp-time
  facts — that's what the mines are for.

## Verification

Full suite 343 passed / 7 skipped (new: meta-enrichment test, era-4 backfill
fixture). 2-iter parallel CPU smoke rendered every token with real values
(`past=nan(0)` correct for a league with no selves yet). Deployed with the
standard ship+STOP+relaunch; box resumed at it=162,402, GPU 97%. Telemetry-
only: no rng consumption, no obs change, no era seam in game construction.
