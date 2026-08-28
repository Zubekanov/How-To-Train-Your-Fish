# 2026-08-27 — scenario_selfplay_frac: the agent takes the scenario's other seat

Joseph's call: with the agent at ~0.64-0.70 vs h1.3 and no 1.4 forthcoming,
promote the agent itself into the scenarios' opposing seat — a dynamically
strong opponent, a second perspective (the caster/punisher side of each
constructed situation, not just the responder), and no script-exploit pressure.

## Design (mix, not replacement)

`Config.scenario_selfplay_frac` (runtime knob, resume-tunable,
`--scenario-selfplay`; deployed 0.6): with that probability a scenario game
skips `make_engine_heuristic` — both seats stay learner-controlled (the base
self-play contract that already existed in `Scenario.engine_seat=None`) and
BOTH seats' transitions train (collect_games drives every surfacing AEC
agent; the engine-seat mode never surfaces p2). Kept a mix deliberately: the
scripted seat is the curriculum's only EXTERNAL pressure, and the
self-play-blind precedent stands (main-phase telegraphing: ~0 err in mirror,
22.5% catastrophic vs 1.3). Mines vs 1.3 remain the watchdog for
mirror-bred habits.

Mechanics worth remembering:
- The roll is the episode rng's FIRST draw, so the constructed board for a
  given seed is byte-identical in either mode (only `is_ai` flips), and the
  trainer recomputes the mode from the game seed alone
  (`scenarios.selfplay_mode(seed)`) — no worker report-back. The extra draw
  shifts the construction rng stream by one vs the previous era (same
  envelope, different draws — distribution-neutral).
- Telemetry: `[scenario]` line gains `| selfplay 0.60 wr self=…(n) script=…(n)`
  and stats.json rows a sparse `scenario_selfplay` dict. The two win-rates
  mean different things: `wr_script` is the legacy sense (skill vs v1.3);
  `wr_self` is the CONSTRUCTED SEAT's advantage-conversion rate (policy vs
  policy from an advantaged start). The scen_league PFSP EMA now blends both —
  still a valid difficulty prior (hard = failing to convert), but per-scenario
  wr trends cross an era seam at this deploy.
- Plumbing: module-global setter in scenarios/base.py (the
  set_text_change_mode pattern), set at train_loop init + pcollect._winit
  (lite carries the frac). Scenario metas now carry the game SEED in the seat
  slot (the learner seat is always p1 for scen games).

## Verification

test_scenario_selfplay.py (4): roll gates the engine seat with a
board-identical-across-modes check, mode recomputable from seed (40 seeds),
self-play surfaces BOTH seats to the policy while script mode keeps p2 off
the policy path, frac=0 reproduces today's behaviour on a registered
scenario. Full suite 341 passed / 7 skipped. Resume smoke on the real
checkpoint with --scenario-selfplay 0.6 (see box numbers below).

## Deployed (b6d00ca; box resumed it=155,430, 2026-08-28)

First full window (it=155,549): 18,734 scenario games, `selfplay 0.60
wr self=0.46(11,295) script=0.60(7,439)` — the split renders and the two
win-rates behave exactly as predicted (self ~= seat-advantage conversion near
0.5; script = skill vs 1.3). Trainer healthy: kl 0.0131, 474.7 it/h, calib
critic acc .74 / brier .18. Note the box had reached ~155k by deploy time —
the next mine window moves to ~165-170k.

## Expected effects to watch at the next mine (~165-170k)

- Scenario transitions per game roughly double in self-play mode (both seats).
- deckout_stack: the agent must LEARN the punisher role 1.3 used to script.
- Watch for mirror-collusion in narrow pockets (the wr_self of a scenario
  drifting to its start-state equity with no behavioural content) — the
  vs-1.3 mine catches what the mirror stops punishing.
