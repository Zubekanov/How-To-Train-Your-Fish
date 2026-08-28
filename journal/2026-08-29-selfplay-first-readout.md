# 2026-08-29 — scenario self-play, first ~6.6k-iteration readout

Joseph's eyeball: "seating the agent on both sides of the scenario seems to
have very slightly degraded winrate vs the heuristics." Reviewed against the
box telemetry (deploy boundary it=155,430; readout through it=162,057, ~17h
of post data). Method: weighted block-aggregation of the eval rows'
anchor_n (train harvest + topup) — the only way these winrates are readable.

## Numbers (all anchors, matched ~5.5-6.6k-it windows)

| anchor | 144-150k | 150k-deploy | post-a (155.4-158.8k) | post-b (158.8-162.1k) | post all |
|---|---|---|---|---|---|
| h1.0 | .9023 | .9087 | .9054 | .9056 | .9055 (n=13.1k) |
| h1.1 | .8535 | .8668 | .8594 | .8616 | .8605 (n=14.9k) |
| h1.2 | .7782 | .7846 | .7797 | .7826 | .7811 (n=18.3k) |
| h1.3 | .7005 | .7139 | .7043 | .7078 | .7061 (n=22.7k) |

- h1.3 pre→post: −0.78pp, z≈1.8 (not significant at 95%, but every anchor
  moved the same direction, −0.3 to −0.8pp — a small real effect is likely).
- Equally important: the pre window was itself a RISING streak (+1.3pp over
  the window before it, z≈3.2). Post is FLAT, and post-b > post-a on h1.3
  (.7043→.7078) — reads as "slope stalled + tiny retrace", not a decline in
  progress. No collapse anywhere; kl 0.0128-0.0134, H 0.42-0.43, aux 0.61,
  critic acc ~0.72 all steady.
- In-scenario telemetry stable: wr_script 0.59-0.61 (same as the deploy
  window's 0.60 — no measured skill loss vs the 1.3 script where it still
  plays), wr_self 0.46-0.47 flat.

## Mechanism (why the h-anchors would stall)

Scenario games are ~43% of all games (~20.5k of ~47k/window). Pre-deploy,
every one had the 1.3 script on the far seat; at frac=0.6 only 40% do, so
the vs-1.3 share of ALL training games dropped ~43% → ~17% (≈2.5×). The
h-anchor curves measure exactly the skill that exposure fed. The self-play
share buys script-independent pressure whose payoff should show as
robustness (mines, mirror behaviour), not as immediate h1.3 movement — that
was the design trade, and the anchors are behaving as the design predicts.

## Verdict + decision

Joseph's read is directionally right but the size is ~0.5-0.8pp against a
noise floor of ±0.6pp — stall, not damage. Options on the resume-tunable
knob (`--scenario-selfplay`): hold 0.6 through the ~165-170k mine (self-play
benefits are what the mine tests), or drop to 0.4-0.5 to tilt back toward
script exposure if h1.3 growth is the priority this week. Recommended: hold
until the mine unless the next ~10k its show post falling below pre-2
(~0.70) — that would be actual regression, not stall.
