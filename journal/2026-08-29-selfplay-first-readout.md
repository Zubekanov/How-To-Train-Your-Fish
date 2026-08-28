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

- CORRECTED (Joseph's pushback, same day): window-mean comparison was the
  wrong instrument on a trending series — each report point is ~380-470
  games (±2.3pp SE), so the pre AGGREGATE was diluting a real slope, and the
  first-pass "hot streak" narrative was wrong. Weighted trend fit
  (140k→deploy): slope +2.34 ± 0.37pp per 10k its, fitted level at deploy
  0.720. Post aggregate 0.706 (n=22.7k, se=0.003) = **−1.4pp below the
  fitted deploy-day level** and −2.2pp below trend continuation; post slope
  +1.2pp/10k = growth HALVED, still positive. This is a real step down at
  the seam plus a slower climb — Joseph's eyeball read, quantified.
- No health regression: kl 0.0128-0.0134, H 0.42-0.43, aux 0.61, critic acc
  ~0.72 steady. In-scenario wr_script stable 0.59-0.61 (no measured skill
  loss vs the 1.3 script where it still plays), wr_self 0.46-0.47 flat.

## Mechanism (why the h-anchors would stall)

Scenario games are ~43% of all games (~20.5k of ~47k/window). Pre-deploy,
every one had the 1.3 script on the far seat; at frac=0.6 only 40% do, so
the vs-1.3 share of ALL training games dropped ~43% → ~17% (≈2.5×). The
h-anchor curves measure exactly the skill that exposure fed. The self-play
share buys script-independent pressure whose payoff should show as
robustness (mines, mirror behaviour), not as immediate h1.3 movement — that
was the design trade, and the anchors are behaving as the design predicts.

## Verdict + decision

Real cost, correctly sized: −1.4pp step at the seam + growth rate halved
(2.3→1.2pp/10k) on h1.3; all four anchors moved the same direction. The
trade bought script-independent pressure whose value is only measurable at
the mine. Options on the resume-tunable knob (`--scenario-selfplay`): hold
0.6 through the ~165-170k mine, or drop to 0.4-0.5 to recover script
exposure at reduced self-play share. Joseph's call — the cost side of the
ledger is now quantified; the benefit side has no number until the mine.

## Lesson (methodology)

Never compare window means across a change boundary on a series with a
known slope — fit the pre-trend (weighted by per-point n) and test the post
window against the fitted level AND continuation. The window mean dilutes
the trend and manufactures a "noise" narrative for the residual.
