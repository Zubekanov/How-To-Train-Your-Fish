# 2026-08-25 — Weakness mine at it≈107.8k (2,000 vs h1.3 + 1,600 mirror)

Panel h1.3 was 0.539 at mine time; the traced vs-h1.3 pool landed at wr 0.530
(consistent). Method: every decision scored by the checkpoint's own hands-view
critic; counterfactual regret = re-score every legal alternative on value drops.
New `mine_self.py` traces mirror self-play with both seats the agent (err > 0 =
the acting seat hurt itself, works for either seat since v is always P(p1 wins)).
Tooling lives in the session scratchpad `mine/` dir; a traced game costs ~6-8
worker-seconds, so thousand-game mines are ~10 minutes on the 28-core PC.

## Findings

1. **Omissions are still the #1 blunder class, but halved.** 0.51/game confirmed
   blunders vs h1.3 (0.9 at the 08-21 ledger), 0.85/game in mirror: pass or
   end-turn while holding castables, at winning positions (mean v 0.64, turns
   24-30, 6+ untapped in 38%). 26% of losses peaked at v ≥ 0.8.
2. **FoF 0-5 splits: 17.4% vs h1.3 but 44.0% in mirror** (n=1,663 / 2,307
   splits), context-independent in mirror. The copy-opponent mispicks too, so
   self-play applies no pressure — only the fof_split scenario's v1.3 picker
   does, and it hasn't generalized beyond the vs-1.3 state distribution.
   fof_pick (teaching the pick side) is the lever that makes league play start
   punishing bad splits.
3. **Deckout endgame** (18.4% of mirror games, 15% of vs-1.3 losses). h1.3 has an
   aggro-deckout punish mode; the loss anatomy vs 1.3: 52/140 = 1.3's chain
   executes the kill, 49 natural draw-step deaths (parity lost upstream), 39
   involve the agent's own draw spell (14 punished mid-stack). At lib ≤ 8 an
   *unanswered* agent draw-spell cast blows up 12.9% vs 3.6% answered. Mirror:
   215/294 deckouts are natural deaths — parity decided upstream — plus 38
   unforced cantrip suicides at ~0 cards. The missing skill is draw-race stack
   timing + upstream parity counting, not a terminal "don't cast at 0" rule.
4. **Vision Charm** wipe casts average −0.070 (n=1,707) vs h1.3 — logged as an
   open observation only (see lessons).
5. Fish-war economy unchanged: never holding a fish lead → wr 0.005 (11% of
   vs-1.3 games); first blood wins 65% of mirrors. Mulligan games 0.413 vs 0.547
   kept (can't be scenario-trained; monitor only).
6. Critic: mirror calibration worst in the low cells (pred 0.1 → actual 0.21;
   Brier 0.200 mirror vs 0.183 vs-heuristic) — the parity underweighting
   persists on-policy.

## Decisions

- Proposed (pending go): **deckout_stack** scenario (lib ~4-14, both clocks
  live, agent hand seeded with one draw spell, natural terminator, v1.3 punish
  mode supplies the pressure) and **fof_pick weight 0.3 → 0.6**.
- Rejected by Joseph: convert_the_lead ("not necessary" — the omission class is
  already halving under general training). Retracted: vision_charm_call.

## Lessons

- **Check the mechanics before narrating.** Two of my first-pass reads were
  wrong: Vision Charm's land-type change is symmetric and one-turn (that's the
  card, not a misuse), and mills act on the *shared* library — "mills self" is
  just the mode choice, there is no self/opponent mill targeting. The −0.070
  delta is most plausibly selection bias (wipe cast when behind on board).
- **1.3's deckout kills are a mode, not noise.** Distinguish "agent suicides at
  0 cards" (degenerate, rare) from "agent commits a draw spell to the stack at a
  small library and gets chained" — scenarios should target the second.
- **AEC terminal lag bites miners.** `env.agents` stays non-empty for one
  termination cycle after the game ends; scoring that state with the critic
  produces garbage (it never trains on post-terminal states). Use
  `is_terminal(g)` + the real winner for the terminal value. The first 400
  mirror games were re-mined after this fix.
- **Pre-game critic values are noise.** choose_play_order "blunders" (regret
  ~0.35 before hands are drawn) are a critic artifact; don't chase them.
- Sample-size note (Joseph's push): 400 games was too small for per-cell stats;
  at ~6-8 worker-sec/game there is no reason not to mine 2,000+.
