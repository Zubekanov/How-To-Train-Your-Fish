"""Scenario #5 — establish a clock from a neutral position (the best-justified one).

Start from an even/neutral position (early turn, near-equal life, ≤1 creature each)
and play a SHORT race: at the horizon, the seat that has dealt more damage wins.
This is a STRUCTURAL win condition (out-race the opponent), NOT a hand-coded
"passivity = loss" rule — so it teaches "establish pressure" without encoding a
strategic judgment into termination. A natural lethal before the horizon just wins
normally (terminator returns None until the horizon).
"""
from __future__ import annotations

from fishrl.forgetful_fish import engine as E
from fishrl.train.scenarios.base import Scenario

HORIZON = 5          # turn-numbers from the start before the race is scored


def _creatures(g, seat) -> int:
    return sum(1 for iid in g.players[seat].battlefield
               if iid in g.objects and E._is_creature(g.objects[iid]))


class EstablishClockScenario(Scenario):
    name = "establish_clock"
    pool_seed = 202

    def predicate(self, env) -> bool:
        g = env.g
        p = g.pending
        return (p is not None and p.player == "p1" and p.type == "priority"
                and g.current_step in ("main1", "main2")
                and 3 <= g.turn_number <= 7
                and abs(g.players["p1"].life - g.players["p2"].life) <= 2
                and _creatures(g, "p1") <= 1 and _creatures(g, "p2") <= 1)

    def on_reset(self, env) -> None:
        g = env.g
        env.scn_ctx["p1_life0"] = g.players["p1"].life
        env.scn_ctx["p2_life0"] = g.players["p2"].life
        env.scn_ctx["turn0"] = g.turn_number

    def terminator(self, env):
        g = env.g
        c = env.scn_ctx
        if g.turn_number - c["turn0"] < HORIZON:
            return None                               # before horizon: natural game-end applies
        d_to_p2 = c["p2_life0"] - g.players["p2"].life     # damage the learner (p1) dealt
        d_to_p1 = c["p1_life0"] - g.players["p1"].life     # damage p1 took
        if d_to_p2 > d_to_p1:
            return "p1"
        if d_to_p1 > d_to_p2:
            return "p2"
        return "draw"
