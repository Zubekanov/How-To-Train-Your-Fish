"""Scenario #2 — condensed board fight (the pivotal-creature interaction).

Both seats start with a creature on board (mid-game), p2 driven by the heuristic.
The pivotal element is board presence: the FIRST seat to an empty board loses
(casting creatures is allowed — you're meant to protect/replace the load-bearing
creature, not hoard it). This directly drills the attrition pattern the diagnosis
found the agent losing: its Dandâns get removed (land-type changers / combat) and
it fails to maintain a board.

Why-it-matters is varied by snapshot diversity (different life totals, hands, and
board states across instances) rather than hard-coded, so the agent learns
"protect the element load-bearing for the win," not "creatures are sacred."
"""
from __future__ import annotations

from fishrl.forgetful_fish import engine as E
from fishrl.train.scenarios.base import Scenario
from fishrl.train.scenarios.surgery import make_engine_heuristic

HORIZON = 4          # p1 turn-numbers before the fight is scored on remaining board


def _creatures(g, seat) -> int:
    return sum(1 for iid in g.players[seat].battlefield
               if iid in g.objects and E._is_creature(g.objects[iid]))


class BoardPresenceScenario(Scenario):
    name = "board_presence"
    pool_seed = 404

    def predicate(self, env) -> bool:
        g = env.g
        p = g.pending
        return (p is not None and p.player == "p1" and p.type == "priority"
                and g.current_step in ("main1", "main2")
                and 4 <= g.turn_number <= 9
                and _creatures(g, "p1") >= 1 and _creatures(g, "p2") >= 1)

    def sample(self, rng):
        g = super().sample(rng)
        make_engine_heuristic(g, "p2")
        return g

    def on_reset(self, env) -> None:
        env.scn_ctx["turn0"] = env.g.turn_number

    def terminator(self, env):
        g = env.g
        c = env.scn_ctx
        cp1, cp2 = _creatures(g, "p1"), _creatures(g, "p2")
        # First to an empty board loses the pivotal-creature fight.
        if cp1 == 0 and cp2 > 0:
            return "p2"
        if cp2 == 0 and cp1 > 0:
            return "p1"
        if cp1 == 0 and cp2 == 0:
            return "draw"
        # Horizon: whoever holds the larger board after a few turns has won the fight.
        if g.turn_number - c["turn0"] >= HORIZON:
            return "p1" if cp1 > cp2 else ("p2" if cp2 > cp1 else "draw")
        return None
