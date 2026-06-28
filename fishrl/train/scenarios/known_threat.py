"""Scenario #3 — known-threat denial (the plan's strongest; hand-author/surgery half).

A known card-advantage card (Accumulated Knowledge) is placed on TOP of the shared
library; p2 is handed to the engine heuristic, which draws and casts it on its turn.
p1 holds a Memory Lapse (and the lands for it) and must deny the threat — counter
it, mill it, or draw it first. One opponent turn: if the threat RESOLVES for p2, p1
loses; if it's denied, p1 wins. Tight, unambiguous credit.

Construction is hybrid: snapshot a real p1-main state where the surgery is possible
(both cards still in the library, ≥3 untapped Islands for p1, ≥2 lands for p2), then
relocate existing instances — no fabricated cards. p2's own copies of the threat are
cleared first so the only one it can land is the one drawn off the top.
"""
from __future__ import annotations

from fishrl.train.scenarios.base import Scenario
from fishrl.train.scenarios.surgery import (
    basics, clear_hand_of, find_library_slot, make_engine_heuristic,
    move_library_card_to_hand, move_library_card_to_top, untapped_basics,
)

THREAT = "Dandân"                  # the opponent's win-condition; the heuristic casts it on sight
TOOL = "Memory Lapse"              # the denial the learner is handed


def _p2_threats_on_bf(g) -> int:
    return sum(1 for iid in g.players["p2"].battlefield
               if iid in g.objects and g.objects[iid].name == THREAT)


class KnownThreatScenario(Scenario):
    name = "known_threat"
    pool_seed = 303
    pool_max_games = 9000
    pool_max_decisions = 500

    def predicate(self, env) -> bool:
        g = env.g
        p = g.pending
        return (p is not None and p.player == "p1" and p.type == "priority"
                and g.current_step in ("main1", "main2")
                and 3 <= g.turn_number <= 8
                and untapped_basics(g, "p1", "Island") >= 3   # p1 can hold up the counter
                and basics(g, "p2", "Island") >= 2            # p2 can pay for the threat
                and find_library_slot(g, THREAT) is not None
                and find_library_slot(g, TOOL) is not None)

    def sample(self, rng):
        g = super().sample(rng)                      # legal neutral p1-main snapshot
        clear_hand_of(g, THREAT, "p2")               # the only threat is the one drawn off top
        move_library_card_to_top(g, THREAT)          # the known incoming threat
        move_library_card_to_hand(g, TOOL, "p1")     # hand the learner its denial
        make_engine_heuristic(g, "p2")               # p2 will draw + cast the threat
        return g

    def on_reset(self, env) -> None:
        # Baseline count of the threat already on p2's battlefield; an INCREASE means
        # p2 resolved a fresh one (the one it drew off the top) -> the threat landed.
        env.scn_ctx["p2_threats0"] = _p2_threats_on_bf(env.g)
        env.scn_ctx["p2_turn_seen"] = False

    def terminator(self, env):
        g = env.g
        c = env.scn_ctx
        # Resolved: p2 landed a fresh threat on the battlefield (a counter sends it to
        # the library instead, so the count never rises) -> loss.
        if _p2_threats_on_bf(g) > c["p2_threats0"]:
            return "p2"
        # Horizon by active-player transition (robust to turn-number semantics): p2's
        # whole turn elapses, then control returns to p1 with no threat landed
        # (countered / milled / drawn by p1) -> denied -> win.
        if g.active_player == "p2":
            c["p2_turn_seen"] = True
        if c["p2_turn_seen"] and g.active_player == "p1":
            return "p1"
        return None
