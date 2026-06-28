"""Scenario #1 — free attack / lethal-close (the unambiguous-correct exemplar).

Start AT p1's declare-attackers step with an eligible attacker and a TRULY empty
opponent board (zero p2 creatures — no blocker, no swing-back, so attacking is
unambiguously right; this is the gating I flagged for the tap-to-block nuance).
Win = deal damage (take the attack). Loss = let combat pass with the attacker
unused (the deliberate hard-rule). Terminates within the turn — tight credit.
"""
from __future__ import annotations

from fishrl.forgetful_fish import engine as E
from fishrl.train.scenarios.base import Scenario


def _p2_creatures(g) -> int:
    return sum(1 for iid in g.players["p2"].battlefield
               if iid in g.objects and E._is_creature(g.objects[iid]))


class FreeAttackScenario(Scenario):
    name = "free_attack"
    pool_seed = 101

    def predicate(self, env) -> bool:
        g = env.g
        p = g.pending
        return (p is not None and p.type == "declare_attackers" and p.player == "p1"
                and len(p.context.get("eligible", [])) >= 1
                and _p2_creatures(g) == 0)            # truly empty board, unpunished

    def on_reset(self, env) -> None:
        env.scn_ctx["p2_life0"] = env.g.players["p2"].life

    def terminator(self, env):
        g = env.g
        if g.players["p2"].life < env.scn_ctx["p2_life0"]:
            return "p1"                               # dealt damage -> took the free attack
        # p1's combat resolved with no damage -> the attacker went unused -> loss.
        if g.current_step in ("main2", "end", "cleanup") or g.active_player != "p1":
            return "p2"
        return None
