"""How do board_presence / deckout games actually END?

board_presence is meant to be a tight stack fight over the one pivotal creature (both seats:
4 life, one 4/1 Dandan, equal lands, 7-card grip; every OTHER creature exiled). But Dandan is
4/1: if the agent attacks, the heuristic blocks and BOTH die -- and the library is creatureless,
so the game "simply continues to a deckout" (the scenario's own docstring).

If that is what usually happens, board_presence is not a second failure -- it is the SAME
deckout failure, measured twice (their win-rates are ~identical: 0.067 vs 0.074, both ~0.4x the
agent's 0.186 baseline vs the same opponent, heuristic v1.2). That would mean one target, not
two, and that the ~16% of training games split across both are buying the same lesson.

Classifies every game by its terminal cause and reports it split by win/loss.

    python -m fishrl.eval.probe_scenario_endings --games 120
"""
from __future__ import annotations

import argparse
import collections

from fishrl.train.belief_env import BeliefAugmentedEnv
from fishrl.train.checkpoint import load_checkpoint
from fishrl.train.collector import actor_act_fn
from fishrl.train.config import Config, resolve_device
from fishrl.train.scenarios import ScenarioEnv, get_scenario
from fishrl.train.train_loop import build_models, config_from_checkpoint, _load_model_state

NETS = ("actor", "critic", "guesser", "public")
SEAT = "p1"                                  # the learner; p2 is the engine heuristic (v1.2)


def _ending(g) -> str:
    """Why did this game end? Deckout is the endgame axis; life is the combat axis.

    Read the reason off g.result, NOT PlayerState.loss_reason: engine._lose() sets has_lost
    and writes the reason into g.result, but never populates p.loss_reason -- the field exists
    (and current_view even exposes it) yet is always None. Reading it silently mislabelled
    every deckout as 'other'."""
    res = g.result or {}
    reason = (res.get("reason") or "").lower()
    winner = res.get("winner")
    if winner not in ("p1", "p2"):
        return "unresolved (cap/draw)"
    who = "bot" if winner == SEAT else "agent"          # the LOSER
    if "empty library" in reason:
        return f"DECKOUT ({who} decked)"
    if "life" in reason:
        return f"LIFE ({who} at 0)"
    return f"other:{reason[:28]} ({who})"


def run(scenario_name, models, n_games, seed, max_decisions):
    endings = collections.Counter()
    wins = collections.Counter()
    creature_gone_at = []
    for i in range(n_games):
        senv = BeliefAugmentedEnv(
            models.guesser, belief=True,
            env=ScenarioEnv(get_scenario(scenario_name), max_decisions=max_decisions))
        senv.reset(seed=seed + i)
        act = actor_act_fn(models.actor)
        traded = None
        for agent in senv.agent_iter(max_iter=max_decisions * 6):
            if senv.terminations[agent] or senv.truncations[agent]:
                senv.step(None)
                continue
            g = senv.g
            if traded is None and not any(
                    g.players[s].battlefield and any(
                        iid in g.objects and (g.objects[iid].power or 0) > 0
                        for iid in g.players[s].battlefield)
                    for s in ("p1", "p2")):
                traded = g.turn_number          # both boards creature-free
            obs = senv.observe(agent)
            a, _ = act(obs)
            senv.step(a)
        g = senv.g
        e = _ending(g)
        endings[e] += 1
        won = g.result.get("winner") == SEAT
        wins[e] += int(won)
        if traded is not None:
            creature_gone_at.append(traded)
    return endings, wins, creature_gone_at


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="checkpoints/latest.pt")
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--seed", type=int, default=77_000)
    ap.add_argument("--max-decisions", type=int, default=2000)
    ap.add_argument("--scenarios", default="board_presence,deckout")
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()
    device = resolve_device(args.gpu)

    pl = load_checkpoint(args.ckpt, map_location="cpu")
    cd = pl["config"]
    cfg = config_from_checkpoint(cd)
    m = build_models(cfg)
    _load_model_state(m, pl["models"])
    for net in (m.actor, m.guesser):
        net.eval()
    print(f"[endings] {args.ckpt}: iter={pl.get('done')} elapsed={pl.get('elapsed',0)/3600:.0f}h "
          f"| {args.games} games/scenario vs heuristic v1.2\n", flush=True)

    for name in args.scenarios.split(","):
        endings, wins, gone = run(name.strip(), m, args.games, args.seed, args.max_decisions)
        tot = sum(endings.values())
        wr = sum(wins.values()) / tot if tot else 0.0
        print(f"=== {name.strip()}  ({tot} games, agent win-rate {wr:.3f}) ===")
        for e, c in endings.most_common():
            w = wins[e]
            print(f"  {e:28s} {c:>4} games ({100*c/tot:>5.1f}%)   agent won {w:>3}/{c:<3} "
                  f"({(w/c if c else 0):.2f})")
        if gone:
            print(f"  board went creature-free in {len(gone)}/{tot} games "
                  f"({100*len(gone)/tot:.0f}%), median turn {sorted(gone)[len(gone)//2]}")
        print(flush=True)


if __name__ == "__main__":
    main()
