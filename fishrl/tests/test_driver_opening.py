"""The opening (roll / play-order / mulligan) is driven via the same action path,
and the acting seat is not assumed to alternate (the roll winner opens)."""
import numpy as np

from fishrl.selfplay.pettingzoo_api import raw_env


def test_opening_first_decision_is_play_order():
    e = raw_env()
    e.reset(seed=0)
    assert e.g.pending is not None
    assert e.g.pending.type == "choose_play_order"
    assert e.agent_selection == e.g.pending.player


def test_roll_winner_varies_across_seeds():
    first = set()
    for seed in range(30):
        e = raw_env()
        e.reset(seed=seed)
        first.add(e.agent_selection)
    assert first == {"p1", "p2"}, "roll winner should sometimes be either seat"


def test_reaches_first_turn_or_terminal():
    rng = np.random.default_rng(1)
    e = raw_env()
    e.reset(seed=2)
    steps = 0
    while e.agents and e.g.turn_number < 1 and steps < 2000:
        a = e.agent_selection
        if e.terminations[a] or e.truncations[a]:
            e.step(None)
            continue
        e.step(int(rng.choice(np.flatnonzero(e.observe(a)["action_mask"]))))
        steps += 1
    assert e.g.turn_number >= 1 or e.g.result["status"] != "ongoing"
