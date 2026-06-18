"""The scripted attacker opponent plays only legal moves, terminates games, and is a
sane aggressive baseline (clearly beats random)."""
import numpy as np

from fishrl.env.aec_env import FishAEC
from fishrl.opponents.attacker import attacker_action


def _play(p1, p2, n, base):
    """p1/p2 are (g, seat, mask, rng) -> action. Returns p1 win-rate over n games.
    A finished game proves legality: FishAEC raises on any illegal atomic action."""
    wins = 0
    for i in range(n):
        env = FishAEC(max_decisions=2000)
        env.reset(seed=base + i)
        rng = np.random.default_rng(base + i)
        steps = 0
        while env.agents and steps < 12000:
            s = env.agent_selection
            if env.terminations[s] or env.truncations[s]:
                env.step(None)
                continue
            mask = env.observe(s)["action_mask"]
            env.step((p1 if s == "p1" else p2)(env.g, s, mask, rng))
            steps += 1
        assert env.g.result.get("status") != "ongoing", "game did not terminate"
        assert steps < 4000, "attacker game ran absurdly long (stall?)"
        wins += int(env.g.result.get("winner") == "p1")
    return wins / n


def _rand(g, s, mask, rng):
    return int(rng.choice(np.flatnonzero(mask)))


def test_attacker_plays_legally_and_terminates_vs_random():
    wr = _play(attacker_action, _rand, 20, 0)
    assert wr >= 0.6, f"attacker should clearly beat random (got {wr})"


def test_attacker_mirror_terminates():
    _play(attacker_action, attacker_action, 10, 500)   # no stall in the mirror
