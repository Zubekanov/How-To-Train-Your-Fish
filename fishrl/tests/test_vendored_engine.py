"""The vendored rules core loads its deck and plays full games unchanged."""
from fishrl.forgetful_fish.cards import load_decklist
from fishrl.opponents.heuristic import HeuristicMatch
from fishrl.opponents.random_masked import RandomMaskedPolicy


def test_decklist_loads():
    deck = load_decklist()
    assert len(deck) == 21
    assert sum(int(c.get("qty") or 1) for c in deck) == 80


def test_random_vs_heuristic_full_games_terminate():
    # A real (random-but-legal) p1 versus the vendored heuristic AI reaches a
    # terminal result every game — exercising the engine end-to-end via fishrl.
    pol = RandomMaskedPolicy(seed=2)
    for seed in range(5):
        m = HeuristicMatch()
        obs = m.reset(seed=seed)
        done = False
        guard = 0
        while not done and guard < 20000:
            guard += 1
            obs, _, done, info = m.step(pol.act(obs))
        assert m.g.result["status"] != "ongoing"
