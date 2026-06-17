"""Sign-convention guard: the shared critic is zero-sum-consistent and seat
outcomes/returns carry the correct signs. A half-flipped sign would silently
train p2 to lose, so this is the most important correctness test."""
import numpy as np

from fishrl.train.advantages import (
    gae, p1_winprob_to_seat_value, seat_outcome,
)


def test_seat_value_zero_sum():
    for p in (0.0, 0.3, 0.5, 0.8, 1.0):
        v1 = p1_winprob_to_seat_value(p, "p1")
        v2 = p1_winprob_to_seat_value(p, "p2")
        assert abs(v1 + v2) < 1e-6           # V_p1 == -V_p2 for the same state
    # a confident p1 win is +1 for p1, -1 for p2
    assert p1_winprob_to_seat_value(1.0, "p1") == 1.0
    assert p1_winprob_to_seat_value(1.0, "p2") == -1.0


def test_seat_outcome_signs():
    assert seat_outcome("p2", "p2") == 1.0
    assert seat_outcome("p2", "p1") == -1.0
    assert seat_outcome("p1", "p1") == 1.0
    assert seat_outcome(None, "p1") == 0.0   # draw / truncation


def test_gae_terminal_only_reward_matches_outcome():
    # With gamma=1, lambda=1 and a perfect zero value baseline, the return at every
    # decision equals the seat's terminal outcome.
    T = 5
    values = np.zeros(T, dtype=np.float32)
    rewards = np.zeros(T, dtype=np.float32)
    rewards[-1] = seat_outcome("p2", "p2")   # this seat won -> +1
    adv, ret = gae(values, rewards, gamma=1.0, lam=1.0)
    assert np.allclose(ret, 1.0)
    # a losing seat gets -1 returns throughout
    rewards[-1] = seat_outcome("p1", "p2")   # winner p1, seat p2 -> -1
    _, ret_lose = gae(values, rewards, gamma=1.0, lam=1.0)
    assert np.allclose(ret_lose, -1.0)


def test_critic_consistency_on_real_state():
    """V_p1(g) == -V_p2(g) using the actual privileged critic on a real state."""
    import torch
    from fishrl.forgetful_fish import engine as E
    from fishrl.forgetful_fish.cards import load_decklist
    from fishrl.data.features import encode_god
    from fishrl.models.estimators import PrivilegedCritic

    g = E.new_multiplayer_game(load_decklist(), p1_name="p1", p2_name="p2", seed=1)
    E.choose_play_order(g, g.pending.player, "first")
    while g.pending is not None and g.pending.type == "mulligan":
        E.mulligan_decision(g, g.pending.player, "keep")
    critic = PrivilegedCritic()
    with torch.no_grad():
        p_p1 = float(critic.p1_winprob(torch.as_tensor(encode_god(g)).unsqueeze(0))[0])
    v1 = p1_winprob_to_seat_value(p_p1, "p1")
    v2 = p1_winprob_to_seat_value(p_p1, "p2")
    assert abs(v1 + v2) < 1e-6
