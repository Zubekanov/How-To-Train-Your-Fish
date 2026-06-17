"""The observation has fixed size and never leaks hidden card identities."""
import numpy as np

from fishrl.forgetful_fish import engine as E
from fishrl.forgetful_fish.cards import load_decklist
from fishrl.obs import vocab as V
from fishrl.obs.encoder import CARD_F, OBS_DIM, SLOTS, encode_observation


def _opened_game(seed=0):
    """A game driven through the opening so both seats hold a 7-card hand."""
    g = E.new_multiplayer_game(load_decklist(), p1_name="p1", p2_name="p2", seed=seed)
    E.choose_play_order(g, g.pending.player, "first")
    guard = 0
    while g.pending is not None and g.pending.type == "mulligan" and guard < 10:
        guard += 1
        E.mulligan_decision(g, g.pending.player, "keep")
    return g


def test_obs_fixed_size_and_finite():
    g = _opened_game()
    obs = encode_observation(g, "p1")
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))


def test_no_opponent_hand_identity_leak():
    g = _opened_game()
    # p1's view of p2's hand must be all-hidden at the opening.
    obs = encode_observation(g, "p1")
    opp_hand_offset = SLOTS["own_hand"] * CARD_F      # opp_hand is the 2nd zone block
    n_opp = len(g.players["p2"].hand)
    assert n_opp > 0
    for r in range(n_opp):
        row = obs[opp_hand_offset + r * CARD_F: opp_hand_offset + (r + 1) * CARD_F]
        name_onehot = row[:V.N_NAMES]
        unknown_bit = row[V.N_NAMES]
        known_bit = row[CARD_F - 1]
        assert name_onehot.sum() == 0.0, "leaked an opponent hand card's identity"
        assert unknown_bit == 1.0
        assert known_bit == 0.0


def test_own_hand_is_visible():
    g = _opened_game()
    obs = encode_observation(g, "p1")
    n = len(g.players["p1"].hand)
    assert n > 0
    # own hand cards should be known (identity present)
    for r in range(n):
        row = obs[r * CARD_F:(r + 1) * CARD_F]
        assert row[:V.N_NAMES].sum() == 1.0
        assert row[CARD_F - 1] == 1.0          # known bit


def test_determinism_same_seed():
    a = encode_observation(_opened_game(3), "p1")
    b = encode_observation(_opened_game(3), "p1")
    assert np.array_equal(a, b)
