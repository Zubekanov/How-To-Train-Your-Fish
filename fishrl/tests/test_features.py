"""Privileged/public feature encoders and the guesser label."""
import numpy as np

from fishrl.forgetful_fish import engine as E
from fishrl.forgetful_fish.cards import load_decklist
from fishrl.data import features as F
from fishrl.obs.encoder import CARD_F
from fishrl.obs import vocab as V


def _opened(seed=0):
    g = E.new_multiplayer_game(load_decklist(), p1_name="p1", p2_name="p2", seed=seed)
    E.choose_play_order(g, g.pending.player, "first")
    guard = 0
    while g.pending is not None and g.pending.type == "mulligan" and guard < 10:
        guard += 1
        E.mulligan_decision(g, g.pending.player, "keep")
    return g


def test_dims_fixed_and_finite():
    g = _opened()
    god, pub = F.encode_god(g), F.encode_public(g)
    assert god.shape == (F.GOD_DIM,) and np.all(np.isfinite(god))
    assert pub.shape == (F.PUB_DIM,) and np.all(np.isfinite(pub))


def test_opponent_hand_counts_matches_state():
    g = _opened()
    for viewer in ("p1", "p2"):
        opp = "p2" if viewer == "p1" else "p1"
        counts = F.opponent_hand_counts(g, viewer)
        assert counts.sum() == len(g.players[opp].hand)
        # spot-check one card name count
        from collections import Counter
        truth = Counter(g.objects[i].name for i in g.players[opp].hand)
        for name, c in truth.items():
            assert counts[V.NAME_INDEX[name]] == c


def test_text_change_exposed_in_observation():
    """A Mind-Bended Island (Island->Swamp) shows its EFFECTIVE basic type and the
    text-altered flag in the observation, not just the printed name 'Island'."""
    from fishrl.forgetful_fish import state as S
    from fishrl.obs.encoder import CARD_F, SLOTS, encode_observation

    g = E.new_multiplayer_game(load_decklist(), p1_name="p1", p2_name="p2", seed=2)
    isl = next(i for i, o in g.objects.items() if o.name == "Island")
    g.library = [s for s in g.library if s.instance_id != isl]
    g.players["p1"].battlefield = [isl]
    g.objects[isl].controller = "p1"
    assert S.add_text_change(g.objects[isl], "Island", "Swamp", eot=False, turn=1)
    g.active_player, g.current_step = "p1", "main1"

    obs = encode_observation(g, "p1")
    off = (SLOTS["own_hand"] + SLOTS["opp_hand"]) * CARD_F      # own battlefield, row 0
    row = obs[off:off + CARD_F]
    B = list(S.BASIC_TYPES)
    tl = V.N_NAMES + 1 + 4                                       # start of type-line basics
    assert row[:V.N_NAMES].argmax() == V.NAME_INDEX["Island"]    # printed name unchanged
    assert row[tl + B.index("Swamp")] == 1.0                     # effective type is now Swamp
    assert row[tl + B.index("Island")] == 0.0
    assert row[tl + 10] == 1.0                                   # text-altered flag set


def test_privileged_sees_opp_hand_public_does_not():
    g = _opened()
    n_p2 = len(g.players["p2"].hand)
    assert n_p2 > 0
    god = F.encode_god(g)
    pub = F.encode_public(g)
    off = F.GOD_SLOTS["p1_hand"] * CARD_F          # p2_hand block start (god)
    god_row0 = god[off:off + CARD_F]
    assert god_row0[:V.N_NAMES].sum() == 1.0        # god knows p2's first hand card
    assert god_row0[CARD_F - 1] == 1.0              # known bit

    poff = F.PUB_SLOTS["p1_hand"] * CARD_F          # p2_hand block start (public)
    pub_row0 = pub[poff:poff + CARD_F]
    assert pub_row0[:V.N_NAMES].sum() == 0.0        # public hides p2's hand identity
    assert pub_row0[V.N_NAMES] == 1.0               # unknown bit
