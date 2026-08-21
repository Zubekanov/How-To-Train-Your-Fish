"""critic_view="hands" (2026-08-21): the public-family critic over both full hands."""
from __future__ import annotations

import numpy as np
import torch

from fishrl.data import features as F
from fishrl.data.features import (HANDS_DIM, HANDS_SLOTS, PUB_DIM, PUBLIC_FAMILY, encode_critic_pub,
                                  encode_hands, encode_public, pub_dim_for)
from fishrl.forgetful_fish import engine as E
from fishrl.forgetful_fish.cards import load_decklist
from fishrl.models.estimators import HandsCritic, PublicCritic, make_critic
from fishrl.obs.encoder import CARD_F
from fishrl.train.config import Config
from fishrl.train.train_loop import build_models, config_from_checkpoint


def _game():
    """A dealt game: play order chosen, both openers kept (hands of 7)."""
    g = E.new_multiplayer_game(load_decklist(), p1_name="p1", p2_name="p2", seed=3)
    E.choose_play_order(g, g.pending.player, "first")
    for _ in range(2):
        E.mulligan_decision(g, g.pending.player, "keep")
    assert len(g.players["p1"].hand) == 7 and len(g.players["p2"].hand) == 7
    return g


def test_hands_layout_and_dims():
    assert "library" not in HANDS_SLOTS and HANDS_SLOTS["stack"] == 6
    assert HANDS_DIM == sum(HANDS_SLOTS.values()) * CARD_F + 12 * 2 + 5 + 3
    assert HANDS_DIM < PUB_DIM
    assert pub_dim_for("hands") == HANDS_DIM and pub_dim_for("public") == PUB_DIM
    assert set(PUBLIC_FAMILY) == {"public", "hands"}


def test_encode_hands_sees_both_hands_and_is_gated():
    g = _game()
    F.set_public_encoding(True)
    x = encode_hands(g)
    assert x.shape == (HANDS_DIM,) and x.dtype == np.float32
    rows = x[:sum(HANDS_SLOTS.values()) * CARD_F].reshape(-1, CARD_F)
    # opening hands are hidden to the public view but present here (7 rows each)
    assert (np.abs(rows[:12]).sum(axis=1) > 0).sum() == len(g.players["p1"].hand) == 7
    assert (np.abs(rows[12:24]).sum(axis=1) > 0).sum() == len(g.players["p2"].hand) == 7
    # the public view carries the same 7 rows as "unknown card" placeholders: no
    # name identity (first N_NAMES columns zero); the hands view names every card
    from fishrl.obs import vocab as V
    pub = encode_public(g)
    prow = pub[:12 * CARD_F].reshape(12, CARD_F)
    assert (np.abs(prow[:7, :V.N_NAMES]).sum(axis=1) > 0).sum() == 0
    assert (np.abs(rows[:7, :V.N_NAMES]).sum(axis=1) > 0).sum() == 7
    # same per-player / game / clock tail as public
    assert np.allclose(x[-(12 * 2 + 5 + 3):], pub[-(12 * 2 + 5 + 3):])
    F.set_public_encoding(False)
    try:
        assert not np.abs(encode_hands(g)).any()
    finally:
        F.set_public_encoding(True)


def test_public_view_dispatch():
    g = _game()
    F.set_public_encoding(True)
    F.set_public_view("public")
    assert encode_critic_pub(g).shape == (PUB_DIM,)
    F.set_public_view("hands")
    try:
        assert encode_critic_pub(g).shape == (HANDS_DIM,)
        assert F.public_view() == "hands"
    finally:
        F.set_public_view("public")
    try:
        F.set_public_view("god"); assert False, "god is not a public-family view"
    except ValueError:
        pass


def test_make_critic_hands_and_config_roundtrip():
    c = make_critic("hands", (64, 64), "entity", 16)
    assert isinstance(c, HandsCritic) and isinstance(c, PublicCritic)
    x = torch.zeros(2, HANDS_DIM)
    v, aux = c.forward_with_aux(x)
    assert v.shape == (2,) and aux.shape == (2,)
    cfg = Config(critic_view="hands", belief_mode="bookkeeper", critic_encoder="entity",
                 actor_encoder="entity", encoder="flat", device="cpu")
    assert not cfg.has_public                                     # no separate public estimator
    m = build_models(cfg)
    assert isinstance(m.critic, HandsCritic) and m.public is None and m.guesser is None
    cd = {"seed": 0, "encoders": {"actor": "entity", "critic": "entity"}, "use_belief": True,
          "critic_hidden": [64, 64], "hidden": [64], "actor_hidden": [64], "card_dim": 16,
          "belief_mode": "bookkeeper", "critic_view": "hands", "critic_deckout_aux": 0.1,
          "text_change_mode": "guided"}
    cfg2 = config_from_checkpoint(cd)
    assert cfg2.critic_view == "hands"
    assert isinstance(build_models(cfg2).critic, HandsCritic)
