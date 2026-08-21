"""Config.obs_counts (2026-08-22): the per-name count block on the actor input and the
hands critic, and the in-place widening of a live checkpoint (function-identical seam)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from fishrl.data import features as F
from fishrl.forgetful_fish import engine as E
from fishrl.forgetful_fish.cards import load_decklist
from fishrl.models.policy import ACTOR_IN, actor_in
from fishrl.obs import vocab as V
from fishrl.train import checkpoint as ckpt
from fishrl.train.config import Config
from fishrl.train.train_loop import _load_model_state, build_models, config_from_checkpoint


@pytest.fixture(autouse=True)
def _reset():
    yield
    F.set_count_block(False)


def _game(seed=3):
    g = E.new_multiplayer_game(load_decklist(), p1_name="p1", p2_name="p2", seed=seed)
    E.choose_play_order(g, g.pending.player, "first")
    for _ in range(2):
        E.mulligan_decision(g, g.pending.player, "keep")
    for _ in range(10):                                        # some graveyard
        g.graveyard.append(g.library.pop(0).instance_id)
    return g


def test_block_contents_and_dims():
    g = _game()
    F.set_public_encoding(True)
    F.set_count_block(False)
    b0 = F.bookkeeper_counts(g, "p1"); h0 = F.encode_hands(g)
    assert b0.shape == (V.N_NAMES,) and h0.shape == (F.HANDS_DIM,)
    F.set_count_block(True)
    assert F.belief_dim() == V.N_NAMES + F.COUNT_DIM and actor_in() == ACTOR_IN + F.COUNT_DIM
    assert F.hands_dim() == F.HANDS_DIM + F.COUNT_DIM and F.pub_dim_for("hands") == F.hands_dim()
    b1 = F.bookkeeper_counts(g, "p1"); h1 = F.encode_hands(g)
    assert b1.shape == (V.N_NAMES + F.COUNT_DIM,) and h1.shape == (F.hands_dim(),)
    assert np.array_equal(b1[:V.N_NAMES], b0) and np.array_equal(h1[:F.HANDS_DIM], h0)
    # actor block: unseen-by-p1 per name = library + p2's (unknown) hand; integers, sum matches
    unseen = b1[V.N_NAMES:V.N_NAMES + V.N_NAMES]
    assert np.all(unseen == np.round(unseen)) and unseen.sum() == len(g.library) + len(g.players["p2"].hand)
    assert b1[-2] == len(g.graveyard) / 40.0 and b1[-1] == len(g.exile) / 8.0
    # critic block: exact library counts by name
    lib = h1[F.HANDS_DIM:F.HANDS_DIM + V.N_NAMES]
    names = [g.objects[s.instance_id].name for s in g.library]
    for n, i in V.NAME_INDEX.items():
        assert lib[i] == names.count(n)
    assert h1[-2] == len(g.graveyard) / 40.0 and h1[-1] == len(g.exile) / 8.0


def _cfg(counts):
    return Config(critic_view="hands", belief_mode="bookkeeper", critic_encoder="entity",
                  actor_encoder="entity", encoder="flat", device="cpu", critic_hidden=(32, 32),
                  actor_hidden=(32,), card_dim=8, obs_counts=counts)


def test_build_models_sets_toggle_and_widths():
    m = build_models(_cfg(True))
    assert F.count_block_on() and m.actor.in_dim == ACTOR_IN + F.COUNT_DIM
    assert m.critic.enc.globals_dim == F.hands_dim() - m.critic.enc.R * F.CARD_F
    m0 = build_models(_cfg(False))
    assert not F.count_block_on() and m0.actor.in_dim == ACTOR_IN
    cd = {"seed": 0, "encoders": {"actor": "entity", "critic": "entity"}, "use_belief": True,
          "critic_hidden": [32, 32], "hidden": [32], "actor_hidden": [32], "card_dim": 8,
          "belief_mode": "bookkeeper", "critic_view": "hands", "critic_deckout_aux": 0.1,
          "text_change_mode": "guided", "obs_counts": True}
    assert config_from_checkpoint(cd).obs_counts is True
    assert config_from_checkpoint({k: v for k, v in cd.items() if k != "obs_counts"}).obs_counts is False


def test_widen_is_function_identical_and_round_trips(tmp_path):
    from fishrl.train.widen_counts import widen
    cfg = _cfg(False)
    torch.manual_seed(1)
    m = build_models(cfg)
    frozen = build_models(cfg)
    opt = torch.optim.Adam(list(m.actor.parameters()) + list(m.critic.parameters()), lr=1e-3)
    # one optimizer step so Adam has moments to widen
    x = torch.randn(2, ACTOR_IN); xc = torch.randn(2, F.HANDS_DIM)
    (m.actor(x).sum() + m.critic(xc).sum()).backward(); opt.step()
    from fishrl.train.train_loop import _model_state
    league = {"anchors": [], "selves": [{"name": "self#1", "wr": 0.5, "games": 3,
                                          "actor": m.actor.state_dict()}]}
    payload = {"format": ckpt.FORMAT,
               "config": {"seed": 0, "encoders": {"actor": "entity", "critic": "entity"},
                          "use_belief": True, "critic_hidden": [32, 32], "hidden": [32],
                          "actor_hidden": [32], "card_dim": 8, "belief_mode": "bookkeeper",
                          "critic_view": "hands", "critic_deckout_aux": 0.1,
                          "text_change_mode": "guided"},
               "done": 123, "elapsed": 1.0, "frozen_it": 100, "handoff_start": 0, "warmup_done": True,
               "models": _model_state(m), "frozen": _model_state(frozen),
               "optim": {"ppo": opt.state_dict()}, "rng": None, "league": league, "scen_league": None}
    path = str(tmp_path / "latest.pt")
    ckpt.save_checkpoint(path, payload)
    assert widen(path)["handoff_start"] == 0                          # default: anchor untouched
    out = widen(path, rearm_kl=True)
    assert out["config"]["obs_counts"] is True and out["handoff_start"] == 123 and out["done"] == 123
    cfg2 = config_from_checkpoint(out["config"], device="cpu")
    m2 = build_models(cfg2)
    _load_model_state(m2, out["models"])
    xw = torch.cat([x, torch.randn(2, F.COUNT_DIM)], dim=1)
    xcw = torch.cat([xc, torch.randn(2, F.COUNT_DIM)], dim=1)
    with torch.no_grad():
        assert torch.allclose(m.actor(x), m2.actor(xw), atol=1e-6)
        assert torch.allclose(m.critic(xc), m2.critic(xcw), atol=1e-6)
        v, aux = m.critic.forward_with_aux(xc); v2, aux2 = m2.critic.forward_with_aux(xcw)
        assert torch.allclose(aux, aux2, atol=1e-6)
    opt2 = torch.optim.Adam(list(m2.actor.parameters()) + list(m2.critic.parameters()), lr=1e-3)
    opt2.load_state_dict(out["optim"]["ppo"])                       # shapes line up
    assert out["league"]["selves"][0]["actor"]["net.0.weight"].shape[1] == m2.actor.net[0].in_features
    assert out["league"]["selves"][0]["wr"] == 0.5
