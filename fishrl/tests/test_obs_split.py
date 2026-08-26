"""Config.obs_split (2026-08-26): the FoF split-context block on the actor tail and
the hands critic, the builder's live mirror into the pending context, and the
in-place widening of a live checkpoint (function-identical seam).

Why it exists: at it=122.7k the critic's V was measured EXACTLY flat across the
five PICK toggles of a Fact-or-Fiction split (the revealed five live only in the
pending context; the arrangement lived only in the env-side builder) — neither
net could see the split being made, and the 0-5 degenerate-split rate sat frozen
across three checkpoints while the critic's pricing of the outcome sharpened."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from fishrl.data import features as F
from fishrl.models.policy import ACTOR_IN, actor_in
from fishrl.obs import vocab as V
from fishrl.spaces import action_space as A
from fishrl.train import checkpoint as ckpt
from fishrl.train.config import Config
from fishrl.train.scenarios import ScenarioEnv
from fishrl.train.scenarios.constructed import FofSplit
from fishrl.train.train_loop import _load_model_state, build_models, config_from_checkpoint


@pytest.fixture(autouse=True)
def _reset():
    yield
    F.set_count_block(False)
    F.set_split_block(False)


def _split_env(seed=0):
    """A FofSplit scenario start driven INTO the split: p1 passes, the FoF resolves,
    the env builds the fof_split CompoundBuilder."""
    scn = FofSplit(); scn.pool_n = 6; scn.pool_max_games = 400
    env = ScenarioEnv(scn, max_decisions=2000)
    env.reset(seed=seed)
    env.step(A.aid("PASS", 0))
    assert env.g.pending is not None and env.g.pending.type == "fof_split"
    assert env.g.pending.player == "p1" and env._builder is not None
    return env


def test_block_zero_outside_fof_and_filled_during_split():
    F.set_public_encoding(True)
    F.set_split_block(True)
    env = _split_env()
    g = env.g
    ctx = g.pending.context
    revealed = list(ctx["revealed"])
    blk = F.split_context_block(g, viewer="p1")
    assert blk.shape == (F.SPLIT_DIM,)
    assert blk[0] == 1.0 and blk[1] == blk[2] == blk[3] == 0.0     # me-splits (viewer p1)
    crit = F.split_context_block(g)                                # p1-oriented (critic)
    assert crit[0] == 1.0
    # before any pick: everything unassigned, piles empty
    n = V.N_NAMES
    assert blk[4:4 + n].sum() == 0 and blk[4 + n:4 + 2 * n].sum() == 0
    assert blk[4 + 2 * n:].sum() == len(revealed) == 5
    # the unassigned counts are the revealed cards by name
    names = [g.objects[i].name for i in revealed]
    for nm, idx in V.NAME_INDEX.items():
        assert blk[4 + 2 * n + idx] == names.count(nm)

    # one PICK_A and one PICK_B: builder mirrors into the context, block follows
    mask = env.observe("p1")["action_mask"]
    a_ids = [A.aid("PICK_A", i) for i in range(5) if mask[A.aid("PICK_A", i)]]
    env.step(a_ids[0])
    b1 = F.split_context_block(g, viewer="p1")
    assert b1[4:4 + n].sum() == 1 and b1[4 + 2 * n:].sum() == 4     # 1 to pile1, 4 open
    mask = env.observe("p1")["action_mask"]
    b_ids = [A.aid("PICK_B", i) for i in range(5) if mask[A.aid("PICK_B", i)]]
    env.step(b_ids[0])
    b2 = F.split_context_block(g, viewer="p1")
    assert b2[4:4 + n].sum() == 1 and b2[4 + n:4 + 2 * n].sum() == 1 and b2[4 + 2 * n:].sum() == 3
    # the hands critic feature rows now DIFFER across toggles (the flat-V bug's fix)
    h1 = F.encode_hands(g)
    assert h1.shape == (F.hands_dim(),)
    assert not np.array_equal(h1[-F.SPLIT_DIM:], np.zeros(F.SPLIT_DIM))
    assert F.split_live(g)

    # outside any FoF resolution: all zeros, and bookkeeper/hands keep their prefix
    env2 = ScenarioEnv(FofSplit(), max_decisions=2000)  # fresh reset, FoF still ON STACK
    env2.scenario.pool_n = 6
    env2.reset(seed=1)
    g2 = env2.g
    assert not F.split_live(g2)                          # on the stack != resolving
    assert F.split_context_block(g2).sum() == 0.0
    F.set_split_block(False)
    b0 = F.bookkeeper_counts(g2, "p1"); h0 = F.encode_hands(g2)
    F.set_split_block(True)
    b1 = F.bookkeeper_counts(g2, "p1"); h1 = F.encode_hands(g2)
    assert b1.shape == (b0.shape[0] + F.SPLIT_DIM,) and np.array_equal(b1[:b0.shape[0]], b0)
    assert h1.shape == (h0.shape[0] + F.SPLIT_DIM,) and np.array_equal(h1[:h0.shape[0]], h0)
    assert b1[-F.SPLIT_DIM:].sum() == 0 and h1[-F.SPLIT_DIM:].sum() == 0


def test_dims_and_config():
    F.set_count_block(True); F.set_split_block(True)
    assert F.belief_dim() == V.N_NAMES + F.COUNT_DIM + F.SPLIT_DIM
    assert actor_in() == ACTOR_IN + F.COUNT_DIM + F.SPLIT_DIM
    assert F.hands_dim() == F.HANDS_DIM + F.COUNT_DIM + F.SPLIT_DIM
    cd = {"seed": 0, "encoders": {"actor": "entity", "critic": "entity"}, "use_belief": True,
          "critic_hidden": [32, 32], "hidden": [32], "actor_hidden": [32], "card_dim": 8,
          "belief_mode": "bookkeeper", "critic_view": "hands", "critic_deckout_aux": 0.1,
          "text_change_mode": "guided", "obs_counts": True, "obs_split": True}
    assert config_from_checkpoint(cd).obs_split is True
    assert config_from_checkpoint({k: v for k, v in cd.items() if k != "obs_split"}).obs_split is False


def _cfg(split):
    return Config(critic_view="hands", belief_mode="bookkeeper", critic_encoder="entity",
                  actor_encoder="entity", encoder="flat", device="cpu", critic_hidden=(32, 32),
                  actor_hidden=(32,), card_dim=8, obs_counts=True, obs_split=split)


def test_build_models_sets_toggle_and_requires_counts():
    m = build_models(_cfg(True))
    assert F.split_block_on() and m.actor.in_dim == ACTOR_IN + F.COUNT_DIM + F.SPLIT_DIM
    assert m.critic.enc.globals_dim == F.hands_dim() - m.critic.enc.R * F.CARD_F
    m0 = build_models(_cfg(False))
    assert not F.split_block_on() and m0.actor.in_dim == ACTOR_IN + F.COUNT_DIM
    with pytest.raises(AssertionError):
        build_models(Config(critic_view="hands", belief_mode="bookkeeper",
                            critic_encoder="entity", actor_encoder="entity", encoder="flat",
                            device="cpu", critic_hidden=(32, 32), actor_hidden=(32,),
                            card_dim=8, obs_counts=False, obs_split=True))


def test_widen_split_is_function_identical_and_round_trips(tmp_path):
    from fishrl.train.widen_split import widen
    cfg = _cfg(False)
    torch.manual_seed(1)
    m = build_models(cfg)                                  # counts ON, split off
    frozen = build_models(cfg)
    opt = torch.optim.Adam(list(m.actor.parameters()) + list(m.critic.parameters()), lr=1e-3)
    xin = ACTOR_IN + F.COUNT_DIM
    xcin = F.HANDS_DIM + F.COUNT_DIM
    x = torch.randn(2, xin); xc = torch.randn(2, xcin)
    (m.actor(x).sum() + m.critic(xc).sum()).backward(); opt.step()
    from fishrl.train.train_loop import _model_state
    league = {"anchors": [], "selves": [{"name": "self#1", "wr": 0.5, "games": 3,
                                          "actor": m.actor.state_dict()}]}
    payload = {"format": ckpt.FORMAT,
               "config": {"seed": 0, "encoders": {"actor": "entity", "critic": "entity"},
                          "use_belief": True, "critic_hidden": [32, 32], "hidden": [32],
                          "actor_hidden": [32], "card_dim": 8, "belief_mode": "bookkeeper",
                          "critic_view": "hands", "critic_deckout_aux": 0.1,
                          "text_change_mode": "guided", "obs_counts": True},
               "done": 123, "elapsed": 1.0, "frozen_it": 100, "handoff_start": 0, "warmup_done": True,
               "models": _model_state(m), "frozen": _model_state(frozen),
               "optim": {"ppo": opt.state_dict()}, "rng": None, "league": league, "scen_league": None}
    path = str(tmp_path / "latest.pt")
    ckpt.save_checkpoint(path, payload)
    assert widen(path)["handoff_start"] == 0                          # default: anchor untouched
    out = widen(path, rearm_kl=True)
    assert out["config"]["obs_split"] is True and out["handoff_start"] == 123 and out["done"] == 123
    cfg2 = config_from_checkpoint(out["config"], device="cpu")
    m2 = build_models(cfg2)
    _load_model_state(m2, out["models"])
    xw = torch.cat([x, torch.randn(2, F.SPLIT_DIM)], dim=1)
    xcw = torch.cat([xc, torch.randn(2, F.SPLIT_DIM)], dim=1)
    with torch.no_grad():
        assert torch.allclose(m.actor(x), m2.actor(xw), atol=1e-6)
        assert torch.allclose(m.critic(xc), m2.critic(xcw), atol=1e-6)
        v, aux = m.critic.forward_with_aux(xc); v2, aux2 = m2.critic.forward_with_aux(xcw)
        assert torch.allclose(aux, aux2, atol=1e-6)
    opt2 = torch.optim.Adam(list(m2.actor.parameters()) + list(m2.critic.parameters()), lr=1e-3)
    opt2.load_state_dict(out["optim"]["ppo"])                       # shapes line up
    assert out["league"]["selves"][0]["actor"]["net.0.weight"].shape[1] == m2.actor.net[0].in_features
