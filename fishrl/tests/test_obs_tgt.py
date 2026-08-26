"""Config.obs_tgt (2026-08-26 re-audit): the choose_targets candidate pack —
per legal-list index, features of the card that PICK_SINGLE index resolves to,
plus stack-target sight extended to the top four objects.

The headline probe is the Spray-fizzle line: WHICH of two same-name fish an
opponent spell targets was aliasing-proved blind in the re-audit; with the pack
on, the targeted-by flags make it distinguishable at the choose_targets
decision (kill the targeted fish -> CR 608.2b counters the Spray -> no draw)."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from fishrl.data import features as F
from fishrl.forgetful_fish import engine as E
from fishrl.forgetful_fish.cards import load_decklist
from fishrl.forgetful_fish.state import PendingDecision
from fishrl.models.policy import ACTOR_IN, actor_in
from fishrl.obs import vocab as V
from fishrl.train import checkpoint as ckpt
from fishrl.train.config import Config
from fishrl.train.scenarios import ScenarioEnv
from fishrl.train.scenarios.constructed import ResponseWindow
from fishrl.train.train_loop import _load_model_state, build_models, config_from_checkpoint


@pytest.fixture(autouse=True)
def _reset():
    F.set_public_encoding(True)
    yield
    F.set_count_block(False)
    F.set_split_block(False)
    F.set_ctx_block(False)
    F.set_tgt_block(False)


def _game(seed=3):
    g = E.new_multiplayer_game(load_decklist(), p1_name="p1", p2_name="p2", seed=seed)
    E.choose_play_order(g, g.pending.player, "first")
    for _ in range(2):
        E.mulligan_decision(g, g.pending.player, "keep")
    return g


def _stack_state(min_fish=1):
    scn = ResponseWindow(); scn.pool_n = 6
    env = ScenarioEnv(scn, max_decisions=2000)
    for seed in range(200):
        env.reset(seed=seed)
        g = env.g
        if not g.stack:
            continue
        so = g.stack[0]
        o = g.objects[so.source_instance_id]
        if o.name not in ("Crystal Spray", "Mind Bend", "Metamorphose") or not so.targets:
            continue
        fish = [i for i in g.players["p1"].battlefield if E._is_creature(g.objects[i])]
        lands = [i for i in g.players["p1"].battlefield if "Land" in (g.objects[i].type_line or "")]
        if len(fish) >= min_fish and lands:
            return g, fish, lands
    pytest.skip("no suitable stack state found")


def test_fizzle_aliasing_resolved():
    """WHICH same-name fish the on-stack spell targets must now be visible at the
    responder's choose_targets (the re-audit's S1, upgraded to material by the
    Spray-fizzle line)."""
    F.set_tgt_block(True)
    g, fish, lands = _stack_state(min_fish=2)
    assert g.objects[fish[0]].name == g.objects[fish[1]].name
    g.pending = PendingDecision(type="choose_targets", player="p1",
                                context={"legal": list(fish) + list(lands)})
    a = copy.deepcopy(g); a.stack[0].targets = [{"type": "object", "id": fish[0]}]
    b = copy.deepcopy(g); b.stack[0].targets = [{"type": "object", "id": fish[1]}]
    ta, tb = F.tgt_block(a, "p1"), F.tgt_block(b, "p1")
    assert not np.array_equal(ta, tb), "targeted fish 0 vs 1 must be distinguishable"
    k = 2 * F.CTX_STACK_OBJ
    assert ta[k + 0 * F.TGT_SLOT_F + 8] == 1.0 and ta[k + 1 * F.TGT_SLOT_F + 8] == 0.0
    assert tb[k + 0 * F.TGT_SLOT_F + 8] == 0.0 and tb[k + 1 * F.TGT_SLOT_F + 8] == 1.0
    # rides into both nets' features
    assert not np.array_equal(F.bookkeeper_counts(a, "p1"), F.bookkeeper_counts(b, "p1"))
    assert not np.array_equal(F.encode_hands(a), F.encode_hands(b))


def test_candidate_features():
    F.set_tgt_block(True)
    g, fish, lands = _stack_state()
    g.objects[lands[0]].tapped = True
    stack_spell = g.stack[0].source_instance_id
    legal = [stack_spell] + list(fish) + list(lands)      # Crystal Spray-style list
    g.pending = PendingDecision(type="choose_targets", player="p1",
                                context={"legal": legal})
    blk = F.tgt_block(g, "p1")
    k = 2 * F.CTX_STACK_OBJ
    n = len(legal)
    valid = [blk[k + i * F.TGT_SLOT_F + 0] for i in range(F.TGT_SLOTS)]
    assert valid[:n] == [1.0] * n and sum(valid) == n
    s = blk[k + 0 * F.TGT_SLOT_F:]                        # the stack-spell candidate
    assert s[4] == 1.0 and s[2] == 0.0 and s[3] == 0.0    # on-stack, not bf
    fslot = blk[k + 1 * F.TGT_SLOT_F:]                    # first fish
    assert fslot[1] == 1.0 and fslot[2] == 1.0            # mine, creature
    lslot = blk[k + (1 + len(fish)) * F.TGT_SLOT_F:]      # first land (tapped above)
    assert lslot[3] == 1.0 and lslot[5] == 1.0            # land, tapped
    # name scalar populated and distinct between fish and land
    assert fslot[11] > 0 and lslot[11] > 0 and fslot[11] != lslot[11]


def test_deep_stack_targets_visible():
    """Stack objects 3-4 from the top now carry their targets (ctx pack has 1-2)."""
    F.set_ctx_block(True); F.set_tgt_block(True)
    g, fish, lands = _stack_state()
    so = g.stack[0]
    g.stack = [copy.deepcopy(so), copy.deepcopy(so), copy.deepcopy(so)]
    a = copy.deepcopy(g); a.stack[0].targets = [{"type": "object", "id": fish[0]}]
    b = copy.deepcopy(g); b.stack[0].targets = [{"type": "object", "id": lands[0]}]
    ta, tb = F.tgt_block(a, "p1"), F.tgt_block(b, "p1")
    assert not np.array_equal(ta, tb), "3rd-from-top target must now be visible"
    n = V.N_NAMES
    assert ta[n + 2] == 1.0 and tb[n + 3] == 1.0          # slot 0 = 3rd from top
    # depth 4: bottom of a 4-deep stack lands in slot 1
    c = copy.deepcopy(g); c.stack.insert(0, copy.deepcopy(so))
    c.stack[0].targets = [{"type": "object", "id": fish[0]}]
    tc = F.tgt_block(c, "p1")
    assert tc[F.CTX_STACK_OBJ + n + 2] == 1.0


def test_gating_and_orientation():
    F.set_tgt_block(True)
    g, fish, lands = _stack_state()
    g.pending = PendingDecision(type="choose_targets", player="p1",
                                context={"legal": list(fish) + list(lands)})
    k = 2 * F.CTX_STACK_OBJ
    assert F.tgt_block(g, "p1")[k:].sum() > 0              # the chooser sees it
    assert F.tgt_block(g, "p2")[k:].sum() == 0             # the other seat does not
    crit = F.tgt_block(g)                                  # critic: p1-oriented, sees it
    assert crit[k:].sum() > 0
    assert crit[k + 0 * F.TGT_SLOT_F + 1] == 1.0           # p1's fish: controller-is-viewer


def test_dims_and_config():
    F.set_count_block(True); F.set_split_block(True); F.set_ctx_block(True); F.set_tgt_block(True)
    assert F.TGT_DIM == 2 * F.CTX_STACK_OBJ + F.TGT_SLOTS * F.TGT_SLOT_F
    assert F.belief_dim() == V.N_NAMES + F.COUNT_DIM + F.SPLIT_DIM + F.CTX_DIM + F.TGT_DIM
    assert actor_in() == ACTOR_IN + F.COUNT_DIM + F.SPLIT_DIM + F.CTX_DIM + F.TGT_DIM
    assert F.hands_dim() == (F.HANDS_DIM + F.COUNT_DIM + F.SPLIT_DIM
                             + F.CTX_DIM + F.CRITIC_CTX_EXTRA + F.TGT_DIM)
    cd = {"seed": 0, "encoders": {"actor": "entity", "critic": "entity"}, "use_belief": True,
          "critic_hidden": [32, 32], "hidden": [32], "actor_hidden": [32], "card_dim": 8,
          "belief_mode": "bookkeeper", "critic_view": "hands", "critic_deckout_aux": 0.1,
          "text_change_mode": "guided", "obs_counts": True, "obs_split": True,
          "obs_ctx": True, "obs_tgt": True}
    assert config_from_checkpoint(cd).obs_tgt is True
    assert config_from_checkpoint({k: v for k, v in cd.items() if k != "obs_tgt"}).obs_tgt is False


def _cfg(tgt):
    return Config(critic_view="hands", belief_mode="bookkeeper", critic_encoder="entity",
                  actor_encoder="entity", encoder="flat", device="cpu", critic_hidden=(32, 32),
                  actor_hidden=(32,), card_dim=8, obs_counts=True, obs_split=True,
                  obs_ctx=True, obs_tgt=tgt)


def test_build_models_widths_and_requires_ctx():
    m = build_models(_cfg(True))
    assert F.tgt_block_on()
    assert m.actor.in_dim == ACTOR_IN + F.COUNT_DIM + F.SPLIT_DIM + F.CTX_DIM + F.TGT_DIM
    assert m.critic.enc.globals_dim == F.hands_dim() - m.critic.enc.R * F.CARD_F
    build_models(_cfg(False))
    assert not F.tgt_block_on()
    with pytest.raises(AssertionError):
        build_models(Config(critic_view="hands", belief_mode="bookkeeper",
                            critic_encoder="entity", actor_encoder="entity", encoder="flat",
                            device="cpu", critic_hidden=(32, 32), actor_hidden=(32,),
                            card_dim=8, obs_counts=True, obs_split=True, obs_ctx=False,
                            obs_tgt=True))


def test_widen_tgt_is_function_identical_and_round_trips(tmp_path):
    from fishrl.train.widen_tgt import widen, K_ACTOR, K_CRITIC
    cfg = _cfg(False)                                  # counts+split+ctx ON, tgt off
    torch.manual_seed(1)
    m = build_models(cfg)
    frozen = build_models(cfg)
    opt = torch.optim.Adam(list(m.actor.parameters()) + list(m.critic.parameters()), lr=1e-3)
    xin = ACTOR_IN + F.COUNT_DIM + F.SPLIT_DIM + F.CTX_DIM
    xcin = F.HANDS_DIM + F.COUNT_DIM + F.SPLIT_DIM + F.CTX_DIM + F.CRITIC_CTX_EXTRA
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
                          "text_change_mode": "guided", "obs_counts": True, "obs_split": True,
                          "obs_ctx": True},
               "done": 123, "elapsed": 1.0, "frozen_it": 100, "handoff_start": 0, "warmup_done": True,
               "models": _model_state(m), "frozen": _model_state(frozen),
               "optim": {"ppo": opt.state_dict()}, "rng": None, "league": league, "scen_league": None}
    path = str(tmp_path / "latest.pt")
    ckpt.save_checkpoint(path, payload)
    out = widen(path, rearm_kl=True)
    assert out["config"]["obs_tgt"] is True and out["handoff_start"] == 123
    cfg2 = config_from_checkpoint(out["config"], device="cpu")
    m2 = build_models(cfg2)
    _load_model_state(m2, out["models"])
    xw = torch.cat([x, torch.randn(2, K_ACTOR)], dim=1)
    xcw = torch.cat([xc, torch.randn(2, K_CRITIC)], dim=1)
    with torch.no_grad():
        assert torch.allclose(m.actor(x), m2.actor(xw), atol=1e-6)
        assert torch.allclose(m.critic(xc), m2.critic(xcw), atol=1e-6)
        v, aux = m.critic.forward_with_aux(xc); v2, aux2 = m2.critic.forward_with_aux(xcw)
        assert torch.allclose(aux, aux2, atol=1e-6)
    opt2 = torch.optim.Adam(list(m2.actor.parameters()) + list(m2.critic.parameters()), lr=1e-3)
    opt2.load_state_dict(out["optim"]["ppo"])
    assert out["league"]["selves"][0]["actor"]["net.0.weight"].shape[1] == m2.actor.net[0].in_features
