"""Config.obs_ctx (2026-08-26 observability audit): the decision-context pack —
stack-spell targets, search eligibility, builder arrangements, blocker focus on
both nets + the critic's step/pending/combat/pay context — and the name-sorted
PICK_SINGLE remap for search_library / choose_graveyard.

Each test is the audit's aliasing probe RESOLVED: states that were bit-identical
to the nets before the pack must now differ."""
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
from fishrl.spaces import action_space as A
from fishrl.spaces.compound import CompoundBuilder
from fishrl.spaces.masking import atomic_mask, pick_list
from fishrl.env.apply import apply_atomic
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


def _game(seed=3):
    g = E.new_multiplayer_game(load_decklist(), p1_name="p1", p2_name="p2", seed=seed)
    E.choose_play_order(g, g.pending.player, "first")
    for _ in range(2):
        E.mulligan_decision(g, g.pending.player, "keep")
    return g


def _stack_state():
    scn = ResponseWindow(); scn.pool_n = 6
    env = ScenarioEnv(scn, max_decisions=2000)
    for seed in range(40):
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
        if fish and lands:
            return g, fish, lands
    pytest.skip("no suitable stack state found")


def test_stack_target_aliasing_resolved():
    F.set_ctx_block(True)
    g, fish, lands = _stack_state()
    a = copy.deepcopy(g); a.stack[0].targets = [{"type": "object", "id": fish[0]}]
    b = copy.deepcopy(g); b.stack[0].targets = [{"type": "object", "id": lands[0]}]
    ca, cb = F.ctx_block(a, "p1"), F.ctx_block(b, "p1")
    assert not np.array_equal(ca, cb), "target fish vs land must now be distinguishable"
    n = V.N_NAMES
    # slot 0 = top of stack; fish target sets the is-creature bit, land the is-land bit
    assert ca[n + 2] == 1.0 and ca[n + 3] == 0.0
    assert cb[n + 2] == 0.0 and cb[n + 3] == 1.0
    assert ca[n + 1] == 1.0, "target controller is the viewer (p1's permanent)"
    # rides into both nets' features
    ba, bb = F.bookkeeper_counts(a, "p1"), F.bookkeeper_counts(b, "p1")
    ha, hb = F.encode_hands(a), F.encode_hands(b)
    assert not np.array_equal(ba, bb) and not np.array_equal(ha, hb)


def test_search_sorted_mapping_and_counts():
    F.set_ctx_block(True)
    g = _game()
    tut = next(iid for iid, o in g.objects.items() if o.name == "Mystical Tutor")
    assert E.start_library_search(g, "p1", tut, types=("instant", "sorcery"))
    ctx = g.pending.context
    ordered = pick_list(g, "search_library", ctx)
    names = [g.objects[i].name for i in ordered]
    assert names == sorted(names), "eligible must be name-sorted under obs_ctx"
    assert set(ordered) == set(ctx["eligible"]), "same SET (mask contract)"
    # the block carries the searcher's eligible counts; the opponent sees zeros
    blk = F.ctx_block(g, "p1")
    k = 2 * F.CTX_STACK_OBJ
    assert blk[k:k + V.N_NAMES].sum() == len(ctx["eligible"])
    blk_opp = F.ctx_block(g, "p2")
    assert blk_opp[k:k + V.N_NAMES].sum() == 0, "library contents are hidden from the non-searcher"
    # apply resolves through the SAME ordering: picking index 0 fetches the
    # alphabetically first eligible name
    first = names[0]
    ok = apply_atomic(g, "p1", A.aid("PICK_SINGLE", 0))
    assert ok and g.library[0].instance_id == ordered[0]
    assert g.objects[g.library[0].instance_id].name == first
    # legacy era: engine order preserved
    F.set_ctx_block(False)
    g2 = _game()
    tut2 = next(iid for iid, o in g2.objects.items() if o.name == "Mystical Tutor")
    E.start_library_search(g2, "p1", tut2, types=("instant", "sorcery"))
    assert pick_list(g2, "search_library", g2.pending.context) == g2.pending.context["eligible"]


def test_builder_mirror_scry_and_blocker_focus():
    F.set_ctx_block(True)
    g = _game()
    who = g.pending.player                                       # whoever holds priority
    opp = "p2" if who == "p1" else "p1"
    assert E.debug_scry(g, who, 3)
    ctx = g.pending.context
    bld = CompoundBuilder(g, who, "scry", ctx)
    items = list(bld.items)
    assert F.ctx_live(g)
    k = 2 * F.CTX_STACK_OBJ + V.N_NAMES
    b0 = F.ctx_block(g, who)
    assert b0[k:k + 2 * V.N_NAMES].sum() == 0                    # nothing placed yet
    bld.feed(g, A.aid("PICK_A", 0))                              # first card to top
    b1 = F.ctx_block(g, who)
    assert b1[k:k + V.N_NAMES].sum() == 1 and b1[k + V.N_NAMES:k + 2 * V.N_NAMES].sum() == 0
    li = V.NAME_INDEX[g.objects[items[0]].name]
    assert b1[k + 2 * V.N_NAMES + li] == 1.0, "last-placed one-hot"
    bld.feed(g, A.aid("PICK_B", 1))                              # second to bottom
    b2 = F.ctx_block(g, who)
    assert b2[k + V.N_NAMES:k + 2 * V.N_NAMES].sum() == 1
    assert not np.array_equal(b1, b2)
    # the opponent must NOT see the arrangement (scry is private)
    assert F.ctx_block(g, opp)[k:k + 2 * V.N_NAMES].sum() == 0

    # blocker focus: PICK_B selecting attacker 0 vs 1 must now differ
    g.pending = PendingDecision(type="declare_blockers", player="p1",
                                context={"attackers": ["a1", "a2"], "eligible": []})
    bl = CompoundBuilder(g, "p1", "declare_blockers", g.pending.context)
    ba = copy.deepcopy(g)
    bl_a = CompoundBuilder(ba, "p1", "declare_blockers", ba.pending.context)
    bl_a.feed(ba, A.aid("PICK_B", 0))
    bb = copy.deepcopy(g)
    bl_b = CompoundBuilder(bb, "p1", "declare_blockers", bb.pending.context)
    bl_b.feed(bb, A.aid("PICK_B", 1))
    fa, fb = F.ctx_block(ba, "p1"), F.ctx_block(bb, "p1")
    assert not np.array_equal(fa, fb), "blocker focus 0 vs 1 must be distinguishable"


def test_critic_ctx_extra_and_dims():
    F.set_ctx_block(True)
    g = _game()
    ex = F.critic_ctx_extra(g)
    assert ex.shape == (F.CRITIC_CTX_EXTRA,)
    assert ex[:V.N_STEPS].sum() == 1.0                           # step one-hot
    pi = V.PENDING_INDEX[g.pending.type] if g.pending else None
    if pi is not None:
        assert ex[V.N_STEPS + pi] == 1.0
    F.set_count_block(True); F.set_split_block(True)
    assert F.belief_dim() == V.N_NAMES + F.COUNT_DIM + F.SPLIT_DIM + F.CTX_DIM
    assert actor_in() == ACTOR_IN + F.COUNT_DIM + F.SPLIT_DIM + F.CTX_DIM
    assert F.hands_dim() == (F.HANDS_DIM + F.COUNT_DIM + F.SPLIT_DIM
                             + F.CTX_DIM + F.CRITIC_CTX_EXTRA)
    cd = {"seed": 0, "encoders": {"actor": "entity", "critic": "entity"}, "use_belief": True,
          "critic_hidden": [32, 32], "hidden": [32], "actor_hidden": [32], "card_dim": 8,
          "belief_mode": "bookkeeper", "critic_view": "hands", "critic_deckout_aux": 0.1,
          "text_change_mode": "guided", "obs_counts": True, "obs_split": True, "obs_ctx": True}
    assert config_from_checkpoint(cd).obs_ctx is True
    assert config_from_checkpoint({k: v for k, v in cd.items() if k != "obs_ctx"}).obs_ctx is False


def _cfg(ctx):
    return Config(critic_view="hands", belief_mode="bookkeeper", critic_encoder="entity",
                  actor_encoder="entity", encoder="flat", device="cpu", critic_hidden=(32, 32),
                  actor_hidden=(32,), card_dim=8, obs_counts=True, obs_split=True, obs_ctx=ctx)


def test_build_models_widths_and_requires_split():
    m = build_models(_cfg(True))
    assert F.ctx_block_on()
    assert m.actor.in_dim == ACTOR_IN + F.COUNT_DIM + F.SPLIT_DIM + F.CTX_DIM
    assert m.critic.enc.globals_dim == F.hands_dim() - m.critic.enc.R * F.CARD_F
    m0 = build_models(_cfg(False))
    assert not F.ctx_block_on()
    with pytest.raises(AssertionError):
        build_models(Config(critic_view="hands", belief_mode="bookkeeper",
                            critic_encoder="entity", actor_encoder="entity", encoder="flat",
                            device="cpu", critic_hidden=(32, 32), actor_hidden=(32,),
                            card_dim=8, obs_counts=True, obs_split=False, obs_ctx=True))


def test_widen_ctx_is_function_identical_and_round_trips(tmp_path):
    from fishrl.train.widen_ctx import widen, K_ACTOR, K_CRITIC
    cfg = _cfg(False)                                  # counts+split ON, ctx off
    torch.manual_seed(1)
    m = build_models(cfg)
    frozen = build_models(cfg)
    opt = torch.optim.Adam(list(m.actor.parameters()) + list(m.critic.parameters()), lr=1e-3)
    xin = ACTOR_IN + F.COUNT_DIM + F.SPLIT_DIM
    xcin = F.HANDS_DIM + F.COUNT_DIM + F.SPLIT_DIM
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
                          "text_change_mode": "guided", "obs_counts": True, "obs_split": True},
               "done": 123, "elapsed": 1.0, "frozen_it": 100, "handoff_start": 0, "warmup_done": True,
               "models": _model_state(m), "frozen": _model_state(frozen),
               "optim": {"ppo": opt.state_dict()}, "rng": None, "league": league, "scen_league": None}
    path = str(tmp_path / "latest.pt")
    ckpt.save_checkpoint(path, payload)
    out = widen(path, rearm_kl=True)
    assert out["config"]["obs_ctx"] is True and out["handoff_start"] == 123
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
