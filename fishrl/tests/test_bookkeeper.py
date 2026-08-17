"""The analytic hand bookkeeper (belief_mode="bookkeeper") and the v3 stack.

The bookkeeper replaces the HandGuesser's 20-dim output with pure arithmetic over
the viewer's own information. Its spec is the Guesser Deposition's "perfect
bookkeeper": known + (handn - known_total) * remaining/remaining_total. These
tests pin the spec against an independent reference implementation on real game
states, plus the v3 collection/update path (public critic + parity aux).
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
import torch

from fishrl.data.features import bookkeeper_counts, opponent_hand_counts
from fishrl.env.aec_env import FishAEC
from fishrl.obs import vocab as V


def _reference_bookkeeper(g, viewer: str) -> np.ndarray:
    """Independent re-derivation (guesser_eval's extraction + the Deposition's
    formula), kept deliberately different in style from the production code."""
    opp = "p2" if viewer == "p1" else "p1"
    obj = g.objects
    known = Counter(obj[iid].name for iid in g.players[opp].hand
                    if viewer in (obj[iid].known_by or []))
    vis = []
    vis += [obj[iid].name for iid in g.players[viewer].hand]
    for pid in ("p1", "p2"):
        vis += [obj[iid].name for iid in g.players[pid].battlefield]
    vis += [obj[iid].name for iid in g.graveyard]
    vis += [obj[iid].name for iid in g.exile]
    vis += [obj[s.source_instance_id].name for s in g.stack
            if s.source_instance_id in obj]
    vis += [obj[s.instance_id].name for s in g.library if s.known_by.get(viewer)]
    visible = Counter(vis)
    copies = Counter(o.name for o in obj.values())
    out = np.zeros(V.N_NAMES, dtype=np.float64)
    remaining = np.zeros(V.N_NAMES, dtype=np.float64)
    for name, idx in V.NAME_INDEX.items():
        out[idx] = known.get(name, 0)
        remaining[idx] = max(copies.get(name, 0) - visible.get(name, 0)
                             - known.get(name, 0), 0)
    fill = len(g.players[opp].hand) - out.sum()
    if fill <= 0 or remaining.sum() <= 0:
        return out.astype(np.float32)
    return (out + fill * remaining / remaining.sum()).astype(np.float32)


def _play_random(seed: int, steps: int = 120):
    env = FishAEC(max_decisions=400)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    states = []
    n = 0
    for agent in env.agent_iter(max_iter=steps * 4):
        if env.terminations[agent] or env.truncations[agent]:
            env.step(None)
            continue
        states.append((env.g, agent))
        mask = env.observe(agent)["action_mask"]
        env.step(int(rng.choice(np.flatnonzero(mask))))
        n += 1
        if n >= steps:
            break
    return env, states


def test_bookkeeper_matches_reference_and_properties():
    env, _ = _play_random(seed=7)
    g = env.g
    for viewer in ("p1", "p2"):
        opp = "p2" if viewer == "p1" else "p1"
        bk = bookkeeper_counts(g, viewer)
        ref = _reference_bookkeeper(g, viewer)
        assert bk.shape == (V.N_NAMES,) and bk.dtype == np.float32
        assert (bk >= -1e-6).all()
        np.testing.assert_allclose(bk, ref, atol=1e-5)
        handn = len(g.players[opp].hand)
        if handn > 0:
            assert abs(float(bk.sum()) - handn) < 1e-4   # exactly hand-size consistent
        # never under-counts a shown card
        known = np.zeros(V.N_NAMES)
        for iid in g.players[opp].hand:
            o = g.objects[iid]
            if viewer in (o.known_by or []) and o.name in V.NAME_INDEX:
                known[V.NAME_INDEX[o.name]] += 1
        assert (bk + 1e-6 >= known).all()


def test_bookkeeper_exact_when_hand_fully_known():
    env, _ = _play_random(seed=11)
    g = env.g
    for iid in g.players["p2"].hand:                 # reveal p2's whole hand to p1
        o = g.objects[iid]
        kb = set(o.known_by or [])
        kb.add("p1")
        o.known_by = list(kb)
    bk = bookkeeper_counts(g, "p1")
    np.testing.assert_allclose(bk, opponent_hand_counts(g, "p1"), atol=1e-6)


def test_belief_env_bookkeeper_mode_fills_the_slot():
    from fishrl.obs.encoder import OBS_DIM
    from fishrl.train.belief_env import BeliefAugmentedEnv
    benv = BeliefAugmentedEnv(None, mode="bookkeeper", max_decisions=200)
    benv.reset(seed=3)
    agent = benv.agent_selection
    obs = benv.observe(agent)["observation"]
    assert obs.shape == (OBS_DIM + V.N_NAMES,)
    np.testing.assert_allclose(obs[OBS_DIM:], bookkeeper_counts(benv.g, agent), atol=1e-5)


# ── the v3 collection/update stack ───────────────────────────────────────────

def _v3_cfg():
    from fishrl.train.config import Config
    return Config(belief_mode="bookkeeper", critic_view="public",
                  critic_deckout_aux=0.1, hidden=(32, 32), critic_hidden=(32, 32),
                  actor_hidden=(32, 32), card_dim=16, warmup_games=2,
                  games_per_iter=2, minibatch=64, max_decisions=200, device="cpu")


def test_v3_models_and_collection_and_update():
    from fishrl.data import features
    from fishrl.models.estimators import PublicCritic
    from fishrl.train.belief_env import BeliefAugmentedEnv
    from fishrl.train.collector import collect_games, fill_critic_values, random_act_fn
    from fishrl.train.ppo import ppo_update
    from fishrl.train.train_loop import build_models

    cfg = _v3_cfg()
    features.set_public_encoding(True)               # the train() gate rule, applied here
    m = build_models(cfg)
    assert m.guesser is None and m.public is None
    assert isinstance(m.critic, PublicCritic) and hasattr(m.critic, "aux_head")

    benv = BeliefAugmentedEnv(None, mode="bookkeeper", max_decisions=cfg.max_decisions)
    rng = np.random.default_rng(0)
    buf = collect_games(benv, random_act_fn(rng), 3, 50, critic=None,
                        max_decisions=cfg.max_decisions, critic_view="public")
    fill_critic_values(buf, m.critic, view="public")
    batch = buf.compute(cfg.gamma, cfg.lam)
    assert float(batch["pub"].abs().sum()) > 0.0     # the critic's food is real
    assert float(batch["god"].abs().sum()) == 0.0    # god encode skipped entirely
    assert "deckout_valid" in batch

    opt = torch.optim.Adam(list(m.actor.parameters()) + list(m.critic.parameters()), lr=1e-3)
    stats = ppo_update(batch, m.actor, m.critic, opt, cfg, ent_coef=0.01, rng_seed=0)
    for k in ("policy_loss", "critic_loss", "deckout_aux_loss"):
        assert np.isfinite(stats[k])


def test_fill_critic_values_asserts_on_zeroed_public_features():
    from fishrl.data import features
    from fishrl.train.belief_env import BeliefAugmentedEnv
    from fishrl.train.collector import collect_games, fill_critic_values, random_act_fn
    from fishrl.train.train_loop import build_models

    cfg = _v3_cfg()
    features.set_public_encoding(False)              # the landmine this guard exists for
    try:
        m = build_models(cfg)
        benv = BeliefAugmentedEnv(None, mode="bookkeeper", max_decisions=100)
        buf = collect_games(benv, random_act_fn(np.random.default_rng(1)), 1, 60,
                            critic=None, max_decisions=100, critic_view="public")
        with pytest.raises(AssertionError, match="set_public_encoding"):
            fill_critic_values(buf, m.critic, view="public")
    finally:
        features.set_public_encoding(True)


def test_deckout_aux_zero_weight_is_a_noop_for_the_critic_gradient():
    from fishrl.data import features
    from fishrl.train.belief_env import BeliefAugmentedEnv
    from fishrl.train.collector import collect_games, fill_critic_values, random_act_fn
    from fishrl.train.ppo import ppo_update
    from fishrl.train.train_loop import build_models

    features.set_public_encoding(True)
    cfg0 = _v3_cfg()
    cfg0.critic_deckout_aux = 0.0
    torch.manual_seed(0)
    m = build_models(cfg0)
    aux_before = {k: v.clone() for k, v in m.critic.aux_head.state_dict().items()}
    benv = BeliefAugmentedEnv(None, mode="bookkeeper", max_decisions=150)
    buf = collect_games(benv, random_act_fn(np.random.default_rng(2)), 2, 80,
                        critic=None, max_decisions=150, critic_view="public")
    fill_critic_values(buf, m.critic, view="public")
    batch = buf.compute(cfg0.gamma, cfg0.lam)
    opt = torch.optim.Adam(list(m.actor.parameters()) + list(m.critic.parameters()), lr=1e-3)
    ppo_update(batch, m.actor, m.critic, opt, cfg0, ent_coef=0.01, rng_seed=1)
    for k, v in m.critic.aux_head.state_dict().items():
        assert torch.equal(v, aux_before[k])         # weight 0 -> the head never moved
