"""BC-handoff knobs on the PPO update (fishrl.imitate -> PPO fine-tune):
`train_actor=False` must leave the actor bit-identical while the critic still
learns, and the KL-to-teacher penalty must actively pull the policy toward the
frozen reference when the advantages carry no signal. Defaults must be the
historic update exactly (kl_teacher reported as 0)."""
from __future__ import annotations

import copy

import numpy as np
import torch

from fishrl.data.features import GOD_DIM
from fishrl.models.policy import ACTOR_IN
from fishrl.spaces import action_space as A
from fishrl.train.config import Config
from fishrl.train.ppo import ppo_update
from fishrl.train.train_loop import build_models


def _cfg(seed=0):
    return Config(device="cpu", seed=seed, hidden=(32, 32), actor_hidden=(32, 32),
                  critic_hidden=(32, 32), minibatch=32, ppo_epochs=2)


def _batch(M=64, seed=0):
    rng = np.random.default_rng(seed)
    x = torch.from_numpy(rng.standard_normal((M, ACTOR_IN)).astype(np.float32))
    mask = torch.zeros((M, A.N), dtype=torch.float32)
    legal = rng.integers(0, A.N, size=(M, 6))
    for i in range(M):
        mask[i, legal[i]] = 1.0
    return {"x_act": x, "mask": mask,
            "action": torch.from_numpy(legal[:, 0].astype(np.int64)),
            "old_logp": torch.zeros(M), "adv": torch.zeros(M),
            "god": torch.from_numpy(rng.standard_normal((M, GOD_DIM)).astype(np.float32)),
            "y_p1": (torch.arange(M) % 2).float(), "valid": torch.ones(M)}


def test_frozen_actor_is_bit_identical_while_critic_learns():
    cfg = _cfg()
    m = build_models(cfg)
    opt = torch.optim.Adam(list(m.actor.parameters()) + list(m.critic.parameters()), lr=1e-3)
    b = _batch()
    b["adv"] = torch.randn(64)                    # real advantages must NOT leak through
    a0 = copy.deepcopy(m.actor.state_dict())
    c0 = copy.deepcopy(m.critic.state_dict())
    stats = ppo_update(b, m.actor, m.critic, opt, cfg, ent_coef=0.01, train_actor=False)
    assert all(torch.equal(m.actor.state_dict()[k], a0[k]) for k in a0)
    assert any(not torch.equal(m.critic.state_dict()[k], c0[k]) for k in c0)
    assert stats["kl_teacher"] == 0.0             # no anchor passed -> reported as zero


def _kl_to(actor, ref, b) -> float:
    with torch.no_grad():
        lp = actor.log_probs(b["x_act"], b["mask"])
        rlp = ref.log_probs(b["x_act"], b["mask"])
    rp = rlp.exp()
    diff = torch.where(rp > 0, rlp - lp, torch.zeros_like(lp))
    return float((rp * diff).sum(-1).mean())


def test_kl_teacher_pulls_policy_toward_reference():
    """Zero advantages + zero entropy coefficient -> the ONLY actor gradient is
    the KL anchor, so KL(ref || pi) must strictly shrink over updates."""
    m = build_models(_cfg(seed=0))
    ref = build_models(_cfg(seed=5)).actor        # a genuinely different policy
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    cfg = _cfg()
    b = _batch()
    before = _kl_to(m.actor, ref, b)
    assert before > 0.01                          # the two inits actually differ
    opt = torch.optim.Adam(list(m.actor.parameters()) + list(m.critic.parameters()), lr=1e-3)
    stats = None
    for i in range(5):
        stats = ppo_update(b, m.actor, m.critic, opt, cfg, ent_coef=0.0, rng_seed=i,
                           ref_actor=ref, kl_ref_coef=1.0)
    after = _kl_to(m.actor, ref, b)
    assert stats["kl_teacher"] > 0.0              # the metric is reported
    assert after < before * 0.8, f"KL did not shrink: {before:.4f} -> {after:.4f}"
