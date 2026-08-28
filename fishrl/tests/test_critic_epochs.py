"""critic_epochs (2026-08-21): the critic fits each batch on only the first N PPO epochs."""
from __future__ import annotations

import copy

import torch

from fishrl.train.config import Config
from fishrl.train.ppo import ppo_update
from fishrl.train.train_loop import build_models


def _cfg(**kw):
    return Config(critic_view="hands", belief_mode="bookkeeper", critic_encoder="entity",
                  actor_encoder="entity", encoder="flat", device="cpu", critic_hidden=(64, 64),
                  actor_hidden=(64,), card_dim=16, minibatch=8, ppo_epochs=3, **kw)


def _batch(m, n=16, seed=0):
    from fishrl.data.features import HANDS_DIM
    from fishrl.models.policy import ACTOR_IN
    from fishrl.spaces import action_space as A
    g = torch.Generator().manual_seed(seed)
    mask = torch.zeros(n, A.N, dtype=torch.bool); mask[:, :5] = True
    x = torch.randn(n, ACTOR_IN, generator=g)
    with torch.no_grad():
        logp = m.actor.log_probs(x, mask)
    action = torch.randint(0, 5, (n,), generator=g)
    return {"x_act": x, "mask": mask, "action": action,
            "old_logp": logp.gather(1, action[:, None]).squeeze(1),
            "adv": torch.randn(n, generator=g), "pub": torch.randn(n, HANDS_DIM, generator=g),
            "y_p1": (torch.rand(n, generator=g) > 0.5).float(), "valid": torch.ones(n, dtype=torch.bool),
            "deckout_valid": torch.zeros(n, dtype=torch.bool)}


def _run(cfg, train_actor=True):
    torch.manual_seed(0)
    m = build_models(cfg)
    batch = _batch(m)
    opt = torch.optim.Adam(list(m.actor.parameters()) + list(m.critic.parameters()), lr=1e-3)
    before = copy.deepcopy(m.critic.state_dict())
    stats = ppo_update(batch, m.actor, m.critic, opt, cfg, 0.01, train_actor=train_actor)
    return m, before, stats


def test_critic_epochs_limits_critic_steps_not_actor_steps():
    _, _, s_all = _run(_cfg(critic_epochs=0))
    assert s_all["n"] == 6 and s_all["n_critic"] == 6          # 3 epochs x 2 minibatches
    _, _, s_one = _run(_cfg(critic_epochs=1))
    assert s_one["n"] == 6 and s_one["n_critic"] == 2          # actor all, critic first epoch
    assert s_one["critic_loss"] > 0.0


def test_critic_untouched_after_its_epochs():
    """With critic_epochs=1 and actor frozen, epochs 2-3 have nothing to fit: the
    update stops after the first pass and the critic moved exactly once per minibatch."""
    m, before, s = _run(_cfg(critic_epochs=1), train_actor=False)
    assert s["n"] == 2 and s["n_critic"] == 2
    assert any(not torch.equal(before[k], v) for k, v in m.critic.state_dict().items())


def test_default_is_legacy_behaviour():
    assert Config().critic_epochs == 0


def test_estimator_metrics_per_turn_buckets():
    """Per-turn calibration buckets come off the same pre-update forward, keyed on the
    turn/40 float in the feature tail."""
    from fishrl.data.features import HANDS_DIM, TURN_BUCKETS, turn_index_for
    from fishrl.eval.metrics import estimator_metrics
    torch.manual_seed(0)
    m = build_models(_cfg())
    n = 40
    pub = torch.zeros(n, HANDS_DIM)
    turns = torch.tensor([1, 5, 12, 25, 40] * 8, dtype=torch.float32)
    pub[:, turn_index_for("hands")] = turns / 40.0
    batch = {"valid": torch.ones(n, dtype=torch.bool), "y_p1": (torch.rand(n) > 0.5).float(), "pub": pub,
             "god": torch.zeros(n, 1)}
    est = estimator_metrics(m, batch)
    assert est["critic_n_t1_10"] == 16 and est["critic_n_t11_20"] == 8 and est["critic_n_t21_30"] == 8 and est["critic_n_t31p"] == 8
    tot = sum(est[f"critic_n_{lbl}"] * est[f"critic_brier_{lbl}"] for lbl, _, _ in TURN_BUCKETS) / n
    assert abs(tot - est["critic_brier"]) < 1e-5
    ct = est["critic_turn"]
    assert len(ct["n"]) == 40 and ct["n"][0] == 8 and ct["n"][4] == 8 and ct["n"][11] == 8 and ct["n"][39] == 8
    assert ct["n"][1] == 0 and ct["brier"][1] is None
    assert abs(sum(ct["n"][i] * ct["brier"][i] for i in range(40) if ct["n"][i]) / n - est["critic_brier"]) < 1e-3


def test_jump_from_sums_reproduces_quantiles():
    """Window |dV| aggregation (2026-08-29): the 1000-bin histogram fold gives the
    mean exactly and any quantile to one bin width; empty windows come back NaN."""
    import numpy as np
    from fishrl.eval.metrics import jump_from_sums
    rng = np.random.default_rng(0)
    j = rng.uniform(0.0, 0.2, size=5000)                 # realistic |dV| scale
    hist = torch.histc(torch.tensor(j, dtype=torch.float32), bins=1000,
                       min=0.0, max=1.0).numpy()
    out = jump_from_sums(len(j), float(j.sum()), hist)
    assert out["critic_jump_n"] == 5000
    assert abs(out["critic_jump_mean"] - j.mean()) < 1e-6
    assert abs(out["critic_jump_p90"] - np.quantile(j, 0.9)) < 2e-3
    # summing a window with itself doubles n and moves neither statistic
    two = jump_from_sums(2 * len(j), 2 * float(j.sum()), hist * 2)
    assert two["critic_jump_n"] == 10000
    assert abs(two["critic_jump_mean"] - out["critic_jump_mean"]) < 1e-9
    assert two["critic_jump_p90"] == out["critic_jump_p90"]
    empty = jump_from_sums(0, 0.0, np.zeros(1000))
    assert empty["critic_jump_n"] == 0
    assert empty["critic_jump_mean"] != empty["critic_jump_mean"]  # NaN


def test_calib_from_sums_matches_per_batch():
    """Window aggregation: folding one batch's _calib_sums through calib_from_sums
    reproduces the per-batch by-turn keys exactly; summing the batch with itself
    doubles every n and leaves every mean unchanged."""
    import numpy as np
    from fishrl.data.features import HANDS_DIM, TURN_BUCKETS, TURN_MAX, turn_index_for
    from fishrl.eval.metrics import calib_from_sums, estimator_metrics
    torch.manual_seed(0)
    m = build_models(_cfg())
    n = 45
    pub = torch.zeros(n, HANDS_DIM)
    # includes turn-0 pre-game decisions: in slot 1 of critic_turn, outside every bucket
    turns = torch.tensor([0, 1, 5, 12, 25, 40, 3, 17, 33] * 5, dtype=torch.float32)
    pub[:, turn_index_for("hands")] = turns / 40.0
    batch = {"valid": torch.ones(n, dtype=torch.bool), "y_p1": (torch.rand(n) > 0.5).float(),
             "pub": pub, "god": torch.zeros(n, 1)}
    est = estimator_metrics(m, batch)
    cs = est["_calib_sums"]
    one = calib_from_sums(np.asarray(cs["turn"]), cs["bucket"])
    assert one["critic_turn"] == est["critic_turn"]
    # the window-pooled HEADLINE (2026-08-29): one batch's fold reproduces the
    # per-batch acc/brier exactly (turn cells partition all scored decisions)
    assert one["n"] == est["n"] == n
    assert abs(one["critic_acc"] - est["critic_acc"]) < 1e-6
    assert abs(one["critic_brier"] - est["critic_brier"]) < 1e-6
    for lbl, _lo, _hi in TURN_BUCKETS:
        assert one[f"critic_n_{lbl}"] == est[f"critic_n_{lbl}"]
        assert abs(one[f"critic_brier_{lbl}"] - est[f"critic_brier_{lbl}"]) < 1e-6
        assert abs(one[f"critic_acc_{lbl}"] - est[f"critic_acc_{lbl}"]) < 1e-6
    # turn-0 rows pool into slot 1 but are outside the buckets
    assert est["critic_turn"]["n"][0] == 10 and est["critic_n_t1_10"] == 15
    two = calib_from_sums(np.asarray(cs["turn"]) * 2,
                          {k: np.asarray(v) * 2 for k, v in cs["bucket"].items()})
    assert two["critic_turn"]["n"] == [2 * v for v in est["critic_turn"]["n"]]
    assert two["critic_turn"]["brier"] == est["critic_turn"]["brier"]
    assert two["n"] == 2 * n
    assert abs(two["critic_acc"] - est["critic_acc"]) < 1e-6
    assert abs(two["critic_brier"] - est["critic_brier"]) < 1e-6
    for lbl, _lo, _hi in TURN_BUCKETS:
        assert two[f"critic_n_{lbl}"] == 2 * est[f"critic_n_{lbl}"]
        assert abs(two[f"critic_brier_{lbl}"] - est[f"critic_brier_{lbl}"]) < 1e-6
