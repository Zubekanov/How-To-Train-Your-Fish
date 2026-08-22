"""critic_td_mix: bootstrapped soft labels for the critic + the deterministic-transition
jump telemetry (ret_p1 / det_next batch keys)."""
from __future__ import annotations

import copy

import numpy as np
import torch

from fishrl.data.buffer import RolloutBuffer, Step
from fishrl.train.advantages import gae
from fishrl.train.config import Config
from fishrl.train.ppo import ppo_update
from fishrl.train.train_loop import build_models
from fishrl.tests.test_critic_epochs import _batch, _cfg


def _step(seat, value, winner, gid=0):
    return Step(seat=seat, x_act=np.zeros(4, dtype=np.float32), mask=np.ones(2, dtype=np.int8),
                action=0, logp=0.0, value=value, god_feat=np.zeros(2, dtype=np.float32),
                pub_feat=np.zeros(2, dtype=np.float32), guess_in=np.zeros(2, dtype=np.float32),
                cnt_target=np.zeros(2, dtype=np.float32), winner=winner, game_id=gid)


def test_buffer_ret_p1_and_det_next():
    buf = RolloutBuffer()
    # game 0: p1, p1, p2, p1  (p2 wins) ; game 1: p2, p2 (p1 wins)
    seq = [("p1", 0.2, "p2", 0), ("p1", 0.1, "p2", 0), ("p2", -0.1, "p2", 0), ("p1", 0.0, "p2", 0),
           ("p2", 0.3, "p1", 1), ("p2", 0.2, "p1", 1)]
    for s, v, w, g in seq:
        buf.add(_step(s, v, w, g))
    b = buf.compute(0.99, 0.9)
    # det_next: same game, same seat, adjacent rows only
    assert b["det_next"].tolist() == [1, -1, -1, -1, 5, -1]
    # ret_p1 = seat sign * lambda-return, independent of the p1 weight / normalisation
    _, r0 = gae(np.array([0.2, 0.1, 0.0], np.float32), np.array([0, 0, -1.0], np.float32), 0.99, 0.9)
    _, r2 = gae(np.array([-0.1], np.float32), np.array([1.0], np.float32), 0.99, 0.9)     # p2 frame: p2 won
    _, r1 = gae(np.array([0.3, 0.2], np.float32), np.array([0, -1.0], np.float32), 0.99, 0.9)  # p2 frame: p1 won
    exp = [r0[0], r0[1], -r2[0], r0[2], -r1[0], -r1[1]]
    assert np.allclose(b["ret_p1"].numpy(), exp, atol=1e-6)
    b2 = buf.compute(0.99, 0.9, p1_adv_weight=2.0)
    assert torch.equal(b2["ret_p1"], b["ret_p1"])


def test_td_mix_changes_critic_target_only_when_set():
    torch.manual_seed(0)
    m0 = build_models(_cfg(critic_td_mix=0.0))
    batch = _batch(m0)
    batch["ret_p1"] = torch.linspace(-1, 1, batch["y_p1"].numel())
    results = []
    for mix in (0.0, 0.5):
        torch.manual_seed(0)
        m = build_models(_cfg(critic_td_mix=mix, critic_epochs=1))
        opt = torch.optim.Adam(list(m.actor.parameters()) + list(m.critic.parameters()), lr=1e-3)
        ppo_update(batch, m.actor, m.critic, opt, m.cfg if hasattr(m, "cfg") else _cfg(critic_td_mix=mix, critic_epochs=1),
                   0.01, train_actor=False)
        results.append(copy.deepcopy(m.critic.state_dict()))
    # same init + same batch, so the critics differ iff the soft label moved the loss
    diff = max(float((results[0][k] - results[1][k]).abs().max()) for k in results[0])
    assert diff > 0.0
    # legacy path: identical whether or not ret_p1 is in the batch when mix == 0
    torch.manual_seed(0)
    m = build_models(_cfg(critic_td_mix=0.0, critic_epochs=1))
    opt = torch.optim.Adam(list(m.actor.parameters()) + list(m.critic.parameters()), lr=1e-3)
    b_no = {k: v for k, v in batch.items() if k != "ret_p1"}
    ppo_update(b_no, m.actor, m.critic, opt, _cfg(critic_td_mix=0.0, critic_epochs=1), 0.01, train_actor=False)
    for k in results[0]:
        assert torch.equal(results[0][k], m.critic.state_dict()[k])


def test_jump_metric_pairs():
    from fishrl.data.features import HANDS_DIM, turn_index_for
    from fishrl.eval.metrics import estimator_metrics
    torch.manual_seed(0)
    m = build_models(_cfg())
    n = 6
    pub = torch.randn(n, HANDS_DIM)
    ti = turn_index_for("hands")
    pub[:, ti] = torch.tensor([3, 3, 3, 4, 7, 7]) / 40.0
    det_next = torch.tensor([1, 2, -1, -1, 5, -1])
    valid = torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.bool)     # row 5 invalid -> pair (4,5) dropped
    batch = {"valid": valid, "y_p1": torch.ones(n), "pub": pub, "god": torch.zeros(n, 1), "det_next": det_next}
    est = estimator_metrics(m, batch)
    assert est["critic_jump_n"] == 2                                 # (0,1) and (1,2); (2,3) not a pair
    with torch.no_grad():
        p = m.critic.p1_winprob(pub)
    exp = ((p[1] - p[0]).abs() + (p[2] - p[1]).abs()) / 2
    assert abs(est["critic_jump_mean"] - float(exp)) < 1e-6
    # a cross-turn pair is excluded
    pub[:, ti] = torch.tensor([3, 4, 4, 4, 7, 7]) / 40.0
    est = estimator_metrics(m, batch)
    assert est["critic_jump_n"] == 1
