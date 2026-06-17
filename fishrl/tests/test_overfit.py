"""The guesser and privileged critic can fit a single small batch — proves the
feature/target wiring and that the heads have the capacity/signal expected."""
import numpy as np
import torch

from fishrl.models.estimators import PrivilegedCritic
from fishrl.models.guesser import HandGuesser
from fishrl.train.belief_env import BeliefAugmentedEnv
from fishrl.train.collector import collect_games, random_act_fn
from fishrl.train.config import Config
from fishrl.train.losses import guesser_poisson, outcome_bce


def _batch():
    cfg = Config(max_decisions=2000)
    benv = BeliefAugmentedEnv(HandGuesser(), max_decisions=cfg.max_decisions)
    buf = collect_games(benv, random_act_fn(np.random.default_rng(0)), 4, 0,
                        critic=None, max_decisions=cfg.max_decisions)
    return buf.compute(cfg.gamma, cfg.lam)


def test_guesser_overfits_batch():
    batch = _batch()
    # take a small slice to overfit
    n = min(128, batch["persp"].shape[0])
    persp, prev, cnt = batch["persp"][:n], batch["prev_guess"][:n], batch["cnt"][:n]
    g = HandGuesser()
    opt = torch.optim.Adam(g.parameters(), lr=3e-3)
    first = guesser_poisson(g(persp, prev), cnt).item()
    for _ in range(300):
        loss = guesser_poisson(g(persp, prev), cnt)
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < first * 0.5            # substantially reduced
    # predicted counts should round close to the true hand sizes on average
    with torch.no_grad():
        pred = g(persp, prev)
    mae = (pred - cnt).abs().mean().item()
    assert mae < 0.5


def test_privileged_critic_overfits_outcome():
    batch = _batch()
    keep = batch["valid"] > 0                     # decided games only
    god, y = batch["god"][keep][:256], batch["y_p1"][keep][:256]
    assert god.shape[0] >= 32, "need decided transitions to overfit"
    c = PrivilegedCritic()
    opt = torch.optim.Adam(c.parameters(), lr=3e-3)
    ones = torch.ones_like(y)
    for _ in range(300):
        loss = outcome_bce(c(god), y, ones)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = (torch.sigmoid(c(god)) > 0.5).float()
    acc = (pred == y).float().mean()
    assert float(acc) > 0.9                       # near-perfect on the training batch
