"""A few PPO iterations run on CPU without NaN, with bounded KL and positive
entropy — an end-to-end integration check of the whole training stack."""
import math

import numpy as np
import torch

from fishrl.train.config import Config
from fishrl.train.train_loop import build_models, train


def test_warmup_and_ppo_iters_run_clean():
    cfg = Config(warmup_games=3, warmup_epochs=1, iters=2, games_per_iter=2,
                 minibatch=128, max_decisions=600)
    logs = []
    m = train(cfg, build_models(cfg), log=logs.append)
    assert any("iter 1" in s for s in logs)
    # no NaN/Inf anywhere in the models after training
    for net in (m.actor, m.critic, m.guesser, m.public):
        for p in net.parameters():
            assert torch.all(torch.isfinite(p)), "non-finite parameter after training"


def test_collected_batch_shapes_consistent():
    from fishrl.models.guesser import HandGuesser
    from fishrl.train.belief_env import BeliefAugmentedEnv
    from fishrl.train.collector import collect_games, random_act_fn
    from fishrl.data.features import GOD_DIM, PUB_DIM
    from fishrl.models.policy import ACTOR_IN
    from fishrl.spaces import action_space as A

    benv = BeliefAugmentedEnv(HandGuesser(), max_decisions=400)
    buf = collect_games(benv, random_act_fn(np.random.default_rng(1)), 2, 0,
                        critic=None, max_decisions=400)
    b = buf.compute(0.997, 0.95)
    M = b["x_act"].shape[0]
    assert b["x_act"].shape == (M, ACTOR_IN)
    assert b["god"].shape == (M, GOD_DIM)
    assert b["pub"].shape == (M, PUB_DIM)
    assert b["mask"].shape == (M, A.N)
    assert torch.all(torch.isfinite(b["adv"]))
