"""Model shapes, masked-softmax legality, and head ranges (flat and entity)."""
import numpy as np
import pytest
import torch

from fishrl.data.features import GOD_DIM, PUB_DIM
from fishrl.models.estimators import PrivilegedCritic, PublicEstimator
from fishrl.models.guesser import HandGuesser, GUESSER_IN, GUESS_DIM
from fishrl.models.policy import ACTOR_IN, MaskedActor, masked_log_softmax
from fishrl.obs.encoder import OBS_DIM
from fishrl.obs import vocab as V
from fishrl.spaces import action_space as A

ENCODERS = ["flat", "entity"]


@pytest.mark.parametrize("encoder", ENCODERS)
def test_actor_in_out_dims(encoder):
    assert ACTOR_IN == OBS_DIM + V.N_NAMES
    actor = MaskedActor(encoder=encoder)
    x = torch.zeros(4, ACTOR_IN)
    assert actor(x).shape == (4, A.N)


def test_masked_softmax_legal_support_and_sampling():
    logits = torch.randn(A.N)
    mask = torch.zeros(A.N, dtype=torch.int8)
    legal = [0, 5, 17, 100]
    for i in legal:
        mask[i] = 1
    logp = masked_log_softmax(logits, mask)
    p = logp.exp()
    assert abs(float(p.sum()) - 1.0) < 1e-5
    illegal = [i for i in range(A.N) if i not in legal]
    assert float(p[illegal].sum()) < 1e-6
    for _ in range(200):
        a = int(torch.multinomial(p, 1))
        assert a in legal


@pytest.mark.parametrize("encoder", ENCODERS)
def test_guesser_nonneg_and_dims(encoder):
    assert GUESSER_IN == OBS_DIM + GUESS_DIM
    gss = HandGuesser(encoder=encoder)
    out = gss(torch.zeros(3, OBS_DIM), torch.zeros(3, GUESS_DIM))
    assert out.shape == (3, GUESS_DIM)
    assert torch.all(out >= 0)


@pytest.mark.parametrize("encoder", ENCODERS)
def test_estimators_winprob_range(encoder):
    pc, pe = PrivilegedCritic(encoder=encoder), PublicEstimator(encoder=encoder)
    with torch.no_grad():
        wp = pc.p1_winprob(torch.zeros(2, GOD_DIM))
    assert float(wp.min()) >= 0.0 and float(wp.max()) <= 1.0
    assert pe.forward(torch.zeros(2, PUB_DIM)).shape == (2,)
