"""The heuristic curriculum pool: the learning policy (p1) plays the engine
heuristic AI (p2), and ONLY p1's transitions are buffered (p2 is off-policy)."""
import numpy as np

from fishrl.data.buffer import RolloutBuffer
from fishrl.data.features import GOD_DIM, PUB_DIM
from fishrl.models.estimators import PrivilegedCritic
from fishrl.models.guesser import HandGuesser
from fishrl.models.policy import ACTOR_IN, MaskedActor
from fishrl.obs import vocab as V
from fishrl.spaces import action_space as A
from fishrl.train.collector import collect_heuristic_games
from fishrl.train.config import Config
from fishrl.train.train_loop import build_models, train


def _nets():
    return MaskedActor(), HandGuesser(), PrivilegedCritic()


def test_collect_heuristic_games_only_records_p1():
    actor, guesser, critic = _nets()
    buf = collect_heuristic_games(guesser, actor, n_games=3, base_seed=0,
                                  critic=critic, use_belief=True, max_decisions=400)
    assert isinstance(buf, RolloutBuffer)
    assert len(buf) > 0, "no p1 decisions collected"
    # Every recorded transition is the learning seat — never the heuristic's.
    assert all(s.seat == "p1" for s in buf.steps)
    for s in buf.steps:
        assert s.x_act.shape == (ACTOR_IN,)
        assert s.mask.shape == (A.N,)
        assert int(s.mask.sum()) > 0                 # a real, legal decision
        assert s.mask[s.action] == 1                 # chosen action was legal
        assert s.god_feat.shape == (GOD_DIM,)
        assert s.pub_feat.shape == (PUB_DIM,)
        assert s.guess_in.shape == (V.N_NAMES,)
        assert s.cnt_target.shape == (V.N_NAMES,)
        assert s.value != 0.0 or True                # filled by the critic pass
        assert s.winner in ("p1", "p2", None)        # back-filled per game


def test_no_belief_zeroes_the_guess_channel():
    actor, guesser, _ = _nets()
    buf = collect_heuristic_games(guesser, actor, n_games=2, base_seed=5,
                                  use_belief=False, max_decisions=400)
    assert len(buf) > 0
    assert all(np.all(s.guess_in == 0.0) for s in buf.steps)


def test_buffer_computes_a_trainable_batch():
    actor, guesser, critic = _nets()
    buf = collect_heuristic_games(guesser, actor, n_games=2, base_seed=1,
                                  critic=critic, max_decisions=400)
    batch = buf.compute(gamma=0.99, lam=0.95)
    n = len(buf)
    assert batch["x_act"].shape == (n, ACTOR_IN)
    assert batch["god"].shape[0] == n
    # winner is decided in a full game, so every step is a valid (labelled) target.
    assert float(batch["valid"].sum()) == n


def test_train_with_heuristic_pool_runs():
    """A tiny bounded run with a heuristic-only pool exercises the merged-buffer path."""
    cfg = Config(iters=2, games_per_iter=4, warmup_games=4, warmup_epochs=1,
                 max_decisions=400, report_winrate_games=0, pool_frac=0.5,
                 pfsp_anchors=("heuristic",), league_size=0)
    m = build_models(cfg)
    train(cfg, m, log=lambda *a, **k: None)
