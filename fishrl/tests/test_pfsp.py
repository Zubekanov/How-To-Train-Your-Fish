"""PFSP: the priority weighting, the league sampler/bookkeeping, the
record-only-the-learner collector, and an end-to-end train with the full league."""
import numpy as np

from fishrl.data.buffer import RolloutBuffer
from fishrl.data.features import GOD_DIM, PUB_DIM
from fishrl.models.estimators import PrivilegedCritic
from fishrl.models.guesser import HandGuesser
from fishrl.models.policy import ACTOR_IN, MaskedActor
from fishrl.obs import vocab as V
from fishrl.spaces import action_space as A
from fishrl.train.collector import collect_vs_opponent
from fishrl.train.config import Config
from fishrl.train.pfsp import LeagueMember, PFSPLeague, priority
from fishrl.train.train_loop import Models, build_models, train


def _learner():
    return Models(MaskedActor(), PrivilegedCritic(), HandGuesser(), None)


# ── priority weighting ────────────────────────────────────────────────────────
def test_priority_hard_prefers_losses():
    # You lose to A (wr 0.1) more than B (wr 0.9) -> A gets the higher weight.
    assert priority(0.1, "hard", 2.0) > priority(0.9, "hard", 2.0)
    assert priority(1.0, "hard", 2.0) == 0.0          # mastered -> zero (before eps floor)


def test_priority_var_prefers_even_matchups():
    assert priority(0.5, "var") > priority(0.1, "var")
    assert priority(0.5, "var") > priority(0.9, "var")


# ── league sampling + bookkeeping ─────────────────────────────────────────────
def test_league_from_config_has_anchors_and_ring():
    cfg = Config(pfsp_anchors=("random", "attacker", "heuristic"), league_size=2)
    lg = PFSPLeague.from_config(cfg)
    assert {m.name for m in lg.members()} == {"random", "attacker", "heuristic"}
    lg.add_snapshot(build_models(cfg), it=10)
    lg.add_snapshot(build_models(cfg), it=20)
    lg.add_snapshot(build_models(cfg), it=30)          # ring of 2 -> oldest evicted
    selves = [m for m in lg.members() if m.kind == "self"]
    assert [m.name for m in selves] == ["self@20", "self@30"]


def test_league_sampling_favours_hard_member():
    lg = PFSPLeague(mode="hard", p=2.0, eps=0.01,
                    anchors=[LeagueMember("easy", "random", wr=0.99),
                             LeagueMember("hard", "attacker", wr=0.05)])
    rng = np.random.default_rng(0)
    picks = [lg.sample(rng).name for _ in range(400)]
    assert picks.count("hard") > picks.count("easy") * 3


def test_league_update_moves_winrate():
    lg = PFSPLeague(wr_ema=0.5)
    m = LeagueMember("x", "random", wr=0.5)
    lg.update(m, learner_won=True)
    assert m.wr == 0.75 and m.games == 1
    lg.update(m, learner_won=False)
    assert m.wr == 0.375 and m.games == 2


def test_empty_league_returns_none():
    assert PFSPLeague().sample(np.random.default_rng(0)) is None


# ── collector records only the learner, seat-balanced ─────────────────────────
def test_collect_vs_random_records_only_learner_p1():
    learner = _learner()
    opp = LeagueMember("random", "random")
    buf = collect_vs_opponent(learner, opp, n_games=2, base_seed=0, critic=learner.critic,
                              max_decisions=400, learner_seat="p1")
    assert isinstance(buf, RolloutBuffer) and len(buf) > 0
    assert all(s.seat == "p1" for s in buf.steps)      # pinned seat honoured
    for s in buf.steps:
        assert s.x_act.shape == (ACTOR_IN,)
        assert s.mask.shape == (A.N,) and int(s.mask.sum()) > 0 and s.mask[s.action] == 1
        assert s.god_feat.shape == (GOD_DIM,) and s.pub_feat.shape == (PUB_DIM,)
        assert s.cnt_target.shape == (V.N_NAMES,)


def test_collect_vs_opponent_seat_balances_by_default():
    learner = _learner()
    opp = LeagueMember("attacker", "attacker")
    buf = collect_vs_opponent(learner, opp, n_games=4, base_seed=3, max_decisions=400)
    seats = {s.seat for s in buf.steps}
    assert seats == {"p1", "p2"}                        # learner played both seats


def test_collect_vs_self_opponent():
    learner = _learner()
    frozen = build_models(Config())
    opp = LeagueMember("self@1", "self", models=frozen)  # Models exposes .actor/.guesser
    buf = collect_vs_opponent(learner, opp, n_games=2, base_seed=1, critic=learner.critic,
                              max_decisions=400)
    assert len(buf) > 0
    batch = buf.compute(gamma=0.99, lam=0.95)
    assert batch["x_act"].shape == (len(buf), ACTOR_IN)


# ── end-to-end ────────────────────────────────────────────────────────────────
def test_train_with_full_pfsp_league_runs():
    cfg = Config(iters=2, games_per_iter=6, warmup_games=4, warmup_epochs=1,
                 max_decisions=400, report_winrate_games=0, pool_frac=0.5,
                 pfsp_anchors=("random", "attacker", "heuristic"), league_size=4)
    train(cfg, build_models(cfg), log=lambda *a, **k: None)
