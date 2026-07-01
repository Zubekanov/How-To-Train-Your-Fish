"""Rollout-buffer credit assignment: GAE must segment per (game, seat) — one game's
terminal reward never leaks into another — truncation bootstraps instead of drawing,
and the buffered guesser input is the seat's carried PREVIOUS guess (the training/
inference contract the belief channel depends on)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from fishrl.data.buffer import RolloutBuffer, Step
from fishrl.obs.encoder import OBS_DIM
from fishrl.obs import vocab as V
from fishrl.train.advantages import gae
from fishrl.train.belief_env import BeliefAugmentedEnv
from fishrl.train.collector import collect_games, random_act_fn


def _step(seat, gid, winner, value=0.0, truncated=False):
    return Step(seat=seat, x_act=np.zeros(4, dtype=np.float32),
                mask=np.ones(2, dtype=np.int8), action=0, logp=0.0, value=value,
                god_feat=np.zeros(2, dtype=np.float32), pub_feat=np.zeros(2, dtype=np.float32),
                guess_in=np.zeros(2, dtype=np.float32), cnt_target=np.zeros(2, dtype=np.float32),
                winner=winner, game_id=gid, truncated=truncated)


# ── per-game GAE segmentation ─────────────────────────────────────────────────
def test_gae_segments_per_game_no_cross_game_leak():
    # Two games for the same seat, opposite outcomes, zero critic values. With
    # gamma=lam=1 each game's advantage is its own terminal ±1 at every step.
    # The old seat-only partition scored game 0 as a draw whose tail bootstrapped
    # into game 1 — its steps would NOT carry +1.
    buf = RolloutBuffer()
    buf.steps = [_step("p1", 0, "p1") for _ in range(3)] + \
                [_step("p1", 1, "p2") for _ in range(3)]
    buf.games = ["p1", "p2"]
    batch = buf.compute(gamma=1.0, lam=1.0)
    a = batch["adv"].numpy()
    assert (a[:3] > 0).all(), "winning game's steps must all carry positive advantage"
    assert (a[3:] < 0).all(), "losing game's steps must all carry negative advantage"
    # pre-normalization the magnitudes are exactly 1 everywhere -> post-normalization
    # they stay equal; any cross-game bootstrap would skew the boundary steps
    assert np.allclose(np.abs(a), np.abs(a[0]))


def test_gae_seat_segments_within_game_still_separate():
    # One game, both seats interleaved: winner's steps positive, loser's negative.
    buf = RolloutBuffer()
    buf.steps = [_step("p1", 0, "p1"), _step("p2", 0, "p1"),
                 _step("p1", 0, "p1"), _step("p2", 0, "p1")]
    buf.games = ["p1"]
    a = buf.compute(gamma=1.0, lam=1.0)["adv"].numpy()
    assert (a[[0, 2]] > 0).all() and (a[[1, 3]] < 0).all()


def test_merge_renumbers_game_ids():
    a, b = RolloutBuffer(), RolloutBuffer()
    a.steps, a.games = [_step("p1", 0, "p1")], ["p1"]
    b.steps, b.games = [_step("p1", 0, "p2")], ["p2"]
    a.merge(b)
    assert [s.game_id for s in a.steps] == [0, 1]
    assert a.games == ["p1", "p2"]
    # and the segments stay separate: opposite signs even though both were game_id 0
    adv = a.compute(gamma=1.0, lam=1.0)["adv"].numpy()
    assert adv[0] > 0 > adv[1]


def test_truncated_game_bootstraps_instead_of_drawing():
    # A confident position (value 0.8) cut by the decision cap must NOT be scored
    # as a draw (reward 0, bootstrap 0 -> advantage ≈ -0.8 at the tail).
    v = np.full(4, 0.8, dtype=np.float32)
    r = np.zeros(4, dtype=np.float32)
    adv_cut, _ = gae(v, r, gamma=0.99, lam=0.95, bootstrap=0.8)
    adv_draw, _ = gae(v, r, gamma=0.99, lam=0.95, bootstrap=0.0)
    assert abs(adv_cut[-1]) < 0.05          # ~neutral: outcome unknown, not "you drew"
    assert adv_draw[-1] < -0.7
    # end-to-end through compute(): normalization centres/rescales, so anchor the
    # scale with a symmetric win + loss pair and compare the SAME cut game scored
    # both ways. Bootstrapped, its advantages sit near the middle of the ±1
    # anchors; scored as a draw (truncated=False, winner None) they get dragged
    # down toward the loss cluster.
    def _mixed(truncated):
        buf = RolloutBuffer()
        buf.steps = ([_step("p1", 0, "p1", value=0.0) for _ in range(3)]
                     + [_step("p1", 1, None, value=0.8, truncated=truncated) for _ in range(3)]
                     + [_step("p1", 2, "p2", value=0.0) for _ in range(3)])
        buf.games = ["p1", None, "p2"]
        return buf.compute(gamma=0.99, lam=0.95)["adv"].numpy()

    a_boot, a_draw = _mixed(True), _mixed(False)
    cut_boot, cut_draw = a_boot[3:6], a_draw[3:6]
    win_b, loss_b = a_boot[:3].mean(), a_boot[6:].mean()
    # draw-scoring punishes the confident cut game; bootstrapping stays ~neutral
    assert (cut_boot > cut_draw + 0.3).all(), (cut_boot, cut_draw)
    assert abs(cut_boot.mean()) < 0.5 * abs(loss_b), (cut_boot, loss_b)
    assert cut_draw.mean() < 0.4 * loss_b < 0        # dragged toward the loss cluster


# ── guesser input contract ────────────────────────────────────────────────────
class _CountingGuesser(torch.nn.Module):
    """Stub guesser: output = prev + 1 elementwise. Makes the carried-previous-guess
    contract directly observable: step k's input is k*ones, output (k+1)*ones."""
    def __init__(self):
        super().__init__()
        self._dev_probe = torch.nn.Parameter(torch.zeros(1))  # device_of() support

    def forward(self, persp, prev):
        return prev + 1.0


def test_collect_games_buffers_the_consumed_previous_guess():
    env = BeliefAugmentedEnv(_CountingGuesser(), max_decisions=200)
    buf = collect_games(env, random_act_fn(np.random.default_rng(0)), 1, 0,
                        critic=None, max_decisions=200)
    assert buf.steps
    per_seat_count = {"p1": 0, "p2": 0}
    for s in buf.steps:
        k = per_seat_count[s.seat]
        # buffered guesser INPUT = the seat's carried previous guess (k ones)...
        assert np.allclose(s.guess_in, k), (s.seat, k, s.guess_in[:3])
        # ...while the actor input carries the CURRENT output (k+1 ones)
        assert np.allclose(s.x_act[OBS_DIM:], k + 1)
        per_seat_count[s.seat] += 1
    # first decision of each seat consumed the zero prior
    assert buf.games and len(buf.games) == 1


def test_collect_games_records_zero_step_games():
    # buf.games must carry one entry per game even if a game yields steps for only
    # one seat or ends early — league updates count games, not transitions.
    env = BeliefAugmentedEnv(_CountingGuesser(), max_decisions=200)
    buf = collect_games(env, random_act_fn(np.random.default_rng(1)), 3, 10,
                        critic=None, max_decisions=200)
    assert len(buf.games) == 3
    assert all(w in ("p1", "p2", None) for w in buf.games)
    # every step's game_id indexes into games and matches its winner label
    for s in buf.steps:
        assert buf.games[s.game_id] == s.winner
