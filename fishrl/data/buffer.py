"""Rollout storage: per-transition records, per-seat GAE, and tensor batching."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from fishrl.obs.encoder import OBS_DIM
from fishrl.train.advantages import gae, seat_outcome


def _pconcat(parts: list) -> np.ndarray:
    """np.concatenate for a few hundred large contiguous blocks, copied by a thread
    pool: the slice-assignment releases the GIL, so ~5 GB/iteration of x_act + pub
    assembly runs at memory bandwidth instead of one core (was ~19% of the trainer)."""
    if len(parts) == 1:
        return np.asarray(parts[0])
    total = sum(p.shape[0] for p in parts)
    if total * parts[0][0].nbytes < (64 << 20):
        return np.concatenate(parts)
    from concurrent.futures import ThreadPoolExecutor
    out = np.empty((total,) + parts[0].shape[1:], dtype=parts[0].dtype)
    offs = np.cumsum([0] + [p.shape[0] for p in parts[:-1]])

    def _cp(i):
        out[offs[i]:offs[i] + parts[i].shape[0]] = parts[i]
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(_cp, range(len(parts))))
    return out


@dataclass
class Step:
    seat: str
    x_act: np.ndarray        # ACTOR_IN = OBS_DIM + N_NAMES (perspective ⊕ current guess)
    mask: np.ndarray         # int8[A.N]
    action: int
    logp: float
    value: float             # seat-frame value from the (old) privileged critic
    god_feat: np.ndarray     # GOD_DIM
    pub_feat: np.ndarray     # PUB_DIM
    guess_in: np.ndarray     # N_NAMES — the guesser's INPUT at this step (the seat's carried
                             # PREVIOUS guess), matching what inference fed. The guess it
                             # produced lives in x_act[OBS_DIM:].
    cnt_target: np.ndarray   # N_NAMES (opponent hand counts — guesser label)
    winner: str | None = None
    game_id: int = 0         # buffer-local game index (renumbered by `merge`) so GAE
                             # never crosses a game boundary
    truncated: bool = False  # decision-cap cut (winner None but the game wasn't decided)
    deckout_end: bool = False  # the game ended by an empty-library draw (engine reason) —
                               # the label domain of the public critic's parity aux head


@dataclass
class RolloutBuffer:
    steps: list = field(default_factory=list)
    games: list = field(default_factory=list)   # per-game winner, indexed by game_id
    meta: list = field(default_factory=list)    # per-game {"truncated"} — how it ended
    # Optional columnar backing (parallel collection): per-game dicts of stacked
    # arrays, each covering a contiguous run of `steps` in order, summing to
    # len(steps). When present, `column()` concatenates them instead of re-stacking
    # ~30k per-step arrays (the np.stack was ~14% of the trainer's wall). The Step
    # arrays are views INTO these, so the two never disagree.
    cols: list = field(default_factory=list)
    _colcache: dict = field(default_factory=dict)   # name -> concatenated array (per cols state)

    def add(self, step: Step):
        self.steps.append(step)
        self.cols = []                              # per-step appends -> no columnar backing
        self._colcache = {}

    _COL_ATTR = {"x_act": "x_act", "mask": "mask", "god": "god_feat", "pub": "pub_feat",
                 "guess_in": "guess_in", "cnt": "cnt_target"}

    def release(self) -> list:
        """Drop the per-step arrays and columnar backing (shared-memory views under
        parallel collection) once the batch tensors exist, so the mappings -- and the
        fd each one pins -- go away now rather than at the next rebinding. Returns the
        pooled block ids the columns lived in (for ParallelCollector.recycle)."""
        ids = [c["_blk"] for c in self.cols if isinstance(c, dict) and "_blk" in c]
        self.steps = []
        self.cols = []
        self._colcache = {}
        return ids

    def _god_all_shared_zero(self) -> bool:
        return (bool(self.cols) and sum(int(c["n"]) for c in self.cols) == len(self.steps)
                and all(c.get("god_shared") and not np.any(c["god"]) for c in self.cols))

    def column(self, name: str) -> np.ndarray:
        """Stacked (T, ...) array for one of x_act/mask/god/pub/guess_in/cnt."""
        if self.cols and sum(int(c["n"]) for c in self.cols) == len(self.steps):
            if name in self._colcache:
                return self._colcache[name]
            parts = []
            for c in self.cols:
                a = c[name]
                if name == "god" and c.get("god_shared"):
                    a = np.broadcast_to(a, (int(c["n"]),) + a.shape)
                parts.append(a)
            out = _pconcat(parts) if len(parts) > 1 else np.array(parts[0])   # copy: never a pool view
            self._colcache[name] = out
            return out
        attr = self._COL_ATTR[name]
        return np.stack([getattr(s, attr) for s in self.steps])

    def __len__(self):
        return len(self.steps)

    def merge(self, other: "RolloutBuffer") -> None:
        """Append another buffer's games, renumbering its game_ids so they stay
        unique — GAE segments must never fuse across buffers that each started
        numbering at 0."""
        base = len(self.games)
        for s in other.steps:
            s.game_id += base
        had_cols = bool(self.cols) or not self.steps
        self.steps.extend(other.steps)
        self.games.extend(other.games)
        self.meta.extend(other.meta)
        self._colcache = {}
        if had_cols and other.cols and sum(int(c["n"]) for c in other.cols) == len(other.steps):
            self.cols.extend(other.cols)
        elif other.steps:
            self.cols = []                          # mixed backing -> fall back to stacking

    def compute(self, gamma: float, lam: float, p1_adv_weight: float = 1.0) -> dict:
        """Assign per-(game, seat) GAE advantages and return stacked torch tensors.

        `p1_adv_weight` (>1) scales the p1-seat advantages before the joint normalization,
        weighting p1 decisions more heavily in the policy gradient. The game is seat-
        symmetric but self-play drifts asymmetric, and the objective (vs-heuristic) only
        ever measures p1 -- so steering capacity toward p1 targets the seat that counts.
        1.0 leaves the advantages untouched (the historic symmetric update)."""
        # Advantages over each seat's ordered subsequence WITHIN one game: each game's
        # terminal ±1 lands on its own last decision and never bootstraps into the next
        # game's opening state. A truncated game (decision cap, no verdict) bootstraps
        # its tail with the critic's own last value instead of pretending it drew.
        adv = np.zeros(len(self.steps), dtype=np.float32)
        # raw seat-frame lambda-return (adv + V, BEFORE the p1 weight and the joint
        # normalisation), flipped to the p1 frame: the critic's bootstrapped target
        # for Config.critic_td_mix. Lives in [-1, 1] like the outcome it blends with.
        ret_p1 = np.zeros(len(self.steps), dtype=np.float32)
        # index of the next decision of the SAME seat in the SAME game with no opponent
        # decision in between (steps are appended in decision order), else -1: the
        # deterministic tap/float/pay/cast chains the critic-jump telemetry scores.
        det_next = np.full(len(self.steps), -1, dtype=np.int64)
        segments: dict[tuple, list[int]] = {}
        for i, s in enumerate(self.steps):
            segments.setdefault((s.game_id, s.seat), []).append(i)
        for (_gid, seat), idxs in segments.items():
            values = np.array([self.steps[i].value for i in idxs], dtype=np.float32)
            rewards = np.zeros(len(idxs), dtype=np.float32)
            last = self.steps[idxs[-1]]
            rewards[-1] = seat_outcome(last.winner, seat)
            bootstrap = values[-1] if last.truncated else 0.0
            a, ret = gae(values, rewards, gamma, lam, bootstrap=bootstrap)
            sgn = 1.0 if seat == "p1" else -1.0
            for j, i in enumerate(idxs):
                ret_p1[i] = sgn * ret[j]
                if j + 1 < len(idxs) and idxs[j + 1] == i + 1:
                    det_next[i] = i + 1
            if seat == "p1" and p1_adv_weight != 1.0:
                a = a * p1_adv_weight
            for j, i in enumerate(idxs):
                adv[i] = a[j]
        # normalize advantages jointly (both seats share the ±1 frame); the p1 up-weight
        # rides through as a preserved p1:p2 magnitude ratio (normalization is a global affine)
        if adv.std() > 1e-6:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        x_act = self.column("x_act")
        out = {
            "x_act": torch.as_tensor(x_act, dtype=torch.float32),
            "persp": torch.as_tensor(x_act[:, :OBS_DIM], dtype=torch.float32),
            "prev_guess": torch.as_tensor(self.column("guess_in"), dtype=torch.float32),
            "mask": torch.as_tensor(self.column("mask"), dtype=torch.float32),
            "action": torch.as_tensor([s.action for s in self.steps], dtype=torch.long),
            "old_logp": torch.as_tensor([s.logp for s in self.steps], dtype=torch.float32),
            "adv": torch.as_tensor(adv, dtype=torch.float32),
            "ret_p1": torch.as_tensor(ret_p1, dtype=torch.float32),
            "det_next": torch.as_tensor(det_next, dtype=torch.long),
            # public-family runs never read "god"; when every game shipped one shared
            # zero row, emit a (T, 1) zero placeholder instead of materialising
            # T x GOD_DIM zeros (3.5 GB at 360 games/iter)
            "god": (torch.zeros(len(self.steps), 1) if self._god_all_shared_zero()
                    else torch.as_tensor(np.ascontiguousarray(self.column("god")), dtype=torch.float32)),
            "pub": torch.as_tensor(self.column("pub"), dtype=torch.float32),
            "cnt": torch.as_tensor(self.column("cnt"), dtype=torch.float32),
            # seat-frame critic value + seat sign: lets estimator_metrics score the
            # pre-update critic from fill_critic_values' forward instead of redoing it
            "value": torch.as_tensor([s.value for s in self.steps], dtype=torch.float32),
            "seat_sign": torch.as_tensor([1.0 if s.seat == "p1" else -1.0 for s in self.steps], dtype=torch.float32),
            "y_p1": torch.as_tensor([1.0 if s.winner == "p1" else 0.0 for s in self.steps], dtype=torch.float32),
            "valid": torch.as_tensor([1.0 if s.winner is not None else 0.0 for s in self.steps], dtype=torch.float32),
            # parity-aux label domain: decided games that ended by decking. y is the
            # same p1-win bit, but the aux loss only reads rows where valid is set.
            "deckout_valid": torch.as_tensor(
                [1.0 if (s.deckout_end and s.winner is not None) else 0.0
                 for s in self.steps], dtype=torch.float32),
        }
        return out
