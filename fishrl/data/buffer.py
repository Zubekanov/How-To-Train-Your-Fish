"""Rollout storage: per-transition records, per-seat GAE, and tensor batching.

Two backings, one API:
  * per-step `Step` objects (serial collectors, tests, eval tools) -- `add()`;
  * columnar (parallel collection): per-game dicts of stacked arrays (`cols`),
    each covering a contiguous run of decisions. The trainer's batch is built
    straight from these -- no ~100k Step objects are ever created (their
    construction + teardown was ~20% of the trainer's wall, 2026-08-23).
`steps` is a property: on a columnar buffer it materialises Step VIEWS lazily,
only for callers that actually index per-step objects.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from fishrl.obs.encoder import OBS_DIM
from fishrl.train.advantages import gae, seat_outcome


def _pconcat(parts: list) -> np.ndarray:
    """np.concatenate for a few hundred large contiguous blocks, copied by a thread
    pool: the slice-assignment releases the GIL, so ~5 GB/iteration of x_act + pub
    assembly runs at memory bandwidth instead of one core (was ~19% of the trainer)."""
    if len(parts) == 1:
        return np.array(parts[0])                       # copy: never hand out a pool view
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


def _dev_concat(parts: list, device) -> torch.Tensor:
    """Assemble per-game blocks straight into one device tensor (the only copy)."""
    total = sum(p.shape[0] for p in parts)
    out = torch.empty((total,) + tuple(parts[0].shape[1:]),
                      dtype=torch.from_numpy(np.asarray(parts[0][:1])).dtype, device=device)
    off = 0
    for p in parts:
        n = p.shape[0]
        out[off:off + n].copy_(torch.from_numpy(np.ascontiguousarray(p)), non_blocking=False)
        off += n
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


# column name -> Step attribute, for the big per-step arrays
_COL_ATTR = {"x_act": "x_act", "mask": "mask", "god": "god_feat", "pub": "pub_feat",
             "guess_in": "guess_in", "cnt": "cnt_target"}


def pack_steps(steps: list) -> dict:
    """Columnar form of a run of Steps (the parallel workers' transport format)."""
    st = steps
    if not st:
        return {"n": 0}
    god_shared = all(x.god_feat is st[0].god_feat for x in st)
    return {
        "n": len(st),
        "seat": np.array([x.seat == "p1" for x in st], dtype=np.bool_),
        "x_act": np.stack([x.x_act for x in st]),
        "mask": np.stack([x.mask for x in st]),
        "action": np.array([x.action for x in st], dtype=np.int64),
        "logp": np.array([x.logp for x in st], dtype=np.float64),
        "value": np.array([x.value for x in st], dtype=np.float64),
        "god": st[0].god_feat if god_shared else np.stack([x.god_feat for x in st]),
        "god_shared": god_shared,
        "pub": np.stack([x.pub_feat for x in st]),
        "guess_in": np.stack([x.guess_in for x in st]),
        "cnt": np.stack([x.cnt_target for x in st]),
        "winner": [x.winner for x in st],
        "game_id": np.array([x.game_id for x in st], dtype=np.int64),
        "truncated": np.array([x.truncated for x in st], dtype=np.bool_),
        "deckout_end": np.array([x.deckout_end for x in st], dtype=np.bool_),
    }


def unpack_steps(d: dict) -> list:
    """Step VIEWS into one columnar dict (no array copies)."""
    n = int(d.get("n", 0))
    if n == 0:
        return []
    god = d["god"]
    shared = bool(d.get("god_shared"))
    return [Step(
        seat="p1" if d["seat"][i] else "p2", x_act=d["x_act"][i], mask=d["mask"][i],
        action=int(d["action"][i]), logp=float(d["logp"][i]), value=float(d["value"][i]),
        god_feat=god if shared else god[i], pub_feat=d["pub"][i],
        guess_in=d["guess_in"][i], cnt_target=d["cnt"][i], winner=d["winner"][i],
        game_id=int(d["game_id"][i]), truncated=bool(d["truncated"][i]),
        deckout_end=bool(d["deckout_end"][i])) for i in range(n)]


class RolloutBuffer:
    def __init__(self, steps: list | None = None, games: list | None = None,
                 meta: list | None = None):
        self._steps: list = list(steps) if steps else []
        self.games: list = list(games) if games else []   # per-game winner, indexed by game_id
        self.meta: list = list(meta) if meta else []      # per-game {"truncated"} — how it ended
        self.cols: list = []                              # columnar backing (per-game dicts)
        self._colcache: dict = {}
        # When set (the trainer's update device), columnar `column()` assembles the big
        # arrays straight into a device tensor -- one copy per game block, no host concat,
        # no second host->device copy in fill_critic_values / ppo_update.
        self.device = None

    # ── backing ────────────────────────────────────────────────────────────────
    @property
    def columnar(self) -> bool:
        return bool(self.cols)

    @property
    def steps(self) -> list:
        """Per-step records. On a columnar buffer these are materialised lazily as
        views (and then kept) -- the trainer's own path never asks for them."""
        if self.cols and not self._steps:
            for c in self.cols:
                self._steps.extend(unpack_steps(c))
        return self._steps

    @steps.setter
    def steps(self, value: list) -> None:
        self._steps = list(value)
        self.cols = []
        self._colcache = {}

    def add(self, step: Step):
        if self.cols:                                     # fall back to per-step backing
            _ = self.steps
            self.cols = []
            self._colcache = {}
        self._steps.append(step)

    def set_cols(self, cols: list) -> None:
        """Adopt columnar backing (parallel collection)."""
        self.cols = list(cols)
        self._steps = []
        self._colcache = {}

    def __len__(self):
        if self.cols:
            return sum(int(c["n"]) for c in self.cols)
        return len(self._steps)

    def release(self) -> list:
        """Drop the per-step arrays and columnar backing (shared-memory views under
        parallel collection) once the batch tensors exist, so the mappings -- and the
        fd each one pins -- go away now rather than at the next rebinding. Returns the
        pooled block ids the columns lived in (for ParallelCollector.recycle)."""
        ids = [c["_blk"] for c in self.cols if isinstance(c, dict) and "_blk" in c]
        self._steps = []
        self.cols = []
        self._colcache = {}
        return ids

    # ── columns ────────────────────────────────────────────────────────────────
    def _god_all_shared_zero(self) -> bool:
        return bool(self.cols) and all(c.get("god_shared") and not np.any(c["god"])
                                       for c in self.cols if int(c["n"]))

    def column(self, name: str):
        """Stacked (T, ...) array for one of x_act/mask/god/pub/guess_in/cnt -- a numpy
        array, or a torch tensor on `self.device` when that is set (columnar only)."""
        if self.cols:
            if name in self._colcache:
                return self._colcache[name]
            parts = []
            for c in self.cols:
                if not int(c["n"]):
                    continue
                a = c[name]
                if name == "god" and c.get("god_shared"):
                    a = np.broadcast_to(a, (int(c["n"]),) + a.shape)
                parts.append(a)
            if not parts:
                out = np.zeros((0,), dtype=np.float32)
            elif self.device is not None and name != "god":
                out = _dev_concat(parts, self.device)
            else:
                out = _pconcat(parts)
            self._colcache[name] = out
            return out
        return np.stack([getattr(s, _COL_ATTR[name]) for s in self._steps])

    def scalar(self, name: str) -> np.ndarray:
        """Per-step scalar column: seat (bool p1), action, logp, value, game_id,
        truncated, deckout_end, y_p1 (bool), valid (bool)."""
        if self.cols:
            if name in ("y_p1", "valid"):
                w = [x for c in self.cols if int(c["n"]) for x in c["winner"]]
                return (np.array([x == "p1" for x in w], dtype=np.bool_) if name == "y_p1"
                        else np.array([x is not None for x in w], dtype=np.bool_))
            parts = [np.asarray(c[name]) for c in self.cols if int(c["n"])]
            return np.concatenate(parts) if parts else np.zeros((0,))
        st = self._steps
        if name == "seat":
            return np.array([s.seat == "p1" for s in st], dtype=np.bool_)
        if name == "y_p1":
            return np.array([s.winner == "p1" for s in st], dtype=np.bool_)
        if name == "valid":
            return np.array([s.winner is not None for s in st], dtype=np.bool_)
        dt = {"action": np.int64, "game_id": np.int64, "truncated": np.bool_,
              "deckout_end": np.bool_}.get(name, np.float64)
        return np.array([getattr(s, name) for s in st], dtype=dt)

    def set_values(self, values: np.ndarray) -> None:
        """Write the critic's seat-frame values for every step (fill_critic_values)."""
        if self.cols:
            off = 0
            for c in self.cols:
                n = int(c["n"])
                if n:
                    c["value"] = np.asarray(values[off:off + n], dtype=np.float64)
                off += n
            return
        for s, v in zip(self._steps, values):
            s.value = float(v)

    # ── composition ────────────────────────────────────────────────────────────
    def merge(self, other: "RolloutBuffer") -> None:
        """Append another buffer's games, renumbering its game_ids so they stay
        unique — GAE segments must never fuse across buffers that each started
        numbering at 0."""
        base = len(self.games)
        self._colcache = {}
        both_cols = other.cols and (self.cols or len(self) == 0)
        if both_cols:
            for c in other.cols:
                c = dict(c)
                if int(c["n"]):
                    c["game_id"] = np.asarray(c["game_id"], dtype=np.int64) + base
                self.cols.append(c)
            self._steps = []
        else:
            mine = self.steps                             # materialise if needed
            for s in other.steps:
                s.game_id += base
            mine.extend(other.steps)
            self.cols = []
        self.games.extend(other.games)
        self.meta.extend(other.meta)

    # ── batch ──────────────────────────────────────────────────────────────────
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
        T = len(self)
        seat_p1 = self.scalar("seat")
        value = self.scalar("value").astype(np.float32)
        game_id = self.scalar("game_id")
        truncated = self.scalar("truncated")
        winners = ([x for c in self.cols if int(c["n"]) for x in c["winner"]] if self.cols
                   else [s.winner for s in self._steps])
        adv = np.zeros(T, dtype=np.float32)
        # raw seat-frame lambda-return (adv + V, BEFORE the p1 weight and the joint
        # normalisation), flipped to the p1 frame: the critic's bootstrapped target
        # for Config.critic_td_mix. Lives in [-1, 1] like the outcome it blends with.
        ret_p1 = np.zeros(T, dtype=np.float32)
        # index of the next decision of the SAME seat in the SAME game with no opponent
        # decision in between (steps are appended in decision order), else -1: the
        # deterministic tap/float/pay/cast chains the critic-jump telemetry scores.
        det_next = np.full(T, -1, dtype=np.int64)
        segments: dict[tuple, list[int]] = {}
        for i in range(T):
            segments.setdefault((int(game_id[i]), bool(seat_p1[i])), []).append(i)
        for (_gid, p1), idxs in segments.items():
            seat = "p1" if p1 else "p2"
            values = value[idxs]
            rewards = np.zeros(len(idxs), dtype=np.float32)
            last = idxs[-1]
            rewards[-1] = seat_outcome(winners[last], seat)
            bootstrap = values[-1] if truncated[last] else 0.0
            a, ret = gae(values, rewards, gamma, lam, bootstrap=bootstrap)
            sgn = 1.0 if p1 else -1.0
            ia = np.asarray(idxs)
            ret_p1[ia] = sgn * ret
            nxt = ia[1:]
            here = ia[:-1]
            adjacent = nxt == here + 1
            det_next[here[adjacent]] = nxt[adjacent]
            if p1 and p1_adv_weight != 1.0:
                a = a * p1_adv_weight
            adv[ia] = a
        # normalize advantages jointly (both seats share the ±1 frame); the p1 up-weight
        # rides through as a preserved p1:p2 magnitude ratio (normalization is a global affine)
        if adv.std() > 1e-6:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        x_act = self.column("x_act")
        y_p1 = self.scalar("y_p1")
        valid = self.scalar("valid")
        deckout = self.scalar("deckout_end")
        out = {
            "x_act": torch.as_tensor(x_act, dtype=torch.float32),
            "persp": torch.as_tensor(x_act[:, :OBS_DIM], dtype=torch.float32),
            "prev_guess": torch.as_tensor(self.column("guess_in"), dtype=torch.float32),
            "mask": torch.as_tensor(self.column("mask")).to(torch.float32),
            "action": torch.as_tensor(self.scalar("action"), dtype=torch.long),
            "old_logp": torch.as_tensor(self.scalar("logp"), dtype=torch.float32),
            "adv": torch.as_tensor(adv, dtype=torch.float32),
            "ret_p1": torch.as_tensor(ret_p1, dtype=torch.float32),
            "det_next": torch.as_tensor(det_next, dtype=torch.long),
            # public-family runs never read "god"; when every game shipped one shared
            # zero row, emit a (T, 1) zero placeholder instead of materialising
            # T x GOD_DIM zeros (3.5 GB at 360 games/iter)
            "god": (torch.zeros(T, 1) if self._god_all_shared_zero()
                    else torch.as_tensor(np.ascontiguousarray(self.column("god")), dtype=torch.float32)),
            "pub": torch.as_tensor(self.column("pub"), dtype=torch.float32),
            "cnt": torch.as_tensor(self.column("cnt"), dtype=torch.float32),
            # seat-frame critic value + seat sign: lets estimator_metrics score the
            # pre-update critic from fill_critic_values' forward instead of redoing it
            "value": torch.as_tensor(value, dtype=torch.float32),
            "seat_sign": torch.as_tensor(np.where(seat_p1, 1.0, -1.0), dtype=torch.float32),
            "y_p1": torch.as_tensor(y_p1, dtype=torch.float32),
            "valid": torch.as_tensor(valid, dtype=torch.float32),
            # parity-aux label domain: decided games that ended by decking. y is the
            # same p1-win bit, but the aux loss only reads rows where valid is set.
            "deckout_valid": torch.as_tensor(deckout & valid, dtype=torch.float32),
        }
        return out
