"""Rollout storage: per-transition records, per-seat GAE, and tensor batching."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from fishrl.obs.encoder import OBS_DIM
from fishrl.train.advantages import gae, seat_outcome


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


@dataclass
class RolloutBuffer:
    steps: list = field(default_factory=list)
    games: list = field(default_factory=list)   # per-game winner, indexed by game_id

    def add(self, step: Step):
        self.steps.append(step)

    def __len__(self):
        return len(self.steps)

    def merge(self, other: "RolloutBuffer") -> None:
        """Append another buffer's games, renumbering its game_ids so they stay
        unique — GAE segments must never fuse across buffers that each started
        numbering at 0."""
        base = len(self.games)
        for s in other.steps:
            s.game_id += base
        self.steps.extend(other.steps)
        self.games.extend(other.games)

    def compute(self, gamma: float, lam: float) -> dict:
        """Assign per-(game, seat) GAE advantages and return stacked torch tensors."""
        # Advantages over each seat's ordered subsequence WITHIN one game: each game's
        # terminal ±1 lands on its own last decision and never bootstraps into the next
        # game's opening state. A truncated game (decision cap, no verdict) bootstraps
        # its tail with the critic's own last value instead of pretending it drew.
        adv = np.zeros(len(self.steps), dtype=np.float32)
        segments: dict[tuple, list[int]] = {}
        for i, s in enumerate(self.steps):
            segments.setdefault((s.game_id, s.seat), []).append(i)
        for (_gid, seat), idxs in segments.items():
            values = np.array([self.steps[i].value for i in idxs], dtype=np.float32)
            rewards = np.zeros(len(idxs), dtype=np.float32)
            last = self.steps[idxs[-1]]
            rewards[-1] = seat_outcome(last.winner, seat)
            bootstrap = values[-1] if last.truncated else 0.0
            a, _ = gae(values, rewards, gamma, lam, bootstrap=bootstrap)
            for j, i in enumerate(idxs):
                adv[i] = a[j]
        # normalize advantages jointly (both seats share the ±1 frame)
        if adv.std() > 1e-6:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        x_act = np.stack([s.x_act for s in self.steps])
        out = {
            "x_act": torch.as_tensor(x_act, dtype=torch.float32),
            "persp": torch.as_tensor(x_act[:, :OBS_DIM], dtype=torch.float32),
            "prev_guess": torch.as_tensor(np.stack([s.guess_in for s in self.steps]), dtype=torch.float32),
            "mask": torch.as_tensor(np.stack([s.mask for s in self.steps]), dtype=torch.float32),
            "action": torch.as_tensor([s.action for s in self.steps], dtype=torch.long),
            "old_logp": torch.as_tensor([s.logp for s in self.steps], dtype=torch.float32),
            "adv": torch.as_tensor(adv, dtype=torch.float32),
            "god": torch.as_tensor(np.stack([s.god_feat for s in self.steps]), dtype=torch.float32),
            "pub": torch.as_tensor(np.stack([s.pub_feat for s in self.steps]), dtype=torch.float32),
            "cnt": torch.as_tensor(np.stack([s.cnt_target for s in self.steps]), dtype=torch.float32),
            "y_p1": torch.as_tensor([1.0 if s.winner == "p1" else 0.0 for s in self.steps], dtype=torch.float32),
            "valid": torch.as_tensor([1.0 if s.winner is not None else 0.0 for s in self.steps], dtype=torch.float32),
        }
        return out
