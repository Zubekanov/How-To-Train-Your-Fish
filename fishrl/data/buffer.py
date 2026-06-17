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
    x_act: np.ndarray        # ACTOR_IN = OBS_DIM + N_NAMES (perspective ⊕ guess_in)
    mask: np.ndarray         # int8[A.N]
    action: int
    logp: float
    value: float             # seat-frame value from the (old) privileged critic
    god_feat: np.ndarray     # GOD_DIM
    pub_feat: np.ndarray     # PUB_DIM
    guess_in: np.ndarray     # N_NAMES (the guess folded into x_act)
    cnt_target: np.ndarray   # N_NAMES (opponent hand counts — guesser label)
    winner: str | None = None


@dataclass
class RolloutBuffer:
    steps: list = field(default_factory=list)

    def add(self, step: Step):
        self.steps.append(step)

    def __len__(self):
        return len(self.steps)

    def compute(self, gamma: float, lam: float) -> dict:
        """Assign per-seat GAE advantages and return stacked torch tensors."""
        # per-seat advantages over each seat's own ordered subsequence
        adv = np.zeros(len(self.steps), dtype=np.float32)
        by_seat: dict[str, list[int]] = {"p1": [], "p2": []}
        for i, s in enumerate(self.steps):
            by_seat[s.seat].append(i)
        for seat, idxs in by_seat.items():
            if not idxs:
                continue
            values = np.array([self.steps[i].value for i in idxs], dtype=np.float32)
            rewards = np.zeros(len(idxs), dtype=np.float32)
            rewards[-1] = seat_outcome(self.steps[idxs[-1]].winner, seat)
            a, _ = gae(values, rewards, gamma, lam)
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
