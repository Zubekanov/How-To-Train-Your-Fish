"""Hyperparameters for the self-play training stack."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    # rollout / credit
    gamma: float = 0.997
    lam: float = 0.95
    max_decisions: int = 2000
    games_per_iter: int = 8

    # PPO
    clip: float = 0.2
    ppo_epochs: int = 4
    minibatch: int = 256
    lr_ppo: float = 3e-4
    grad_clip: float = 1.0
    ent_start: float = 0.02
    ent_end: float = 0.005
    critic_coef: float = 0.5

    # auxiliary supervised heads
    lr_guesser: float = 1e-3
    lr_public: float = 1e-3
    aux_steps: int = 2          # SGD steps on guesser/public per iteration (between PPO updates)

    # schedule
    warmup_games: int = 64
    warmup_epochs: int = 3
    iters: int = 100
    hidden: tuple = (256, 256)
    seed: int = 0
    ckpt_dir: str = "checkpoints"

    def ent_coef(self, it: int) -> float:
        if self.iters <= 1:
            return self.ent_end
        frac = min(it / (self.iters - 1), 1.0)
        return self.ent_start + frac * (self.ent_end - self.ent_start)
