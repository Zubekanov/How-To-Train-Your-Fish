"""Hyperparameters for the self-play training stack."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


def resolve_device(gpu: bool) -> str:
    """Map a --gpu flag to a torch device string, falling back to CPU (with a
    warning) when CUDA was requested but isn't available."""
    if gpu and torch.cuda.is_available():
        return "cuda"
    if gpu:
        print("[warn] --gpu requested but CUDA is not available; running on CPU", flush=True)
    return "cpu"


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
    # Front-end encoder: "flat" (MLP over the raw observation) or "entity" (shared
    # card-embedding encoder; far fewer params, composes relational structure).
    # Default "flat" so behaviour is unchanged until the A/B (eval/ab_encoder) backs
    # flipping it. The critic is off the inference path, so it gets more capacity;
    # under MC-ish returns (lam≈1) that lowers advantage VARIANCE — lower `lam` to
    # make the bigger critic bias-relevant via bootstrapping.
    encoder: str = "flat"
    critic_hidden: tuple = (512, 512, 256)
    device: str = "cpu"          # "cpu" | "cuda" (set via resolve_device / --gpu)
    seed: int = 0
    ckpt_dir: str = "checkpoints"

    def ent_coef(self, it: int) -> float:
        if self.iters <= 1:
            return self.ent_end
        frac = min(it / (self.iters - 1), 1.0)
        return self.ent_start + frac * (self.ent_end - self.ent_start)
