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
    # Front-end encoder: "flat" (MLP over the raw observation), "entity" (shared
    # card-embedding encoder, masked mean/max pool; far fewer params), or "attention"
    # (the entity front-end + cross-zone self-attention and a learned per-zone
    # attention pool — relational, less lossy than mean/max). Default "flat" so
    # behaviour is unchanged until the A/B (eval/ab_encoder) backs flipping it.
    #
    # `encoder` is the BASE applied to any net without an explicit override below.
    # The four nets have different throughput exposure, so the encoder is decoupled
    # per-net (resolve via `enc_for`):
    #   * actor / guesser run per-decision DURING collection -> on the throughput path.
    #   * critic runs per-decision too (value baseline) but its god_feat is buffered,
    #     so it is batchable post-collection; and it is gone at deployment entirely.
    #     -> an entity critic is a near-free calibration win once values are batched.
    #   * public is diagnostic-only.
    # On-policy A/B (eval/ab_encoder): entity Brier 0.261 vs flat 0.342 -> entity is
    # the calibration winner, so it's the natural critic override.
    encoder: str = "flat"
    # when False, the actor's belief channel is fed zeros (guesser skipped) -- the
    # controlled "is the belief worth anything?" ablation. Architecture is unchanged.
    use_belief: bool = True
    actor_encoder: str | None = None
    # critic defaults to entity: it's the measured on-policy calibration winner
    # (Brier 0.261 vs flat 0.342), off the deployment path, and ~free now that the
    # value pass is batched post-collection. The actor/guesser stay "flat" until a
    # win-rate head-to-head backs flipping them.
    critic_encoder: str | None = "entity"
    guesser_encoder: str | None = None
    public_encoder: str | None = None
    critic_hidden: tuple = (512, 512, 256)
    device: str = "cpu"          # "cpu" | "cuda" (set via resolve_device / --gpu)
    seed: int = 0
    ckpt_dir: str = "checkpoints"

    def enc_for(self, net: str) -> str:
        """Resolve the encoder for one net ('actor'|'critic'|'guesser'|'public'),
        falling back to the base `encoder` when no per-net override is set."""
        return getattr(self, f"{net}_encoder") or self.encoder

    def ent_coef(self, it: int) -> float:
        if self.iters <= 1:
            return self.ent_end
        frac = min(it / (self.iters - 1), 1.0)
        return self.ent_start + frac * (self.ent_end - self.ent_start)
