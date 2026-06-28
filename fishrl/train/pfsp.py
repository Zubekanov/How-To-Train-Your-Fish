"""Prioritized Fictitious Self-Play (PFSP) league + opponent sampler.

The learner trains a `pool_frac` slice of each iteration against opponents drawn
from a LEAGUE rather than always its mirror self. League members are:

  * scripted anchors -- ``random``, ``attacker``, ``heuristic`` (also the eval
    anchors, so the pool optimizes directly toward what the dashboard measures), and
  * past selves -- a bounded ring of frozen actor+guesser snapshots appended each
    status report (this is the *fictitious* in fictitious self-play: training
    against a population of past versions fights cycling and catastrophic forgetting).

Each member carries a running (EMA) win-rate of the LEARNER against it. Opponents
are sampled with probability proportional to a priority of that win-rate
(:func:`priority`): ``hard`` weights the opponents you lose to, ``var`` weights even
matchups. Mastered opponents (win-rate -> 1) decay toward the ``eps`` floor so the
learner stops wasting games on them but never drops them entirely.

The collection of a single pool game lives in the collector
(``collect_vs_opponent`` / ``collect_heuristic_games``); this module owns only the
membership, sampling, and win-rate bookkeeping. Everything here is in-memory and
launch-time -- nothing is serialized into the checkpoint, so resume is unaffected
(the past-self ring simply refills over the first few reports after a restart).
"""
from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field

import numpy as np

# Scripted anchors the league can include (engine `heuristic` is driven separately
# via the sandbox path, but is still a first-class league member here).
SCRIPTED_KINDS = ("random", "attacker", "heuristic")


@dataclass
class LeagueMember:
    name: str
    kind: str                    # "random" | "attacker" | "heuristic" | "self" | "scenario"
    models: object = None        # frozen actor+guesser holder (kind == "self"); else None
    wr: float = 0.5              # EMA win-rate OF THE LEARNER vs this member / scenario
    games: int = 0               # pool games played vs this member (diagnostics)
    weight: float = 1.0          # fixed prior multiplier on the sampling priority (config bias)


@dataclass
class _FrozenSelf:
    """Minimal frozen opponent holder: just the actor + guesser the learner faces.
    Mirrors the attribute surface (`.actor`, `.guesser`) the match/collector code
    reads, without copying the critic/public (unused as an opponent)."""
    actor: object
    guesser: object


def _freeze_self(models, it: int) -> LeagueMember:
    """Deep-copy the learner's actor + guesser into a frozen (eval, grad-free)
    league member tagged with the iteration it was taken at."""
    actor = copy.deepcopy(models.actor)
    guesser = copy.deepcopy(models.guesser)
    for net in (actor, guesser):
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
    return LeagueMember(name=f"self@{it}", kind="self", models=_FrozenSelf(actor, guesser))


def priority(wr: float, mode: str = "hard", p: float = 2.0) -> float:
    """Unnormalized sampling weight for a member from the learner's win-rate `wr`.

    ``hard`` -> (1 - wr)^p : higher for opponents the learner loses to.
    ``var``  -> wr*(1 - wr): higher for even (~0.5) matchups.
    """
    x = min(max(float(wr), 0.0), 1.0)
    if mode == "var":
        return x * (1.0 - x)
    return (1.0 - x) ** p


@dataclass
class PFSPLeague:
    """The opponent population + prioritized sampler. Construct from a Config."""
    mode: str = "hard"
    p: float = 2.0
    eps: float = 0.05
    wr_ema: float = 0.1
    anchors: list = field(default_factory=list)        # fixed scripted members
    selves: deque = field(default_factory=deque)       # bounded ring of past selves

    @classmethod
    def from_config(cls, cfg) -> "PFSPLeague":
        anchors = [LeagueMember(name=k, kind=k)
                   for k in cfg.pfsp_anchors if k in SCRIPTED_KINDS]
        return cls(mode=cfg.pfsp_mode, p=cfg.pfsp_p, eps=cfg.pfsp_eps,
                   wr_ema=cfg.pfsp_wr_ema, anchors=anchors,
                   selves=deque(maxlen=max(0, int(cfg.league_size))))

    @classmethod
    def scenario_league(cls, cfg, names) -> "PFSPLeague":
        """A dedicated PFSP league over the curriculum SCENARIOS (one member each,
        `names` with a positive `scenario_weights` entry). Reuses the same priority
        sampler + EMA win-rate as the opponent league, so within the scenario-game
        budget each scenario is drawn by the learner's difficulty on it (``hard``
        favours scenarios it is losing) rather than a fixed share. `scenario_weights`
        survives as a fixed prior multiplier (and as the on/off switch via 0)."""
        anchors = [LeagueMember(name=n, kind="scenario",
                                weight=float(cfg.scenario_weights.get(n, 1.0)))
                   for n in names if cfg.scenario_weights.get(n, 1.0) > 0]
        return cls(mode=cfg.pfsp_mode, p=cfg.pfsp_p, eps=cfg.pfsp_eps,
                   wr_ema=cfg.pfsp_wr_ema, anchors=anchors, selves=deque(maxlen=0))

    def add_snapshot(self, models, it: int) -> None:
        """Append a frozen snapshot of the current learner as a past-self member
        (no-op when the ring is disabled, i.e. league_size == 0)."""
        if self.selves.maxlen and self.selves.maxlen > 0:
            self.selves.append(_freeze_self(models, it))

    def members(self) -> list:
        return list(self.anchors) + list(self.selves)

    def sample(self, rng: np.random.Generator) -> LeagueMember | None:
        """Draw one opponent ∝ priority(win-rate). Returns None if the league is
        empty (caller then falls back to mirror self-play for that game)."""
        members = self.members()
        if not members:
            return None
        weights = np.array([(priority(m.wr, self.mode, self.p) + self.eps) * m.weight
                            for m in members], dtype=np.float64)
        total = weights.sum()
        if not np.isfinite(total) or total <= 0:
            idx = int(rng.integers(len(members)))
        else:
            idx = int(rng.choice(len(members), p=weights / total))
        return members[idx]

    def update(self, member: LeagueMember, learner_won: bool) -> None:
        """EMA-update a member's learner win-rate from one decided pool game."""
        member.games += 1
        member.wr = (1.0 - self.wr_ema) * member.wr + self.wr_ema * (1.0 if learner_won else 0.0)

    def summary(self) -> str:
        """Compact 'name=wr(games)' line for the status log (sorted by hardest)."""
        ms = sorted(self.members(), key=lambda m: m.wr)
        return " ".join(f"{m.name}={m.wr:.2f}({m.games})" for m in ms)
