"""Scenario-based curriculum: short, targeted start-states mixed INTO self-play so
terminal win/loss reward lands close to the skill it teaches. Reward stays terminal
±1; scenarios shape only the initial-state distribution + termination.

Public API:
  get_scenario(name)            -> cached Scenario instance (pool built once, lazily)
  scenario_names()              -> registered names
  sample_scenario_name(w, rng)  -> weighted pick over names with weight > 0
  ScenarioEnv                   -> the FishAEC subclass that runs a scenario
"""
from __future__ import annotations

import numpy as np

from fishrl.train.scenarios.board_presence import BoardPresenceScenario
from fishrl.train.scenarios.deckout import DeckoutScenario
from fishrl.train.scenarios.env import ScenarioEnv
from fishrl.train.scenarios.known_threat import KnownThreatScenario

_REGISTRY = {c.name: c for c in (KnownThreatScenario, BoardPresenceScenario,
                                 DeckoutScenario)}
_INSTANCES: dict = {}


def scenario_names() -> list:
    return list(_REGISTRY)


def get_scenario(name: str):
    """Cached singleton per name, so each scenario's snapshot pool is built once."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown scenario {name!r}; have {scenario_names()}")
    if name not in _INSTANCES:
        _INSTANCES[name] = _REGISTRY[name]()
    return _INSTANCES[name]


def sample_scenario_name(weights: dict, rng: np.random.Generator) -> str:
    """Weighted choice over registered names with positive weight (unknown names
    ignored). Falls back to uniform over all registered names if none are positive."""
    names = [n for n in _REGISTRY if weights.get(n, 0.0) > 0.0]
    if not names:
        names = scenario_names()
        w = np.ones(len(names))
    else:
        w = np.array([weights[n] for n in names], dtype=float)
    return names[int(rng.choice(len(names), p=w / w.sum()))]


__all__ = ["ScenarioEnv", "get_scenario", "scenario_names", "sample_scenario_name"]
