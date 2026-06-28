"""ScenarioEnv: a FishAEC that resets from a scenario start-state and ends the
episode via the scenario's terminator (through FishAEC's `_terminal_override` hook).

Everything else — observation encoding, masks, the compound builder, the belief
wrapper, the collector — is reused unchanged, so scenario transitions are byte-
identical in shape to self-play transitions (same encoder + belief contract).
"""
from __future__ import annotations

import numpy as np

from fishrl.env.aec_env import FishAEC
from fishrl.env.driver import STOPS_MODES
from fishrl.forgetful_fish import engine as E


class ScenarioEnv(FishAEC):
    def __init__(self, scenario, stops_mode: str = "default", max_decisions: int = 2000):
        super().__init__(stops_mode=stops_mode, max_decisions=max_decisions)
        self.scenario = scenario
        self.scn_ctx: dict = {}

    def reset(self, seed=None, options=None):
        rng = np.random.default_rng(seed)
        g = self.scenario.sample(rng)               # legal, deep-copied start-state
        for pid in self.possible_agents:
            E.set_player_stops(g, pid, STOPS_MODES[self.stops_mode])
        self.g = g
        self.agents = list(self.possible_agents)
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self._builder = None
        self._decisions = 0
        self._scenario_result = None
        self.scn_ctx = {}
        self.scenario.on_reset(self)                 # capture start references
        self.agent_selection = self.possible_agents[0]
        self._refresh()

    def _terminal_override(self):
        return self.scenario.terminator(self)
