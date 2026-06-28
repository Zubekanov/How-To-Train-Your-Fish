"""PettingZoo AEC environment for Forgetful Fish self-play.

Both seats are driven as controlled ("human") seats of the vendored engine, which
pauses at every decision by setting ``g.pending``. ``g.pending.player`` is the
authority on whose turn it is; the env never assumes alternation. Each agent sees
a Dict observation ``{"observation", "action_mask"}``; only the acting agent has a
non-empty mask. Reward is sparse terminal ±1 (win/lose), 0 otherwise.
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv

from fishrl.forgetful_fish import engine as E
from fishrl.forgetful_fish.cards import load_decklist
from fishrl.env.apply import apply_atomic
from fishrl.env.driver import STOPS_MODES, is_terminal, terminal_winner
from fishrl.obs.encoder import OBS_DIM, encode_observation
from fishrl.spaces import action_space as A
from fishrl.spaces.compound import COMPOUND_TYPES, CompoundBuilder
from fishrl.spaces.masking import atomic_mask


class FishAEC(AECEnv):
    metadata = {"render_modes": [], "name": "forgetful_fish_v0", "is_parallelizable": False}

    def __init__(self, stops_mode: str = "default", max_decisions: int = 4000):
        super().__init__()
        if stops_mode not in STOPS_MODES:
            raise ValueError(f"stops_mode must be one of {tuple(STOPS_MODES)}")
        self.possible_agents = ["p1", "p2"]
        self.stops_mode = stops_mode
        self.max_decisions = max_decisions
        self.action_spaces = {a: spaces.Discrete(A.N) for a in self.possible_agents}
        self.observation_spaces = {
            a: spaces.Dict({
                "observation": spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32),
                "action_mask": spaces.Box(0, 1, (A.N,), np.int8),
            })
            for a in self.possible_agents
        }

    @property
    def decision_id(self) -> int:
        """Monotonic counter that increments iff the engine actually ADVANCED (an atomic
        action applied, or a compound builder finalized) -- it stays CONSTANT across the
        sub-steps of one compound decision (the builder accumulates env-side; `g` is
        frozen until finalize). Lets the collector dedupe the god/public encodes that are
        otherwise recomputed identically on every sub-step."""
        return self._decisions

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    # ── lifecycle ────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        self.g = E.new_multiplayer_game(load_decklist(), p1_name="p1", p2_name="p2", seed=seed)
        for pid in self.possible_agents:
            E.set_player_stops(self.g, pid, STOPS_MODES[self.stops_mode])
        self.agents = list(self.possible_agents)
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self._builder: CompoundBuilder | None = None
        self._decisions = 0
        self._scenario_result: str | None = None   # set by a scenario terminator (subclass)
        self.agent_selection = self.possible_agents[0]
        self._refresh()

    def step(self, action):
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return
        self._cumulative_rewards[agent] = 0.0
        action = int(action)
        if self._builder is not None:
            if self._builder.feed(self.g, action):     # finalized -> engine advanced
                self._builder = None
                self._decisions += 1
        else:
            if not apply_atomic(self.g, agent, action):
                raise AssertionError(f"illegal atomic action {A.decode(action)} for "
                                     f"{self.g.pending.type} (mask contract violated)")
            self._decisions += 1
        self.rewards = {a: 0.0 for a in self.agents}
        self._refresh()
        self._accumulate_rewards()

    # ── observation ──────────────────────────────────────────────────────────
    def observe(self, agent):
        acting = agent == self.agent_selection and not self.terminations[agent]
        mask = self._current_mask() if acting else np.zeros(A.N, dtype=np.int8)
        prog = self._builder.progress() if (acting and self._builder is not None) else 0.0
        return {"observation": encode_observation(self.g, agent, prog),
                "action_mask": mask.astype(np.int8)}

    def _current_mask(self) -> np.ndarray:
        if self._builder is not None:
            return self._builder.mask()
        return atomic_mask(self.g, self.agent_selection)

    # ── scenario hook ────────────────────────────────────────────────────────
    def _terminal_override(self):
        """Scenario subclasses return 'p1' / 'p2' / 'draw' to end the episode early
        with that result (terminal ±1 reward, no shaping), or None to defer to the
        normal engine game-end. Base env: always None, so behaviour is unchanged."""
        return None

    @property
    def winner(self):
        """Terminal winner seat, or None on a draw. A scenario terminator's result
        (cached when it fired in `_refresh`) wins over the engine result; otherwise
        the engine's terminal winner — identical to the previous behaviour."""
        if self._scenario_result is not None:
            return None if self._scenario_result == "draw" else self._scenario_result
        return terminal_winner(self.g)

    # ── internal: advance to the next decision point ─────────────────────────
    def _refresh(self):
        g = self.g
        while True:
            ov = self._terminal_override()
            if ov is not None:
                self._scenario_result = ov                # cache so `winner` is stable
            if ov is not None or is_terminal(g) or self._decisions >= self.max_decisions:
                if ov is not None:                        # scenario decided the outcome
                    done, winner = True, (None if ov == "draw" else ov)
                else:
                    done, winner = is_terminal(g), terminal_winner(g)
                for a in self.agents:
                    self.terminations[a] = done
                    self.truncations[a] = not done
                    self.rewards[a] = 0.0 if winner is None else (1.0 if a == winner else -1.0)
                return
            pend = g.pending
            assert pend is not None, "engine yielded with no pending decision and no result"
            self.agent_selection = pend.player
            if pend.type in COMPOUND_TYPES:
                if self._builder is None:
                    self._builder = CompoundBuilder(g, pend.player, pend.type, pend.context or {})
                if int(self._builder.mask().sum()) == 0:   # nothing to choose (empty-library scry/reorder)
                    self._builder.autofinalize(g)          # auto-resolve; engine advances
                    self._builder = None
                    continue
            else:
                self._builder = None
                # A pay decision must never present an empty mask (it would make the
                # masked policy uniform and sample an illegal action). If affordability
                # gating ever lets a payment strand, abort it transparently here — the
                # agent never sees a cancel action, so this is not a reversible stall
                # lever. Mirrors the empty-builder autofinalize above.
                if pend.type == "pay" and int(atomic_mask(g, pend.player).sum()) == 0:
                    import os
                    if os.environ.get("FISH_DEBUG_STRAND"):
                        print(f"[STRAND] aborting empty-mask payment: ctx={pend.context}",
                              flush=True)
                    E.cancel_payment(g, pend.player)
                    continue
            return
