"""Teacher-game generation for behaviour cloning: play adapter-driven games and
record every labelled decision as ``(obs, mask, action)``.

Two game shapes, mixed by ``mirror_frac``:

* vs-internal — :class:`TeacherEnv`: p1 is the adapter-driven (labelled) seat,
  p2 an ENGINE-INTERNAL scripted profile, exactly the human-vs-AI testbench
  flow. This is the state distribution the RL agent faces against anchors.
* mirror — a plain FishAEC with BOTH seats adapter-driven; both are labelled
  (twice the samples, and p2-seat states for the self-play distribution).

Observations are stored WITHOUT the belief channel (zeros are appended at
train time): a fresh clone has no meaningful guesser, and per the belief
ablation the actor leans on that channel only weakly.

No torch in this module -- worker processes stay light.
"""
from __future__ import annotations

import pickle
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from fishrl.env.aec_env import FishAEC
from fishrl.env.driver import STOPS_MODES
from fishrl.forgetful_fish import engine as E
from fishrl.forgetful_fish.cards import load_decklist
from fishrl.imitate.adapter import HeuristicAdapter


class TeacherEnv(FishAEC):
    """FishAEC whose p2 is an engine-internal scripted AI: p1 is the only
    env-driven seat. Built on ``new_sandbox_game`` so the roll and the AI's own
    pregame decisions are already resolved when the env takes over."""

    def __init__(self, opponent: str = "heuristic_1_2", stops_mode: str = "default",
                 max_decisions: int = 2000):
        super().__init__(stops_mode=stops_mode, max_decisions=max_decisions)
        self.opponent = opponent

    def reset(self, seed=None, options=None):
        self.g = E.new_sandbox_game(load_decklist(), seed=seed, p1_name="p1",
                                    ai_profile=self.opponent)
        E.set_player_stops(self.g, "p1", STOPS_MODES[self.stops_mode])
        self.agents = list(self.possible_agents)
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self._builder = None
        self._decisions = 0
        self._scenario_result = None
        self.agent_selection = self.possible_agents[0]
        self._refresh()


def play_teacher_game(seed: int, teacher: str = "heuristic_1_2",
                      opponent: str | None = "heuristic_1_2",
                      stops_mode: str = "default", max_decisions: int = 2000) -> dict:
    """One adapter-driven game. ``opponent=None`` -> mirror (both seats labelled);
    else p2 is that engine-internal profile and only p1 is labelled.

    Returns {"obs": (N, OBS_DIM) f16, "mask": (N, A.N) i8, "act": (N,) i32,
    "seat": (N,) i8 (0=p1), "winner": "p1"/"p2"/None, "fallbacks": int,
    "forced_targets": int}."""
    if opponent is None:
        env = FishAEC(stops_mode=stops_mode, max_decisions=max_decisions)
        seats = ("p1", "p2")
    else:
        env = TeacherEnv(opponent, stops_mode=stops_mode, max_decisions=max_decisions)
        seats = ("p1",)
    adapters = {s: HeuristicAdapter(teacher) for s in seats}
    env.reset(seed=seed)
    obs_l, mask_l, act_l, seat_l = [], [], [], []
    guard = max_decisions * 8
    while env.agents and guard > 0:
        guard -= 1
        agent = env.agent_selection
        if env.terminations[agent] or env.truncations[agent]:
            env.step(None)
            continue
        if agent not in adapters:
            raise RuntimeError(f"engine surfaced a decision for the internal seat {agent}")
        o = env.observe(agent)
        a = adapters[agent].act(env, o["action_mask"])
        obs_l.append(o["observation"].astype(np.float16))
        mask_l.append(o["action_mask"].astype(np.int8))
        act_l.append(a)
        seat_l.append(0 if agent == "p1" else 1)
        env.step(a)
    return {
        "obs": np.stack(obs_l) if obs_l else np.zeros((0, 1), np.float16),
        "mask": np.stack(mask_l) if mask_l else np.zeros((0, 1), np.int8),
        "act": np.array(act_l, dtype=np.int32),
        "seat": np.array(seat_l, dtype=np.int8),
        "winner": env.winner,
        "fallbacks": sum(ad.fallbacks for ad in adapters.values()),
        "forced_targets": sum(ad.forced_targets for ad in adapters.values()),
    }


def _schedule(games: int, opponents: list, mirror_frac: float, seed0: int) -> list:
    """Per-game (seed, opponent_or_None): mirrors interleaved at mirror_frac, the
    rest round-robin over the opponent profiles. Deterministic in seed0."""
    out, acc = [], 0.0
    ri = 0
    for i in range(games):
        acc += mirror_frac
        if acc >= 1.0:
            acc -= 1.0
            out.append((seed0 + i, None))
        else:
            out.append((seed0 + i, opponents[ri % len(opponents)]))
            ri += 1
    return out


def _gen_chunk(args: tuple) -> bytes:
    """Worker: play a chunk of scheduled games, return one compressed blob."""
    jobs, teacher, stops_mode, max_decisions = args
    parts = [play_teacher_game(seed, teacher, opp, stops_mode, max_decisions)
             for seed, opp in jobs]
    return zlib.compress(pickle.dumps(parts), level=3)


def generate(games: int, teacher: str = "heuristic_1_2",
             opponents: tuple = ("heuristic", "heuristic_1_1", "heuristic_1_2"),
             mirror_frac: float = 0.5, seed: int = 0, workers: int = 0,
             stops_mode: str = "default", max_decisions: int = 2000,
             chunk: int = 8, log=print) -> dict:
    """Play `games` teacher games (optionally across worker processes) and return
    the concatenated dataset with a per-sample game id for split-by-game."""
    sched = _schedule(games, list(opponents), mirror_frac, seed)
    chunks = [sched[i:i + chunk] for i in range(0, len(sched), chunk)]
    results: list = []
    if workers and workers > 0:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_gen_chunk, (c, teacher, stops_mode, max_decisions))
                    for c in chunks]
            for n, f in enumerate(futs, 1):
                results.extend(pickle.loads(zlib.decompress(f.result())))
                log(f"[datagen] chunk {n}/{len(chunks)} done "
                    f"({sum(len(r['act']) for r in results)} samples)")
    else:
        for n, c in enumerate(chunks, 1):
            results.extend(pickle.loads(zlib.decompress(
                _gen_chunk((c, teacher, stops_mode, max_decisions)))))
            log(f"[datagen] chunk {n}/{len(chunks)} done "
                f"({sum(len(r['act']) for r in results)} samples)")
    gid = np.concatenate([np.full(len(r["act"]), i, dtype=np.int32)
                          for i, r in enumerate(results)])
    data = {
        "obs": np.concatenate([r["obs"] for r in results]),
        "mask": np.concatenate([r["mask"] for r in results]),
        "act": np.concatenate([r["act"] for r in results]),
        "seat": np.concatenate([r["seat"] for r in results]),
        "game": gid,
        "winners": [r["winner"] for r in results],
        "fallbacks": int(sum(r["fallbacks"] for r in results)),
        "forced_targets": int(sum(r["forced_targets"] for r in results)),
    }
    n = len(data["act"])
    log(f"[datagen] {games} games -> {n} labelled decisions "
        f"({n / max(games, 1):.0f}/game); fallbacks={data['fallbacks']} "
        f"forced_targets={data['forced_targets']}")
    return data
