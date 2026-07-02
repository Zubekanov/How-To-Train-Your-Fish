"""Self-play rollout collection through the belief-augmented env.

One acting function drives BOTH seats (shared policy). Each env step is one
transition (compound builder sub-steps included — the mask changes per sub-step,
so each is a distinct masked decision). Features are captured at decision time,
before the engine mutates the state; the winner label is back-filled per game.
"""
from __future__ import annotations

import numpy as np
import torch

from fishrl.data.buffer import RolloutBuffer, Step
from fishrl.data.features import encode_god, encode_public, opponent_hand_counts
from fishrl.models import device_of
from fishrl.obs import vocab as V
from fishrl.obs.encoder import OBS_DIM
from fishrl.train.advantages import SEAT_SIGN


def random_act_fn(rng: np.random.Generator):
    def act(obs):
        legal = np.flatnonzero(obs["action_mask"])
        return int(rng.choice(legal)), 0.0
    return act


def actor_act_fn(actor):
    dev = device_of(actor)

    def act(obs):
        mask = obs["action_mask"]
        legal = np.flatnonzero(mask)
        if legal.size == 1:
            # Forced decision: with a 1-hot legal mask the masked policy assigns prob 1 to the
            # sole action, so logp == log(1) == 0 exactly -- skip the (~20-25% of) forwards whose
            # output is determined anyway. Bit-equivalent to computing it (PPO recomputes the same
            # logp=0 from the same mask), so this is free compute, not a learning change.
            return int(legal[0]), 0.0
        x = torch.as_tensor(obs["observation"], dtype=torch.float32).unsqueeze(0).to(dev)
        m = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0).to(dev)
        with torch.no_grad():
            logp_all = actor.log_probs(x, m)[0]
            p = logp_all.exp()
            a = int(torch.multinomial(p, 1))
        return a, float(logp_all[a])
    return act


def _stamp_game(buf: RolloutBuffer, start: int, winner, truncated: bool = False,
                forced=None) -> None:
    """Back-fill one finished game's outcome onto its steps and register the game.

    Every step gets the winner label plus a buffer-local game_id so GAE segments
    per game (never bootstrapping across a boundary), and `truncated` marks a
    decision-cap cut (winner None but not a decided draw) so the advantage tail
    bootstraps from the critic instead of scoring the cut as a draw. The winner is
    ALSO recorded in `buf.games` even when the game produced zero learner steps —
    league win-rate updates must count games the learner lost before its first
    decision. `forced` is the winning seat when the free-attack hard rule (or any
    forced result) ended the game, else None — kept in `buf.meta` for the window
    telemetry."""
    gid = len(buf.games)
    for i in range(start, len(buf.steps)):
        buf.steps[i].winner = winner
        buf.steps[i].game_id = gid
        buf.steps[i].truncated = truncated
    buf.games.append(winner)
    buf.meta.append({"truncated": truncated, "forced": forced})


def fill_critic_values(buf: RolloutBuffer, critic, batch: int = 8192) -> None:
    """Batched post-collection value pass for the privileged critic.

    The critic is the ONLY value head in the advantage loop (asymmetric actor-critic),
    but it is FIXED during collection -- it's updated only afterward in `ppo_update`.
    So computing each step's seat-frame value baseline in one batched forward over the
    buffered `god_feat` is numerically identical to the old per-decision batch=1 calls,
    while lifting the critic forward out of the per-decision hot loop (it was ~14% of
    collection wall-clock at batch=1). No extra state: god_feat is already buffered."""
    if not buf.steps:
        return
    dev = device_of(critic)
    god = np.stack([s.god_feat for s in buf.steps])
    sign = np.array([SEAT_SIGN[s.seat] for s in buf.steps], dtype=np.float32)
    out = np.empty(len(buf.steps), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(god), batch):
            gt = torch.as_tensor(god[i:i + batch], dtype=torch.float32).to(dev)
            p1 = critic.p1_winprob(gt).cpu().numpy()
            out[i:i + batch] = sign[i:i + batch] * (2.0 * p1 - 1.0)
    for s, v in zip(buf.steps, out):
        s.value = float(v)


def collect_heuristic_games(guesser, actor, n_games, base_seed, critic=None,
                            use_belief=True, max_decisions=2000,
                            stops_mode="default") -> RolloutBuffer:
    """Collect rollouts where the LEARNING policy (p1) plays the engine's heuristic
    AI (p2) — the curriculum/pool opponent, in contrast to `collect_games`'
    shared-policy self-play.

    The heuristic is the vendored engine AI driven re-entrantly by the engine (the
    same sandbox path eval uses), so we never choose its actions here: only p1 ever
    receives a pending. Critically, ONLY p1's transitions are recorded — p2's moves
    are off-policy (not produced by `actor`) and would corrupt the PPO update if
    buffered. Every Step matches the self-play shape (belief-augmented obs ⊕ guess,
    p1-oriented god/public features, opponent-hand-count targets) so the merged
    buffer trains identically. The opponent always takes the p2 seat (sandbox
    convention); self-play already supplies the p2-perspective transitions.
    """
    from fishrl.opponents.heuristic import SEAT, HeuristicMatch

    act = actor_act_fn(actor)
    dev = device_of(guesser)
    zeros = np.zeros(V.N_NAMES, dtype=np.float32)
    buf = RolloutBuffer()
    for gi in range(n_games):
        match = HeuristicMatch(stops_mode=stops_mode, max_decisions=max_decisions)
        obs = match.reset(seed=base_seed + gi)
        prev_guess = zeros.copy()
        start = len(buf)
        done = False
        guard = 0
        while not done and guard < max_decisions * 6:
            guard += 1
            persp = obs["observation"]
            mask = obs["action_mask"]
            if int(mask.sum()) == 0:                 # not p1's decision / terminal — stop
                break
            g = match.g
            # g is the decision-time state (p2 already resolved by the engine). god/public
            # are p1-oriented like the self-play path; recomputed per decision (the pool is
            # a minority of games, so the self-play decision_id dedupe isn't worth the coupling).
            god, pub = encode_god(g), encode_public(g)
            if use_belief:
                with torch.no_grad():
                    guess = guesser(
                        torch.as_tensor(persp, dtype=torch.float32).unsqueeze(0).to(dev),
                        torch.as_tensor(prev_guess, dtype=torch.float32).unsqueeze(0).to(dev),
                    ).squeeze(0).cpu().numpy().astype(np.float32)
            else:
                guess = zeros.copy()
            x_act = np.concatenate([persp, guess]).astype(np.float32)
            action, logp = act({"observation": x_act, "action_mask": mask})
            buf.add(Step(
                seat=SEAT, x_act=x_act,
                mask=mask.astype(np.int8), action=action, logp=logp,
                value=0.0, god_feat=god, pub_feat=pub,
                guess_in=prev_guess.astype(np.float32),   # the guesser's INPUT (carried prev)
                cnt_target=opponent_hand_counts(g, SEAT),
            ))
            prev_guess = guess
            obs, _reward, done, _info = match.step(action)
        winner = match.g.result.get("winner")
        _stamp_game(buf, start, winner,
                    truncated=winner is None and match.g.result.get("status") == "ongoing")
    if critic is not None:
        fill_critic_values(buf, critic)
    return buf


def _belief_guess(guesser, persp, prev, dev):
    """One guesser forward: opponent-hand belief from a seat's perspective ⊕ its
    carried previous guess. Returns the new guess (float32[N_NAMES])."""
    with torch.no_grad():
        return guesser(
            torch.as_tensor(persp, dtype=torch.float32).unsqueeze(0).to(dev),
            torch.as_tensor(prev, dtype=torch.float32).unsqueeze(0).to(dev),
        ).squeeze(0).cpu().numpy().astype(np.float32)


def collect_vs_opponent(learner, opponent, n_games, base_seed, critic=None,
                        use_belief=True, max_decisions=2000,
                        learner_seat=None, enforce_free_attack=False) -> RolloutBuffer:
    """Collect rollouts where the LEARNER plays a fixed pool ``opponent`` through the
    two-seat AEC env, recording ONLY the learner's transitions (the opponent is
    off-policy and must never enter the PPO buffer). Seat-balanced: the learner plays
    p1 on even games and p2 on odd games, cancelling first-player bias in the data.

    ``opponent`` is a PFSP league member with ``kind`` in {"random", "attacker",
    "self"}; the engine "heuristic" is driven by ``collect_heuristic_games`` instead
    (the engine resolves it on p2, so it can't take an arbitrary seat here). A "self"
    opponent acts through ITS OWN frozen actor+guesser (its own belief channel).

    Every recorded Step matches the self-play shape (belief-augmented obs ⊕ guess,
    p1-oriented god/public features, opponent-hand-count targets) so the merged
    buffer trains identically to mirror self-play.
    """
    from fishrl.env.aec_env import FishAEC

    kind = opponent.kind
    if kind == "heuristic":
        raise ValueError("heuristic is collected via collect_heuristic_games, not collect_vs_opponent")
    learn_act = actor_act_fn(learner.actor)
    ldev = device_of(learner.guesser)
    zeros = np.zeros(V.N_NAMES, dtype=np.float32)

    rng = np.random.default_rng(base_seed)
    opp_actor = opp_guesser = opp_dev = None
    attacker_action = None
    if kind == "self":
        opp_actor, opp_guesser = opponent.models.actor, opponent.models.guesser
        opp_dev = device_of(opp_guesser)
    elif kind == "attacker":
        from fishrl.opponents.attacker import attacker_action

    buf = RolloutBuffer()
    for gi in range(n_games):
        # Seat-balance across games unless the caller pins a seat (single-game pool calls
        # alternate the seat themselves so the parity isn't always p1).
        lseat = learner_seat or ("p1" if gi % 2 == 0 else "p2")
        env = FishAEC(max_decisions=max_decisions, enforce_free_attack=enforce_free_attack)
        env.reset(seed=base_seed + gi)
        prev = {"p1": zeros.copy(), "p2": zeros.copy()}   # per-seat belief carry
        start = len(buf)
        cache_id = cache_god = cache_pub = None
        guard = 0
        while env.agents and guard < max_decisions * 6:
            guard += 1
            seat = env.agent_selection
            if env.terminations[seat] or env.truncations[seat]:
                env.step(None)
                continue
            base = env.observe(seat)
            mask = base["action_mask"]
            persp = base["observation"]
            if seat == lseat:
                g = env.g
                did = env.decision_id
                if did != cache_id:                        # engine advanced -> fresh encode
                    cache_id, cache_god, cache_pub = did, encode_god(g), encode_public(g)
                    god, pub = cache_god, cache_pub
                else:                                      # frozen state (compound sub-step)
                    god, pub = cache_god.copy(), cache_pub.copy()
                prev_in = prev[seat]
                if use_belief:
                    guess = _belief_guess(learner.guesser, persp, prev_in, ldev)
                    prev[seat] = guess
                else:
                    guess = zeros.copy()
                x_act = np.concatenate([persp, guess]).astype(np.float32)
                action, logp = learn_act({"observation": x_act, "action_mask": mask})
                buf.add(Step(
                    seat=seat, x_act=x_act,
                    mask=mask.astype(np.int8), action=action, logp=logp,
                    value=0.0, god_feat=god, pub_feat=pub,
                    guess_in=prev_in.astype(np.float32),   # the guesser's INPUT (carried prev)
                    cnt_target=opponent_hand_counts(g, seat),
                ))
            elif kind == "random":
                action = int(rng.choice(np.flatnonzero(mask)))
            elif kind == "attacker":
                action = attacker_action(env.g, seat, mask, rng)
            else:                                          # "self": frozen policy + own belief
                if use_belief:
                    guess = _belief_guess(opp_guesser, persp, prev[seat], opp_dev)
                    prev[seat] = guess
                else:
                    guess = zeros
                ox = torch.as_tensor(np.concatenate([persp, guess]), dtype=torch.float32).unsqueeze(0).to(opp_dev)
                om = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0).to(opp_dev)
                with torch.no_grad():
                    action = int(torch.multinomial(opp_actor.log_probs(ox, om)[0].exp(), 1))
            env.step(action)
        # env.winner, NOT g.result: a game ended by the free-attack hard rule (or any
        # terminal override) records its verdict only on the env — g.result stays
        # "ongoing" and would mislabel a decided game as winner=None.
        winner = env.winner
        _stamp_game(buf, start, winner,
                    truncated=winner is None and env.g.result.get("status") == "ongoing",
                    forced=env._forced_result)
    if critic is not None:
        fill_critic_values(buf, critic)
    return buf


def collect_games(belief_env, act_fn, n_games, base_seed, critic=None,
                  max_decisions=2000) -> RolloutBuffer:
    buf = RolloutBuffer()
    for gi in range(n_games):
        belief_env.reset(seed=base_seed + gi)
        start = len(buf)
        # god/public features depend only on `g`, which is FROZEN across the sub-steps of
        # a compound decision -> encode once per distinct engine state (keyed on the env's
        # decision_id) and reuse across sub-steps. A reused vector is copied so each Step
        # owns an independent array (defensive; they're only ever read via np.stack today).
        cache_id, cache_god, cache_pub = None, None, None
        for agent in belief_env.agent_iter(max_iter=max_decisions * 6):
            if belief_env.terminations[agent] or belief_env.truncations[agent]:
                belief_env.step(None)
                continue
            obs = belief_env.observe(agent)
            x = obs["observation"]
            g = belief_env.g
            did = belief_env.decision_id
            if did != cache_id:                       # engine advanced -> fresh encode
                cache_id, cache_god, cache_pub = did, encode_god(g), encode_public(g)
                god, pub = cache_god, cache_pub
            else:                                     # same frozen state (compound sub-step)
                god, pub = cache_god.copy(), cache_pub.copy()
            action, logp = act_fn(obs)
            buf.add(Step(
                seat=agent, x_act=x.astype(np.float32),
                mask=obs["action_mask"].astype(np.int8), action=action, logp=logp,
                value=0.0, god_feat=god, pub_feat=pub,
                # the guesser's INPUT at this observe (the seat's carried previous
                # guess) — its output is x[OBS_DIM:], already inside x_act
                guess_in=belief_env.last_prev[agent].astype(np.float32),
                cnt_target=opponent_hand_counts(g, agent),
            ))
            belief_env.step(action)
        winner = belief_env.winner            # scenario terminator result, or engine winner
        _stamp_game(buf, start, winner,
                    truncated=winner is None and belief_env.g.result.get("status") == "ongoing",
                    forced=belief_env.forced_result)
    if critic is not None:
        fill_critic_values(buf, critic)            # batched, off the per-decision path
    return buf
