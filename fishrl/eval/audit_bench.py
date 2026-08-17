"""Audit benchmark worker (external audit, 2026-08-17).

One process = one benchmark chunk, pinned to a single logical processor. Plays
games from a checkpoint and appends one JSON row per game to --out (flushed per
game, so a killed process keeps its finished games). Modes:

  anchor  — vs a scripted opponent. heuristic* profiles run the engine sandbox
            (learner fixed on p1, engine convention); random/attacker run through
            FishAEC with the learner alternating seats by game parity.
  mirror  — shared-policy mirror self-play; records winner seat, first player,
            play-order choices, decisions, end reason.
  calib   — collect mirror games and emit one row per decision with the
            privileged critic's and public estimator's P(p1 win) vs the outcome.

Ablation flags (anchor/mirror): --no-belief feeds the actor a zero belief vector
(guesser skipped, matching use_belief=False); --no-clock zeroes the 3 deckout-
clock floats at the tail of the perspective observation before the guesser and
actor see it. torch is seeded PER GAME (seed-base + game index) so arms that
share --seed-base are paired: same deck shuffle, same sampling stream at the
root of each game.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np


def build(ckpt_path: str):
    import torch
    from fishrl.train.checkpoint import load_checkpoint
    from fishrl.train.train_loop import build_models, config_from_checkpoint, _load_model_state
    torch.set_num_threads(1)
    pl = load_checkpoint(ckpt_path, map_location="cpu")
    cfg = config_from_checkpoint(pl["config"])
    m = build_models(cfg)
    _load_model_state(m, pl["models"])
    if m.guesser is None or m.public is None:
        raise SystemExit("this tool interrogates the legacy guesser/public nets; "
                         "the given checkpoint is a v3 (bookkeeper/public-critic) "
                         "payload -- point it at a v1/v2 lineage instead")
    return m, pl


def _zero_clock(persp: np.ndarray) -> np.ndarray:
    from fishrl.obs.encoder import OBS_DIM
    p = persp.copy()
    p[OBS_DIM - 3:OBS_DIM] = 0.0
    return p


def _act(models, persp, mask, prev, use_belief, no_clock):
    """Belief-augment + sample one action. Returns (action, new_prev_guess)."""
    import torch
    from fishrl.obs import vocab as V
    if no_clock:
        persp = _zero_clock(persp)
    z = np.zeros(V.N_NAMES, dtype=np.float32)
    if use_belief:
        with torch.no_grad():
            guess = models.guesser(
                torch.as_tensor(persp, dtype=torch.float32).unsqueeze(0),
                torch.as_tensor(prev, dtype=torch.float32).unsqueeze(0),
            ).squeeze(0).numpy().astype(np.float32)
    else:
        guess = z
    x = np.concatenate([persp, guess]).astype(np.float32)
    xt = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)
    mt = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logp = models.actor.log_probs(xt, mt)[0]
        a = int(torch.multinomial(logp.exp(), 1))
    return a, guess


def run_heuristic(models, args, out):
    import torch
    from fishrl.opponents.heuristic import HeuristicMatch
    from fishrl.obs import vocab as V
    z = np.zeros(V.N_NAMES, dtype=np.float32)
    for gi in range(args.start, args.start + args.count):
        seed = args.seed_base + gi
        torch.manual_seed(seed)
        t0 = time.perf_counter()
        m = HeuristicMatch(max_decisions=args.max_decisions, profile=args.opponent)
        obs = m.reset(seed=seed)
        prev = z.copy()
        done, r, guard = False, 0.0, 0
        while not done and guard < args.max_decisions * 6:
            guard += 1
            a, prev = _act(models, obs["observation"], obs["action_mask"], prev,
                           not args.no_belief, args.no_clock)
            obs, r, done, _ = m.step(a)
        g = m.g
        row = {"mode": "anchor", "opp": args.opponent, "tag": args.tag, "gi": gi,
               "seed": seed, "agent_seat": "p1",
               "win": int(r > 0), "loss": int(r < 0),
               "draw_or_trunc": int(r == 0.0),
               "decisions": m._decisions, "turns": g.turn_number,
               "lib_left": len(g.library),
               "status": g.result.get("status"), "reason": g.result.get("reason"),
               "secs": round(time.perf_counter() - t0, 3)}
        out.write(json.dumps(row) + "\n")
        out.flush()


def run_aec_opponent(models, args, out):
    """random/attacker through FishAEC, learner seat alternating by game parity."""
    import torch
    from fishrl.env.aec_env import FishAEC
    from fishrl.obs import vocab as V
    from fishrl.opponents.attacker import attacker_action
    z = np.zeros(V.N_NAMES, dtype=np.float32)
    for gi in range(args.start, args.start + args.count):
        seed = args.seed_base + gi
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        agent_seat = "p1" if gi % 2 == 0 else "p2"
        t0 = time.perf_counter()
        env = FishAEC(max_decisions=args.max_decisions)
        env.reset(seed=seed)
        prev = z.copy()
        guard = ndec = 0
        while env.agents and guard < args.max_decisions * 6:
            guard += 1
            seat = env.agent_selection
            if env.terminations[seat] or env.truncations[seat]:
                env.step(None)
                continue
            base = env.observe(seat)
            if seat == agent_seat:
                a, prev = _act(models, base["observation"], base["action_mask"], prev,
                               not args.no_belief, args.no_clock)
            elif args.opponent == "random":
                a = int(rng.choice(np.flatnonzero(base["action_mask"])))
            else:
                a = attacker_action(env.g, seat, base["action_mask"], rng)
            ndec += 1
            env.step(a)
        g = env.g
        w = g.result.get("winner")
        row = {"mode": "anchor", "opp": args.opponent, "tag": args.tag, "gi": gi,
               "seed": seed, "agent_seat": agent_seat,
               "win": int(w == agent_seat), "loss": int(w not in (agent_seat, None)),
               "draw_or_trunc": int(w is None),
               "decisions": ndec, "turns": g.turn_number, "lib_left": len(g.library),
               "status": g.result.get("status"), "reason": g.result.get("reason"),
               "secs": round(time.perf_counter() - t0, 3)}
        out.write(json.dumps(row) + "\n")
        out.flush()


def run_mirror(models, args, out):
    import torch
    from fishrl.env.aec_env import FishAEC
    from fishrl.obs import vocab as V
    z = np.zeros(V.N_NAMES, dtype=np.float32)
    for gi in range(args.start, args.start + args.count):
        seed = args.seed_base + gi
        torch.manual_seed(seed)
        t0 = time.perf_counter()
        env = FishAEC(max_decisions=args.max_decisions)
        env.reset(seed=seed)
        prev = {"p1": z.copy(), "p2": z.copy()}
        guard = ndec = 0
        choose_first = choose_total = 0
        while env.agents and guard < args.max_decisions * 6:
            guard += 1
            seat = env.agent_selection
            if env.terminations[seat] or env.truncations[seat]:
                env.step(None)
                continue
            base = env.observe(seat)
            pend = env.g.pending
            a, prev[seat] = _act(models, base["observation"], base["action_mask"],
                                 prev[seat], not args.no_belief, args.no_clock)
            if pend is not None and pend.type == "choose_play_order":
                # NOTE: metrics.seat_diag_counts counts `a == 0` here, but the legal
                # ids during choose_play_order are the PLAY_ORDER block (offset ~146),
                # so its choose_first_frac is identically 0.0 — a telemetry bug this
                # audit found. Count the actual block-local index instead.
                from fishrl.spaces import action_space as A
                choose_total += 1
                choose_first += int(a == A.aid("PLAY_ORDER", 0))
            ndec += 1
            env.step(a)
        g = env.g
        w = env.winner
        row = {"mode": "mirror", "tag": args.tag, "gi": gi, "seed": seed,
               "winner": w, "first_player": g.first_player,
               "draw_or_trunc": int(w is None),
               "choose_first": choose_first, "choose_total": choose_total,
               "decisions": ndec, "turns": g.turn_number, "lib_left": len(g.library),
               "status": g.result.get("status"), "reason": g.result.get("reason"),
               "secs": round(time.perf_counter() - t0, 3)}
        out.write(json.dumps(row) + "\n")
        out.flush()


def run_calib(models, args, out):
    import torch
    from fishrl.train.belief_env import BeliefAugmentedEnv
    from fishrl.train.collector import actor_act_fn, collect_games
    torch.manual_seed(args.seed_base + args.start)
    benv = BeliefAugmentedEnv(models.guesser, max_decisions=args.max_decisions)
    buf = collect_games(benv, actor_act_fn(models.actor), args.count,
                        args.seed_base + args.start, critic=None,
                        max_decisions=args.max_decisions)
    god = np.stack([s.god_feat for s in buf.steps])
    pub = np.stack([s.pub_feat for s in buf.steps])
    priv_p = np.empty(len(buf.steps), dtype=np.float32)
    pub_p = np.empty(len(buf.steps), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(god), 4096):
            priv_p[i:i + 4096] = models.critic.p1_winprob(
                torch.as_tensor(god[i:i + 4096], dtype=torch.float32)).numpy()
            pub_p[i:i + 4096] = models.public.p1_winprob(
                torch.as_tensor(pub[i:i + 4096], dtype=torch.float32)).numpy()
    # per-game step totals for progress fractions
    totals: dict = {}
    for s in buf.steps:
        totals[s.game_id] = totals.get(s.game_id, 0) + 1
    counters: dict = {}
    for i, s in enumerate(buf.steps):
        j = counters.get(s.game_id, 0)
        counters[s.game_id] = j + 1
        if s.winner is None:
            continue
        row = {"mode": "calib", "tag": args.tag, "game": int(s.game_id) + args.start,
               "frac": round(j / max(totals[s.game_id] - 1, 1), 4),
               "priv": round(float(priv_p[i]), 4), "pub": round(float(pub_p[i]), 4),
               "y": int(s.winner == "p1"), "seat": s.seat}
        out.write(json.dumps(row) + "\n")
    out.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", required=True, choices=["anchor", "mirror", "calib"])
    ap.add_argument("--opponent", default="heuristic")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--seed-base", type=int, default=5_000_000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--affinity", default="")
    ap.add_argument("--no-belief", action="store_true")
    ap.add_argument("--no-clock", action="store_true")
    ap.add_argument("--max-decisions", type=int, default=2000)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    from fishrl.train.pcollect import _apply_affinity, parse_affinity
    _apply_affinity(parse_affinity(args.affinity))
    models, _pl = build(args.ckpt)
    with open(args.out, "a", encoding="utf-8") as out:
        if args.mode == "anchor" and args.opponent.startswith("heuristic"):
            run_heuristic(models, args, out)
        elif args.mode == "anchor":
            run_aec_opponent(models, args, out)
        elif args.mode == "mirror":
            run_mirror(models, args, out)
        else:
            run_calib(models, args, out)


if __name__ == "__main__":
    main()
