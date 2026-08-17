"""Audit: how accurate is the hand guesser, and against what baseline?

Plays mirror self-play games (actor belief-augmented exactly as in training) and
records, per decision, everything needed to score the guesser against analytic
baselines computed from the viewer's own information:

  guess    -- the carried-recurrence guess the actor actually consumed
  shadow   -- guesser(persp, zeros): same state, recurrence ablated
  target   -- true opponent hand counts (the Poisson label)
  known    -- per-name counts of opponent hand cards the viewer KNOWS (known_by)
  visible  -- per-name counts of every card whose identity the viewer can see
              and that cannot be in the opponent's hand (own hand, both
              battlefields, graveyard, exile, stack sources, viewer-known
              library slots)
  handn    -- opponent hand size (public), plus turn / progress bookkeeping

The "perfect bookkeeper" baseline is then: known + (handn - known_total) x
remaining/remaining_total, where remaining = copies - visible - known. If the
guesser cannot beat that, it is doing arithmetic, not inference.

One .npz per invocation (chunked; safe under process kills).
"""
from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=50)
    ap.add_argument("--seed-base", type=int, default=7_000_000)
    ap.add_argument("--out", required=True, help=".npz path for this chunk")
    ap.add_argument("--affinity", default="")
    ap.add_argument("--max-decisions", type=int, default=2000)
    args = ap.parse_args()

    import torch
    from fishrl.train.pcollect import _apply_affinity, parse_affinity
    _apply_affinity(parse_affinity(args.affinity))
    torch.set_num_threads(1)

    from fishrl.eval.audit_bench import build
    from fishrl.env.aec_env import FishAEC
    from fishrl.obs import vocab as V

    models, _pl = build(args.ckpt)
    guesser, actor = models.guesser, models.actor
    z = np.zeros(V.N_NAMES, dtype=np.float32)
    zt = torch.zeros(1, V.N_NAMES)

    rows = {k: [] for k in ("guess", "shadow", "target", "known", "visible",
                            "handn", "turn", "frac", "seat", "game")}

    def name_counts(names) -> np.ndarray:
        v = np.zeros(V.N_NAMES, dtype=np.uint8)
        for n in names:
            i = V.NAME_INDEX.get(n)
            if i is not None:
                v[i] += 1
        return v

    for gi in range(args.start, args.start + args.count):
        seed = args.seed_base + gi
        torch.manual_seed(seed)
        env = FishAEC(max_decisions=args.max_decisions)
        env.reset(seed=seed)
        prev = {"p1": z.copy(), "p2": z.copy()}
        game_rows = {k: [] for k in rows}
        guard = 0
        while env.agents and guard < args.max_decisions * 6:
            guard += 1
            seat = env.agent_selection
            if env.terminations[seat] or env.truncations[seat]:
                env.step(None)
                continue
            base = env.observe(seat)
            persp = base["observation"]
            g = env.g
            obj = g.objects
            opp = "p2" if seat == "p1" else "p1"

            pt = torch.as_tensor(persp, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                guess = guesser(pt, torch.as_tensor(prev[seat]).unsqueeze(0)).squeeze(0).numpy()
                shadow = guesser(pt, zt).squeeze(0).numpy()

            # ground truth + viewer-side bookkeeping
            target = name_counts(obj[iid].name for iid in g.players[opp].hand)
            known = name_counts(obj[iid].name for iid in g.players[opp].hand
                                if seat in (obj[iid].known_by or []))
            vis = []
            vis += [obj[iid].name for iid in g.players[seat].hand]
            for pid in ("p1", "p2"):
                vis += [obj[iid].name for iid in g.players[pid].battlefield]
            vis += [obj[iid].name for iid in g.graveyard]
            vis += [obj[iid].name for iid in g.exile]
            vis += [obj[s.source_instance_id].name for s in g.stack
                    if s.source_instance_id in obj]
            vis += [obj[s.instance_id].name for s in g.library if s.known_by.get(seat)]
            visible = name_counts(vis)

            game_rows["guess"].append(guess.astype(np.float16))
            game_rows["shadow"].append(shadow.astype(np.float16))
            game_rows["target"].append(target)
            game_rows["known"].append(known)
            game_rows["visible"].append(visible)
            game_rows["handn"].append(len(g.players[opp].hand))
            game_rows["turn"].append(min(g.turn_number, 255))
            game_rows["seat"].append(0 if seat == "p1" else 1)
            game_rows["game"].append(gi)

            # act exactly as training-time inference does (carried guess -> actor)
            x = np.concatenate([persp, guess.astype(np.float32)])
            with torch.no_grad():
                logp = actor.log_probs(
                    torch.as_tensor(x, dtype=torch.float32).unsqueeze(0),
                    torch.as_tensor(base["action_mask"], dtype=torch.float32).unsqueeze(0))[0]
                a = int(torch.multinomial(logp.exp(), 1))
            prev[seat] = guess.astype(np.float32)
            env.step(a)

        n = len(game_rows["game"])
        fr = (np.arange(n) / max(n - 1, 1)).astype(np.float16)
        game_rows["frac"] = list(fr)
        for k in rows:
            rows[k].extend(game_rows[k])

    np.savez_compressed(
        args.out,
        guess=np.stack(rows["guess"]), shadow=np.stack(rows["shadow"]),
        target=np.stack(rows["target"]), known=np.stack(rows["known"]),
        visible=np.stack(rows["visible"]),
        handn=np.array(rows["handn"], dtype=np.uint8),
        turn=np.array(rows["turn"], dtype=np.uint8),
        frac=np.array(rows["frac"], dtype=np.float16),
        seat=np.array(rows["seat"], dtype=np.uint8),
        game=np.array(rows["game"], dtype=np.uint16),
    )
    print(f"wrote {args.out}: {len(rows['game'])} decisions from {args.count} games")


if __name__ == "__main__":
    main()
