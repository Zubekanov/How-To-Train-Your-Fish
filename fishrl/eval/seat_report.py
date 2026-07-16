"""On-demand seat / play-draw diagnostic for a checkpoint.

    python -m fishrl.eval.seat_report [--ckpt checkpoints/latest.pt] [--games 200]

Prints how the shared self-play policy splits by SEAT (p1 vs p2) and, separately,
by PLAY/DRAW (who takes the first turn). The engine is seat-symmetric (random and
attacker mirror both sit at ~0.50), so any p1/p2 gap here is a LEARNED asymmetry:
the shared net plays one seat better than the other. It matters because the
objective — beating the engine heuristic — is only ever measured from p1 (the
engine drives the heuristic on p2), so the headline vs-heuristic number is taken
from whichever seat the policy happens to be worse at.
"""
from __future__ import annotations

import argparse

import torch

from fishrl.eval.metrics import seat_diag_counts, seat_diag_rates
from fishrl.train import checkpoint as ckpt
from fishrl.train.config import Config
from fishrl.train.train_loop import build_models, _load_model_state


def load_models(ckpt_path: str):
    pl = ckpt.load_checkpoint(ckpt_path, map_location="cpu")
    enc = pl["config"]["encoders"]
    cfg = Config(device="cpu", seed=pl["config"].get("seed", 0),
                 use_belief=pl["config"].get("use_belief", True),
                 critic_hidden=tuple(pl["config"].get("critic_hidden", (512, 512, 256))),
                 **{f"{n}_encoder": enc[n] for n in ("actor", "critic", "guesser", "public")})
    m = build_models(cfg)
    _load_model_state(m, pl["models"])
    for net in (m.actor, m.critic, m.guesser, m.public):
        net.eval()
    return m, cfg, int(pl.get("done", 0))


def main() -> None:
    ap = argparse.ArgumentParser(description="Seat / play-draw diagnostic for a checkpoint.")
    ap.add_argument("--ckpt", default="checkpoints/latest.pt")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--max-decisions", type=int, default=2000)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    m, cfg, done = load_models(args.ckpt)
    c = seat_diag_counts(m, n_games=args.games, seed=args.seed,
                         max_decisions=args.max_decisions, use_belief=cfg.use_belief)
    r = seat_diag_rates(c)

    print(f"seat/play-draw diagnostic  ckpt={args.ckpt}  it={done}  "
          f"games={r['n_games']} decided={r['decided']}")
    print(f"  SEAT       p1={r['seat_p1_wr']:.3f}  p2={r['seat_p2_wr']:.3f}   "
          f"(gap {r['seat_p2_wr'] - r['seat_p1_wr']:+.3f}; engine-symmetric baseline 0.500/0.500)")
    print(f"  PLAY/DRAW  play={r['play_wr']:.3f}  draw={r['draw_wr']:.3f}   "
          f"(separate axis from seat; first_player is seat-neutral)")
    print(f"  roll-winner chooses PLAY-first {r['choose_first_frac']:.3f} of the time")
    print(f"  sanity: first_player is p1 {r['first_player_p1_frac']:.3f} of games (want ~0.500)")


if __name__ == "__main__":
    main()
