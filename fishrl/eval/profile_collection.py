"""Decompose one self-play collection rollout into engine vs network wall-clock.

The throughput question for the encoder choice hinges on ONE unmeasured ratio: in a
collection rollout, what fraction of per-decision wall-clock is network forward passes
(actor + guesser + critic) versus pure-Python engine stepping and feature encoding?
If the engine dominates, a 6x-slower network forward is a small tax on collection; if
the network dominates, encoder speed matters a lot.

This profiles `collect_games` with a flat agent under cProfile and buckets the time.
We measure on CPU by default: it matches the deployment-speed thesis, the engine cost
is device-independent, and cProfile times CPU work synchronously (CUDA kernels are
async and would be mis-timed). Weights are untrained -- forward-pass COST does not
depend on the weights, only the network shape, so this is a valid cost decomposition.

    python -m fishrl.eval.profile_collection --games 40
    python -m fishrl.eval.profile_collection --games 40 --encoder entity   # project the tax

Note: `collect_games` runs the critic per decision (batch=1) to store the value
baseline. That forward is therefore ON the training-collection path even though the
critic is gone at deployment -- but `god_feat` is already buffered, so it could be
batched post-collection instead. We break the critic out separately so that tradeoff
is visible rather than assumed.
"""
from __future__ import annotations

import argparse
import cProfile
import pstats

from fishrl.train.collector import actor_act_fn, collect_games
from fishrl.train.config import Config
from fishrl.train.belief_env import BeliefAugmentedEnv
from fishrl.train.train_loop import build_models


def _cum(st: pstats.Stats, file_sub: str, func: str) -> float:
    """Sum cumulative time over stats entries matching (file substring, funcname).

    cumtime (not tottime) is the honest metric here: a network forward's real cost is
    mostly in torch's C-level ops, which cProfile attributes to built-in '~' entries,
    NOT to the Python frame. tottime-by-module would undercount the network; the
    cumtime of the on-path entry point rolls those built-in children back in."""
    total = 0.0
    for (fn, _ln, name), (_cc, _nc, _tt, ct, _callers) in st.stats.items():
        if name == func and file_sub in fn.replace("\\", "/"):
            total += ct
    return total


def main():
    ap = argparse.ArgumentParser(description="Profile engine vs network in collection.")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--encoder", default="flat",
                    help="encoder for ALL nets (flat|entity|attention); project the tax")
    ap.add_argument("--gpu", action="store_true",
                    help="profile on CUDA (async kernels make cProfile times unreliable)")
    args = ap.parse_args()

    device = "cuda" if args.gpu else "cpu"
    cfg = Config(device=device, encoder=args.encoder)
    m = build_models(cfg)
    for net in (m.actor, m.critic, m.guesser, m.public):
        net.eval()
    benv = BeliefAugmentedEnv(m.guesser, max_decisions=cfg.max_decisions)
    act = actor_act_fn(m.actor)

    print(f"=== collection profile | encoder={args.encoder} | device={device} | "
          f"{args.games} games ===", flush=True)

    pr = cProfile.Profile()
    pr.enable()
    buf = collect_games(benv, act, args.games, base_seed=0, critic=m.critic,
                        max_decisions=cfg.max_decisions)
    pr.disable()

    st = pstats.Stats(pr)
    total = st.total_tt
    n_dec = len(buf)

    # network forwards, by net, via the on-path entry points
    actor_t = _cum(st, "fishrl/train/collector.py", "act")          # tensor build + forward + sample
    guesser_t = _cum(st, "fishrl/models/guesser.py", "forward")     # guesser fwd (inside observe)
    critic_t = _cum(st, "fishrl/models/estimators.py", "p1_winprob")  # critic fwd (value baseline)
    network = actor_t + guesser_t + critic_t

    # engine + feature encoding (pure Python / numpy)
    step_t = _cum(st, "fishrl/env/aec_env.py", "step")
    observe_t = _cum(st, "fishrl/train/belief_env.py", "observe")
    obs_net = observe_t - guesser_t           # perspective build inside observe (engine-side)
    god_t = _cum(st, "fishrl/data/features.py", "encode_god")
    pub_t = _cum(st, "fishrl/data/features.py", "encode_public")
    cnt_t = _cum(st, "fishrl/data/features.py", "opponent_hand_counts")
    feat_t = god_t + pub_t + cnt_t

    def row(label, t):
        print(f"  {label:34s} {t*1e3/n_dec:8.3f} ms/dec  {100*t/total:6.1f}%", flush=True)

    print(f"  {n_dec} decisions over {total:.2f}s  ->  {total*1e3/n_dec:.3f} ms/decision\n", flush=True)
    print("  NETWORK (forward passes on the collection path):", flush=True)
    row("actor  (inference path)", actor_t)
    row("guesser (inference path)", guesser_t)
    row("critic (value baseline; batchable)", critic_t)
    row("-> network subtotal", network)
    print("\n  ENGINE + FEATURES (pure Python / numpy):", flush=True)
    row("env.step (engine)", step_t)
    row("observe perspective build", obs_net)
    row("encode_god / public / counts", feat_t)
    print(flush=True)
    row("network total", network)
    row("non-network total (rest)", total - network)
    print(f"\n  network share of collection: {100*network/total:.1f}%", flush=True)
    print(f"  critic share alone:          {100*critic_t/total:.1f}%  "
          f"(removable from path via batched post-collection value pass)", flush=True)
    actor_guesser = actor_t + guesser_t
    print(f"  actor+guesser (true deploy path): {100*actor_guesser/total:.1f}%", flush=True)
    print("\n  Reading: a 6x-slower encoder multiplies ONLY the network share. If the "
          f"network is {100*network/total:.0f}% of collection, 6x -> "
          f"{(total - network + 6*network)/total:.2f}x total collection time.", flush=True)


if __name__ == "__main__":
    main()
