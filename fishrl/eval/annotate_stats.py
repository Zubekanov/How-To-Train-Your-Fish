"""Add a note / marker to the skill-graph data (stats.json "annotations").

Pins an event to a training iteration so a plotter can draw it as a vertical line / label on the
win-rate (skill) curves -- e.g. a change in the training pool or hyperparameters that explains a
shift in the trend. Lock-safe (same flock + atomic write as the live writers) and idempotent
(deduped by iteration + text), so it is safe to run while the services are up.

    python -m fishrl.eval.annotate_stats --it 9508 \\
        --text "added heuristic to the training pool"
"""
from __future__ import annotations

import argparse

from fishrl.train import stats as stats_io


def main() -> None:
    ap = argparse.ArgumentParser(description="Annotate the skill graph (stats.json).")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--it", type=int, required=True, help="iteration the event is pinned to")
    ap.add_argument("--text", required=True, help="the note to show on the graph")
    ap.add_argument("--kind", default="n.b.", help="marker kind/label (default: n.b.)")
    args = ap.parse_args()

    added = stats_io.add_annotation(args.ckpt_dir, args.it, args.text, kind=args.kind)
    verb = "added" if added else "already present"
    print(f"[annotate] {verb}: it={args.it} {args.kind}: {args.text}")


if __name__ == "__main__":
    main()
