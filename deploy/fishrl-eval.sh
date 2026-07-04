#!/usr/bin/env bash
set -euo pipefail

# Out-of-band parallel win-rate panel. Reads <repo>/checkpoints/latest.pt and evaluates the
# current policy vs random / attacker / heuristic / frozen-self across worker processes, then
# logs one [eval ...] line to journald. Decoupled from the trainer so training never blocks
# for win rates; driven hourly by fishrl-eval.timer.
REPO_DIR="/home/zubekanov/Repositories/How-To-Train-Your-Fish"
VENV="/home/zubekanov/Repositories/Website_Dev/.venv"

cd "$REPO_DIR"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

# Tiered targets (also the code defaults, spelled out for visibility): the panel
# HARVESTS anchor outcomes from the trainer's own pool games (wr_train in the latest
# report row) and only plays each anchor's deficit below target -- the big eval-compute
# saving on this box. --no-harvest restores the full-target panel.
exec python -m fishrl.eval.parallel_panel \
  --ckpt-dir "${REPO_DIR}/checkpoints" \
  --n-games 100 --n-attacker 50 --n-random 30 --n-frozen 50
