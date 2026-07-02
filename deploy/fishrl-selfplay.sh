#!/usr/bin/env bash
set -euo pipefail

# Long-running fishrl self-play trainer. Auto-resumes from <repo>/checkpoints/latest.pt and
# runs indefinitely (--iters 0) until the service is stopped; SIGTERM triggers a graceful
# final checkpoint. Started by the fishrl-selfplay systemd user unit.
REPO_DIR="/home/zubekanov/Repositories/How-To-Train-Your-Fish"
VENV="/home/zubekanov/Repositories/Website_Dev/.venv"

cd "$REPO_DIR"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

# exec so python becomes the service's main PID and receives SIGTERM directly.
# warmup/eval sizes kept modest -- this shares a busy board (see thread caps in the unit).
exec python -m fishrl.train \
  --resume \
  --iters 0 \
  --encoder flat --critic-encoder entity \
  --ckpt-dir "${REPO_DIR}/checkpoints" \
  --scenario-pool \
  --pool-frac 0.5 \
  --warmup-games 16 \
  --report-every-seconds 3600 \
  --report-winrate-games 0 \
  --checkpoint-every-seconds 900
