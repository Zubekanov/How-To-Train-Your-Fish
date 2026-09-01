# How-To-Train-Your-Fish

Adversarial machine learning agents for the game of Forgetful Fish aka Dandân.

> **Status (2026-09-02): training concluded.** The final agent is
> `checkpoints-v3/best.pt` (iteration 196,263; ~0.75 win-rate vs the v1.3
> heuristic across 2,000-game benches). The full story — run history, final
> benchmarks, lessons — is in [`docs/FINAL-REPORT.md`](docs/FINAL-REPORT.md).
> The lineage remains locally resumable via `deploy\fishrl-pc.bat`.

`fishrl` wraps a complete Forgetful Fish rules engine (vendored from the Website
project under `fishrl/forgetful_fish/`, logic unmodified) as a **PettingZoo AEC**
two-agent, imperfect-information self-play environment with a flat masked action
space and a fixed-size tensor observation.

This README covers **how to run** the stack. For **why it is built this way** — the RL
theory and techniques, the design decisions and the measurements behind them, and the
full telemetry field reference (plus the traps that bite anyone plotting the history) —
see [`docs/DESIGN.md`](docs/DESIGN.md).

## Install

```bash
pip install -e .            # numpy, pettingzoo, gymnasium, torch  (+ .[dev] for pytest)
```

`requirements.txt` is the deploy lock (torch pinned to the running minor version);
`pyproject.toml` keeps loose floors for library use.

## Quickstart — random self-play

```python
import numpy as np
from fishrl.selfplay.pettingzoo_api import env

e = env()
e.reset(seed=0)
for agent in e.agent_iter():
    obs, reward, term, trunc, info = e.last()
    if term or trunc:
        e.step(None)
    else:
        legal = np.flatnonzero(obs["action_mask"])   # mask is the legality contract
        e.step(int(np.random.choice(legal)))
print(e.unwrapped.g.result)                          # {"status": "p1_wins", ...}
```

## Evaluate a policy vs the built-in heuristic AI

```python
from fishrl.opponents.heuristic import evaluate
from fishrl.opponents.random_masked import RandomMaskedPolicy
print(evaluate(RandomMaskedPolicy().act, n_games=20))   # win/loss record
```

## Self-play training stack

Adversarial self-play (masked PPO) plus two auxiliary belief/outcome models:

- **Hand-guesser** (`models/guesser.py`): per-seat, takes the fair perspective view
  ⊕ its previous guess, predicts the **opponent's hand as per-name expected counts**
  (Poisson). Its output augments the actor's observation via `BeliefAugmentedEnv`.
- **Advantage estimators** (`models/estimators.py`): both output **P(p1 wins)**,
  trained vs the terminal winner — a **privileged** one (full hidden info; also the
  PPO critic, asymmetric actor-critic) and a **public** one (mutual knowledge only).

```python
from fishrl.train.config import Config
from fishrl.train.train_loop import train, build_models
cfg = Config(iters=50, games_per_iter=8)      # CPU-friendly defaults
models = train(cfg, build_models(cfg))         # warmup -> PPO self-play

from fishrl.eval.metrics import winrate_vs_random, winrate_vs_heuristic, estimator_metrics, collect_eval_batch
print(winrate_vs_random(models), winrate_vs_heuristic(models))
print(estimator_metrics(models, collect_eval_batch(models)))   # privileged should beat public
```

Or run the end-to-end demo: `python -m fishrl.eval.smoke`.

### Running (CPU by default, `--gpu` to use CUDA)

The runnable entry points take a `--gpu` flag (falls back to CPU with a warning if
CUDA isn't available). Models and per-update minibatches move to the device; rollout
collection stays on CPU (the engine is pure Python), so the GPU mainly accelerates
the network updates.

```bash
python -m fishrl.train --iters 200 --encoder entity         # bounded run
python -m fishrl.eval.smoke --gpu                            # short demo + metrics
python -m fishrl.eval.ab_encoder --gpu                       # flat-vs-entity A/B
```

### Opponent pool / PFSP league

By default a fraction of each iteration's games is played against a **league**
opponent instead of mirror self-play: the scripted anchors (random / attacker /
heuristic / heuristic_1_1) plus a ring of frozen past-self snapshots (one appended
per status report). Only the learner seat's transitions are trained. Opponents are
sampled by prioritized fictitious self-play over the learner's per-opponent
win-rate. Two heuristic versions exist: `heuristic` is **v1.0**
(`forgetful_fish/ai.py`) — the long-standing training opponent, the vs-heuristic
eval anchor, and the scenario bot; `heuristic_1_1` is **v1.1**
(`forgetful_fish/ai_v1_1.py`, the stronger testbench line, ~65% vs v1.0) and is a
POOL opponent only, so the eval baseline stays comparable across the run.

- `--pool-frac` (default 0.25) — fraction of games vs a league opponent; 0 = pure self-play
- `--pfsp-mode` (`hard` default | `var`) — `hard` favours opponents you lose to; `var` favours even matchups
- `--league-size` (default 8) — length of the frozen past-self ring; 0 = anchors only

### Scenario curriculum (`--scenario-frac`)

`--scenario-frac` (default 0.0 = off; the service runs 0.3) seeds that fraction of
each iteration's games from a short, targeted scenario start-state instead of a
full game. Scenarios shape only the initial state + termination; reward stays
terminal ±1. Six are registered (`fishrl/train/scenarios/`):

- `known_threat` — a card-advantage spell sits on top of the shared library; deny it (counter it or manipulate a dud on top) before the empty-handed bot draws it. Curated counter+manipulation grip.
- `known_threat_random` — same denial test with a random 7-card grip: answer the threat (or recognise you can't) with whatever you hold.
- `board_presence` — both seats at 4 life, one creature each, all other creatures stripped: a tight stack fight over the pivotal creature.
- `deckout` — creatures exiled, 40-card shared library: a pure card-advantage / deckout race.
- `survive_lethal` — agent at 4 life facing untapped Dandâns with a random grip; survive the swing back.
- `survive_lethal_vision` — same, but a Vision Charm is guaranteed in grip (its land mode turns off the Islands the Dandâns need) — measures whether the agent finds and casts the clean answer.

Scenario win-rates are logged per scenario but are a **curriculum signal only** —
judge progress on the full-game vs-heuristic eval.

### Long offline runs: checkpointing, resume, and the systemd service

`python -m fishrl.train` checkpoints to `<ckpt-dir>/latest.pt` (atomically) every
`--checkpoint-every-seconds` (default 900) and on SIGTERM/SIGINT, and writes a numbered
milestone at each status report. The checkpoint carries everything needed to resume
seamlessly — all four model state_dicts, the frozen-self anchor, the three optimizers, RNG
state, the iteration counter, and cumulative wall-clock — so a crash/restart loses at most the
in-flight iteration (and never re-runs warmup).

```bash
python -m fishrl.train --resume --iters 0       # resume latest.pt; run forever until stopped
python -m fishrl.train --resume --iters 0 --max-hours 48
```

`--iters 0` runs unbounded; `--resume` continues `latest.pt` if present (a fresh start over an
existing checkpoint requires `--fresh`). On resume the architecture/seed are taken from the
checkpoint (the `--encoder*` flags are ignored, and a mismatch is rejected).

A `Restart=always` systemd **system** service (`/etc/systemd/system/fishrl-selfplay.service`,
running as the user via `User=`) drives this offline and auto-resumes on crash/reboot.
The unit files and launcher scripts are committed under [`deploy/`](deploy/) (see its
README for install paths and the stale `--user` unit to ignore). Windows is also
supported — see [`deploy/WINDOWS.md`](deploy/WINDOWS.md) for setup, `--reserve-cores`,
`--collect-workers`, the `fishrl.serve` dashboard, relay training, and
`fishrl.transfer` (zip export/import of a run between machines):

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now fishrl-selfplay
journalctl -u fishrl-selfplay -f                 # follow the hourly status lines
```

The unit caps BLAS/torch threads (`OMP_NUM_THREADS` etc.) to share the box politely and sets
`TimeoutStopSec=150` so the graceful checkpoint finishes before SIGKILL on stop/restart.

**Win rates run out-of-band.** The inline panel is single-threaded and blocks the loop, so the
service passes `--report-winrate-games 0` (skip it) and a separate timer evaluates instead.
`fishrl.eval.parallel_panel` reads `latest.pt` and fans the games for each anchor across worker
processes (one BLAS thread each — parallelism is across processes), so a 100-game panel finishes
in a few minutes on the spare cores *without* pausing training. torch is seeded per chunk, so the
panel is reproducible for a fixed checkpoint + worker count (hour-over-hour deltas reflect the
policy, not sampling noise).

```bash
python -m fishrl.eval.parallel_panel --ckpt-dir checkpoints --n-games 100   # one-off
sudo systemctl enable --now fishrl-eval.timer                               # hourly, logs [eval ...]
journalctl -u fishrl-eval.service -f
```

**Best checkpoint.** Each panel also rolls `<ckpt-dir>/best.pt` — the highest win-rate-vs-heuristic
checkpoint seen so far (a full, resume-able snapshot; the winning panel is recorded in `best.json`).
It is saved from the in-memory payload that was just evaluated, so it always matches the reported
rate even though the trainer overwrites `latest.pt` mid-eval. Pass `--no-best` to skip it. (n=100
has ±5% binomial noise, so treat `best.pt` as the best *measured* checkpoint, not a certainty.)

**Machine-readable history: `stats.json`.** Both processes also append to
`<ckpt-dir>/stats.json` (flock + atomic write, see `fishrl/train/stats.py`): the trainer
adds a `"reports"` record at each status report, the eval service an `"evals"` record at
each panel — the plottable companion to the journald lines. History from before
`stats.json` existed can be recovered from journald with
`python -m fishrl.eval.backfill_stats --ckpt-dir checkpoints` (idempotent; live rows win).

Key design points: one shared policy plays both seats; each env step (including
compound-decision sub-steps) is one PPO transition; per-seat GAE uses a single
zero-sum sign convention (`V_p1 = -V_p2`, guarded by `tests/test_perspective.py`);
the guesser is frozen within each PPO update and slow-refreshed between iterations.

### Per-card observation features (`obs/encoder.py`, `CARD_F`)

Each card slot encodes: name one-hot (20-card vocab) + unknown bit; generic type
flags (land/creature/instant/sorcery); **effective basic land type** (5-way
multi-hot read from the *rewritten* type line, so a Mind-Bended Island reads as
Swamp); **basic types referenced in the oracle text** (5-way, e.g. a Dandân's
"Island" clause after a text change); a **text-altered flag**; power/toughness/
damage/counters; tapped / summoning-sick / controller-is-self; and a known bit.
Text changes (Mind Bend / Crystal Spray / Vision Charm) are thus fully visible to
agents, including which basic type a permanent currently is.

### Front-end encoder (`Config.encoder`)

`"flat"` is an MLP over the raw observation. `"entity"` reshapes the same
flat vector into per-card rows and applies a **shared card encoder** (name embedding
+ feature MLP, learned once and reused across every zone/slot), with zone and
positional embeddings and masked mean+max pooling per zone — far fewer params
(actor 1.48M→0.43M) and built to compose relational structure. `"attention"` adds
cross-zone self-attention and a learned per-zone attention pool on top of the entity
front-end. The critic gets more capacity (`Config.critic_hidden`) since it is off the
inference path.

The encoder is resolved **per net** (`--encoder` is the base; `--actor-encoder`,
`--critic-encoder`, `--guesser-encoder`, `--public-encoder` override it). Current
defaults (`fishrl/train/config.py`): base `encoder="flat"`, and
**`critic_encoder="entity"`** — on the on-policy privileged-critic A/B, entity is the
calibration winner (Brier 0.261 vs flat 0.342), it is off the deployment path, and
it is ~free now that the value pass is batched post-collection. The actor/guesser/
public stay `"flat"` until a win-rate head-to-head backs flipping them
(`fishrl.eval.actor_headtohead` is that experiment). Compare critics with
`python -m fishrl.eval.ab_encoder`.

## Layout

```
fishrl/
  forgetful_fish/   vendored rules engine (state, engine, cards, ai) — do not edit
  data/             fish_cards.json (the 80-card decklist), buffer.py (rollout
                    buffer), features.py (god-state features for the critic)
  env/              aec_env.py (FishAEC), driver.py, apply.py
  spaces/           action_space.py, masking.py, compound.py
  obs/              encoder.py, vocab.py
  models/           policy.py, estimators.py, guesser.py, mlp.py, entity_encoder.py
  opponents/        random_masked.py, heuristic.py (sandbox eval vs ai.py)
  selfplay/         pettingzoo_api.py (env / raw_env factory)
  train/            config.py, train_loop.py, ppo.py, collector.py, pfsp.py,
                    scenarios/, checkpoint.py, stats.py, ...
  eval/             metrics.py, parallel_panel.py, smoke.py, ab_encoder.py,
                    actor_headtohead.py, backfill_stats.py, profilers
  tests/            pytest suite
deploy/             systemd units + launchers for the offline service (see deploy/README.md)
```

## Design notes

- **Agent-as-human-seat.** The engine is run as a two-human game
  (`new_multiplayer_game`, both seats `is_ai=False`); it pauses at every decision
  by setting `g.pending`. The env treats each learning agent as such a seat — no
  edits to the rules core. `g.pending.player` is the sole turn authority (turn
  order is *not* strictly alternating).
- **Masked action space** (`Discrete`, see `fishrl/spaces/action_space.py`): cards
  are addressed by zone-slot index; combinatorial decisions (scry, blocks,
  Fact-or-Fiction split, ...) are accumulated by an env-side builder and only sent
  to the engine once well-formed. Every unmasked action is guaranteed accepted.
- **Observations** come strictly from `state.current_view`, which enforces
  hidden-information legality (no opponent-hand / unknown-library leakage).
- **Reward**: sparse terminal ±1 (zero-sum). `stops_mode="default"` keeps the
  decision space small (act at own mains + responses); `"full"` gives priority at
  every step.

## Test

pytest is the only dev dependency (`pip install -e .[dev]`):

```bash
python -m pytest -q
```
