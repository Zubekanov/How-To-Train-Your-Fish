# How-To-Train-Your-Fish

Adversarial machine learning agents for the game of Forgetful Fish aka Dandân.

`fishrl` wraps a complete Forgetful Fish rules engine (vendored from the Website
project under `fishrl/forgetful_fish/`, logic unmodified) as a **PettingZoo AEC**
two-agent, imperfect-information self-play environment with a flat masked action
space and a fixed-size tensor observation.

## Install

```bash
pip install -e .            # numpy, pettingzoo, gymnasium  (+ .[dev] for pytest)
```

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
python -m fishrl.train --gpu --iters 200 --encoder entity   # train + save checkpoints/fishrl.pt
python -m fishrl.eval.smoke --gpu                            # short demo + metrics
python -m fishrl.eval.ab_encoder --gpu                       # flat-vs-entity A/B
```

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

`"flat"` (default) is an MLP over the raw observation. `"entity"` reshapes the same
flat vector into per-card rows and applies a **shared card encoder** (name embedding
+ feature MLP, learned once and reused across every zone/slot), with zone and
positional embeddings and masked mean+max pooling per zone — far fewer params
(actor 1.48M→0.43M) and built to compose relational structure. Both nets share the
trunk; the critic gets more capacity (`Config.critic_hidden`) since it is off the
inference path. Compare them with `python -m fishrl.eval.ab_encoder`. NOTE: on the
current privileged-critic A/B, flat still wins (held-out Brier 0.23 vs 0.35), so the
default stays `"flat"` — the entity encoder likely needs attention (not just
pooling) to beat it; that's the noted follow-up before flipping the default.

## Layout

```
fishrl/
  forgetful_fish/   vendored rules engine (state, engine, cards, ai) — do not edit
  data/             fish_cards.json (the 80-card decklist)
  env/              aec_env.py (FishAEC), driver.py, apply.py
  spaces/           action_space.py, masking.py, compound.py
  obs/              encoder.py, vocab.py
  opponents/        random_masked.py, heuristic.py (sandbox eval vs ai.py)
  selfplay/         pettingzoo_api.py (env / raw_env factory)
  tests/            pytest suite
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

```bash
python -m pytest -q
```
