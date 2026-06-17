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

Key design points: one shared policy plays both seats; each env step (including
compound-decision sub-steps) is one PPO transition; per-seat GAE uses a single
zero-sum sign convention (`V_p1 = -V_p2`, guarded by `tests/test_perspective.py`);
the guesser is frozen within each PPO update and slow-refreshed between iterations.

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
