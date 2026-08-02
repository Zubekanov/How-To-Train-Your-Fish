# Scope: heuristic→action-space adapter (BC bootstrap)

Goal: label every RL decision state with "what would heuristic vX do here", as a flat
`Discrete(285)` action id, so a policy can be behaviour-cloned from the scripted
heuristics and then fine-tuned with PPO (AlphaGo-style pipeline). This document scopes
the adapter; nothing here is implemented yet.

## Why an adapter is needed at all

Scripted seats never touch the action space today: the engine drives them internally
(`is_ai=True` → `_give_priority` calls `take_priority`, pendings route through
`resolve_pending`, combat through `choose_attackers`/`choose_blocks`). There is no
heuristic→action-id mapping anywhere. BC needs the heuristic's answer *as the student
seat*, expressed in the same `(obs, mask, action)` alphabet the RL agent uses.

## Three findings that shrink the problem

1. **Four hooks are already functional** (return values, no interception needed):
   `choose_attackers` (list), `choose_blocks` (dict), `choose_trigger_target` (id),
   `choose_discards` (list). These map directly to builder sub-action sequences.

2. **Every `resolve_pending` handler is compute-then-act**: each `_PENDING_HANDLERS`
   entry (ai.py:774-789) does read-only evaluation and then makes exactly **one**
   `E.complete_*` / decision call at the end. So a record-and-abort shim works:
   monkeypatch the target `E.*` functions to capture `(fn, args)` and raise a sentinel
   exception. The handler's logic runs read-only up to the call; the real game state is
   never mutated; no deepcopy required (keep a deepcopy in v1 anyway as belt-and-braces).
   `take_priority` (imperative, ≤1 action per call) is shimmed the same way; recording
   nothing = the heuristic declines = label `PASS`.

3. **Pay is a pure function**: `_best_land_to_tap` (ai.py:193-226) computes the tap
   choice without side effects. Pay pendings are labeled by calling it directly —
   `TAP_LAND[bf.index(land)]`, or `CANCEL_PAY` when it returns None.

## Translation table (pending type → label)

Atomic (answer = one action id, from the recorded call's args):

| pending | heuristic source | flat label |
|---|---|---|
| `priority` | `take_priority` shim | `PLAY_HAND[i]` / `PLAY_HAND_ALT[i]` / `CYCLE_HAND[i]` / `ACTIVATE[..]` / `TAP_LAND[i]` (float) from the recorded call; nothing recorded → `PASS` |
| `pay` | `_best_land_to_tap` (direct call) | `TAP_LAND[bf.index(iid)]`; None → `CANCEL_PAY` |
| `choose_play_order` | `_decide_play_order` | `PLAY_ORDER[0/1]` |
| `mulligan` | `_decide_mulligan` | `MULLIGAN[0/1]` |
| `fof_choose` | `_decide_fof_choose` | `FOF_CHOOSE[pile-1]` |
| `choose_text_change` | `_decide_text_change` | `TEXT_CHANGE[5*from+to]` |
| `choose_targets` | recorded `complete_targets` | `PICK_SINGLE[ctx["legal"].index(t)]`; cancel → `TARGET_CANCEL` |
| `choose_graveyard` / `search_library` / `put_from_hand` | handlers | `PICK_SINGLE[ctx["eligible"].index(pick)]`; None → `PICK_NONE` |
| `name_card` | `_decide_name_card` | `PICK_SINGLE[ctx["names"].index(name)]` |
| `order_triggers` | replicate the engine's AI auto-order (engine.py:739) | `PICK_SINGLE[trigger index]` |

Compound (answer = a **queue** of sub-actions drained one per env step; builder item
lists are frozen at creation, so indices computed once stay valid):

| pending | heuristic answer | sub-action sequence |
|---|---|---|
| `scry` | `(tops, bottoms)` | `PICK_A[items.index(t)]` per top **in order** (append order = final order), then `PICK_B` per bottom |
| `reorder` | `(order, shuffle)` | shuffle → `SHUFFLE`; else `PICK_A` per card in order |
| `putback` / `bottom` / `discard` | ordered list | `PICK_A[items.index(c)]` per card in order |
| `fof_split` | `(pile1, pile2)` | `PICK_A`/`PICK_B` per card, canonical items order |
| `declare_attackers` | list | `PICK_A` per attacker, then `COMMIT` |
| `declare_blockers` | dict attacker→blockers | attackers sorted **ascending by index** (builder focus is forward-only): `PICK_B[j]`, then `PICK_A[i]` per blocker; finish with `COMMIT` |

Adapter contract: **one expert query per engine pending**, then drain the queue without
re-querying (the heuristic answered the whole decision; re-querying mid-builder would
ask it about a partial state it never sees).

## Data generation

Run games through the normal AEC env with the student seat driven by the adapter
(exactly the RL decision distribution, including pay/priority micro-decisions) against
a mixed opponent roster (h1.0/1.1/1.2, attacker, random, adapter-vs-adapter) so the
clone sees diverse states — not just the teacher's own line. Every decision point emits
`(obs, mask, expert_action_id)`. No NN forward in the loop → collection is fast; the
existing worker pool infrastructure applies.

## Correctness: the golden parity test

The decisive test: same seed, two runs — (a) heuristic as engine-internal `is_ai` seat,
(b) heuristic via adapter driving the flat action space — must produce **identical game
transcripts**. Any divergence is a translation bug. Risks to parity: env stop-schedule
differences (extra priority windows are harmless if the heuristic declines them),
engine RNG stream alignment. If exact parity proves brittle, fall back to
distributional checks (win rates, action histograms per pending type) plus per-type
unit tests against handcrafted states.

## Known wrinkles

- `END_TURN` and `ALLOC_MANA` are never labeled (the heuristic doesn't use them) — a
  benign style artifact; PPO can rediscover them.
- The v1.0 `resolve_pending` guard (ai.py:606) checks `ai_profile == "heuristic"`;
  the adapter bypasses it by calling the version module's handlers directly — also how
  one adapter serves v1.0/1.1/1.2 (same hook surface per module).
- Trigger targeting for a non-AI seat may surface as a different pending than the AI
  path — verify which pending type carries Mystic Sanctuary targeting for humans and
  map `choose_trigger_target` onto it.
- `take_priority`'s per-turn budget attr (`g._ai_actions`) is irrelevant under the
  adapter (one decision per query) but the shim must not corrupt it.

## Effort estimate

| piece | size |
|---|---|
| shim + translators + queue (`fishrl/imitate/adapter.py`) | ~300-400 lines + per-type unit tests |
| golden parity test | ~100 lines, most of the debugging time lands here |
| data-gen driver (worker-pool fan-out, buffer of (obs, mask, label)) | ~150 lines |
| BC trainer (masked cross-entropy, eval vs anchors) | ~150 lines |
| PPO handoff (critic-only warmup, KL-to-teacher anneal, entropy floor) | config + ~100 lines, plus babysat experiments |

Roughly: adapter + parity 2-3 focused sessions, BC pipeline 1, handoff experimentation
GPU-bound. Sequencing per the plan: validate with **h1.2 now** (exists today); the
payoff compounds at the **next architecture restart**, and h2.0 slots in later for free.

## Status: IMPLEMENTED (adapter + datagen + BC trainer)

* `fishrl/imitate/adapter.py` — `HeuristicAdapter` per the design above, any
  heuristic profile. Extra findings over the scope:
  * `_choose_action` / `_best_land_to_tap` are pure decision halves, so priority
    and pay need no shim at all; the cast TARGET rides along in the ("cast", ...)
    tuple and is cached for the following `choose_targets` pending.
  * `take_priority`'s per-turn action budget (`g._ai_actions`, 30/turn, counted
    per priority grant, SHARED across AI seats via a game attribute) is part of
    observable teacher behaviour — replicated verbatim.
* `fishrl/imitate/datagen.py` — `TeacherEnv` (p1 env-driven, p2 engine-internal,
  built on `new_sandbox_game`) + mirror games (both seats adapter-driven, both
  labelled); worker-pool chunked generation; obs stored f16 WITHOUT the belief
  channel (zeros appended at train time).
* `fishrl/imitate/bc.py` — masked cross-entropy BC trainer; split-by-game holdout
  accuracy (overall + non-forced); writes a RESUME-COMPATIBLE checkpoint
  (architecture record + fresh optimizers + frozen snapshot + RNG), verified to
  load through `config_from_checkpoint` + the trainer's resume path.
* `fishrl/tests/test_imitate.py` — pure translation units, a vs-internal datagen
  smoke, and GOLDEN PARITY: adapter-mirror (full stops) vs both-seats-internal
  engine games, 8 seeds: **zero mask-forced deviations on 8/8, strict transcript
  equality on 7/8**. The exempted seed is a driving-flow property, not a
  translation gap: the engine-internal hold-priority flow grants fewer
  post-resolution windows than the env flow; the env re-offers the window to the
  adapter AND to the RL agent in deployment, so the adapter's answer is the
  deployment-correct one. (One cosmetic normalization: the env path logs
  "X's ability targets Y." at trigger placement; the internal path sets the same
  target silently.)

Not yet implemented: the PPO handoff protocol (critic/guesser warmup before
trusting advantages, KL-to-teacher anneal, entropy floor) — deliberately a
separate, babysat experiment.
