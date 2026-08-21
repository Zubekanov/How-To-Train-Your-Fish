# How To Train Your Fish — Design, Theory, and Telemetry

An explanatory companion to the `README`. The README tells you **how to run** the stack; this
document explains **what it is doing and why**, and specifies the **telemetry contract** that
downstream consumers (the dashboard, the website collector, the grapher agent, `backfill_stats`)
depend on.

Everything below is stated against the code as of `02ac079`. File:line references are given so
claims stay checkable — if this doc and the code disagree, the code wins and this doc is a bug.

**Contents**
- [Part 0 — The game, and why it is an interesting RL problem](#part-0)
- [Part 1 — Theory and techniques](#part-1)
- [Part 2 — Design decisions and the evidence behind them](#part-2)
- [Part 3 — Telemetry reference](#part-3)
- [Part 4 — Traps for telemetry consumers](#part-4)
- [Part 5 — The current run](#part-5)

---

<a name="part-0"></a>
## Part 0 — The game, and why it is an interesting RL problem

**Forgetful Fish** (aka *Dandân*) is a two-player mono-blue Magic: the Gathering format. Its
defining structural feature, and the reason it is worth training on, is that **both players share
a single ordered hidden library** (`fishrl/forgetful_fish/state.py:5-7`). There is one 80-card
deck, one graveyard, one exile. Cards are communal (`CardInstance.owner = None`,
`state.py:52`), and each player carries a *separate* record of which cards they personally know
(`known_by`, `state.py:56`).

The 80-card deck (`fishrl/data/fish_cards.json`, 21 entries): 20 Island, 8 Memory Lapse, 10 Dandân
(two printings collapsing to one name), 4 Accumulated Knowledge, 4 Lonely Sandbar, 3 each
Brainstorm / Crystal Spray / Ponder / Predict, 2 each Day's Undoing, Fact or Fiction, Halimar
Depths, Metamorphose, Mind Bend, Mystic Sanctuary, Mystical Tutor, Svyelunite Temple, Temple of
Epiphany, The Surgical Bay, Vision Charm.

Two win conditions (`engine.py`): an opponent's life hits 0 (`_lose`, `engine.py:2290`), or a
player **draws from an empty library** (`engine.py:61-71`, for *any* draw including spell draws,
per CR 104.3c/120.3). Starting life is 20.

Three properties make it a genuinely hard learning problem rather than a toy:

1. **Asymmetric, divergent knowledge of a shared secret.** Both players are drawing from the same
   hidden library, and their beliefs about it diverge based on what each has seen. Brainstorm,
   Ponder, and Halimar Depths let you learn — and rearrange — the top of a deck your *opponent*
   also draws from. `shuffle_library` clears all knowledge (`state.py:373`);
   `forget_rearranged` (`state.py:387`) models partial forgetting.
2. **Card identity is mutable.** Mind Bend, Crystal Spray, and Vision Charm rewrite type lines and
   oracle text mid-game. Because Dandân has a state trigger — *"when you control no Islands,
   sacrifice it"* (`_check_state_triggers`, `engine.py:2301`) — turning an Island into a Swamp is
   **removal**. The observation therefore encodes the *rewritten* type line and the basic types
   named in the *current* oracle text, not the printed card.
3. **The deckout race is the real game.** Because the library is shared and finite, every extra
   draw is not merely card advantage — it advances a shared clock toward a loss condition, and
   *parity* decides who hits the empty library first. This is the strategic core the agent has
   most struggled with (see [the deckout clock](#deckout-clock)).

The engine under `fishrl/forgetful_fish/` is **vendored** from the Website project. Its logic is
treated as read-only; genuine engine fixes found here must be mirrored upstream to the
`heuristic-testbench` repo.

---

<a name="part-1"></a>
## Part 1 — Theory and techniques

### 1.1 The formulation

A two-player, zero-sum, imperfect-information game, framed as a **PettingZoo AEC** environment
(`fishrl/env/aec_env.py:25`) with two seats `p1`/`p2`. Formally each seat faces a POMDP whose
hidden state includes the opponent's hand and the library order.

**Agent-as-human-seat.** The engine is run as a two-*human* game (both seats `is_ai=False`); it
pauses at every decision by setting `g.pending`. The env presents each learning agent as such a
seat, so **no edits to the rules core are needed** to make the game controllable.
`g.pending.player` is the sole turn authority — turn order is *not* strictly alternating, and the
env never assumes it is (`aec_env.py:3-5, 144`).

**Reward is sparse and terminal only**: ±1 zero-sum at game end, 0 everywhere else
(`aec_env.py:140`). There is no reward shaping anywhere in the stack. This is a deliberate
commitment: shaped rewards change the optimal policy unless carefully constructed, and the whole
point is to learn what actually wins.

**Observations come strictly from `state.current_view`**, which enforces hidden-information
legality at the engine level — the actor physically cannot see the opponent's hand or unknown
library cards. This is what makes the privileged critic (below) sound rather than cheating.

### 1.2 Masked action space

A single flat `Discrete(285)` head (`fishrl/spaces/action_space.py`), with a legality mask
supplied per decision. Illegal logits are set to `-1e9` before log-softmax
(`models/policy.py:25-29`), so illegal actions receive zero probability and contribute no gradient.

Cards are addressed by **zone-slot index** — the ordinal position in the engine's ordered zone
list — and resolved to a per-game UUID only at apply time (`action_space.py:9-14`). This gives the
key invariant: **observation row *k* and action slot *k* always refer to the same card.**

Capacities: `HAND=12`, `BF=34`, `PICK_K=64`, `CMP_K=20`, `ABIL_SLOTS=2`. Blocks: `PASS`,
`END_TURN`, `PLAY_HAND`, `PLAY_HAND_ALT`, `CYCLE_HAND`, `TAP_LAND`, `ACTIVATE`, `ALLOC_MANA`,
`CANCEL_PAY`, `PLAY_ORDER`, `MULLIGAN`, `FOF_CHOOSE`, `TEXT_CHANGE`, `PICK_SINGLE`, `PICK_NONE`,
`TARGET_CANCEL`, `PICK_A`, `PICK_B`, `COMMIT`, `SHUFFLE`.

**Compound decisions.** Genuinely combinatorial choices — scry, reorder, discard, declare
attackers, declare blockers, Fact-or-Fiction split (`spaces/compound.py:23-26`) — are not
expressible as one flat pick. A `CompoundBuilder` accumulates them **env-side** through a shared
sub-alphabet (`PICK_A`/`PICK_B`/`COMMIT`/`SHUFFLE`) and calls the engine's completion function
only once the decision is well-formed. The engine never sees a partial or illegal call, and the
policy keeps one flat head plus one mask. The builder deliberately lives in the env, **not** in
the serialized game state (`compound.py:13-14`).

Two anti-stall invariants are worth knowing, because both were learned the hard way: declaring
attackers is **add-only** (a reversible toggle lets the agent oscillate forever,
`compound.py:87-88`), and blocker focus moves **forward only** (`compound.py:92-94`).

**Every unmasked action is guaranteed accepted** — `apply_atomic` raises
`AssertionError("mask contract violated")` otherwise. Two empty-mask guards exist in `_refresh`,
because an empty mask makes the masked softmax uniform and would let an illegal action be sampled:
an empty compound builder auto-finalizes (`aec_env.py:148-151`), and a payment decision with an
empty mask cancels payment (`aec_env.py:159-165`, debuggable via `FISH_DEBUG_STRAND`).

### 1.3 Self-play with a shared policy

**One policy plays both seats.** Combined with seat-equivariant observations and masks, this has a
strong theoretical consequence that we lean on repeatedly:

> If the observation is seat-equivariant, the mask is seat-equivariant, the policy is shared, and
> the game is symmetric under the seat-swap involution σ, then the mirror win-rate **must** be
> 50/50. A persistent deviation is *proof* that one of those premises is false.

That syllogism is what turned a vague "the agent prefers p2" observation into a located engine
bug — see [§2.3](#seat-symmetry).

Each env step — **including each compound sub-step** — is one PPO transition.

### 1.4 Masked PPO

Standard clipped-surrogate PPO (`train/ppo.py`, `train/losses.py`):

```
loss = policy_loss  -  ent_coef * entropy  +  critic_coef * critic_loss     # ppo.py:41
```

- **Policy**: `-min(ratio*adv, clamp(ratio, 1-clip, 1+clip)*adv).mean()`, `clip=0.2`
  (`losses.py:14-18`). Entropy is computed over **legal actions only** (`policy.py:47-51`).
- **Critic**: **binary cross-entropy, not MSE** (`losses.py:25-29`). The critic predicts
  *P(p1 wins)* against the terminal winner, masked to decided games. This is the natural
  parameterization when the only reward is a ±1 terminal outcome: the value function *is* a win
  probability, so a classification loss is better conditioned than regressing a discounted return.
  There is no value clipping and no return regression.
- 4 epochs × minibatch 256, grad-clipped at 1.0 over actor+critic jointly, single Adam over both
  (`train_loop.py:171`), shuffle seeded per-iteration for reproducibility (`ppo.py:29`).
- `gamma=0.997` — near 1 because reward is terminal-only and episodes run ~200+ decisions.
  `lam=0.95`.

### 1.5 Asymmetric actor-critic (the privileged critic)

| Net | Input | Dim | Output | Trained by |
|---|---|---|---|---|
| `MaskedActor` | perspective ⊕ belief | 6517 | 285 logits, **no value head** | PPO clip |
| `PrivilegedCritic` | god state | 8920 | 1 logit = P(p1 wins) | BCE |
| `HandGuesser` | perspective ⊕ prev guess | 6517 | 20 softplus counts | Poisson NLL |
| `PublicEstimator` | public state | 6456 | 1 logit = P(p1 wins) | BCE, **diagnostic only** |

The critic sees **everything**: both hands and the full ordered 64-slot library
(`GOD_SLOTS`, `features.py:25-28`), with no visibility filter. The actor sees only its legal view.

This is sound because **the critic is only ever a training-time baseline**. It never selects an
action and is not shipped. Centralized-critic / decentralized-actor is the standard resolution
(the CTDE family, à la MADDPG/OpenAI Five): a value function conditioned on the true state has
dramatically lower variance than one that must average over the opponent's possible hands, and
lower-variance advantages mean a better-directed policy gradient — without leaking a single bit
into the deployed policy.

The **`PublicEstimator` is a firewalled diagnostic** (`estimators.py:5-11`). It predicts the same
quantity from mutual knowledge only. It is never in the advantage loop; its sole job is to make
`brier_gap = pub_brier − priv_brier` a *measurement of how much hidden information is worth* at
this point in the run. It can be disabled with `--train-public 0` to buy back collection time,
at the cost of that telemetry.

Both heads are **p1-oriented and seat-agnostic**. A seat's value is derived by a sign convention
rather than by re-encoding the state — see next.

### 1.6 The zero-sum sign convention and per-seat GAE

```python
SEAT_SIGN = {"p1": +1.0, "p2": -1.0}                        # advantages.py:19
V_seat   = SEAT_SIGN[seat] * (2*P(p1 wins) - 1)             # advantages.py:23
z_seat   = +1 if winner == seat, -1 if winner == opp, 0 if draw/truncated
```

Both seats read the **same** `P(p1 wins)` with opposite sign, which enforces `V_p1 ≡ −V_p2`
exactly, by construction rather than by hope (`advantages.py:11-13`, guarded by
`tests/test_perspective.py`). One critic, one calibration, no drift between two heads.

**GAE runs per `(game_id, seat)` segment** (`buffer.py:70-71`) — each seat's own ordered
subsequence of decisions within one game. The reward vector is all zeros except the last element
(`buffer.py:76`). The bootstrap is `values[-1]` if the game was **truncated** by the decision cap
and `0.0` if it genuinely ended (`buffer.py:77`) — a cap hit is an unknown outcome, not a draw,
and scoring it as one would teach the agent that stalling is safe.

Advantages are normalized **jointly across both seats** after the per-segment `p1_adv_weight`
scaling (`buffer.py:79-86`). Since normalization is a global affine map, any intentional p1:p2
magnitude ratio survives it. (`p1_adv_weight` is currently 1.0 — see [§2.3](#seat-symmetry) for
why it was reverted.)

Critic values are filled **after** collection in one batched forward (`fill_critic_values`,
`collector.py:69-90`), not per-decision. This is numerically identical — the critic is frozen
during collection — and reclaims ~14% of collection wall-clock.

### 1.7 Belief modeling

The `HandGuesser` predicts the **opponent's hand as per-name expected counts** (a length-20
non-negative vector, softplus output), trained with **Poisson NLL** against the true counts
(`losses.py:32-34`). Poisson is the right likelihood here: the target is a vector of small
non-negative *counts*, not a distribution over one card.

The label `opponent_hand_counts` is a god-state quantity used **only as a supervised target** —
never as an input (`features.py:49-57`).

Three design commitments make this honest:

1. **Supervised-only, never end-to-end.** The guess is computed under `no_grad` and baked into the
   stored observation, so the only gradient reaching the guesser is its own Poisson NLL
   (`guesser.py:9-13`). Were it trained through the policy, it would stop being a posterior over
   the opponent's hand and become "whatever scalar helps the actor" — an unfalsifiable extra layer
   rather than an interpretable belief.
2. **Per-seat carry.** The guesser is recurrent through its own previous guess, and each seat
   carries its **own** previous guess (`belief_env.py:4-7`). The seats interleave, so a shared
   belief buffer would leak one seat's guess into the other's input — a real information leak
   masquerading as memory.
3. **Frozen within an update, slow-refreshed between.** This keeps the actor's input distribution
   stationary across a PPO update (`train_loop.py:3-4`).

The actor's input is `concat([perspective, guess])` → 6497 + 20 = 6517. With `use_belief=False`
the channel is zeroed and no guesser forward runs.

### 1.8 The PFSP league

Pure self-play is prone to cycling and to forgetting how to beat styles it no longer plays.
The league is **prioritized fictitious self-play** (`train/pfsp.py`), following the AlphaStar
line:

```
hard -> (1 - wr)^p      # p = 2.0; focus on opponents you LOSE to     (pfsp.py:73-82)
var  -> wr * (1 - wr)   # focus on even matchups
weights = (priority(wr) + eps) * member.weight    # eps = 0.05 floors starvation
```

Win-rates are tracked as an EMA over decided games (`wr_ema=0.1`, prior 0.5).

**Members** = scripted anchors + a ring of frozen past selves:

| Anchor | Role |
|---|---|
| `random` | masked-uniform floor |
| `attacker` | barely above random — *not* a skill test |
| `heuristic` (v1.0) | **the eval anchor**, held fixed for run-long comparability |
| `heuristic_1_1` | stronger testbench line — **pool only** |
| `heuristic_1_2` | testbench line frozen at release, plays deckout — **pool only** |
| `heuristic_1_3` | current testbench mainline (evaluator off; ~83% vs v1.0 in the testbench arena) — **pool only** |

The versioning discipline matters: **only v1.0 is ever the eval anchor**, so the headline
vs-heuristic curve remains comparable across the entire run even as better opponents enter the
pool (`pfsp.py:36-38`).

The **past-self ring** (`deque(maxlen=league_size=8)`) deep-copies actor+guesser at each status
report (`pfsp.py:61-70`). A frozen self acts through **its own** actor *and its own guesser* —
i.e. its own belief channel — so it is a faithful snapshot of a past agent, not a hybrid.

League state **is checkpointed** (member EMAs, game counts, and the ring's weights), matched by
name on load so membership changes are safe (`pfsp.py:147-185`). Without this, every restart reset
all EMAs to 0.5 and emptied the ring for ~8 reports — a self-inflicted amnesia that made long
runs strictly worse than short ones.

Seat assignment by kind (`train_loop.py:620-634`): scenarios and `heuristic*` force the learner to
p1 (the engine resolves the heuristic on p2 — the sandbox convention); `random`/`attacker`/`self`
alternate seats by parity.

### 1.9 The scenario curriculum

Sparse terminal reward over 200-decision games is a brutal credit-assignment problem, and some
strategic situations (a tight deckout race) essentially never arise from a random start. Scenarios
address **exploration**, not reward.

> **A scenario shapes only the initial state distribution and the termination condition. Reward
> stays terminal ±1** (`scenarios/base.py:3-6`).

This is the crucial discipline: it is *not* reward shaping, so it cannot change what the optimal
policy is — it only changes which states the agent practises. `ScenarioEnv` overrides `reset` to
sample a deep-copied legal start state and `_terminal_override` to return an early verdict
(`scenarios/env.py:17-45`). The verdict is cached and **wins over** the engine result.

Seven are registered: `known_threat`, `known_threat_random`, `board_presence`, `deckout`,
`survive_lethal`, `survive_lethal_vision`, `survive_lethal_single` (one Dandân, one
guaranteed answer from Metamorphose/Mind Bend/Crystal Spray, Vision Charm excluded from
the filler — targets the text-change/removal surfaces the 2026-08-07 weakness probe
measured at ~0 exemplar agreement). All run the engine bot on p2 at profile
`heuristic_1_3` (the current testbench mainline; v1.2 until 2026-08-06, v1.0 before
2026-07-10 — `scenario_wr` steps down at each upgrade).

Two modes, mutually exclusive:
- **`--scenario-pool`** (current): each scenario becomes a league member competing for the
  `pool_frac` budget, so **its play rate floats with its difficulty** — PFSP automatically
  practises what the agent is losing.
- **`--scenario-frac`** (legacy): a fixed carve-out of each iteration.

`--scenario-weight NAME=W` sets a fixed prior multiplier (and `W=0` is the off switch).
In pool mode `--scenario-boost B` (default 3.0) additionally multiplies **every** scenario
member's weight: scenario episodes are far shorter than full games, so boosting their
play-count share costs sub-proportional wall-clock, and `pool_frac` still caps the whole
pool slice (mirror self-play keeps the rest). `B=1` restores the unboosted behaviour.

**2026-08-21 — the hands critic (`critic_view="hands"`, swapped in place on the v3 lineage).** A supervised
benchmark on 88.7k v1.3-mirror states (same MLP head, 3 split seeds) ranked the critic's
possible inputs: public Brier .190, **public + both hands .183**, + each player's known top-8
.184, god .188. The hands carry the value the full library order drowns (god is *worse* than
hands with strictly more information), and known-top slots add nothing. Encode cost is 18–33 µs
for every view with the C fast path, so the old omniscient-critic throughput argument no longer
applies. `HANDS_SLOTS` = the public layout minus the library rows, both hands fully visible
(`features.encode_hands`, `HandsCritic`, same aux head and gates as public; `set_public_view`
selects the pub_feat encoding). Installed IN PLACE on the v3 lineage with
`fishrl.train.swap_critic` at it 47,538: only the critic and its slice of the Adam state are
fresh; actor, league, scenario EMAs, counters and elapsed continue. The checkpoint's
`handoff_start` makes `--freeze-actor-iters` / `--kl-teacher-*` count from the swap (a
1,500-iteration critic-only freeze, KL-to-teacher 0.1 annealed over 10k) without resetting
`done`. (A first attempt bootstrapped a separate `checkpoints-v4` lineage with `done=0`; that
was unnecessary and was discarded after ~1 h of critic warmup.)

**2026-08-21 — envelope-constructed curriculum** (`scenarios/envelope.py`, `scenarios/constructed.py`).
The seven original scenarios were *complete manufactures from hand constants* (both players at
4 life, every other Dandân exiled, ten lands on turn 2) — legal states in corners no real game
visits, four of them ending on proxy terminators. They stay registered at weight 0. The
replacement set is built by one shared sampler, `envelope_sample`, which randomises inside a
measured envelope of real game states (per-turn-bucket bands of life / lands / Island-typed lands
/ hand / fish / library / graveyard, the graveyard drawn from the real cast mix, exile = resolved
Undoings, zone conservation) and each scenario adds only a few `Overrides`. All end by the
natural game result. Members: `fish_war`, `response_window` (+`_bend`), `protect_the_fish`,
`removal_in_hand`, `deckout_short`, `deckout_with_fish`, `lethal_on_board`, `steer_the_top`,
`undoing_call` — each aimed at a gap measured by the 13k-game trace / 700-game critic ledger
(first-blood race, passing with an instant up, removal aimed at lands, the parity endgame).
v3-best starting win rates 0.15–0.57.

**Scenario win-rates are a curriculum signal, never a success metric** (`config.py:90-91`). Judge
progress on the full-game vs-heuristic eval only.

### 1.10 The entropy schedule

Linear anneal from `ent_start=0.02` to `ent_end=0.005` over a horizon, then hold at the floor
(`config.py:226-244`). The horizon is `iters` when bounded, else `ent_anneal_iters`.

The **cosine sawtooth re-heat** (`ent_reheat_period`, default 0 = off) exists because the plain
anneal was shaped for a ~3k-iteration run. On a multi-100k-iteration run it pins exploration at
the floor for essentially the whole run — fine for sharpening a nearly-converged policy, bad for
escaping a self-play local optimum. The re-heat fires only *after* the initial anneal and replays
the range on a cosine sawtooth (LR warm restarts, applied to entropy). It costs a mild win-rate
wobble at each re-heat, so the standing guidance is **enable it when a run stalls, not
preemptively**.

### 1.11 Observation encoding

Three encodings, all built from a shared 44-float **card row** (`CARD_F=44`,
`obs/encoder.py:52`):

```
name one-hot (20) | unknown (1) | type flags (4) | effective basic type from type_line (5)
| basic types named in oracle_text (5) | text_variant (1) | P/T/damage/counters (4)
| tapped / summoning-sick / controller-is-self (3) | known (1)
```

Note what rows 21–31 buy: the **effective** basic type is read from the *rewritten* type line, so a
Mind-Bended Island reads as a Swamp, and the basic types named in the *current* oracle text are
encoded separately. Text changes are therefore fully visible to the agent — which they must be,
since text-changing *is* the removal suite.

| Encoding | Dim | Zones | Consumer |
|---|---|---|---|
| perspective | **6497** | own/opp hand 12, own/opp bf 34, gy 32, exile 8, stack 6, **library 8** | actor, guesser |
| god | **8920** | p1/p2 hand 12, p1/p2 bf 34, gy 32, exile 8, stack 6, **library 64** | critic |
| public | **6456** | as god but **library 8**, hands filtered to what the non-owner knows | public estimator |

A library slot is public iff **both** players know it (`features.py:220-222`).

The globals tail (73 for perspective; 32 for god/public) carries per-player scalars (life/20,
hand count, lands, untapped lands, mana pool by colour, mulligans, has_lost), game scalars
(turn/40, active_is_self, priority_is_self, **len(library)/80**, stack size), and for perspective
also the step one-hot, combat state, pending type, payment state, and compound-builder progress.

`set_public_encoding(False)` short-circuits `encode_public` to a shared zero vector
(`features.py:182-201`), skipping ~40µs/decision on the collection hot path.

### 1.12 Front-end encoders: flat vs entity

- **`flat`** — an MLP straight over the raw vector. The input projection alone dominates the
  parameter count (the 1.81M-param flat actor spent 6517×256 = 1.67M — **92%** — on its first
  layer); it must learn "a card in hand slot 3" and "the same card in hand slot 7" as unrelated
  inputs.
- **`entity`** — reshapes the same flat vector into per-card rows and applies a **shared
  `CardEncoder`** (name embedding + feature MLP, learned once and reused for every row in every
  zone), plus zone and positional embeddings, then **masked mean+max pooling per zone**
  (`models/entity_encoder.py`). `enc_dim = n_zones * 2 * d + globals_dim`.
- **`attention`** — adds cross-zone self-attention and a learned attention pool. Implemented,
  **shelved** (no measured win).

The entity encoder is the right inductive bias for this domain: cards are exchangeable within a
zone, and the same card means the same thing wherever it sits. It reaches a comparable
representation with a fraction of the parameters.

Encoders are resolved **per net** (`--encoder` is the base; `--actor-encoder`, `--critic-encoder`,
`--guesser-encoder`, `--public-encoder` override).

---

<a name="part-2"></a>
## Part 2 — Design decisions and the evidence behind them

This section records decisions that are *not* derivable from the code — the ones where the
reasoning, and especially the measurement that settled it, is the valuable part.

### 2.1 The privileged critic is entity; the actor was flat until it wasn't

An on-policy A/B of the critic front-end measured **entity Brier 0.261 vs flat 0.342** — a large
calibration win. Entity became the critic default (`config.py:213`). It is safe to spend
capacity/complexity here precisely because the critic is off the deployment path.

The actor stayed `flat` far longer, on the principle that the actor is the thing that ships and a
change there needs a **win-rate** head-to-head, not a proxy metric. A 2-seed 60-minute
head-to-head came back genuinely ambiguous: entity won against learned-opponent anchors and lost
against the scripted attacker. That was not sufficient evidence, so the actor stayed flat.

What eventually moved it was **shape analysis, not the A/B**: the 1.81M-parameter flat actor had
**92% of its parameters in the input projection**. It was almost entirely a linear map of a
sparse one-hot vector, with very little left for reasoning. That, plus a hard plateau, motivated
the restart (§2.5).

### 2.2 Belief is used, but weakly

A probe at 25 iterations found the actor uses the belief channel only weakly (per-dimension
sensitivity ~0.85× the perspective channels, small absolute sway). The honest conclusion recorded
at the time: settle it with a **win-rate on/off ablation**, which has not been run. The channel is
cheap and theoretically sound, so it stays on pending that measurement.

<a name="seat-symmetry"></a>
### 2.3 The seat asymmetry — a worked example of the method

**Symptom:** mirror self-play settled at **p1 0.37 / p2 0.63**. With a shared policy on a
symmetric game this is impossible, so one premise had to be false.

The investigation is worth recording because the method generalizes:

1. **Prove the premises one at a time.** Observations and masks were verified seat-equivariant
   over 11,743 states — 0 violations. So the obs weren't it.
2. **Rule out the plausible-but-wrong explanation.** Belief was a natural suspect (per-seat carry,
   asymmetric information). Turning belief **off** made it *worse* (0.35/0.65) — ruling it out.
3. **Therefore: the engine.** By elimination, the σ-symmetry premise itself was false.
4. **Build the instrument the hypothesis demands.** An obs-only audit found nothing, because the
   shared library's *hidden order* is not observable. A **lockstep full-hidden-state σ-mirror
   audit** (no re-syncing between steps — an early version re-synced each step and silently
   absorbed the divergence) localized it.

**Root causes found**, all the same bug in different clothes — *p1-first iteration over a shared
resource*:
- Day's Undoing shuffled and dealt p1-first over the shared library/RNG.
- `_check_sba` resolved creature deaths into the shared graveyard, and checked life loss, p1-first.
- `_check_state_triggers` (Dandân sacrifice) ran p1-first.

Fixed by resolving in **APNAP order** (active player first, CR 101.4) — `243325a`. Mirror was
**unchanged** at 0.366. The APNAP fixes were correct but not the dominant term.

**The dominant cause** was subtler and is the most transferable lesson here. `Crystal Spray` /
`Mind Bend` built their target list as `[iid for pl in g.players.values() for iid in pl.battlefield]`
— always p1-first. The action mask is `PICK_SINGLE k` = *index into that list*. Under seat swap
**the mask bits matched perfectly** — the equivariance check passed — while the *meaning* of index
`k` differed. Making the list **caster-first** (`51e69da`) fixed it:

```
mirror p1 win-rate:  0.366  ->  0.480 / 0.520
lockstep audit:      23/40  ->  0/200 violations
```

> **The lesson:** an equivariance check on mask *bits* is not a check on action *semantics*. When
> an action is an index into a list, the list order is part of the action space contract.

**Two honest corrections came out of this**, both recorded because they were mistakes:

- **`p1_adv_weight` was built on a wrong model** and reverted (`b3438c5`). It was designed to
  counteract the seat drift by up-weighting p1 advantages — but with equivariant obs and masks, a
  *shared* policy **cannot** seat-specialize in the first place. The knob was treating a symptom
  that the premises said couldn't exist. It survives at 1.0 as a no-op.
- **The seat bug was not the plateau.** After the fix the baseline was unchanged (h1.0 0.375,
  h1.2 0.225, deckout 0.042). Two real problems had been conflated. The plateau is a genuine
  strategic gap: the agent is strong at defence and tempo (`survive_lethal` 0.61, `known_threat`
  0.99) and loses the deckout race.

Mirrored upstream (testbench `522e9c5`).

<a name="deckout-clock"></a>
### 2.4 The deckout clock — shipping a feature, then proving it is ignored

**The problem.** Who decks out first is a **parity** question: given the shared library size and
who draws next, the loser is determined. But the observation carries library size as the smooth
float `len(g.library)/80.0`. A ReLU network cannot extract `n % 2` from that in any useful way —
parity is maximally non-smooth.

**The evidence** (`cee4c88`, `bf6474d`):
- `board_presence` **is** the deckout game — 81% of those games are decided by decking.
- Probing the 592h checkpoint's trunk for the deckout winner: **0.525** accuracy vs a 0.514 base
  rate. With a direct supervised label from the raw observation: 0.508. An oracle: 1.000.
  The information is *present* but *inaccessible*.
- A control probe: the opponent-library-knowledge channel does **not** predict the winner (tight
  null). We did not migrate for it. Negative results are cheap here and worth running.

**The fix** (`eb4d683`): `deckout_clock(g, viewer)` (`encoder.py:89-103`) emits 3 bits —
`(library_parity, next_drawer_is_viewer, viewer_decks_first)` — appended to all three encodings.
It is a **re-encoding of information already present**, not new information, which is why it is
legal for the actor as well as the critic. It is zeroed in pregame so the two seats cannot
disagree.

**The outcome, honestly:** an ablation showed the actor's behavior changes by **Δ0.000** with the
clock on vs off. The feature is shipped, provably computable, and **provably ignored**. This is an
*information-vs-signal* gap: the clock tells the agent the state, but nothing in a sparse terminal
reward tells it that the state *matters* 200 decisions before the payoff.

The designed-but-**unimplemented** lever is potential-based reward shaping (Ng et al. 1999):
`F = γΦ(s′) − Φ(s)` with `Φ = ±c/decksize`, antisymmetric across seats. Potential-based shaping is
the one shaping form that is provably **policy-invariant** — it changes the credit-assignment
gradient without moving the optimum. This has been repeatedly proposed and **never approved**; it
remains the most promising lever on the plateau.

### 2.5 The v2 restart: bigger entity actor

Given the 92%-in-input-projection finding and a hard plateau, the flat 622h run was preserved and
a fresh run started in `checkpoints-v2` with an **entity actor at 768/768/384, d=128** (`9bf5766`).

The prerequisite was making architecture **fully resizable and persisted** (`eba07f3`), because
the shapes had previously been implicit — a resized model would silently fail to load.
`config_from_checkpoint` (`train_loop.py:69-91`) is now documented as **the one place** that maps
checkpoint → architecture, with historic defaults so pre-resize checkpoints still reconstruct.
`tests/test_architecture_roundtrip.py` guards the contract, including backward compatibility.

### 2.6 Removing `enforce_free_attack`

A legacy rule (added early, when training was slow) forfeited a seat that declared no attackers
when attacking was strictly free. The intent was to teach the *agent* a strictly-correct move.

The bug: it fired on **whichever seat declared attackers, including scripted AI opponents**. A
random opponent that failed to swing simply lost — random went 20/20 in one harvest window.

Removed entirely in `02ac079`. Scope of the distortion, which matters for reading historical
telemetry:

| Game type | Driver | Rule fired? |
|---|---|---|
| random / attacker / past-self pool | `FishAEC` | **yes** — opponent could forfeit ⇒ **inflated** |
| mirror self-play, scenarios | `FishAEC` / `ScenarioEnv` | yes — learner was spoon-fed |
| **heuristic (all versions)** | `HeuristicMatch` | **no** — different driver |

So the headline `heuristic`/`heuristic11`/`heuristic12` curves were **never affected**. Only
`random` and `attacker` were inflated. See [Part 4](#part-4).

### 2.7 Throughput decisions

Measured, not assumed:

- **Entity ≈ flat end-to-end** (~1670–1820 vs ~1800 it/h). The encoder is not the bottleneck.
- **Encoder vectorization 1.9×** (`0edc42c`): the 8-zone Python pooling loop became a
  `scatter_add`/`scatter_reduce` (666→344 µs, maxdiff 2e-7). `jit.script` was tried and rejected —
  it failed on module-global ints and would have risked deepcopy/pickle for frozen anchors and
  worker shipping.
- **Weak worker scaling** (8→16 workers = **+6%**). Hence the resource policy: 8–12 P-core workers
  and a generous E-core reserve, which is nearly free. E-core drift was measured at **2.3× slower
  per decision**, so affinity pinning is not cosmetic.
- **Pipelining** (`--pipeline-collect`, +51% transitions/h): overlaps iteration N+1's collection
  with N's GPU update, making the behavior policy **one update stale**. This is theoretically fine
  — PPO's importance ratio and clip absorb exactly this lag; it shows up as a higher `approx_kl`
  floor. Rows are marked `"pipeline": true` so A/B comparisons stay honest.
- **Collection is CPU-bound regardless of trainer device** — the engine is pure Python. CUDA
  accelerates the *update*, not collection.
- Batched critic values + regex memoization sped collection **2.44×**. Entity's collection cost is
  a ~1.4× tax, not 6×; `engine.step` is only 1.9% of wall-clock and feature encoding is 41%.

A **benchmark methodology trap** worth recording: fresh short runs have rapidly-growing game
length (134→210 decisions), so `it/h` between two fresh runs is **not comparable**. Tier-3
throughput changes (`--games-per-iter 24-32`, `--minibatch 1024`) therefore need a settled-policy
A/B and remain unmeasured offline.

---

<a name="part-3"></a>
## Part 3 — Telemetry reference

### 3.1 Artifacts on disk

All live in `<ckpt-dir>`; all writes are atomic (tmp + `os.replace`).

| File | Writer | Contents |
|---|---|---|
| `stats.json` | trainer + eval service | `{schema:1, reports:[…], evals:[…]}` — **the plottable history** |
| `ticks.json` | trainer | `{schema:1, ticks:[…]}`, ring capped at **2000** |
| `best.pt` / `best.json` | eval panel | best-measured checkpoint + the winning panel |
| `latest.pt` | trainer | resume point (every 900s and on SIGTERM/SIGINT) |
| `step_%08d.pt` / `archive_%08d.pt` | trainer | milestone / per-10k archives |
| `owner.json`, `peer.json`, `trainer.lock`, `STOP` | trainer/serve | ownership + control plane |

Two processes write `stats.json`; they coordinate with `flock` on `stats.json.lock`
(`train/stats.py:36-45`).

### 3.2 The report row — `stats.json → reports[]`

Emitted every `report_every_seconds` (default 3600). **Deliberately pure trainer metrics — no
win-rates from a blocking panel.**

**Identity / regime**

| Field | Meaning |
|---|---|
| `it` | global iteration counter |
| `elapsed_h` | cumulative training hours **across restarts** |
| `wall_time` | epoch seconds |
| `host` | `platform.node()` — relay handoffs interleave hosts in one history |
| `device` | `cpu`/`cuda` — added after 3 sessions silently ran on CPU |
| `pipeline` | **sparse**; present (`true`) only under `--pipeline-collect`. Absent ⇒ strictly on-policy |
| `source` | `"live"` |

**Throughput**: `iters` (this window), `iters_per_h`, `transitions`, `collect_s`, `update_s`,
`collect_frac` = `collect_s/(collect_s+update_s)`.

**Losses** (window means): `policy_loss`, `critic_loss`, `entropy`, `approx_kl`, `clip_frac`,
`guesser_loss`, `public_loss`.

**Calibration** (on the last batch only):

| Field | Meaning |
|---|---|
| `priv_acc`, `priv_brier` | privileged critic — accuracy `(p>0.5)==y`, Brier `mean((p−y)²)` |
| `pub_acc`, `pub_brier` | public estimator, same |
| `brier_gap` | `pub_brier − priv_brier` — **the value of hidden information** |
| `gmae` | guesser mean-absolute-error on hand counts |

`pub_*` keys are **stripped** when `--train-public 0`.

**Game telemetry**

| Field | Meaning |
|---|---|
| `games` | games this window |
| `dec_per_game` | episode length, **full games only** |
| `scen_dec_per_game` | episode length, scenario games |
| `trunc_rate` | fraction hitting the decision cap |
| `draw_rate` | fraction drawn |
| `mirror_p1_wr` | p1 share of **decided mirror self-play** — drift from 0.5 = seat exploitation |
| `forced_dec_frac` | fraction of steps with exactly 1 legal action — decision dilution |

**Opponent mix** — `opp_trained` (= self+past), `opp_self`, `opp_past`, `opp_heuristic` (v1.0
**only**), `opp_heuristic11`, `opp_heuristic12`, `opp_heuristic13`, `opp_attacker`, `opp_random`,
`opp_scenario`.
**Denominator = grand total (league + scenario games).**

**Nested**

- `scenario_mix` — `{name: share}`, same denominator.
- `scenario_wr` — `{name: PFSP EMA}`. **Curriculum difficulty, not skill.**
- `league_wr` — `{name: EMA}` for scripted anchors with games > 0. The only WR signal for
  `heuristic_1_1`/`heuristic_1_2`/`heuristic_1_3` in the report row.
- `wr_train` — `{anchor: [wins, games]}` — **the harvest** (§3.4). Window-scoped, reset each report.

NaN/inf are sanitized to `null`.

### 3.3 The eval row — `stats.json → evals[]`

Written by `fishrl.eval.parallel_panel`, out-of-band on a timer, reading `latest.pt` so it never
pauses training.

| Field | Meaning |
|---|---|
| `it`, `frozen_at` | policy iteration evaluated |
| `elapsed_h`, `wall_time`, `took_s`, `n`, `workers` | run metadata |
| `random`, `attacker`, `heuristic`, `heuristic11`, `heuristic12` | anchor win-rates |
| `frozen` | vs the frozen self (0.5 if nothing decided) |
| `anchor_n` | `{anchor: {train, topup}}` — the sample provenance |
| `harvest_from` | high-water report `it` consumed |
| `new_best` | did this panel roll `best.pt` |
| `source` | `"eval"` (or `"inline"`, `"journald-inline"`) |
| `seat_p1_wr`, `seat_p2_wr` | same shared policy in each seat — any gap is learned asymmetry |
| `play_wr`, `draw_wr` | win-rate on the play vs on the draw — **a separate axis from seat** |
| `choose_first_frac` | how often the policy chooses to play first |
| `seat_diag_n` | decided games behind the seat diagnostics |

**The draw convention** (`metrics.py:8-11`, module-wide): **denominator = ALL games including
draws and truncations; numerator = strict wins only.** This holds for every anchor, and it is what
makes harvested and freshly-played games poolable into one estimate.

Anchors are seeded per-chunk, so a panel is **reproducible for a fixed checkpoint + worker count**
— hour-over-hour deltas reflect the policy, not sampling noise.

`best.pt` is ratcheted on the **maximin over the scripted anchors** (2026-08-07, was v1.0-only
before): a panel's `best_score` is its *lowest* anchor win-rate — the wr vs its hardest opponent,
in practice the newest testbench heuristic — and best.pt rolls when that minimum strictly
improves. A legacy v1.0-keyed `best.json` (no `best_score`) is superseded by the first panel
after the change. At n=100 the binomial noise is ±5%, so it is the best *measured* checkpoint,
not a certainty.

### 3.4 The harvest — why win-rates are "out of more than 100"

Every iteration, training plays real games against the very anchors we want to measure. Throwing
those outcomes away and re-playing 100 fresh games would be wasteful.

So the trainer counts them (`wr_train`, under the **eval convention** — all games in the
denominator, strict wins in the numerator) and the panel pools them with its own games:

```
combined[anchor] = (harvested_wins + topup_wins) / (harvested_games + topup_games)
```

`anchor_n` records the split, e.g. `{"heuristic": {"train": 281, "topup": 100}}` → that 0.189 is
out of **381 games**, not 100. This is why panel win-rates have denominators above the nominal
`--n-games`.

> **Note — this changed.** `plan_topup` **used to** play only each anchor's *deficit*
> (`max(0, target − n_train)`), capping the combined sample at `target` — a *cheaper* panel with
> the same noise. It now plays the **full target unconditionally** and harvested games are **extra
> samples on top** (`parallel_panel.py:165-176`). **Harvest now buys precision, not cheapness.**

`find_harvest` uses a **high-water mark**: each eval records the highest report `it` it consumed,
and the next takes everything strictly above it. This replaced a "newest unconsumed row only" rule
that silently dropped reports landing while an eval was in flight. Rows older than
`--harvest-max-age-seconds` (7200) are skipped as a stalled-trainer guard. **Any harvest miss
degrades to a full panel, never a thinner one.**

Harvestable anchors: `heuristic`, `heuristic11`, `heuristic12`, `attacker`, `random`. The frozen
self is **not** harvestable (league past-selves are ring members, not the eval's snapshot).

### 3.5 The tick row — `ticks.json → ticks[]`

Per iteration, flushed at most every 60s, ring-capped at 2000:
`it`, `wall_time`, `T`, `games`, `policy_loss`, `critic_loss`, `entropy`, `approx_kl`,
`guesser_loss`, `public_loss`, `collect_s`, `update_s`, `cpu`, `ram`, `gpu`.

`cpu`/`ram`/`gpu` are percentages sampled by a daemon thread every 5s; **`null` where unavailable**
(the ODROID has no GPU). Ticks are re-seeded from disk on resume so the ring is continuous.

### 3.6 The checkpoint payload

`format: 1`. Carries everything needed to resume **seamlessly** — a crash loses at most the
in-flight iteration and never re-runs warmup:

- `config` — the architecture record: `seed`, `encoders{actor,critic,guesser,public}`,
  `use_belief`, `hidden`, `actor_hidden`, `critic_hidden`, `card_dim`
- `done` (iteration), `elapsed` (cumulative seconds), `frozen_it`, `warmup_done`
- `models` + `frozen` — four state_dicts each
- `optim` — all three optimizers (`ppo`, `g`, `p`)
- `rng` — torch/numpy (+ CUDA). Restored via `.cpu()` because a `--gpu` resume maps everything to
  CUDA and torch demands CPU ByteTensors — a real bug that was fixed (`2b87d37`)
- `league` / `scen_league` — member EMAs, game counts, and the past-self ring weights

**Not** in the checkpoint: `wr_train`, `opp_mix`, `gwin` — all window-scoped and durable only via
`stats.json`. An encoder mismatch on resume is a hard `ValueError`.

### 3.7 Log lines (a parsed contract)

The journald/stdout lines are **parsed** by `backfill_stats` and the website collector, so their
format is a contract, not cosmetics:

- `[status …]` — the per-report human line; the float class in the parser is `[-\d.naif]+`
  specifically so it matches `nan` **and** `inf`. A float class that can't match them makes the
  **whole line** unparseable and drops the entire row, not just one field.
- `[league it=] games= trained= (self= past= heuristic= attacker= random=) h11= h12=` — the paren
  block is a **strict 5-token sequence** for parsers, so `heuristic=` there is the **merged**
  v1.0+v1.1+v1.2 count; the split rides *after* the paren.
- `[scenario …]`, `[eval …]`, `[warmup]`, `[resume]`, `[stop]`, `[archive]`, `[pcollect]`.

`fishrl.eval.backfill_stats` recovers history from journald into the exact live schemas
(idempotent; live rows win). Dedupe keys: reports on `it`; **evals on `(it, wall_time)`** — the
eval service can legitimately re-score the same iteration later, and both points are wanted.

### 3.8 The dashboard (`fishrl.serve`)

Four panels: **win-rates** (5000-it running mean, faint raw dots), **system utilization** (250-tick
mean), **throughput**, **opponent mix** (stacked). SSE first for no gap, then a full range fetch,
`/api/summary` polled every 5s.

Endpoints: `/api/summary` (never redirected — this machine's state), `/api/reports|evals|ticks?since_it=N`
(responses carry `run` = ckpt-dir basename, since iteration counters reset across runs),
`/api/stream` (SSE), `/api/archives` + `/archives/<file>` (local only), `/api/action` (gated by
`--allow-actions`).

Plotted series: `heuristic`, `heuristic11`, `heuristic12`, `random`, `attacker`, `frozen`,
`seat_p1_wr`. **★ stars mark `new_best`, and only on the `heuristic` series.**

---

<a name="part-4"></a>
## Part 4 — Traps for telemetry consumers

Read this section before plotting anything.

1. **`random` and `attacker` are inflated before `02ac079`.** The `enforce_free_attack` rule
   forfeited *scripted opponents* that declined a free attack (§2.6). It applies to every row
   written before the next trainer restart after that commit. **`heuristic*` was never affected**
   (different driver). Expect `random`/`attacker` to **drop** to honest values at the restart —
   that is the fix landing, not a regression.
2. **`scenario_wr` has a step DOWN on 2026-07-10.** The scenario engine seat was upgraded from
   heuristic v1.0 to v1.2 (`a8f7151`). Stronger opponent, not an agent regression
   (`scenarios/base.py:34-37`).
3. **`heuristic=` means two different things.** Merged all-versions inside the `[league]` paren
   block; **v1.0 only** in `stats.json`'s `opp_heuristic` and in every WR field.
4. **Two denominators for the opponent mix.** `stats.json`'s `opp_*` uses the grand total
   (including scenarios); the human `[status]`/`[league]` log lines stay league-normalized.
5. **Win-rate denominators exceed `n`.** See the harvest (§3.4) — use `anchor_n`, not `n`, to know
   the real sample size.
6. **`scenario_wr` is not a skill metric.** It is a PFSP difficulty EMA. Judge progress on the
   full-game vs-heuristic eval only.
7. **`pipeline` is sparse.** Absent means strictly on-policy; don't read absence as `false` from a
   schema that never had the key.
8. **`elapsed_h` is cumulative across restarts**, and `it` **resets across runs** — segment by
   `run` (ckpt-dir basename), not by `it`. Merged relay histories also interleave `host`.
9. **Persisted but never plotted**: `seat_p2_wr`, `play_wr`, `draw_wr`, `choose_first_frac`,
   `seat_diag_n`. Available to any consumer that wants them.
10. **Computed but never persisted**: `first_player_p1_frac`, `n_games` from `seat_diag_rates`.
11. **Seat diagnostics are off by default** (`--seat-diag-games 0`); the PC launcher passes 48.
12. **`pub_*` and `brier_gap` vanish** under `--train-public 0`.
13. **`gpu` is `null` on the ODROID.** Absent series ≠ zero.

---

<a name="part-5"></a>
## Part 5 — The current run

> **v3 addendum (2026-08-17).** The run described below (`checkpoints-v2`) is now the
> PRESERVED previous lineage; the active run is **`checkpoints-v3`**, restarted after two
> independent audits ("The Dandân Audit", "The Guesser Deposition") measured both
> privileged-information consumers as no longer paying for themselves. What changed:
>
> - **`belief_mode="bookkeeper"`** — the HandGuesser is GONE (no net). The actor's 20-dim
>   belief slot carries the analytic hand bookkeeper (`features.bookkeeper_counts`): known
>   opponent cards + unseen-pool proportions × public hand size. Measured: the guesser's
>   learned edge over this was 0.026 nats (11% of headroom), decaying to ~0.005 by the
>   endgame, while its forward cost ~32% of collection.
> - **`critic_view="public"`** — the PPO critic (`PublicCritic`) reads the PUBLIC encoding;
>   the god encode is off the hot path and the separate PublicEstimator no longer exists.
>   §1.5's asymmetric-critic rationale and §3.2's `brier_gap` therefore describe the
>   pre-v3 stack; measured 2026-08-17, the privileged head's Brier edge had depreciated
>   to ~0 (0.202 vs 0.197 on fresh mirror play).
> - **`critic_deckout_aux`** — the public critic carries a deckout-winner auxiliary head
>   trained on empty-library-draw-ended games (the parity-credit lever; §2.4's PBRS
>   proposal re-scoped: the 2026-08-17 ablation showed the ACTOR now uses the clock —
>   deckout-scenario wr 0.380 with it / 0.164 zeroed / 0.080 inverted — while the critic
>   still underweighted parity).
> - **`text_change_mode="guided"`** — the 5×5 `choose_text_change` block is masked down to
>   {EFFECT, NO-OP} (`spaces.masking`): the type written on the targeted card → a type
>   absent from its controller's permanents, plus one provably-inert pair. `auto` masks to
>   {EFFECT} alone (played by the forced-decision fast path). `A.N` is unchanged.
> - **Warm start** — the v3 actor initializes from `checkpoints-v2/best.pt` via
>   `fishrl.train.bootstrap_v3` (shapes identical; the belief slot keeps its width), with
>   a freeze-actor phase + KL-to-teacher anneal + lowered `--ent-start` in `train.args`.
> - **Telemetry**: v3 report rows carry `critic_acc`/`critic_brier`/`deckout_aux_loss`
>   instead of `guesser_loss`/`public_loss`/`priv_*`/`pub_*`/`brier_gap`/`gmae`; the
>   `[status]` line prints `calib critic(acc=…,brier=…) aux=…`; `backfill_stats` parses
>   both eras. Historical NOTE: `gmae` was never anchored (an all-zeros predictor scores
>   0.203 vs the guesser's 0.207) and `choose_first_frac` was pinned at 0.0 by a bug
>   until 2026-08-17 (`fc3d9eb`).
>
> Legacy checkpoints (v1/v2/bc) load everywhere via `config_from_checkpoint`'s
> era-defaulting; `critic_watch`/`ab_encoder`/`guesser_eval`/`belief_sensitivity`/
> `actor_headtohead`/`probe_knowledge` are legacy-only tools and refuse v3 payloads with
> a clear error.

`checkpoints-v2`, launched from `deploy/train.args`:

```
--encoder flat --critic-encoder entity --actor-encoder entity
--actor-hidden 768,768,384 --card-dim 128
--scenario-pool --pool-frac 0.6 --warmup-games 16
--ent-end 0.008 --ent-anneal-iters 250000
--scenario-weight board_presence=0.5 --scenario-weight deckout=0.5
```

Live architecture (read back from `latest.pt`):

| Net | Params | Encoder | `enc_dim` |
|---|---|---|---|
| actor | **2,677,021** | entity (d=128) | 2141 |
| critic | 1,494,785 | entity (d=128) | 2080 |
| guesser | 1,740,564 | flat | — |
| public | 1,720,065 | flat | — |

The two scenario weights at 0.5 are pointed **directly at the known weakness**: `board_presence`
and `deckout` are the two deckout-race scenarios, and the deckout race is the gap (§2.4).

Known-open items, none approved:
- **Deckout potential-based shaping** — the designed lever on the proven-ignored clock (§2.4).
- **Tier-3 throughput** (`--games-per-iter 24-32`, `--minibatch 1024`) — flags exposed, needs a
  settled-policy A/B.
- **Cyclical entropy re-heat** — implemented, off. Enable if the run stalls
  (`--ent-reheat-period 150000 --ent-reheat-peak 0.015`).
- Belief on/off win-rate ablation; an `it=0` eval to capture the untrained baseline.

---

## Appendix — Known code-level traps

Surfaced during the audit behind this document; recorded so they are not rediscovered.

- **`_pspecs` duplicates the serial sampling logic.** `train_loop.py:600-642` and `711-793` are two
  implementations of the same seat/seed/member-kind dispatch, kept in lockstep **by comment only**,
  with no test asserting equivalence. This is the highest-risk drift surface in the loop.
- **`max_seconds` silently bypasses `ent_coef`** (`train_loop.py:807-811`) and anneals on
  wall-clock instead. A fixed-wall-clock A/B of `--ent-reheat-period` would therefore be a no-op.
- **`ent_start` has no CLI flag** (only `--ent-end`, `--ent-anneal-iters`, `--ent-reheat-period`,
  `--ent-reheat-peak`).
- **`FishAEC` defaults `max_decisions=4000`; `Config` uses 2000.** Every training path passes the
  cfg value explicitly, so this only bites a direct `FishAEC()` construction.
- **`ent_reheat_peak` defaults to exactly `ent_start`** (both 0.02), so a re-heat replays the full
  original range. Probably intentional, previously undocumented.
