"""Self-play integration: games terminate, rewards are zero-sum, masks are sound."""
import numpy as np

from fishrl.forgetful_fish.state import GameState
from fishrl.env.apply import apply_atomic
from fishrl.selfplay.pettingzoo_api import env as make_env, raw_env
from fishrl.spaces import action_space as A
from fishrl.spaces.compound import COMPOUND_TYPES
from fishrl.spaces.masking import atomic_mask


def _clone(g):
    return GameState.from_dict(g.to_dict())


def test_random_selfplay_terminates_zero_sum():
    rng = np.random.default_rng(0)
    for seed in range(6):
        e = raw_env()
        e.reset(seed=seed)
        steps = 0
        while e.agents:
            a = e.agent_selection
            if e.terminations[a] or e.truncations[a]:
                e.step(None)
                continue
            mask = e.observe(a)["action_mask"]
            legal = np.flatnonzero(mask)
            assert legal.size > 0, f"empty mask at {e.g.pending.type}"
            e.step(int(rng.choice(legal)))
            steps += 1
            assert steps < 8000, "game failed to terminate"
        assert e.g.result["status"] != "ongoing"


def test_rewards_delivered_via_api():
    rng = np.random.default_rng(7)
    e = make_env()
    e.reset(seed=1)
    totals = {"p1": 0.0, "p2": 0.0}
    for agent in e.agent_iter(max_iter=20000):
        obs, rew, term, trunc, _ = e.last()
        totals[agent] += rew
        if term or trunc:
            e.step(None)
        else:
            e.step(int(rng.choice(np.flatnonzero(obs["action_mask"]))))
    assert totals["p1"] + totals["p2"] == 0.0
    assert {totals["p1"], totals["p2"]} == {1.0, -1.0}


def test_atomic_mask_is_sound():
    """Every unmasked atomic action is accepted by the engine (tested on clones)."""
    rng = np.random.default_rng(3)
    checked = 0
    for seed in range(2):
        e = raw_env()
        e.reset(seed=seed)
        steps = 0
        while e.agents and steps < 600:
            a = e.agent_selection
            if e.terminations[a] or e.truncations[a]:
                e.step(None)
                continue
            pend = e.g.pending
            if e._builder is None and pend.type not in COMPOUND_TYPES:
                mask = atomic_mask(e.g, a)
                for act in np.flatnonzero(mask):
                    g2 = _clone(e.g)
                    assert apply_atomic(g2, a, int(act)), \
                        f"{A.decode(int(act))} rejected for {pend.type}"
                    checked += 1
            mask = e.observe(a)["action_mask"]
            e.step(int(rng.choice(np.flatnonzero(mask))))
            steps += 1
    assert checked > 100       # exercised a meaningful number of (state, action) pairs


def test_empty_library_compound_auto_resolves():
    """A scry/reorder on an empty library (deck-out) has zero legal sub-actions; the
    env must auto-resolve it rather than hand the agent an empty mask (which would
    make the masked softmax uniform and let an illegal action be sampled)."""
    from fishrl.forgetful_fish import engine as E
    for start in (lambda g, s: E.start_reorder(g, "p1", s, 3),
                  lambda g, s: E.start_scry(g, "p1", s, 3, resolving_kind="ability")):
        e = raw_env()
        e.reset(seed=1)
        g = e.g
        g.active_player, g.current_step = "p1", "main1"
        g.library = []                              # deck-out
        start(g, next(iter(g.objects)))
        assert g.pending.type in ("reorder", "scry")
        e._refresh()                                # must auto-finalize, not stall
        assert e._builder is None
        assert g.pending is None or g.pending.type not in ("reorder", "scry")


def test_spell_draw_from_empty_library_decks_via_sba():
    """A spell/ability draw (NOT just the turn-based draw step) into an empty library
    must lose the game at the next state-based-action check (CR 704.5c). Regression for
    the bug where only the draw step enforced decking, so spell/ability draws into an
    empty library silently drew nothing."""
    from fishrl.forgetful_fish import engine as E
    from fishrl.forgetful_fish.state import draw_card, draw_cards
    for drawer in (lambda g: draw_card(g, "p1"), lambda g: draw_cards(g, "p1", 3)):
        e = raw_env()
        e.reset(seed=1)
        g = e.g
        g.library = []                              # deck-out
        assert drawer(g) in (None, [])              # the draw returns nothing...
        assert g.players["p1"].drew_from_empty      # ...but flags the player
        assert not g.players["p1"].has_lost         # not lost until the next SBA
        E._check_sba(g)
        assert g.players["p1"].has_lost
        assert g.result["winner"] == "p2"
        assert "empty library" in g.result["reason"]


def test_draw_step_from_empty_library_still_decks():
    """The turn-based draw step still decks an empty-library player -- guards the change
    that routes the loss through the SBA instead of an inline check in the draw step."""
    from fishrl.forgetful_fish import engine as E
    e = raw_env()
    e.reset(seed=1)
    g = e.g
    g.library = []
    g.turn_number = 3                               # past the first-turn draw skip
    g.active_player = "p1"
    E._enter_step(g, "draw")
    assert g.players["p1"].has_lost
    assert g.result["winner"] == "p2"


def test_compound_types_are_exercised():
    """Compound builders actually run (and finalize) in real games."""
    rng = np.random.default_rng(5)
    seen = set()
    for seed in range(8):
        e = raw_env()
        e.reset(seed=seed)
        steps = 0
        while e.agents and steps < 4000:
            a = e.agent_selection
            if e.terminations[a] or e.truncations[a]:
                e.step(None)
                continue
            if e.g.pending.type in COMPOUND_TYPES:
                seen.add(e.g.pending.type)
            e.step(int(rng.choice(np.flatnonzero(e.observe(a)["action_mask"]))))
            steps += 1
    assert seen, "no compound decisions encountered across 20 games"
