"""Envelope-constructed scenarios (scenarios/constructed.py + envelope.py):
every one resets into a legal p1 decision, plays to the NATURAL result, keeps the
construction invariants (life in steps of 4, fish only with an Island, zone
conservation, exile = Undoings only), honours its overrides, and stays inside the
measured envelope's p10-p90 bands in aggregate. Pools are shrunk to keep it fast."""
from __future__ import annotations

import numpy as np
import pytest

from fishrl.forgetful_fish import engine as E
from fishrl.train.scenarios import ScenarioEnv
from fishrl.train.scenarios.constructed import (CONSTRUCTED, DeckoutShort, FishWar, RemovalInHand,
                                                ResponseWindow, ResponseWindowBend, UndoingCall)
from fishrl.train.scenarios.envelope import ENVELOPE, LIFE_LEVELS, REMOVAL, UNDOING
from fishrl.train.scenarios.surgery import is_island, is_land

_CACHE: dict = {}


def _small(cls):
    if cls not in _CACHE:
        scn = cls(); scn.pool_n = 6; scn.pool_max_games = 400
        _CACHE[cls] = scn
    return _CACHE[cls]


def _fish(g, s):
    return sum(1 for i in g.players[s].battlefield if E._is_creature(g.objects[i]))


def _islands(g, s):
    return sum(1 for i in g.players[s].battlefield if is_island(g.objects[i]))


def _names(g, ids):
    return [g.objects[i].name for i in ids]


def _rollout(env, seed, cap=3000):
    rng = np.random.default_rng(seed); d = 0
    while env.agents and d < cap:
        a = env.agent_selection
        if env.terminations[a] or env.truncations[a]:
            env.step(None); continue
        legal = np.flatnonzero(env.observe(a)["action_mask"])
        if legal.size == 0:
            env.step(None); continue
        env.step(int(rng.choice(legal))); d += 1
    return d


@pytest.mark.parametrize("cls", CONSTRUCTED, ids=[c.name for c in CONSTRUCTED])
def test_constructed_scenario_resets_legally_and_ends_naturally(cls):
    scn = _small(cls)
    env = ScenarioEnv(scn, max_decisions=2000)
    for seed in range(4):
        env.reset(seed=seed)
        g = env.g
        assert env.agent_selection == "p1"
        assert int(env.observe("p1")["action_mask"].sum()) > 0, "p1 must have a legal move"
        assert g.players["p2"].is_ai and g.players["p2"].ai_profile == "heuristic_1_3"
        assert scn.terminator(env) is None                       # natural result only
        for s in ("p1", "p2"):
            assert g.players[s].life in LIFE_LEVELS
            assert _fish(g, s) == 0 or _islands(g, s) >= 1
            assert _islands(g, s) >= 1
        total = (sum(len(g.players[s].hand) + len(g.players[s].battlefield) for s in g.players)
                 + len(g.library) + len(g.graveyard) + len(g.exile) + len(g.stack))
        assert total == len(g.objects), "zone conservation"
        assert all(g.objects[i].name == UNDOING for i in g.exile), "exile holds only Undoings"
        _rollout(env, seed)
        assert env.g.result.get("status") in ("p1_wins", "p2_wins"), "must reach the engine's own result"
        assert env.winner == env.g.result.get("winner")


def test_overrides_are_honoured():
    env = ScenarioEnv(_small(RemovalInHand), max_decisions=2000)
    for seed in range(6):
        env.reset(seed=seed); g = env.g
        assert any(n in REMOVAL for n in _names(g, g.players["p1"].hand))
        assert 1 <= _fish(g, "p2") <= 2
        assert g.active_player == "p1" and not g.stack
    env = ScenarioEnv(_small(UndoingCall), max_decisions=2000)
    for seed in range(6):
        env.reset(seed=seed); g = env.g
        assert UNDOING in _names(g, g.players["p1"].hand)
        assert 2 <= len(g.library) <= 10
    env = ScenarioEnv(_small(DeckoutShort), max_decisions=2000)
    for seed in range(6):
        env.reset(seed=seed)
        assert 2 <= len(env.g.library) <= 12


def test_response_window_starts_with_p2_spell_on_the_stack():
    for cls in (ResponseWindow, ResponseWindowBend):
        env = ScenarioEnv(_small(cls), max_decisions=2000)
        for seed in range(6):
            env.reset(seed=seed); g = env.g
            assert g.active_player == "p2" and g.priority_player == "p1"
            assert len(g.stack) == 1 and g.stack[0].controller == "p2"
            spell = g.objects[g.stack[0].source_instance_id]
            assert spell.name == g.stack[0].description
            untapped = sum(1 for i in g.players["p1"].battlefield
                           if is_land(g.objects[i]) and not g.objects[i].tapped)
            assert untapped >= 2
            if cls is ResponseWindowBend:
                assert spell.name == "Mind Bend"
                assert "Memory Lapse" in _names(g, g.players["p1"].hand)
                assert g.stack[0].targets and g.stack[0].targets[0]["id"] in g.players["p1"].battlefield
            # p1 passing resolves the spell (p2 already passed) -- the engine moves on
            E.pass_priority(g, "p1")
            assert not g.stack or g.result.get("status") != "ongoing" or g.pending is not None


def test_reset_is_deterministic():
    scn = _small(FishWar)
    a = ScenarioEnv(scn, max_decisions=800); a.reset(seed=7)
    b = ScenarioEnv(scn, max_decisions=800); b.reset(seed=7)
    for s in ("p1", "p2"):
        assert a.g.players[s].battlefield == b.g.players[s].battlefield
        assert a.g.players[s].hand == b.g.players[s].hand
        assert a.g.players[s].life == b.g.players[s].life
    assert [x.instance_id for x in a.g.library] == [x.instance_id for x in b.g.library]
    assert a.g.turn_number == b.g.turn_number


def test_samples_stay_inside_the_envelope_bands():
    """Aggregate plausibility audit: over many samples the medians of lands, hand,
    library and graveyard sit inside each bucket's p10-p90 band, life follows the
    20-4k ladder with 20 the mode, and fish counts stay in the 0/1/2 range."""
    scn = _small(FishWar)
    env = ScenarioEnv(scn, max_decisions=800)
    per = {}
    for seed in range(120):
        env.reset(seed=seed)
        per.setdefault(scn.last["bucket"], []).append(scn.last)
    for bucket, ps in per.items():
        band = ENVELOPE[bucket]
        lands = float(np.median([v for x in ps for v in x["lands"].values()]))
        hand = float(np.median([v for x in ps for v in x["hand"].values()]))
        lib = float(np.median([x["library"] for x in ps]))
        gy = float(np.median([x["graveyard"] for x in ps]))
        assert band["lands"][0] <= lands <= band["lands"][1], (bucket, lands)
        assert band["hand"][0] <= hand <= band["hand"][1], (bucket, hand)
        assert band["lib"][0] <= lib <= band["lib"][1], (bucket, lib)
        assert band["gy"][0] <= gy <= band["gy"][1], (bucket, gy)
        lives = [v for x in ps for v in x["life"].values()]
        assert set(lives) <= set(LIFE_LEVELS)
        assert np.mean([v == 20 for v in lives]) >= 0.4
        assert max(v for x in ps for v in x["fish"].values()) <= 2


def test_fof_split_starts_with_fof_on_the_stack_and_reaches_the_split():
    from fishrl.train.scenarios.constructed import FofSplit
    env = ScenarioEnv(_small(FofSplit), max_decisions=2000)
    for seed in range(6):
        env.reset(seed=seed); g = env.g
        assert g.active_player == "p2" and g.priority_player == "p1"
        assert len(g.stack) == 1 and g.stack[0].controller == "p2"
        assert g.objects[g.stack[0].source_instance_id].name == "Fact or Fiction"
        assert len(g.library) >= 5, "FoF must have a full top five to reveal"
        # p1 passing resolves it (p2 already passed): the SPLIT lands on p1, and the
        # revealed five are literally the top of the constructed library
        top5 = [c.instance_id for c in g.library[:5]]
        E.pass_priority(g, "p1")
        assert g.pending is not None and g.pending.type == "fof_split"
        assert g.pending.player == "p1"
        assert sorted(g.pending.context["revealed"]) == sorted(top5)


def test_fof_pick_holds_the_spell_with_mana_up():
    from fishrl.train.scenarios.constructed import FofPick
    env = ScenarioEnv(_small(FofPick), max_decisions=2000)
    for seed in range(6):
        env.reset(seed=seed); g = env.g
        assert "Fact or Fiction" in _names(g, g.players["p1"].hand)
        untapped = sum(1 for i in g.players["p1"].battlefield
                       if is_land(g.objects[i]) and not g.objects[i].tapped)
        assert untapped >= 4, "must be able to actually cast it"
        assert not g.stack


def test_hold_the_answer_has_the_instant_and_no_target_yet():
    from fishrl.train.scenarios.constructed import HoldTheAnswer
    env = ScenarioEnv(_small(HoldTheAnswer), max_decisions=2000)
    for seed in range(6):
        env.reset(seed=seed); g = env.g
        assert any(n in ("Vision Charm", "Crystal Spray", "Mind Bend")
                   for n in _names(g, g.players["p1"].hand))
        assert _fish(g, "p2") == 0, "the answer must have no fish target yet"
        untapped = sum(1 for i in g.players["p1"].battlefield
                       if is_land(g.objects[i]) and not g.objects[i].tapped)
        assert untapped >= 2, "the sorcery-speed temptation must be live"
        assert g.active_player == "p1" and not g.stack


def test_deckout_stack_is_a_small_library_with_a_draw_spell():
    from fishrl.train.scenarios.constructed import DeckoutStack
    env = ScenarioEnv(_small(DeckoutStack), max_decisions=2000)
    for seed in range(6):
        env.reset(seed=seed); g = env.g
        assert 4 <= len(g.library) <= 14
        assert any(n in ("Accumulated Knowledge", "Brainstorm", "Predict",
                         "Fact or Fiction", "Crystal Spray")
                   for n in _names(g, g.players["p1"].hand))
        untapped = sum(1 for i in g.players["p1"].battlefield
                       if is_land(g.objects[i]) and not g.objects[i].tapped)
        assert untapped >= 2
        assert not g.stack


def test_opening_race_is_an_early_game():
    from fishrl.train.scenarios.constructed import OpeningRace
    scn = _small(OpeningRace)
    env = ScenarioEnv(scn, max_decisions=2000)
    fish_rel = []
    for seed in range(12):
        env.reset(seed=seed); g = env.g
        assert g.turn_number <= 6
        assert g.players["p1"].life in (20, 16) and g.players["p2"].life in (20, 16)
        band = ENVELOPE["1-5"]
        # zone conservation may clamp a couple below the band's low edge
        assert band["lib"][0] - 4 <= scn.last["library"] <= band["lib"][1]
        assert band["hand"][0] <= min(scn.last["hand"].values())
        fish_rel.append((_fish(g, "p1"), _fish(g, "p2")))
    assert any(a < b for a, b in fish_rel), "behind-on-fish starts must occur"
