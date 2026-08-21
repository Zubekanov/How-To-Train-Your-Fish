"""Graveyard overflow ordering (2026-08-22): at <= 32 cards the fill is unchanged; past
that, AKs first, one of each distinct instant/sorcery name, then the newest of the rest --
identically in every view (actor ref/fast, god, public, hands)."""
from __future__ import annotations

import numpy as np

from fishrl.data import features as F
from fishrl.data.features import encode_god, encode_god_ref, encode_hands, encode_public, encode_public_ref
from fishrl.forgetful_fish import engine as E
from fishrl.forgetful_fish.cards import load_decklist
from fishrl.obs import vocab as V
from fishrl.obs.encoder import CARD_F, SLOTS, encode_observation, encode_observation_ref, graveyard_order

N = SLOTS["graveyard"]


def _game(n_gy: int, seed=5):
    g = E.new_multiplayer_game(load_decklist(), p1_name="p1", p2_name="p2", seed=seed)
    E.choose_play_order(g, g.pending.player, "first")
    for _ in range(2):
        E.mulligan_decision(g, g.pending.player, "keep")
    # mill the top n_gy cards of the shared library into the graveyard, in draw order
    for _ in range(n_gy):
        g.graveyard.append(g.library.pop(0).instance_id)
    return g


def _gy_rows(x, offset_rows):
    rows = x[offset_rows * CARD_F:(offset_rows + N) * CARD_F].reshape(N, CARD_F)
    return [V.CARD_NAMES[int(np.argmax(r[:V.N_NAMES]))] if r[:V.N_NAMES].any() else None for r in rows]


def test_order_helper():
    items = [("Island", "Land"), ("Memory Lapse", "Instant"), ("Accumulated Knowledge", "Instant"),
             ("Memory Lapse", "Instant"), ("Dandân", "Creature — Fish"), ("Ponder", "Sorcery"),
             ("Accumulated Knowledge", "Instant"), ("Dandân", "Creature — Fish")]
    nm, tl = (lambda x: x[0]), (lambda x: x[1])
    assert graveyard_order(items, 8, nm, tl) == items                   # no overflow: untouched
    out = graveyard_order(items, 5, nm, tl)
    assert [x[0] for x in out] == ["Accumulated Knowledge", "Accumulated Knowledge",
                                   "Memory Lapse", "Ponder",           # one per distinct name
                                   "Dandân"]                            # newest of the rest
    assert graveyard_order(items, 2, nm, tl) == [items[2], items[6]]    # cap respected


def test_no_change_below_capacity():
    g = _game(N)
    names = [g.objects[i].name for i in g.graveyard]
    x = encode_observation(g, "p1")
    off = SLOTS["own_hand"] + SLOTS["opp_hand"] + SLOTS["own_bf"] + SLOTS["opp_bf"]
    assert _gy_rows(x, off) == names                                    # append order, all 32


def test_overflow_orders_by_value_in_every_view():
    g = _game(46)
    gy = [g.objects[i] for i in g.graveyard]
    exp = [o.name for o in graveyard_order(gy, N, lambda o: o.name, lambda o: o.type_line)]
    n_ak = sum(o.name == "Accumulated Knowledge" for o in gy)
    assert exp[:n_ak] == ["Accumulated Knowledge"] * n_ak
    assert len(exp) == N and set(exp) <= {o.name for o in gy}
    # the newest card of the remainder survives; the oldest non-priority card does not
    rest = [o for o in gy if o.name != "Accumulated Knowledge"]
    assert rest[-1].name in exp
    off_a = SLOTS["own_hand"] + SLOTS["opp_hand"] + SLOTS["own_bf"] + SLOTS["opp_bf"]
    for fn in (encode_observation, encode_observation_ref):
        assert _gy_rows(fn(g, "p1"), off_a) == exp
    F.set_public_encoding(True)
    off_c = 12 + 12 + 34 + 34
    for fn in (encode_god, encode_god_ref, encode_public, encode_public_ref, encode_hands):
        assert _gy_rows(fn(g), off_c) == exp, fn.__name__
    assert np.array_equal(encode_observation(g, "p1"), encode_observation_ref(g, "p1"))
    assert np.array_equal(encode_public(g), encode_public_ref(g))
