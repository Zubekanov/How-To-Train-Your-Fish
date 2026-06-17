"""Compound-decision builders produce engine-accepted, correctly-ordered results."""
from fishrl.forgetful_fish import engine as E
from fishrl.forgetful_fish import state as S
from fishrl.forgetful_fish.cards import load_decklist
from fishrl.spaces import action_space as A
from fishrl.spaces.compound import CompoundBuilder


def _fresh():
    g = E.new_multiplayer_game(load_decklist(), p1_name="p1", p2_name="p2", seed=1)
    # Place the game in p1's main phase so a resolved decision can hand back
    # priority (new_multiplayer_game leaves the pregame active_player empty).
    g.active_player = "p1"
    g.current_step = "main1"
    g.pending = None
    return g


def test_scry_orders_top_and_bottom():
    g = _fresh()
    E.start_scry(g, "p1", None, 3, resolving_kind="ability")
    ids = [c["instance_id"] for c in g.pending.context["cards"]]
    b = CompoundBuilder(g, "p1", "scry", g.pending.context)
    assert not b.feed(g, A.aid("PICK_B", 0))     # card0 -> bottom
    assert not b.feed(g, A.aid("PICK_A", 1))     # card1 -> top
    assert b.feed(g, A.aid("PICK_A", 2))         # card2 -> top, finalizes
    assert g.pending.type != "scry"              # resolution finished
    lib = [s.instance_id for s in g.library]
    assert lib[0] == ids[1] and lib[1] == ids[2]   # top order = pick order
    assert lib[-1] == ids[0]                        # bottomed card is last


def test_putback_puts_first_pick_on_top():
    g = _fresh()
    drawn = S.draw_cards(g, "p1", 3)
    E.start_putback(g, "p1", None, 2)
    b = CompoundBuilder(g, "p1", "putback", g.pending.context)
    hand = list(g.players["p1"].hand)
    # pick hand[2] then hand[0]; hand[2] should end up on top.
    assert not b.feed(g, A.aid("PICK_A", 2))
    assert b.feed(g, A.aid("PICK_A", 0))
    assert g.pending.type != "putback"           # resolution finished
    lib = [s.instance_id for s in g.library]
    assert lib[0] == hand[2] and lib[1] == hand[0]


def test_fof_split_partitions_revealed():
    g = _fresh()
    E.start_fof_split(g, "p1", _any_instance(g))   # caster p1 -> split belongs to p2
    ctx = g.pending.context
    revealed = list(ctx["revealed"])
    b = CompoundBuilder(g, "p2", "fof_split", ctx)
    # send first card to pile2, the rest to pile1
    done = False
    for i in range(len(revealed)):
        done = b.feed(g, A.aid("PICK_B" if i == 0 else "PICK_A", i))
    assert done
    # now the caster (p1) must choose a pile
    assert g.pending is not None and g.pending.type == "fof_choose" and g.pending.player == "p1"


def _any_instance(g):
    return next(iter(g.objects))
