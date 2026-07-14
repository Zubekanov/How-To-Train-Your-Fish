"""The deckout clock: who is forced to draw from the SHARED empty library.

The library is shared and both seats draw one per turn, so the loser is decided by
(library_count mod 2) XOR (whose draw is next) -- and every EXTRA card drawn flips it. The
observation already carried the count, but only as len/80.0: an MLP given a direct supervised
label cannot recover parity from that float (0.508 vs a 0.514 base rate) and the 592h actor had
not learned it (trunk 0.525) -- see fishrl/eval/probe_deckout_clock.py. These features are a
RE-ENCODING of information already present, so the rule itself must be exactly right.
"""
import numpy as np

from fishrl.obs.encoder import OBS_DIM, deckout_clock, encode_observation


class _G:
    """Minimal stand-in: deckout_clock reads only these three attributes."""

    def __init__(self, n, active, step):
        self.library = [None] * n
        self.active_player = active
        self.current_step = step


def _loser_by_simulation(n, active, step):
    """Ground truth: actually alternate the draws and see who hits the empty library."""
    drawn_this_turn = step not in ("untap", "upkeep", "draw", "")
    nxt = ("p2" if active == "p1" else "p1") if drawn_this_turn else active
    left = n
    while True:
        if left == 0:
            return nxt                       # this seat must draw and cannot
        left -= 1
        nxt = "p2" if nxt == "p1" else "p1"  # the other seat draws next turn


def test_clock_matches_a_simulated_draw_schedule():
    for n in range(0, 25):
        for active in ("p1", "p2"):
            for step in ("main1", "main2", "upkeep", "draw", "combat"):
                truth = _loser_by_simulation(n, active, step)
                for viewer in ("p1", "p2"):
                    parity, nxt_is_me, i_deck = deckout_clock(_G(n, active, step), viewer)
                    assert parity == float(n % 2)
                    assert i_deck == float(truth == viewer), (n, active, step, viewer)


def test_an_extra_draw_flips_who_decks_out():
    """The whole strategic point: drawing a card flips the parity, so a seat losing the
    deckout race can flip it by drawing. Brainstorm/Predict are deckout-tempo, not just
    card advantage."""
    before = deckout_clock(_G(10, "p1", "main1"), "p1")[2]
    after = deckout_clock(_G(9, "p1", "main1"), "p1")[2]      # one extra card drawn
    assert before != after


def test_pre_draw_step_gives_the_next_draw_to_the_active_player():
    # in upkeep the active player has NOT drawn yet, so the next draw is theirs
    assert deckout_clock(_G(4, "p1", "upkeep"), "p1")[1] == 1.0
    # by main1 they have drawn, so the next draw belongs to the opponent
    assert deckout_clock(_G(4, "p1", "main1"), "p1")[1] == 0.0


def test_clock_is_zeroed_in_the_pregame():
    """Before the play-order roll there is no active player, so the clock is undefined. It must
    not fall back to anything viewer-dependent -- that made BOTH seats read 'I draw next'."""
    from fishrl.env.aec_env import FishAEC
    env = FishAEC(max_decisions=200)
    env.reset(seed=3)                        # lands on choose_play_order: active_player == ""
    assert env.g.active_player not in ("p1", "p2")
    assert deckout_clock(env.g, "p1") == (0.0, 0.0, 0.0)
    assert deckout_clock(env.g, "p2") == (0.0, 0.0, 0.0)


def test_clock_is_in_the_observation_and_is_seat_relative():
    """It must actually reach the actor, and be viewer-oriented: exactly one seat draws next
    and exactly one decks first, so those bits must DISAGREE between the seats."""
    from fishrl.env.aec_env import FishAEC
    env = FishAEC(max_decisions=400)
    env.reset(seed=3)
    rng = np.random.default_rng(0)
    while env.agents and env.g.active_player not in ("p1", "p2"):   # play past the pregame
        s = env.agent_selection
        m = env.observe(s)["action_mask"]
        env.step(int(rng.choice(np.flatnonzero(m))))
    g = env.g
    o1, o2 = encode_observation(g, "p1"), encode_observation(g, "p2")
    assert o1.shape == (OBS_DIM,)
    c1, c2 = o1[-3:], o2[-3:]
    assert c1[0] == c2[0] == float(len(g.library) % 2)   # parity is seat-agnostic
    assert c1[1] != c2[1]                                # exactly one seat draws next
    assert c1[2] != c2[2]                                # exactly one seat decks first
