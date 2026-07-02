"""The vendored rules core loads its deck and plays full games unchanged."""
from fishrl.forgetful_fish.cards import load_decklist
from fishrl.opponents.heuristic import HeuristicMatch
from fishrl.opponents.random_masked import RandomMaskedPolicy


def test_decklist_loads():
    deck = load_decklist()
    assert len(deck) == 21
    assert sum(int(c.get("qty") or 1) for c in deck) == 80


def test_random_vs_heuristic_full_games_terminate():
    # A real (random-but-legal) p1 versus the vendored heuristic AI reaches a
    # terminal result every game — exercising the engine end-to-end via fishrl.
    pol = RandomMaskedPolicy(seed=2)
    for seed in range(5):
        m = HeuristicMatch()
        obs = m.reset(seed=seed)
        done = False
        guard = 0
        while not done and guard < 20000:
            guard += 1
            obs, _, done, info = m.step(pol.act(obs))
        assert m.g.result["status"] != "ongoing"


# ── rules parity with the upstream (website) engine ──────────────────────────
def _started(seed=0):
    from fishrl.forgetful_fish import engine as E
    g = E.new_sandbox_game(load_decklist(), seed=seed, game_id="g", ai_profile="passive")
    E.choose_play_order(g, "p1", "first")
    E.mulligan_decision(g, "p1", "keep")            # both keep -> human auto-stops at main1
    return g


def _to_main1(g):
    from fishrl.forgetful_fish import engine as E
    for _ in range(8):
        if g.current_step == "main1":
            break
        E.pass_priority(g, "p1")
    return g


def test_response_holds_the_phase_until_a_manual_pass():
    # Pulled into a phase with no stop by an opponent's spell: once the stack
    # clears, the phase must behave as if it had a stop (manual pass to leave)
    # instead of auto-advancing. (Upstream commit b9f5c42.)
    from fishrl.forgetful_fish import engine as E
    from fishrl.forgetful_fish import state as S
    g = _to_main1(_started())
    E.set_stop(g, "p1", "mine", "main1")                 # remove the main1 stop
    assert not E._has_stop(g, "p1")
    g.players["p1"].held_step = []                       # simulate never having stopped here
    g.stack.append(S.StackObject(stack_id="x", controller="p2"))
    E._give_priority(g, "p1")                            # pulled in to respond
    assert g.priority_player == "p1" and g.current_step == "main1"
    g.stack.pop()                                        # the object resolves / leaves the stack
    E._give_priority(g, "p1")
    # Previously this auto-passed out of the phase; the step is now held.
    assert g.priority_player == "p1" and g.current_step == "main1"
    assert g.pending.type == "priority"
    E.pass_priority(g, "p1")                             # the manual pass releases it
    assert g.current_step != "main1"                     # phase finally advances


def test_draw_from_empty_loses_immediately_mid_effect():
    # The upstream rule: ANY draw from an empty library loses on the spot and the
    # rest of the effect stops (previously: a flag -> loss at the next state-based
    # check, after the effect finished resolving).
    from fishrl.forgetful_fish import engine as E
    g = _to_main1(_started())
    g.library = g.library[:1]                            # one card left, need two
    lost = E._draw_or_lose(g, "p1", 2)
    assert lost is True
    assert g.players["p1"].has_lost
    assert g.result.get("status") != "ongoing" and g.result.get("winner") == "p2"
    # exactly the one available card was drawn before the loss registered
    assert len(g.library) == 0
