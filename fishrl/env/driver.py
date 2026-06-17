"""Engine-driving configuration and helpers for the environment.

The engine, run as a two-human game (`new_multiplayer_game`, both seats
`is_ai=False`), pauses at every decision a controlled seat must make by setting
``g.pending`` and returning. The environment treats each learning agent as such a
seat: ``g.pending.player`` is the sole authority on whose turn it is (turn order
is NOT strictly alternating — the roll winner opens, blocks/splits/triggers can
belong to the non-active player, mulligan order follows `first_player`).
"""
from __future__ import annotations

from fishrl.forgetful_fish import engine as E

# All turn steps, for the "full" stops profile.
ALL_STEPS = list(E._STEPS)

# Priority-stop profiles seeded at reset via engine.set_player_stops.
#   "default" — act at own main phases + when responding to an opponent's stack
#               object (the engine's _human_should_stop default). Small decision
#               space; the agent is only asked when it can do something meaningful.
#   "full"    — take priority at every step (ablations / instant-heavy curricula).
STOPS_MODES = {
    "default": {"mine": ["main1", "main2"], "theirs": []},
    "full": {"mine": list(ALL_STEPS), "theirs": list(ALL_STEPS)},
}


def is_terminal(g) -> bool:
    return g.result.get("status") != "ongoing"


def terminal_winner(g):
    """Winning seat ("p1"/"p2") or None (draw / unfinished)."""
    return g.result.get("winner")
