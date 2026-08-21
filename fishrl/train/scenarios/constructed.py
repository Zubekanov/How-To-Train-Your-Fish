"""Envelope-constructed scenarios (2026-08-21) — the curriculum that replaced the
hand-constant manufactures (board_presence / deckout / known_threat* /
survive_lethal*), which are kept registered at weight 0 for telemetry continuity.

Each scenario is a skeleton predicate (which priority configuration to borrow from
the pool: p1's main phase, p2's main phase with p1 holding priority, or p2's main
with a p2 spell on the stack) + a turn bucket + a few `Overrides` on the shared
`envelope_sample`. Every one ends by the NATURAL game result — no proxy
terminators — and the learner always plays p1 against the v1.3 engine seat.

Which gap each one aims at (Field Guide / Critic's Ledger, 2026-08-21):
  fish_war          the redeploy-and-answer attrition war; first blood lost 63%
  response_window   passing on the opponent's turn with an instant + mana up (the
                    largest confirmed-blunder class); `_bend` = the Mind Bend the
                    agent lets resolve 10% of the time at -0.17 each
  protect_the_fish  the leak peak: one fish up, none opposing, ~16 life, then lost
  removal_in_hand   25% of removal aimed at lands (value-negative per the critic)
  deckout_short     the parity endgame, lost 5:1 vs v1.3 from <=12 cards
  deckout_with_fish both clocks live at once (45% of real deckouts have a fish up)
  lethal_on_board   survive_lethal with real boards and the real terminator
  steer_the_top     Mystical Tutor value-negative; put-back / split leaks
  undoing_call      Day's Undoing cast 50% more than v1.3, worse when it resolves
"""
from __future__ import annotations

from fishrl.train.scenarios.base import Scenario
from fishrl.train.scenarios.envelope import (DANDAN, INSTANTS, REMOVAL, TOP_MANIP, UNDOING,
                                             Overrides, envelope_sample)


def _p1_prio(env, actives, stack_len):
    g = env.g; p = g.pending
    if not (p is not None and p.player == "p1" and p.type == "priority"
            and g.active_player in actives and g.current_step in ("main1", "main2")
            and 2 <= g.turn_number <= 30):
        return False
    if stack_len == 0:
        return not g.stack
    return (len(g.stack) == 1 and g.stack[0].kind == "spell"
            and g.stack[0].controller == "p2")


def p1_main(env):       return _p1_prio(env, ("p1",), 0)
def p2_main(env):       return _p1_prio(env, ("p2",), 0)
def p2_main_stack(env): return _p1_prio(env, ("p2",), 1)
def any_main(env):      return _p1_prio(env, ("p1", "p2"), 0)


class Constructed(Scenario):
    """Base: predicate + buckets + overrides → envelope_sample. Subclasses set
    `skeleton`, `buckets` and implement `overrides(rng)`; `last` keeps the sampled
    parameters of the most recent manufacture (tests / plausibility audit)."""
    engine_seat = "p2"
    skeleton = staticmethod(p1_main)
    buckets: tuple = ("11-16",)
    last: dict | None = None

    def predicate(self, env) -> bool:
        return type(self).skeleton(env)

    def overrides(self, rng) -> Overrides:
        return Overrides()

    def _manufacture(self, g, rng) -> None:
        bucket = str(rng.choice(list(self.buckets)))
        self.last = envelope_sample(g, rng, bucket, self.overrides(rng))

    def terminator(self, env):
        return None                                    # natural result, always


class FishWar(Constructed):
    name = "fish_war"; pool_seed = 1101
    buckets = ("6-10", "11-16", "17-24")

    def overrides(self, rng):
        rel = str(rng.choice(["ahead", "behind", "level"]))
        return Overrides(total_fish_min=1, fish_relation=rel)


_STACK_MIX = {DANDAN: 0.35, "Crystal Spray": 0.10, "Mind Bend": 0.08, "Metamorphose": 0.07,
              "Accumulated Knowledge": 0.10, "Fact or Fiction": 0.07, "Brainstorm": 0.08,
              "Vision Charm": 0.15}


class ResponseWindow(Constructed):
    name = "response_window"; pool_seed = 1102
    skeleton = staticmethod(p2_main_stack)
    buckets = ("11-16", "17-24", "25+")

    def overrides(self, rng):
        names = list(_STACK_MIX); p = [_STACK_MIX[n] for n in names]
        spell = names[int(rng.choice(len(names), p=[x / sum(p) for x in p]))]
        p1_fish = {1: 0.8, 2: 0.2} if spell in ("Crystal Spray", "Mind Bend", "Metamorphose") else None
        return Overrides(stack_spell={"choices": {spell: 1.0}}, p1_fish=p1_fish,
                         p1_hand_require_one_of=INSTANTS, p1_untapped_min=2)


class ResponseWindowBend(ResponseWindow):
    name = "response_window_bend"; pool_seed = 1103

    def overrides(self, rng):
        return Overrides(stack_spell={"choices": {"Mind Bend": 1.0}}, p1_fish={1: 0.8, 2: 0.2},
                         p1_hand_require=("Memory Lapse",), p1_untapped_min=2)


class ProtectTheFish(Constructed):
    name = "protect_the_fish"; pool_seed = 1104
    skeleton = staticmethod(p2_main)
    buckets = ("11-16", "17-24")

    def overrides(self, rng):
        return Overrides(p1_fish={1: 0.8, 2: 0.2}, p2_fish=0, p1_life_min=12, p1_untapped_min=2)


class RemovalInHand(Constructed):
    name = "removal_in_hand"; pool_seed = 1105
    buckets = ("6-10", "11-16", "17-24")

    def overrides(self, rng):
        last_island = bool(rng.random() < 0.5)          # half: the last-Island line exists
        return Overrides(p1_hand_require_one_of=REMOVAL, p2_fish={1: 0.8, 2: 0.2},
                         p2_islands=1 if last_island else None)


class DeckoutShort(Constructed):
    name = "deckout_short"; pool_seed = 1106
    skeleton = staticmethod(any_main)
    buckets = ("lib12",)

    def overrides(self, rng):
        return Overrides(library=(2, 12))


class DeckoutWithFish(Constructed):
    name = "deckout_with_fish"; pool_seed = 1107
    skeleton = staticmethod(any_main)
    buckets = ("lib12",)

    def overrides(self, rng):
        return Overrides(library=(4, 14), total_fish_min=1)


class LethalOnBoard(Constructed):
    name = "lethal_on_board"; pool_seed = 1108
    buckets = ("11-16", "17-24", "25+")

    def overrides(self, rng):
        return Overrides(p2_fish={1: 0.6, 2: 0.3, 3: 0.1}, lethal_on_p1=True)


class SteerTheTop(Constructed):
    name = "steer_the_top"; pool_seed = 1109
    buckets = ("25+",)

    def overrides(self, rng):
        return Overrides(library=(8, 20), p1_hand_require_one_of=TOP_MANIP,
                         top_known_to_p1=bool(rng.random() < 0.5))


class UndoingCall(Constructed):
    name = "undoing_call"; pool_seed = 1110
    skeleton = staticmethod(any_main)
    buckets = ("lib12",)

    def overrides(self, rng):
        return Overrides(library=(2, 10), p1_hand_require=(UNDOING,))


CONSTRUCTED = (FishWar, ResponseWindow, ResponseWindowBend, ProtectTheFish, RemovalInHand,
               DeckoutShort, DeckoutWithFish, LethalOnBoard, SteerTheTop, UndoingCall)
