"""Scenario #3 — known-threat denial (manufactured; deny the opponent's card advantage).

The opponent is a minimal bot: 10 random untapped lands (every land in this mono-blue
deck taps for U), NO cards in hand, no creatures, driven by the engine heuristic — so
on its turn it draws the top card and (having nothing else) casts it. A card-advantage
spell sits on top of the shared library, so by default the bot draws it, resolves it,
and gains cards. The scenario starts on the AGENT's turn (also with 10 random untapped
lands; the unused lands stay in the library) and ends at the end of the bot's first
turn; the agent LOSES if the bot's hand ever exceeds two cards. The agent
must therefore either manipulate a dud to the top of the deck before the bot draws, or
counter the bot's spell (it is handed a Memory Lapse + manipulation, with the mana).

The "> 2 cards" test is card-agnostic: it fires for any genuine card-advantage line
(Day's Undoing draws 7, Fact or Fiction nets ≥3, ...) and not for a card-neutral one.

`KnownThreatRandomScenario` is the same denial test but deals the agent a RANDOM
7-card grip instead of the curated counter + manipulation — measuring whether it can
answer (or recognise it can't) the known threat with whatever it happens to hold.
"""
from __future__ import annotations

from fishrl.forgetful_fish import engine as E
from fishrl.forgetful_fish.state import LibrarySlot
from fishrl.train.scenarios.base import Scenario
from fishrl.train.scenarios.surgery import make_engine_heuristic, pool_all_zones

THREAT = "Day's Undoing"     # card-advantage spell: drawn + cast -> bot hand >> 2
TOOL = "Memory Lapse"        # the counter the agent is handed
MANIP = ("Brainstorm", "Ponder", "Predict")   # top-of-deck manipulation spells in the curated grip
# Mana base per seat: a RANDOM selection of lands from the deck (every land in this
# mono-blue deck taps for U), not Islands specifically; the unused lands stay in the
# library. Halimar Depths is one such land (it manipulates the top on ETB, but here
# it is just a mana source unless drawn and replayed).
P2_LANDS = 10
P1_LANDS = 10
P1_HAND = 7


class KnownThreatScenario(Scenario):
    name = "known_threat"
    pool_seed = 303
    RANDOM_HAND = False          # subclass flips this for the random-grip variant

    def predicate(self, env) -> bool:
        # Any clean p1 main-phase priority; the manufacture rebuilds both seats and
        # only needs the deck (pooled across zones) to contain the threat + a counter.
        g = env.g
        p = g.pending
        if not (p is not None and p.player == "p1" and p.type == "priority"
                and g.current_step in ("main1", "main2") and not g.stack):
            return False
        names = {g.objects[i].name for i in g.objects}
        return THREAT in names and TOOL in names

    def sample(self, rng):
        g = super().sample(rng)
        self._manufacture(g, rng)
        make_engine_heuristic(g, "p2")
        return g

    def _manufacture(self, g, rng) -> None:
        pool = pool_all_zones(g)
        lands, threat, tools, manip, other = [], None, [], [], []
        for iid in pool:
            o = g.objects.get(iid)
            if o is None:
                continue
            nm = o.name
            if "Land" in (o.type_line or ""):      # any land -> the random mana-base pool
                lands.append(iid)
            elif nm == THREAT and threat is None:
                threat = iid
            elif nm == TOOL:
                tools.append(iid)
            elif nm in MANIP:
                manip.append(iid)
            else:
                other.append(iid)
        rng.shuffle(lands); rng.shuffle(other); rng.shuffle(manip)

        def put_land(seat):
            if not lands:
                return
            iid = lands.pop()
            o = g.objects[iid]
            o.tapped = False; o.controller = seat; o.entered_this_turn = False
            g.players[seat].battlefield.append(iid)

        for _ in range(P2_LANDS):                  # the bot: 10 random untapped lands, empty hand
            put_land("p2")
        g.players["p2"].hand = []
        for _ in range(P1_LANDS):                  # the agent: ample mana to answer the threat
            put_land("p1")
        if self.RANDOM_HAND:
            # variant: a RANDOM 7-card grip (no curated answers) — does the agent
            # answer the known threat with whatever it happens to hold?
            leftover = lands + tools + manip + other
            rng.shuffle(leftover)
            hand = [leftover.pop() for _ in range(min(P1_HAND, len(leftover)))]
            rest = leftover
        else:
            hand = []                              # curated grip: a counter + manipulation + filler
            if tools:
                hand.append(tools.pop())
            if manip:
                hand.append(manip.pop())
            while len(hand) < P1_HAND and other:
                hand.append(other.pop())
            rest = lands + tools + manip + other     # leftover lands stay in the library
        for iid in hand:
            g.objects[iid].controller = "p1"
        g.players["p1"].hand = hand
        # library: the threat on TOP, the rest below
        rng.shuffle(rest)
        g.library = [LibrarySlot(instance_id=threat, known_by={"p1": True, "p2": False})]
        g.library += [LibrarySlot(instance_id=iid, known_by={"p1": False, "p2": False})
                      for iid in rest]
        g.graveyard = []

    def on_reset(self, env) -> None:
        env.scn_ctx["p2_seen"] = False

    def terminator(self, env):
        g = env.g
        c = env.scn_ctx
        if len(g.players["p2"].hand) > 2:          # bot gained card advantage -> loss
            return "p2"
        if g.active_player == "p2":
            c["p2_seen"] = True
        if c["p2_seen"] and g.active_player == "p1":  # bot's turn ended with hand <=2 -> denied
            return "p1"
        return None


class KnownThreatRandomScenario(KnownThreatScenario):
    """known_threat with a RANDOM 7-card grip instead of the curated counter +
    manipulation. Same threat-on-top denial test and terminator; the agent must
    answer the threat (or recognise it can't) with whatever a random hand holds."""
    name = "known_threat_random"
    pool_seed = 313
    RANDOM_HAND = True
