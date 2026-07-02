"""
Forgetful Fish game state — a serializable "god-state" that can be saved to and
loaded from the database, plus a per-player projection.

This is a state *container* only (no rules engine yet). The format's defining
feature is a shared, ordered, hidden library/graveyard/exile, so every physical
card is a stable instance referenced by id; zones are ordered lists of ids.

Continuous text-changing effects (Mind Bend / Crystal Spray) are intentionally
not modeled — in this format they are used as targeted removal for a Dandân and
are handled as actions rather than a layering engine.
"""
from __future__ import annotations

import dataclasses
import random
import re
import uuid
from dataclasses import dataclass, field, asdict

SCHEMA_VERSION = 1

PLAYERS = ("p1", "p2")

# Reference list of turn steps (state stores the current one as a plain string).
STEPS = (
    "untap", "upkeep", "draw", "main1", "begin_combat", "declare_attackers",
    "declare_blockers", "combat_damage", "main2", "end", "cleanup",
)


def _kw(cls, d: dict) -> dict:
    """Keep only keys that are fields of `cls` (forward/backward-compatible loads)."""
    names = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in (d or {}).items() if k in names}


# ---------------------------------------------------------------------------
# Card instances (one per physical card, wherever it currently is)
# ---------------------------------------------------------------------------

@dataclass
class CardInstance:
    instance_id: str
    card_key: str = ""            # which printing, e.g. "MIR/338"
    name: str = ""
    type_line: str = ""           # e.g. "Creature — Fish", "Instant", "Land"
    oracle_text: str = ""         # rules text (parsed for "enters tapped", etc.)
    mana_cost: str = ""           # e.g. "{2}{U}" (empty for lands)
    power: int = 0                # creatures; 0 for non-creatures
    toughness: int = 0
    owner: str | None = None      # None = communal (shared deck); set when relevant
    controller: str | None = None
    known_by: list = field(default_factory=list)   # pids who know this card's identity
    #                                                in a hidden zone (drawn from a known
    #                                                library slot, or seen in a public zone)
    # permanent flags
    tapped: bool = False
    entered_this_turn: bool = True   # summoning sickness
    damage_marked: int = 0
    counters: dict = field(default_factory=dict)
    attached_to: str | None = None
    attachments: list = field(default_factory=list)   # instance ids
    phased_out: bool = False
    face_down: bool = False
    is_token: bool = False
    chosen: dict = field(default_factory=dict)        # modes / named card / chosen colour, etc.
    # Text-changing effects (Crystal Spray / Mind Bend / Vision Charm). Each entry
    # {"frm","to","eot","turn"} rewrites a basic land type word. `type_line` and
    # `oracle_text` above hold the *effective* (already-rewritten) text; text_orig
    # keeps the pristine values so the change can revert. Changes apply only while
    # on the stack or battlefield; leaving those zones reverts (recompute_text).
    text_changes: list = field(default_factory=list)
    text_orig: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "CardInstance":
        return cls(**_kw(cls, d))


# ---------------------------------------------------------------------------
# Library slot (instance + per-player knowledge of this position)
# ---------------------------------------------------------------------------

@dataclass
class LibrarySlot:
    instance_id: str
    known_by: dict = field(default_factory=lambda: {"p1": False, "p2": False})

    @classmethod
    def from_dict(cls, d: dict) -> "LibrarySlot":
        return cls(**_kw(cls, d))


# ---------------------------------------------------------------------------
# Stack objects (spells AND abilities) and pending triggers
# ---------------------------------------------------------------------------

@dataclass
class StackObject:
    stack_id: str
    kind: str = "spell"                  # spell | triggered | activated
    source_instance_id: str | None = None
    controller: str | None = None
    targets: list = field(default_factory=list)   # [{"type":"object"/"player","id":...}]
    modes: list = field(default_factory=list)
    chosen: dict = field(default_factory=dict)
    x: int | None = None
    description: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "StackObject":
        return cls(**_kw(cls, d))


# ---------------------------------------------------------------------------
# Combat (attacker -> blockers + damage order). Source of truth for combat.
# ---------------------------------------------------------------------------

@dataclass
class Combat:
    # attacker_instance_id -> {"target": "p1"/"p2", "blockers": [ids], "damage_order": [ids]}
    attackers: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "Combat":
        return cls(**_kw(cls, d))


# ---------------------------------------------------------------------------
# Pending decision (awaiting a player's input — lets us save mid-action)
# ---------------------------------------------------------------------------

@dataclass
class PendingDecision:
    type: str            # choose_targets | scry | fact_or_fiction | discard_to_hand | ...
    player: str
    context: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "PendingDecision":
        return cls(**_kw(cls, d))


# ---------------------------------------------------------------------------
# Player state
# ---------------------------------------------------------------------------

@dataclass
class PlayerState:
    pid: str
    name: str = ""
    is_ai: bool = False
    # How an AI player plays: "heuristic" (plays lands/spells/combat via
    # fishrl.forgetful_fish.ai) or "passive" (auto-passes everything — the original
    # sandbox opponent, still used by engine tests that want a quiet opponent).
    ai_profile: str = "heuristic"
    life: int = 20
    mana_pool: dict = field(default_factory=dict)
    max_hand_size: int = 7
    land_played_this_turn: bool = False
    has_lost: bool = False
    loss_reason: str | None = None
    hand: list = field(default_factory=list)         # instance ids (ordered)
    battlefield: list = field(default_factory=list)  # instance ids
    tap_undo: list = field(default_factory=list)     # lands tapped since the last
    #                                                  other action (newest last)
    mulligans: int = 0                               # mulligans taken this game
    kept: bool = False                               # has kept their opening hand
    yield_mode: str = ""                             # "" | conditional | unconditional
    #                                                  (pass rest of turn auto-yield)
    # Manual-priority stops, keyed by turn context ("mine" = this player's turn,
    # "theirs" = the opponent's turn) -> list of step names. Default: own mains.
    stops: dict = field(default_factory=lambda: {"mine": ["main1", "main2"], "theirs": []})
    yield_here: dict = field(default_factory=dict)   # {"context","phase"} target, or {}
    # The last [turn_number, step] where this player was interactively held with
    # priority (a stop, or pulled in to respond to an opponent's spell). Once
    # held in a step, later priority grants in the SAME step keep stopping — so
    # responding in a phase with no stop doesn't auto-pass you out of the phase
    # the moment the stack clears. Self-expires when the turn/step moves on.
    held_step: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "PlayerState":
        return cls(**_kw(cls, d))


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    schema_version: int = SCHEMA_VERSION
    game_id: str = ""
    seed: int = 0
    rng_state: list | None = None          # serialized random.Random().getstate()
    turn_number: int = 0
    active_player: str = "p1"
    first_player: str = ""        # who takes the first turn (set in the opening)
    current_step: str = "untap"
    priority_player: str | None = None
    passed: dict = field(default_factory=lambda: {"p1": False, "p2": False})
    players: dict = field(default_factory=dict)        # pid -> PlayerState
    library: list = field(default_factory=list)        # LibrarySlot (index 0 = top)
    graveyard: list = field(default_factory=list)       # instance ids (index 0 = bottom)
    exile: list = field(default_factory=list)
    stack: list = field(default_factory=list)           # StackObject (last = top)
    objects: dict = field(default_factory=dict)         # instance_id -> CardInstance
    combat: Combat = field(default_factory=Combat)
    pending: "PendingDecision | None" = None
    pending_triggers: list = field(default_factory=list)  # StackObject awaiting ordering
    result: dict = field(default_factory=lambda: {"status": "ongoing", "winner": None, "reason": None})
    turns_since_stop: int = 0     # safety: force a human stop if auto-pass runs away
    log: list = field(default_factory=list)
    chat: list = field(default_factory=list)            # table talk: {"by": pid, "text": str}

    # -- serialization ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "game_id": self.game_id,
            "seed": self.seed,
            "rng_state": self.rng_state,
            "turn_number": self.turn_number,
            "active_player": self.active_player,
            "first_player": self.first_player,
            "current_step": self.current_step,
            "priority_player": self.priority_player,
            "passed": self.passed,
            "players": {pid: asdict(p) for pid, p in self.players.items()},
            "library": [asdict(s) for s in self.library],
            "graveyard": list(self.graveyard),
            "exile": list(self.exile),
            "stack": [asdict(s) for s in self.stack],
            "objects": {iid: asdict(o) for iid, o in self.objects.items()},
            "combat": asdict(self.combat),
            "pending": asdict(self.pending) if self.pending else None,
            "pending_triggers": [asdict(s) for s in self.pending_triggers],
            "result": self.result,
            "log": list(self.log),
            "chat": [dict(m) for m in self.chat],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GameState":
        g = cls()
        g.schema_version = d.get("schema_version", SCHEMA_VERSION)
        g.game_id = d.get("game_id", "")
        g.seed = d.get("seed", 0)
        g.rng_state = d.get("rng_state")
        g.turn_number = d.get("turn_number", 0)
        g.active_player = d.get("active_player", "p1")
        g.first_player = d.get("first_player", "")
        g.current_step = d.get("current_step", "untap")
        g.priority_player = d.get("priority_player")
        g.passed = d.get("passed", {"p1": False, "p2": False})
        g.players = {pid: PlayerState.from_dict(pd) for pid, pd in (d.get("players") or {}).items()}
        g.library = [LibrarySlot.from_dict(s) for s in (d.get("library") or [])]
        g.graveyard = list(d.get("graveyard") or [])
        g.exile = list(d.get("exile") or [])
        g.stack = [StackObject.from_dict(s) for s in (d.get("stack") or [])]
        g.objects = {iid: CardInstance.from_dict(o) for iid, o in (d.get("objects") or {}).items()}
        g.combat = Combat.from_dict(d.get("combat") or {})
        g.pending = PendingDecision.from_dict(d["pending"]) if d.get("pending") else None
        g.pending_triggers = [StackObject.from_dict(s) for s in (d.get("pending_triggers") or [])]
        g.result = d.get("result") or {"status": "ongoing", "winner": None, "reason": None}
        g.log = list(d.get("log") or [])
        g.chat = [dict(m) for m in (d.get("chat") or [])]
        return g

    @property
    def status(self) -> str:
        return "active" if self.result.get("status") == "ongoing" else "finished"


# ---------------------------------------------------------------------------
# RNG (de)serialization — store the full Random state for deterministic resume
# ---------------------------------------------------------------------------

def _serialize_rng(rng: random.Random) -> list:
    version, internal, gauss = rng.getstate()
    return [version, list(internal), gauss]


def rng_from_state(state: list | None) -> random.Random:
    rng = random.Random()
    if state:
        version, internal, gauss = state
        rng.setstate((version, tuple(internal), gauss))
    return rng


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def draw_card(state: "GameState", player: str, *, log: bool = True) -> str | None:
    """Move the top card of the shared library into `player`'s hand.

    Returns the moved instance id, or None if the library is empty. Drawing
    from an empty library loses the game, but that rule lives in the engine
    (which turns a None return into a loss for ANY draw — natural or from a
    spell/ability — via _draw_or_lose).
    `log=False` suppresses the per-draw log line (used by draw_cards, which logs
    the aggregate instead).
    """
    _seal_taps(state)
    if not state.library:
        state.log.append(f"{state.players[player].name} cannot draw from an empty library.")
        return None
    slot = state.library.pop(0)
    # whoever knew this top card now knows it in the drawer's hand
    state.objects[slot.instance_id].known_by = [pid for pid, k in slot.known_by.items() if k]
    state.players[player].hand.append(slot.instance_id)
    if log:
        state.log.append(f"{state.players[player].name} draws a card.")
    return slot.instance_id


def draw_cards(state: "GameState", player: str, n: int) -> list:
    """Draw `n` cards (each its own event, per the rules) but log them as one
    line: 'X draws a card.' / 'X draws N cards.'. Returns the drawn instance ids."""
    drawn = []
    for _ in range(max(n, 0)):
        iid = draw_card(state, player, log=False)
        if iid is None:
            break
        drawn.append(iid)
    if drawn:
        word = "a card" if len(drawn) == 1 else f"{len(drawn)} cards"
        state.log.append(f"{state.players[player].name} draws {word}.")
    return drawn


def _mark_seen(state: "GameState", instance_id: str) -> None:
    """A card in a public zone is seen by everyone; record it so the knowledge
    survives if the card later moves to a hidden zone (e.g. back to hand)."""
    o = state.objects.get(instance_id)
    if o:
        o.known_by = list(state.players.keys())


def mark_library_known(state: "GameState", instance_id: str, players) -> None:
    """Record that `players` know the identity of the library card `instance_id`.
    Use this wherever knowledge of the library is gained: a player who looks at
    the top and controls its order knows those cards (pass that one player); a
    card moved into the library from a public zone (graveyard, stack, …) is known
    to everyone (pass both players). Shuffling clears this again."""
    for slot in state.library:
        if slot.instance_id == instance_id:
            for pid in players:
                slot.known_by[pid] = True
            return


def _reconcile_played_name(state: "GameState", player: str, played: CardInstance) -> None:
    """When `player` plays a card, an opponent who knew a card of the same name
    in `player`'s hand (but not this exact card) attributes the play to it — so
    that known card's identity is consumed rather than leaving a phantom known
    card alongside a separate blank-face play."""
    prior = set(played.known_by)                     # who already knew this exact card
    for o in state.players:
        if o == player or o in prior:
            continue
        for aid in state.players[player].hand:       # the cards still in hand
            a = state.objects.get(aid)
            if a and a.name == played.name and o in (a.known_by or []):
                a.known_by.remove(o)
                break


def shuffle_library(state: "GameState") -> None:
    """Shuffle the shared library and clear all knowledge of it — positions are
    randomised and nobody knows the identity of any library card any more."""
    rng = rng_from_state(state.rng_state)
    rng.shuffle(state.library)
    for slot in state.library:
        slot.known_by = {pid: False for pid in state.players}
        o = state.objects.get(slot.instance_id)
        if o:
            o.known_by = []
    state.rng_state = _serialize_rng(rng)


def forget_rearranged(state: "GameState", slots, arranger: str) -> None:
    """`arranger` privately rearranged a section of the library — clear every
    *other* player's knowledge of those slots. Opponents can neither learn the
    new order (the arrangement was private) nor keep relying on the old position
    of any card they previously knew there, since it may have moved. The
    arranger's own knowledge is untouched. Use this at any site where a player
    reorders/keeps-or-bottoms a section of the library out of sight (reorder,
    scry). `slots` is an iterable of LibrarySlot (e.g. state.library[:n])."""
    for slot in slots:
        for pid in state.players:
            if pid != arranger:
                slot.known_by[pid] = False


_PERMANENT_TYPES = ("creature", "artifact", "enchantment", "land", "planeswalker", "battle")


def _is_land(type_line: str) -> bool:
    return "land" in (type_line or "").lower()


# ---------------------------------------------------------------------------
# Text-changing effects: rewrite one basic land type word with another, fixing
# the article (a/an) and the plural. Crystal Spray / Mind Bend / Vision Charm.
# ---------------------------------------------------------------------------

BASIC_TYPES = ["Plains", "Island", "Swamp", "Mountain", "Forest"]
_PLURAL = {"Plains": "Plains", "Island": "Islands", "Swamp": "Swamps",
           "Mountain": "Mountains", "Forest": "Forests"}


def _swap_basic_type(text: str, frm: str, to: str) -> str:
    """Replace every instance of basic land type `frm` with `to`, fixing 'a/an'
    and the plural. Leaves unrelated words (and the card name) untouched."""
    if not text:
        return text
    art = "an" if to[0].lower() in "aeiou" else "a"

    def repl_art(m):
        return (art.capitalize() if m.group(1)[0].isupper() else art) + " " + to
    text = re.sub(r"\b([Aa]n?) " + re.escape(frm) + r"\b", repl_art, text)
    text = re.sub(r"\b" + re.escape(_PLURAL[frm]) + r"\b", _PLURAL[to], text)
    text = re.sub(r"\b" + re.escape(frm) + r"\b", to, text)
    return text


def change_relevant(o: "CardInstance", frm: str) -> bool:
    """Whether rewriting `frm` would actually change this object's text/type."""
    hay = (o.text_orig.get("type_line", o.type_line) + " "
           + o.text_orig.get("oracle_text", o.oracle_text))
    return bool(re.search(r"\b" + re.escape(frm) + r"s?\b", hay))


def recompute_text(o: "CardInstance") -> None:
    """Re-derive the effective type_line/oracle_text from the pristine originals
    plus the active text_changes (in application order). With no changes, restore
    the originals and drop the saved copy."""
    if not o.text_changes:
        if o.text_orig:
            o.type_line = o.text_orig.get("type_line", o.type_line)
            o.oracle_text = o.text_orig.get("oracle_text", o.oracle_text)
            o.text_orig = {}
        return
    base_tl = o.text_orig.get("type_line", o.type_line)
    base_ot = o.text_orig.get("oracle_text", o.oracle_text)
    tl, ot = base_tl, base_ot
    for ch in o.text_changes:
        tl = _swap_basic_type(tl, ch["frm"], ch["to"])
        ot = _swap_basic_type(ot, ch["frm"], ch["to"])
    o.type_line, o.oracle_text = tl, ot


def add_text_change(o: "CardInstance", frm: str, to: str, *, eot: bool, turn: int) -> bool:
    """Apply a basic-land-type rewrite to `o`. No-op (returns False) when `frm`
    isn't present in the object's text/type."""
    if frm == to or not change_relevant(o, frm):
        return False
    if not o.text_orig:
        o.text_orig = {"type_line": o.type_line, "oracle_text": o.oracle_text}
    o.text_changes.append({"frm": frm, "to": to, "eot": bool(eot), "turn": turn})
    recompute_text(o)
    return True


def revert_text_changes(o: "CardInstance") -> None:
    """Drop all text changes (e.g. the card left the stack/battlefield)."""
    if o.text_changes or o.text_orig:
        o.text_changes = []
        recompute_text(o)


def expire_eot_text_changes(o: "CardInstance") -> None:
    """Drop 'until end of turn' changes; keep indefinite ones (Mind Bend)."""
    if any(c.get("eot") for c in o.text_changes):
        o.text_changes = [c for c in o.text_changes if not c.get("eot")]
        recompute_text(o)


def text_variant(o: "CardInstance") -> str | None:
    """Which rendered variant image to show: follow the card's original 'Island'
    through its active changes. Returns e.g. 'swamp', or None for the base art."""
    if not o.text_changes:
        return None
    cur = "Island"
    for ch in o.text_changes:
        if ch["frm"] == cur:
            cur = ch["to"]
    return cur.lower() if cur != "Island" and cur in BASIC_TYPES else None


def controls_basic_type(state: "GameState", player: str, typ: str) -> bool:
    """Whether `player` controls a permanent of basic land type `typ` (reading the
    effective, possibly text-changed, type line)."""
    pat = re.compile(r"\b" + re.escape(typ) + r"\b", re.I)
    return any(pat.search(state.objects[iid].type_line or "")
               for iid in state.players[player].battlefield if iid in state.objects)


_TYPE_COLOR = {"plains": "W", "island": "U", "swamp": "B", "mountain": "R", "forest": "G"}


def land_mana_color(o: "CardInstance") -> str:
    """The mana colour a land taps for: from its (effective) basic land type, or
    the colour in its printed '{T}: Add {X}' ability, defaulting to {U}."""
    tl = (o.type_line or "").lower()
    for typ, sym in _TYPE_COLOR.items():
        if re.search(r"\b" + typ + r"\b", tl):
            return sym
    m = re.search(r"add \{([wubrg])\}", (o.oracle_text or "").lower())
    return m.group(1).upper() if m else "U"


# ---------------------------------------------------------------------------
# Activated abilities of permanents on the battlefield. A card with more than
# one is offered as a click menu. `adds` > 0 is a mana ability (resolves at
# once, off the stack); `effect` names a non-mana effect that uses the stack.
# `cost` is generic mana paid from the pool (this format treats all mana as one
# colour). Cards with no entry keep the default "click a land to tap for {U}".
# ---------------------------------------------------------------------------

# `cost` is the ability's mana cost as a mana-cost STRING (parsed by engine._parse_cost,
# like a spell's mana_cost), so coloured pips are preserved — {1}{U} is one generic AND
# one blue, NOT two generic. "" means no mana cost (the {T} part is paid by tapping).
PERMANENT_ABILITIES = {
    "Svyelunite Temple": [
        {"text": "{T}: Add {U}.", "tap": True, "sac": False, "cost": "", "adds": 1, "effect": None},
        {"text": "{T}, Sacrifice: Add {U}{U}.", "tap": True, "sac": True, "cost": "", "adds": 2, "effect": None},
    ],
    "The Surgical Bay": [
        {"text": "{T}: Add {U}.", "tap": True, "sac": False, "cost": "", "adds": 1, "effect": None},
        {"text": "{1}{U}, {T}, Sacrifice: Draw a card.", "tap": True, "sac": True, "cost": "{1}{U}", "adds": 0, "effect": "draw_1"},
    ],
}


def _ability_view(o: "CardInstance") -> list:
    """Per-permanent abilities for the click menu: index, text, whether it is a
    mana ability, and a base availability (a tap ability needs it untapped).
    current_view refines `available` with the timing context for the viewer."""
    out = []
    for i, a in enumerate(PERMANENT_ABILITIES.get(o.name, [])):
        out.append({"index": i, "text": a["text"], "mana": bool(a["adds"]),
                    "available": not (a["tap"] and o.tapped)})
    return out


# Cards with cycling (CR 702.29): name -> number of blue ({U}) pips in the
# cycling cost. Cycling is an activated ability from the hand — pay the cost and
# discard the card to draw one.
CYCLING = {"Lonely Sandbar": 1}


def _available_mana(state: "GameState", player: str) -> int:
    """Mana the player could spend right now: their floating pool plus one for
    each untapped land they control."""
    pool = sum(state.players[player].mana_pool.values())
    lands = sum(1 for iid in state.players[player].battlefield
                if _is_land(state.objects[iid].type_line) and not state.objects[iid].tapped)
    return pool + lands


def _can_play_now(state: "GameState", viewer: str, card: "CardInstance") -> bool:
    """Whether `viewer` may play this hand card right now (timing only)."""
    p = state.pending
    if not (p and p.type == "priority" and p.player == viewer):
        return False
    tl = (card.type_line or "").lower()
    is_land = "land" in tl
    if is_land and state.players[viewer].land_played_this_turn:
        return False
    sorcery_speed = is_land or "instant" not in tl
    if sorcery_speed and (state.active_player != viewer
                          or state.current_step not in ("main1", "main2")
                          or state.stack):
        return False
    return True


# Cycling (CR 702.29): an activated ability usable from the hand — pay the mana
# cost and discard the card to draw one. Name -> number of blue ({U}) pips in the
# cost. current_view tells the front end whether a held card can be played and/or
# cycled right now.
CYCLING = {"Lonely Sandbar": 1}

# Modal spells (CR 700.2): the mode is chosen as the spell is cast (a menu on the
# hand card, like cycling). `text` is the mode's rules line — logged after the
# cast and used to pick the on-stack art variant. Vision Charm is the only one.
MODAL_SPELLS = {
    "Vision Charm": [
        {"key": "mill", "text": "Target player mills four cards."},
        {"key": "land", "text": "Change each land of one type into another type."},
    ],
}


def modal_mode_text(name: str, mode: str) -> str:
    """The rules line for a modal spell's chosen mode (empty if unknown)."""
    return next((m["text"] for m in MODAL_SPELLS.get(name, []) if m["key"] == mode), "")


def _available_mana(state: "GameState", player: str) -> int:
    """Mana the player could spend now: floating pool plus untapped lands."""
    pool = sum(state.players[player].mana_pool.values())
    lands = sum(1 for iid in state.players[player].battlefield
                if _is_land(state.objects[iid].type_line) and not state.objects[iid].tapped)
    return pool + lands


def _can_play_now(state: "GameState", viewer: str, card: "CardInstance") -> bool:
    """Whether `viewer` may play `card` from hand right now (land/sorcery speed
    needs their main phase with an empty stack; lands also need their land drop)."""
    p = state.pending
    if not (p and p.type == "priority" and p.player == viewer):
        return False
    tl = (card.type_line or "").lower()
    is_land = "land" in tl
    if is_land and state.players[viewer].land_played_this_turn:
        return False
    sorcery_speed = is_land or "instant" not in tl
    if sorcery_speed and (state.active_player != viewer
                          or state.current_step not in ("main1", "main2")
                          or state.stack):
        return False
    return True


def _is_permanent(type_line: str) -> bool:
    tl = (type_line or "").lower()
    return any(t in tl for t in _PERMANENT_TYPES)


# ---------------------------------------------------------------------------
# "Enters tapped" — parsed from oracle text. The subject of the clause is
# templated inconsistently: "This <type> enters tapped", "<Cardname> enters
# tapped", or (for a legendary) just its short name, e.g. "Alirios enters the
# battlefield tapped". We accept all three, plus the legacy "enters the
# battlefield tapped" wording, and an optional "unless <condition>".
# ---------------------------------------------------------------------------

_ENTERS_TAPPED_RE = re.compile(r"\benters(?:\s+the\s+battlefield)?\s+tapped\b", re.I)
_CARD_TYPES = ("land", "creature", "artifact", "enchantment", "planeswalker", "battle", "permanent")
_NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4,
                 "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def _self_subjects(name: str, type_line: str) -> list:
    """The phrases a card uses to refer to itself in its own rules text."""
    subs = ["it", "this permanent"]
    n = (name or "").strip().lower()
    if n:
        subs.append(n)
        short = n.split(",")[0].strip()          # legendary short name ("Alirios, Enraptured" -> "alirios")
        if short and short != n:
            subs.append(short)
    types = (type_line or "").split("—")[0].lower()
    for t in _CARD_TYPES:
        if re.search(r"\b" + t + r"\b", types):
            subs.append("this " + t)
    return subs


def parse_enters_tapped(oracle_text: str, name: str, type_line: str) -> "dict | None":
    """Find a self-referential 'enters tapped' clause. Returns None when there is
    none, {"unless": None} when always tapped, or {"unless": condition_text}."""
    if not oracle_text:
        return None
    subjects = _self_subjects(name, type_line)
    for raw in re.split(r"(?<=[.;])\s+|\n", oracle_text):
        s = raw.strip().lstrip("•").strip().lower().rstrip(".")
        if "tapped" not in s:
            continue
        m = _ENTERS_TAPPED_RE.search(s)
        if not m or s[:m.start()].strip() not in subjects:   # subject must be this card
            continue
        cond = re.search(r"\bunless\s+(.+)$", s[m.end():].strip())
        return {"unless": cond.group(1).strip() if cond else None}
    return None


def _condition_met(state: "GameState", controller: str, cond: str, self_iid: str) -> "bool | None":
    """Evaluate an 'unless' condition. Returns None when we can't parse it."""
    m = re.search(r"control[s]?\s+(\w+)\s+or\s+more\s+other\s+([a-z]+)", cond)
    if m:
        n = _NUMBER_WORDS.get(m.group(1))
        typ = m.group(2).rstrip("s")                         # "islands" -> "island"
        if n is None:
            return None
        count = sum(
            1 for iid in state.players[controller].battlefield
            if iid != self_iid and re.search(r"\b" + re.escape(typ) + r"\b",
                                              (state.objects[iid].type_line or "").lower())
        )
        return count >= n
    return None


def enters_tapped(state: "GameState", controller: str, card: "CardInstance") -> bool:
    """Whether `card` enters the battlefield tapped for `controller`."""
    info = parse_enters_tapped(card.oracle_text, card.name, card.type_line)
    if info is None:
        return False
    if not info["unless"]:
        return True                                          # unconditional
    # "enters tapped unless C": tapped when C is false — or can't be evaluated.
    return not bool(_condition_met(state, controller, info["unless"], card.instance_id))


def _seal_taps(state: "GameState") -> None:
    """An "other action" happened — mana taps can no longer be undone with z."""
    for p in state.players.values():
        p.tap_undo = []


def _drop_pool(pool: dict, sym: str = "U", n: int = 1) -> None:
    pool[sym] = pool.get(sym, 0) - n
    if pool.get(sym, 0) <= 0:
        pool.pop(sym, None)


def _reverse_source(state: "GameState", player: str, rec) -> None:
    """Undo the cost a mana action paid: untap the source, and (for a
    tap-and-sacrifice mana ability) return it from the graveyard untapped. `rec`
    is an undo record `{kind, iid, mana}` (a bare id is treated as a plain tap)."""
    if isinstance(rec, str):
        rec = {"kind": "tap", "iid": rec}
    iid = rec.get("iid")
    obj = state.objects.get(iid)
    if not obj:
        return
    if rec.get("kind") == "sac":
        if iid in state.graveyard:
            state.graveyard.remove(iid)
        if iid not in state.players[player].battlefield:
            state.players[player].battlefield.append(iid)
    obj.tapped = False


def tap_land(state: "GameState", player: str, instance_id: str) -> bool:
    """Tap one of `player`'s untapped lands for mana (floating in their pool).
    Records the tap so it can be undone with untap_last (z)."""
    p = state.players[player]
    if instance_id not in p.battlefield:
        return False
    obj = state.objects[instance_id]
    if not _is_land(obj.type_line) or obj.tapped:
        return False
    obj.tapped = True
    sym = land_mana_color(obj)                        # Island {U}, Swamp {B}, ... (text-aware)
    p.mana_pool[sym] = p.mana_pool.get(sym, 0) + 1
    p.tap_undo.append({"kind": "tap", "iid": instance_id, "mana": 1, "sym": sym})
    return True


def untap_last(state: "GameState", player: str) -> str | None:
    """Undo the most recent mana action (z): untap (or un-sacrifice) the source
    and remove the mana it made. Only actions since the last other action are
    undoable (tap_undo is sealed)."""
    p = state.players[player]
    if not p.tap_undo:
        return None
    rec = p.tap_undo.pop()
    if isinstance(rec, str):
        rec = {"kind": "tap", "iid": rec, "mana": 1}
    _reverse_source(state, player, rec)
    _drop_pool(p.mana_pool, rec.get("sym", "U"), rec.get("mana", 1))
    return rec.get("iid")


def play_card(state: "GameState", player: str, instance_id: str) -> bool:
    """Play a card from `player`'s hand: lands go straight to the battlefield,
    everything else goes onto the stack as a spell. Returns False if the card
    isn't in that hand."""
    p = state.players[player]
    if instance_id not in p.hand:
        return False
    _seal_taps(state)
    p.hand.remove(instance_id)
    obj = state.objects[instance_id]
    obj.controller = player
    _reconcile_played_name(state, player, obj)   # attribute the play to a known same-named card
    _mark_seen(state, instance_id)               # entering a public zone
    if _is_land(obj.type_line):
        obj.entered_this_turn = True
        obj.tapped = enters_tapped(state, player, obj)
        p.battlefield.append(instance_id)
        state.log.append(f"{p.name} plays {obj.name}.")
    else:
        state.stack.append(StackObject(
            stack_id=uuid.uuid4().hex, kind="spell",
            source_instance_id=instance_id, controller=player, description=obj.name,
        ))
        state.log.append(f"{p.name} casts {obj.name}.")
    return True


def resolve_top(state: "GameState") -> str | None:
    """Resolve the top object of the stack: permanents enter the battlefield
    under their controller, other spells go to the shared graveyard."""
    if not state.stack:
        return None
    _seal_taps(state)
    obj = state.stack.pop()
    iid = obj.source_instance_id
    if not iid or iid not in state.objects:
        return None
    inst = state.objects[iid]
    _mark_seen(state, iid)
    if _is_permanent(inst.type_line):
        inst.entered_this_turn = True
        inst.tapped = enters_tapped(state, obj.controller, inst)
        state.players[obj.controller].battlefield.append(iid)
        state.log.append(f"{inst.name} resolves.")
    else:
        state.graveyard.append(iid)
        state.log.append(f"{inst.name} resolves.")
    return iid


def reveal_library(state: "GameState", player: str) -> None:
    """Mark every library position known to `player` (e.g. looking at the whole
    deck). The per-player projection then shows the order to that player."""
    for slot in state.library:
        slot.known_by[player] = True
    state.log.append(f"{state.players[player].name} examines the deck.")


def reveal_hand(state: "GameState", viewer: str) -> None:
    """Reveal the opponent's hand to `viewer` (debug): every card they hold
    becomes known to the viewer."""
    opp = "p2" if viewer == "p1" else "p1"
    for iid in state.players[opp].hand:
        o = state.objects.get(iid)
        if o and viewer not in (o.known_by or []):
            o.known_by.append(viewer)
    state.log.append(f"{state.players[viewer].name} looks at {state.players[opp].name}'s hand.")


def take_from_library(state: "GameState", player: str, instance_id: str) -> bool:
    """Debug: move a specific card from the library into `player`'s hand."""
    for idx, slot in enumerate(state.library):
        if slot.instance_id == instance_id:
            state.library.pop(idx)
            o = state.objects.get(instance_id)
            if o:                                  # whoever knew that slot knows it in hand
                o.known_by = [pid for pid, k in slot.known_by.items() if k]
            state.players[player].hand.append(instance_id)
            state.log.append(f"{state.players[player].name} takes a card from the deck.")
            return True
    return False


def reorder_hand(state: "GameState", player: str, order: list) -> bool:
    """Reorder `player`'s hand to match `order` (a list of instance ids); any
    ids not in the current hand are ignored, any omitted are kept at the end."""
    hand = state.players[player].hand
    in_hand = set(hand)
    new = [iid for iid in order if iid in in_hand]
    seen = set(new)
    new += [iid for iid in hand if iid not in seen]
    state.players[player].hand = new
    return True


def new_game(decklist, *, p1_name="Player 1", p2_name="Player 2",
             seed: int | None = None, game_id: str | None = None) -> GameState:
    """Build a fresh game: every card copy becomes an instance in the shuffled
    shared library. `decklist` is a list of dicts with qty/name/set/collector_number
    (the Forgetful Fish card data)."""
    if seed is None:
        seed = random.randrange(2 ** 31)
    rng = random.Random(seed)

    g = GameState()
    g.game_id = game_id or str(uuid.uuid4())
    g.seed = seed
    g.players = {
        "p1": PlayerState(pid="p1", name=p1_name),
        "p2": PlayerState(pid="p2", name=p2_name),
    }
    def _stat(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0
    for card in decklist:
        card_key = f"{card.get('set', '')}/{card.get('collector_number', '')}".strip("/")
        for _ in range(int(card.get("qty") or 1)):
            iid = uuid.uuid4().hex
            g.objects[iid] = CardInstance(
                instance_id=iid,
                card_key=card_key,
                name=card.get("name", ""),
                type_line=card.get("type_line", ""),
                oracle_text=card.get("oracle_text", ""),
                mana_cost=card.get("mana_cost", ""),
                power=_stat(card.get("power")),
                toughness=_stat(card.get("toughness")),
                entered_this_turn=False,
            )
            g.library.append(LibrarySlot(instance_id=iid))

    rng.shuffle(g.library)
    g.rng_state = _serialize_rng(rng)
    return g


# ---------------------------------------------------------------------------
# Per-player projection (what a given player is allowed to see)
# ---------------------------------------------------------------------------

def _public_object(o: CardInstance) -> dict:
    return {
        "instance_id": o.instance_id,
        "card_key": o.card_key,
        "name": o.name,
        "type_line": o.type_line,
        "oracle_text": o.oracle_text,
        "power": o.power,
        "toughness": o.toughness,
        "tapped": o.tapped,
        "entered_this_turn": o.entered_this_turn,
        "damage_marked": o.damage_marked,
        "counters": o.counters,
        "attached_to": o.attached_to,
        "attachments": o.attachments,
        "phased_out": o.phased_out,
        "is_token": o.is_token,
        "chosen": o.chosen,
        "controller": o.controller,
        "abilities": _ability_view(o),
        # which rendered variant image to show when text-changed (e.g. "swamp")
        "text_variant": text_variant(o),
    }


def _stack_view(state: GameState) -> list:
    """The stack as public objects (the stack is visible to everyone)."""
    out = []
    for s in state.stack:
        entry = asdict(s)
        src = state.objects.get(s.source_instance_id)
        entry["card"] = _public_object(src) if src else None
        # A modal spell on the stack shows the chosen mode's art variant.
        if src and src.name in MODAL_SPELLS and src.chosen.get("mode"):
            entry["card"]["mode_variant"] = src.chosen["mode"]
        out.append(entry)
    return out


def _pending_public(state: GameState) -> dict | None:
    """The part of a pending decision that BOTH players (and a spectator) may
    see: its type, whose decision it is, and the public 'X is resolving Y' note
    and Fact-or-Fiction split it surfaces — never the private decision context
    (a scry/reorder/search/putback/put-from-hand/name-card hides the cards it
    would otherwise reveal)."""
    p = state.pending
    if not p:
        return None
    out = {"type": p.type, "player": p.player, "waiting": True}
    ctx = p.context or {}
    if ctx.get("resolving_name"):
        out["resolving_name"] = ctx["resolving_name"]
        out["resolving_by"] = ctx.get("resolving_by")
    # Fact or Fiction reveals the top cards to both players before the split, so
    # the (already mutual) piles are safe to surface.
    if p.type == "fof_split":
        out["context"] = {
            "cards": ctx.get("cards", []),
            "pile1": ctx.get("pile1", list(ctx.get("revealed", []))),
            "pile2": ctx.get("pile2", []),
        }
    return out


def _chat_view(state: "GameState") -> list:
    """Table-talk, with each sender's display name resolved. Chat is public —
    both players and spectators see the same transcript."""
    out = []
    for m in state.chat:
        by = m.get("by")
        out.append({"by": by, "text": m.get("text", ""),
                    "name": state.players[by].name if by in state.players else (by or "?")})
    return out


def current_view(state: GameState, viewer: str) -> dict:
    """Filter the god-state down to what `viewer` ("p1"/"p2") may see: their own
    hand and the library positions they know, everyone's public zones, and only
    counts for the opponent's hand and the unknown library."""
    opp = "p2" if viewer == "p1" else "p1"

    def pub(iid):
        return _public_object(state.objects[iid])

    players_out = {}
    for pid, p in state.players.items():
        players_out[pid] = {
            "name": p.name,
            "life": p.life,
            "mana_pool": p.mana_pool,
            "hand_count": len(p.hand),
            "battlefield": [pub(iid) for iid in p.battlefield],
            "has_lost": p.has_lost,
            "loss_reason": p.loss_reason,
        }
    # Refine the viewer's own activated abilities with the current timing: a
    # mana ability is usable at your priority OR while you pay a cost; a non-mana
    # ability only at your priority (CR 605.3a). The front end skips the menu
    # when exactly one is usable.
    pend = state.pending
    your_priority = bool(pend) and pend.type == "priority" and pend.player == viewer
    in_your_pay = bool(pend) and pend.type == "pay" and pend.player == viewer
    for card in players_out[viewer]["battlefield"]:
        for a in card.get("abilities", []):
            if a["available"]:                        # already passed the untapped check
                a["available"] = (your_priority or in_your_pay) if a["mana"] else your_priority

    # Reveal the viewer's own hand fully; the opponent's hand as known/hidden
    # slots (a card the viewer knows shows its face, otherwise a card back).
    players_out[viewer]["hand"] = [pub(iid) for iid in state.players[viewer].hand]
    # Annotate the viewer's own hand with what they can do with each card now:
    # play it, and (for cards with cycling) cycle it.
    for card_out, iid in zip(players_out[viewer]["hand"], state.players[viewer].hand):
        card = state.objects[iid]
        card_out["can_play"] = _can_play_now(state, viewer, card)
        cyc = CYCLING.get(card.name)
        if cyc is not None:
            card_out["cycle"] = cyc
            card_out["can_cycle"] = your_priority and _available_mana(state, viewer) >= cyc
        if card.name in MODAL_SPELLS:                  # modal spell: pick the mode as you cast
            card_out["modes"] = MODAL_SPELLS[card.name]
    players_out[opp]["hand"] = [
        {"known": True, **pub(iid)} if viewer in (state.objects[iid].known_by or [])
        else {"known": False}
        for iid in state.players[opp].hand
    ]

    library_out = []
    for slot in state.library:
        if slot.known_by.get(viewer):
            library_out.append({"known": True, **pub(slot.instance_id)})
        else:
            library_out.append({"known": False})

    stack_out = _stack_view(state)

    pending_out = None
    if state.pending:
        if state.pending.player == viewer:
            pending_out = asdict(state.pending)
            # A spell mid-resolution: surface "X is resolving Y" at the top level
            # too (the front end reads it there).
            ctx = state.pending.context or {}
            if ctx.get("resolving_name"):
                pending_out["resolving_name"] = ctx["resolving_name"]
                pending_out["resolving_by"] = ctx.get("resolving_by")
        else:
            # The non-decider sees only the public part — same filter a spectator
            # gets (see _pending_public).
            pending_out = _pending_public(state)

    return {
        "schema_version": state.schema_version,
        "game_id": state.game_id,
        "you": viewer,
        "opponent": opp,
        "turn_number": state.turn_number,
        "active_player": state.active_player,
        "current_step": state.current_step,
        "priority_player": state.priority_player,
        "yield_mode": state.players[viewer].yield_mode,
        "stops": state.players[viewer].stops,
        "yield_here": state.players[viewer].yield_here,
        "players": players_out,
        "library_count": len(state.library),
        "library": library_out,
        "graveyard": [pub(iid) for iid in state.graveyard],
        "exile": [pub(iid) for iid in state.exile],
        "stack": stack_out,
        "combat": asdict(state.combat),
        "pending": pending_out,
        "result": state.result,
        "log": list(state.log),
        "chat": _chat_view(state),
    }


def spectator_view(state: GameState) -> dict:
    """A read-only projection for a neutral spectator: the game from the host's
    seat ("p1" at the bottom) but showing ONLY what BOTH players know — never
    either player's secret hand or private library knowledge. This is the
    intersection of both players' knowledge, so it is safe to serve to anyone:
    it can reveal nothing a player doesn't already know."""
    def pub(iid):
        return _public_object(state.objects[iid])

    def hand_out(pid):
        # A hand card is mutual knowledge iff the NON-owner knows it (the owner
        # always sees their own hand).
        opp = "p2" if pid == "p1" else "p1"
        out = []
        for iid in state.players[pid].hand:
            if opp in (state.objects[iid].known_by or []):
                out.append({"known": True, **pub(iid)})
            else:
                out.append({"known": False})
        return out

    players_out = {}
    for pid, p in state.players.items():
        players_out[pid] = {
            "name": p.name,
            "life": p.life,
            "mana_pool": p.mana_pool,
            "hand_count": len(p.hand),
            "hand": hand_out(pid),
            "battlefield": [pub(iid) for iid in p.battlefield],
            "has_lost": p.has_lost,
            "loss_reason": p.loss_reason,
        }

    library_out = []
    for slot in state.library:
        if slot.known_by.get("p1") and slot.known_by.get("p2"):
            library_out.append({"known": True, **pub(slot.instance_id)})
        else:
            library_out.append({"known": False})

    return {
        "schema_version": state.schema_version,
        "game_id": state.game_id,
        "spectator": True,
        "you": "p1",                                   # host orientation (host at the bottom)
        "opponent": "p2",
        "turn_number": state.turn_number,
        "active_player": state.active_player,
        "current_step": state.current_step,
        "priority_player": state.priority_player,
        "players": players_out,
        "library_count": len(state.library),
        "library": library_out,
        "graveyard": [pub(iid) for iid in state.graveyard],
        "exile": [pub(iid) for iid in state.exile],
        "stack": _stack_view(state),
        "combat": asdict(state.combat),
        "pending": _pending_public(state),
        "result": state.result,
        "log": list(state.log),
        "chat": _chat_view(state),
    }


# The lobby's in-progress preview is a battlefield IMAGE rendered from this
# spectator_view's public battlefield by snapshot.render_battlefield_png — the
# battlefield is public, so drawing the real cards leaks nothing.
