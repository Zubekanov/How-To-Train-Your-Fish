"""Heuristic->action-space adapter: drive a seat of the AEC env with a vendored
heuristic AI, emitting flat ``Discrete(A.N)`` action ids (the BC teacher).

Scripted seats normally never touch the action space -- the engine drives them
internally through imperative hooks that ACT on the game state rather than
return choices. This adapter turns those hooks into labels (docs/BC_ADAPTER.md):

* ``priority`` uses ``_choose_action`` (the decision half of ``take_priority``,
  which computes without acting) and translates its ("land"/"cycle"/"cast"/
  "activate") tuple; the chosen cast TARGET is cached for the ``choose_targets``
  pending that follows the cast.
* ``pay`` uses ``_best_land_to_tap`` (pure).
* every ``resolve_pending`` handler is compute-then-act with exactly one engine
  completion call at the end, so a record-and-abort shim (:func:`_intercept`)
  captures that call's arguments without the game state ever being touched.
* combat + compound decisions translate a single structured answer into a QUEUE
  of builder sub-actions, drained one env-step at a time without re-querying
  (the teacher answered the whole decision; it never sees partial builders).

Deviations the action space forces (CANCEL_PAY / TARGET_CANCEL are masked out
as anti-stall measures) fall back to a legal action and are counted on
``fallbacks`` / ``forced_targets`` -- the parity test pins both to zero on
reference games.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np

from fishrl.forgetful_fish import engine as E
from fishrl.forgetful_fish.state import MODAL_SPELLS
from fishrl.spaces import action_space as A


def teacher_module(profile: str):
    """The vendored AI module for a heuristic profile (same map as engine._ai_mod)."""
    if profile not in E._HEURISTIC_PROFILES:
        raise ValueError(f"unknown teacher profile {profile!r}; have {E._HEURISTIC_PROFILES}")
    if profile == "heuristic_1_1":
        from fishrl.forgetful_fish import ai_v1_1
        return ai_v1_1
    if profile == "heuristic_1_2":
        from fishrl.forgetful_fish import ai_v1_2
        return ai_v1_2
    from fishrl.forgetful_fish import ai
    return ai


class _Recorded(Exception):
    """Raised by an intercepted engine function: carries the call instead of applying it."""

    def __init__(self, fn: str, args: tuple, kwargs: dict):
        super().__init__(fn)
        self.fn, self.args, self.kwargs = fn, args, kwargs


# Every engine entry point a teacher may call to complete a decision or act on the
# game. All are patched during a query: the first call recorded aborts the handler
# before any mutation, so no snapshot/deepcopy of the game state is ever needed.
_DECISION_FNS = (
    # resolve_pending completions
    "choose_play_order", "mulligan_decision", "bottom_cards", "complete_scry",
    "complete_reorder", "complete_putback", "complete_library_search",
    "complete_fof_split", "complete_fof_choose", "complete_name_card",
    "complete_text_change", "complete_put_from_hand", "discard_to_hand_size",
    "complete_graveyard_choice", "complete_targets", "place_trigger",
    # priority/pay actions (belt-and-braces: _choose_action must not act at all)
    "play", "cycle", "activate_ability", "tap", "allocate_mana", "pass_priority",
    "end_turn", "cancel_payment", "declare_attackers", "declare_blockers",
)


@contextmanager
def _intercept():
    saved = [(n, getattr(E, n)) for n in _DECISION_FNS]

    def stub(name):
        def f(*a, **k):
            raise _Recorded(name, a, k)
        return f

    for n, _fn in saved:
        setattr(E, n, stub(n))
    try:
        yield
    finally:
        for n, fn in saved:
            setattr(E, n, fn)


def _pick(items: list, x) -> int | None:
    """PICK_SINGLE id for item x, or None if unaddressable."""
    if x in items and items.index(x) < A.PICK_K:
        return A.aid("PICK_SINGLE", items.index(x))
    return None


def _pick_or_none(items: list, x) -> int | None:
    return A.aid("PICK_NONE") if x is None else _pick(items, x)


class HeuristicAdapter:
    """One teacher-driven seat. Query once per engine pending; drain the queue."""

    def __init__(self, profile: str = "heuristic_1_2"):
        self.profile = profile
        self.mod = teacher_module(profile)
        self.queue: list = []
        self.cast_target = None       # target chosen with a ("cast", ...) decision
        self.fallbacks = 0            # translations the mask rejected
        self.forced_targets = 0       # teacher would cancel; cancel is masked out

    def reset(self) -> None:
        self.queue.clear()
        self.cast_target = None

    # ── the public entry point ────────────────────────────────────────────────
    def act(self, env, mask: np.ndarray) -> int:
        """Next action id for the acting seat of `env` (FishAEC or a wrapper)."""
        base = getattr(env, "env", env)
        if not self.queue:
            self._plan(base)
        a = self.queue.pop(0)
        if a is None or not mask[a]:
            self.fallbacks += 1
            self.queue.clear()
            a = self._legal_default(mask)
        return int(a)

    @staticmethod
    def _legal_default(mask: np.ndarray) -> int:
        pa = A.aid("PASS")
        return pa if mask[pa] else int(np.flatnonzero(mask)[0])

    # ── planning: one teacher query -> one or more queued sub-actions ─────────
    def _plan(self, base) -> None:
        g, agent = base.g, base.agent_selection
        b = base._builder
        if b is not None:
            self.cast_target = None
            self.queue.extend(self._plan_compound(g, agent, b))
            return
        p = g.pending
        assert p is not None and p.player == agent, "adapter queried off-turn"
        t, ctx = p.type, (p.context or {})
        if t != "choose_targets":
            self.cast_target = None       # a stale cast cache must not leak forward
        if t == "priority":
            self.queue.append(self._plan_priority(g, agent))
        elif t == "pay":
            self.queue.append(self._plan_pay(g, agent))
        elif t == "choose_targets":
            self.queue.append(self._plan_targets(g, agent, ctx))
        elif t == "order_triggers":
            # the engine-internal AI auto-places its triggers in holding-area order
            self.queue.append(A.aid("PICK_SINGLE", 0))
        else:
            self.queue.append(self._plan_decision(g, agent, t, ctx))

    def _plan_priority(self, g, agent):
        # Replicate take_priority's per-turn action budget VERBATIM (same shared
        # game attribute, same count-every-grant semantics): the engine-internal
        # teacher passes once the budget is spent, and the h1.2 the RL agent
        # actually faces runs with this throttle -- so the labels must too.
        budget = getattr(g, "_ai_actions", None)
        if not budget or budget[0] != g.turn_number:
            budget = [g.turn_number, 0]
            g._ai_actions = budget                        # plain attr -- not serialized
        budget[1] += 1
        if budget[1] > getattr(self.mod, "_MAX_ACTIONS_PER_TURN", 30):
            return A.aid("PASS")
        try:
            with _intercept():
                act = self.mod._choose_action(g, agent)   # decision only, never acts
        except _Recorded as r:  # pragma: no cover - purity violation = teacher bug
            raise RuntimeError(f"{self.profile}._choose_action acted ({r.fn})") from r
        kind = act[0] if act else None
        hand = list(g.players[agent].hand)
        if kind in ("land", "cast", "cycle"):
            iid = act[1]
            if iid not in hand or hand.index(iid) >= A.HAND:
                return None
            i = hand.index(iid)
            if kind == "land":
                return A.aid("PLAY_HAND", i)
            if kind == "cycle":
                return A.aid("CYCLE_HAND", i)
            _, _, mode, target = act
            self.cast_target = target
            modes = MODAL_SPELLS.get(g.objects[iid].name) or []
            if mode is not None and len(modes) > 1 and mode == modes[1]["key"]:
                return A.aid("PLAY_HAND_ALT", i)
            return A.aid("PLAY_HAND", i)
        if kind == "activate":
            _, iid, k = act
            bf = list(g.players[agent].battlefield)
            if iid not in bf or bf.index(iid) >= A.BF or k >= A.ABIL_SLOTS:
                return None
            return A.aid("ACTIVATE", bf.index(iid) * A.ABIL_SLOTS + k)
        return A.aid("PASS")

    def _plan_pay(self, g, agent):
        land = self.mod._best_land_to_tap(g, agent)       # pure
        bf = list(g.players[agent].battlefield)
        if land is None or land not in bf or bf.index(land) >= A.BF:
            return None       # teacher would cancel_payment; cancel is masked out
        return A.aid("TAP_LAND", bf.index(land))

    def _plan_targets(self, g, agent, ctx):
        legal = list(ctx.get("legal", []))
        t, self.cast_target = self.cast_target, None
        if t is not None and t not in legal:
            self.forced_targets += 1      # teacher would back out; cancel is masked out
            t = None
        if t is None:
            t = max(legal, key=lambda iid: self.mod.card_value(g, agent, iid), default=None)
        return _pick(legal, t)

    def _plan_decision(self, g, agent, ptype, ctx):
        """Atomic pendings that resolve through a _PENDING_HANDLERS entry."""
        rec = self._query_handler(g, agent, ptype, ctx)
        fn, a = rec.fn, rec.args
        if fn == "choose_play_order":
            return A.aid("PLAY_ORDER", 0 if a[2] == "first" else 1)
        if fn == "mulligan_decision":
            return A.aid("MULLIGAN", 0 if a[2] == "keep" else 1)
        if fn == "complete_fof_choose":
            return A.aid("FOF_CHOOSE", 0 if int(a[2]) == 1 else 1)
        if fn == "complete_text_change":
            frm, to = a[2], a[3]
            if frm not in A.BASICS or to not in A.BASICS or frm == to:
                return None
            return A.aid("TEXT_CHANGE", A.BASICS.index(frm) * len(A.BASICS) + A.BASICS.index(to))
        if fn == "complete_name_card":
            return _pick(list(ctx.get("names", [])), a[2])
        if fn == "complete_library_search":
            return _pick_or_none(list(ctx.get("eligible", [])), a[2])
        if fn == "complete_put_from_hand":
            return _pick_or_none(list(ctx.get("eligible", [])), a[2])
        if fn == "complete_graveyard_choice":
            return _pick_or_none(list(ctx.get("eligible", [])), a[2])
        return None       # a handler completing through an unexpected call

    def _plan_compound(self, g, agent, b) -> list:
        """One structured teacher answer -> the builder sub-action sequence."""
        items = list(b.items[:A.CMP_K])
        idx = {iid: i for i, iid in enumerate(items)}
        t = b.ptype
        if t == "declare_attackers":
            chosen = self.mod.choose_attackers(g, agent, list(b.items)) or []
            return [A.aid("PICK_A", idx[x]) for x in chosen if x in idx] + [A.aid("COMMIT")]
        if t == "declare_blockers":
            blocks = self.mod.choose_blocks(g, agent, list(b.items)) or {}
            seq = []
            # builder focus moves FORWARD only -> emit in ascending attacker index
            for j, att in enumerate(b.attackers[:A.CMP_K]):
                mine = [x for x in (blocks.get(att) or []) if x in idx]
                if mine:
                    seq.append(A.aid("PICK_B", j))
                    seq += [A.aid("PICK_A", idx[x]) for x in mine]
            return seq + [A.aid("COMMIT")]
        rec = self._query_handler(g, agent, t, g.pending.context or {})
        a, k = rec.args, rec.kwargs
        if t == "scry":       # complete_scry(g, p, tops, bottoms); append order = final order
            return ([A.aid("PICK_A", idx[x]) for x in a[2] if x in idx]
                    + [A.aid("PICK_B", idx[x]) for x in a[3] if x in idx])
        if t == "reorder":    # complete_reorder(g, p, order, shuffle=False)
            shuffle = bool(k.get("shuffle", a[3] if len(a) > 3 else False))
            if shuffle and b.allow_shuffle:
                return [A.aid("SHUFFLE")]
            return [A.aid("PICK_A", idx[x]) for x in a[2] if x in idx]
        if t == "fof_split":  # complete_fof_split(g, p, pile1, pile2)
            pile2 = set(a[3])
            return [A.aid("PICK_B" if x in pile2 else "PICK_A", idx[x]) for x in items]
        if t in ("putback", "bottom", "discard"):
            order = [x for x in a[2] if x in idx]
            if b.count:
                order = order[:b.count]
            return [A.aid("PICK_A", idx[x]) for x in order]
        raise RuntimeError(f"unhandled compound pending {t!r}")

    def _query_handler(self, g, agent, ptype, ctx) -> _Recorded:
        handler = self.mod._PENDING_HANDLERS.get(ptype)
        if handler is None:
            raise RuntimeError(f"{self.profile}: no handler for pending {ptype!r}")
        try:
            with _intercept():
                handler(g, agent, ctx)
        except _Recorded as r:
            return r
        raise RuntimeError(f"{self.profile}: {ptype!r} handler completed nothing")
