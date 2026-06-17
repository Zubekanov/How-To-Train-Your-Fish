"""Env-side builders for compound (combinatorial) decisions.

The engine completes several decisions with structured arguments — an ordered
top/bottom partition (scry), a two-pile partition (Fact-or-Fiction split), a
bipartite blocker assignment, an ordered subset (putback / bottom / discard), a
full permutation (reorder), a subset (declare attackers). Rather than enumerate
those joint spaces, we accumulate them through a small shared sub-action alphabet
(``PICK_A`` / ``PICK_B`` / ``COMMIT`` / ``SHUFFLE``) and only call the engine's
completion function once the builder forms a well-formed, exactly-partitioning
argument. This keeps a single flat ``Discrete`` head + one mask, and the engine
never sees a partial/illegal call.

A builder is created when the env first encounters one of :data:`COMPOUND_TYPES`
and lives in the env (NOT in the serialized game state) until it finalizes.
"""
from __future__ import annotations

import numpy as np

from fishrl.forgetful_fish import engine as E
from fishrl.spaces import action_space as A

COMPOUND_TYPES = {
    "scry", "reorder", "putback", "bottom", "discard",
    "declare_attackers", "declare_blockers", "fof_split",
}


class CompoundBuilder:
    """Accumulates sub-actions for one compound decision, then calls the engine."""

    def __init__(self, g, player: str, ptype: str, ctx: dict):
        self.player = player
        self.ptype = ptype
        self.allow_shuffle = bool(ctx.get("allow_shuffle", False))
        # count of cards the agent must place (slots for putback, count otherwise)
        self.count = int(ctx.get("slots") or ctx.get("count") or 0)

        # Resolve the working item-id list(s) for this decision.
        if ptype in ("scry", "reorder"):
            self.items = [c["instance_id"] for c in ctx.get("cards", [])]
        elif ptype in ("putback", "bottom", "discard"):
            self.items = list(g.players[player].hand)
        elif ptype == "declare_attackers":
            self.items = list(ctx.get("eligible", []))
        elif ptype == "declare_blockers":
            self.attackers = list(ctx.get("attackers", []))
            self.items = list(ctx.get("eligible", []))   # candidate blockers
        elif ptype == "fof_split":
            self.items = list(ctx.get("revealed", []))
        else:  # pragma: no cover - guarded by COMPOUND_TYPES
            raise ValueError(f"not a compound decision: {ptype}")

        # builder accumulators
        self.top: list[str] = []          # scry
        self.bottom: list[str] = []       # scry
        self.order: list[str] = []        # reorder / putback / bottom / discard
        self.pile2: list[str] = []        # fof_split (pile1 = the remainder)
        self.assigned: set[str] = set()   # scry / fof_split placement tracking
        self.attacking: list[str] = []    # declare_attackers
        self.blocks: dict[str, list] = {}  # declare_blockers: attacker_id -> [blocker_id]
        self.sel_attacker: int | None = None
        self.used_blockers: set[int] = set()

    # ── masking ────────────────────────────────────────────────────────────
    def mask(self) -> np.ndarray:
        m = np.zeros(A.N, dtype=np.int8)
        n = min(len(self.items), A.CMP_K)
        t = self.ptype
        if t == "scry":
            for i in range(n):
                if self.items[i] not in self.assigned:
                    m[A.aid("PICK_A", i)] = 1   # to top
                    m[A.aid("PICK_B", i)] = 1   # to bottom
        elif t == "reorder":
            for i in range(n):
                if self.items[i] not in self.order:
                    m[A.aid("PICK_A", i)] = 1
            if self.allow_shuffle and not self.order:
                m[A.aid("SHUFFLE")] = 1
        elif t in ("putback", "bottom", "discard"):
            for i in range(n):
                if self.items[i] not in self.order:
                    m[A.aid("PICK_A", i)] = 1
        elif t == "declare_attackers":
            for i in range(n):
                m[A.aid("PICK_A", i)] = 1       # toggle in/out
            m[A.aid("COMMIT")] = 1
        elif t == "declare_blockers":
            for j in range(min(len(self.attackers), A.CMP_K)):
                m[A.aid("PICK_B", j)] = 1       # select an attacker to assign to
            if self.sel_attacker is not None:
                for i in range(n):
                    if i not in self.used_blockers:
                        m[A.aid("PICK_A", i)] = 1
            m[A.aid("COMMIT")] = 1
        elif t == "fof_split":
            for i in range(n):
                if self.items[i] not in self.assigned:
                    m[A.aid("PICK_A", i)] = 1   # pile 1
                    m[A.aid("PICK_B", i)] = 1   # pile 2
        return m

    # ── feeding sub-actions ──────────────────────────────────────────────────
    def feed(self, g, action: int) -> bool:
        """Apply one sub-action. Returns True iff the builder finalized (and the
        engine has been advanced); False if it merely accumulated."""
        name, i = A.decode(action)
        t = self.ptype
        if t == "scry":
            iid = self.items[i]
            if name == "PICK_A":
                self.top.append(iid)
            else:
                self.bottom.append(iid)
            self.assigned.add(iid)
            if len(self.assigned) == len(self.items):
                return self._finalize(E.complete_scry(g, self.player, self.top, self.bottom))
            return False
        if t == "reorder":
            if name == "SHUFFLE":
                return self._finalize(E.complete_reorder(g, self.player, [], shuffle=True))
            self.order.append(self.items[i])
            if len(self.order) == len(self.items):
                return self._finalize(E.complete_reorder(g, self.player, self.order))
            return False
        if t in ("putback", "bottom", "discard"):
            self.order.append(self.items[i])
            if len(self.order) >= self.count:
                if t == "putback":
                    return self._finalize(E.complete_putback(g, self.player, self.order))
                if t == "bottom":
                    return self._finalize(E.bottom_cards(g, self.player, self.order))
                return self._finalize(E.discard_to_hand_size(g, self.player, self.order))
            return False
        if t == "declare_attackers":
            if name == "COMMIT":
                return self._finalize(E.declare_attackers(g, self.player, self.attacking))
            iid = self.items[i]
            if iid in self.attacking:
                self.attacking.remove(iid)
            else:
                self.attacking.append(iid)
            return False
        if t == "declare_blockers":
            if name == "COMMIT":
                return self._finalize(E.declare_blockers(g, self.player, self.blocks))
            if name == "PICK_B":
                self.sel_attacker = i
                return False
            if name == "PICK_A" and self.sel_attacker is not None:
                attacker = self.attackers[self.sel_attacker]
                self.blocks.setdefault(attacker, []).append(self.items[i])
                self.used_blockers.add(i)
            return False
        if t == "fof_split":
            iid = self.items[i]
            if name == "PICK_B":
                self.pile2.append(iid)
            self.assigned.add(iid)
            if len(self.assigned) == len(self.items):
                p1 = [x for x in self.items if x not in self.pile2]
                return self._finalize(E.complete_fof_split(g, self.player, p1, self.pile2))
            return False
        return False  # pragma: no cover

    def autofinalize(self, g) -> bool:
        """Resolve a builder that has NO legal sub-action — only `scry`/`reorder`
        on an empty library (deck-out), where there are zero cards to arrange.
        Completing with empty arguments is the correct no-op; the env calls this so
        the agent is never handed an empty action mask (which would otherwise make
        the masked softmax uniform and let an illegal action be sampled)."""
        if self.ptype == "scry":
            return self._finalize(E.complete_scry(g, self.player, [], []))
        if self.ptype == "reorder":
            return self._finalize(E.complete_reorder(g, self.player, []))
        raise RuntimeError(f"cannot autofinalize a {self.ptype} builder with an empty mask")

    def progress(self) -> float:
        """Fraction of this compound decision already specified (0..1)."""
        n = len(self.items) or 1
        t = self.ptype
        if t in ("scry", "fof_split"):
            return len(self.assigned) / n
        if t == "reorder":
            return len(self.order) / n
        if t in ("putback", "bottom", "discard"):
            return len(self.order) / (self.count or 1)
        if t == "declare_blockers":
            return len(self.used_blockers) / n
        return 0.0

    @staticmethod
    def _finalize(ok: bool) -> bool:
        if not ok:
            raise RuntimeError("compound completion rejected by engine — malformed builder")
        return True
