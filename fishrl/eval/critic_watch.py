"""Two-critic ledger over v1.3 mirror games: what does the value function know, when?

Plays N games of heuristic_1_3 vs heuristic_1_3 (both seats the engine AI, the
testbench-arena pump-loop pattern) while the checkpoint's TWO outcome heads watch
every state change:

  * PrivilegedCritic  (god features: both hands + full ordered library)
  * PublicEstimator   (mutual-knowledge features only)

Both output P(p1 wins); the games are policy-free (no actor, no guesser), so this
measures the CRITICS against a fixed, strong, scripted play distribution -- not the
policy. Per state-change we record both probabilities plus cheap scalars (life,
library); per AI decision we record an event class (cast:<name>, land, activate,
pass, pend:<type>, attackers, ...). Value deltas between consecutive states are
attributed to the event that caused them, so action classes can be ranked by how
much game-value they move.

Encoding cost discipline: a state VERSION key (log length, turn, stack, life, hand,
library) dedupes encodes -- pass-chains that change nothing are never re-encoded.
Critic forwards run batched per game inside each worker (CPU, 1 torch thread);
weights cross the process boundary as numpy (torch 2.12 plain-unpickle leaks).

    python -m fishrl.eval.critic_watch --games 4000 --pin --out watch_v13
    python -m fishrl.eval.critic_watch --analyze watch_v13.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

# ── worker globals ────────────────────────────────────────────────────────────
_W: dict = {}

# One logical processor per physical P-core (i7-14700KF: LPs 0-15 = 8 P-cores in
# HT pairs, 16-27 = E-cores). Same rationale + measurement as the testbench arena.
_PCORE_LPS = tuple(range(0, 16, 2))

# Deep all-AI games recurse through the re-entrant engine to 8k+ frames; the
# default C stack hard-crashes (0xC00000FD) on Windows. Every game runs on a
# thread with a big explicit stack (virtual reservation, not committed memory).
_GAME_STACK_BYTES = 128 * 1024 * 1024
_RECURSION_LIMIT = 50_000

WINNER_CODE = {"p1": 1, "p2": 0, None: -1}


def _apply_affinity(lp: int) -> None:
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetProcessAffinityMask.argtypes = (wintypes.HANDLE, ctypes.c_size_t)
        k32.SetProcessAffinityMask.restype = wintypes.BOOL
        k32.SetProcessAffinityMask(k32.GetCurrentProcess(), 1 << lp)
    except Exception as e:                                    # noqa: BLE001 - best effort
        print(f"[affinity] skipped: {e}", flush=True)


def _build_heads(config_dict: dict, weights: dict):
    """Rebuild ONLY the two outcome heads from the checkpoint's architecture keys
    and numpy weights (no actor/guesser -- the games are engine-AI driven)."""
    import torch
    from fishrl.models.estimators import PrivilegedCritic, PublicEstimator
    from fishrl.train.train_loop import config_from_checkpoint
    cfg = config_from_checkpoint(config_dict)
    heads = {
        "critic": PrivilegedCritic(cfg.head_hidden("critic"), cfg.enc_for("critic"), cfg.card_dim),
        "public": PublicEstimator(cfg.head_hidden("public"), cfg.enc_for("public"), cfg.card_dim),
    }
    for name, net in heads.items():
        net.load_state_dict({k: torch.from_numpy(v) for k, v in weights[name].items()})
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
    return heads


def _init_worker(config_dict, weights, pin_counter=None) -> None:
    sys.setrecursionlimit(_RECURSION_LIMIT)
    if pin_counter is not None:
        with pin_counter.get_lock():
            idx = pin_counter.value
            pin_counter.value += 1
        if idx < len(_PCORE_LPS):
            _apply_affinity(_PCORE_LPS[idx])
    import torch
    torch.set_num_threads(1)
    from fishrl.forgetful_fish.cards import load_decklist
    _W["decklist"] = load_decklist()
    _W["heads"] = _build_heads(config_dict, weights)
    _install_recorder()


# ── event recording ───────────────────────────────────────────────────────────
# The v1.3 module's decision entry points are wrapped ONCE per worker; each call
# appends an event to the current game's recorder before running the original.
# Wrapping the module (not the engine) captures decisions wherever the engine
# re-enters them, including nested stack wars inside a single pump.

class _Recorder:
    def __init__(self, g):
        self.g = g
        self.states: list = []          # [turn, step, p1_life, p2_life, lib, hand1, hand2]
        self.god: list = []             # god feature rows, aligned with .states
        self.pub: list = []
        self.events: list = []          # [state_idx, turn, step, actor, class_str, stacktop_str, top_ctrl]
        self._last_version = None
        self.encode_s = 0.0

    def _version(self):
        g = self.g
        return (len(g.log), g.turn_number, len(g.stack),
                g.players["p1"].life, g.players["p2"].life,
                len(g.players["p1"].hand), len(g.players["p2"].hand),
                len(g.library), len(g.graveyard))

    def snapshot(self) -> int:
        """Encode the current state iff it changed since the last snapshot;
        return its index."""
        v = self._version()
        if v == self._last_version and self.states:
            return len(self.states) - 1
        from fishrl.data.features import encode_god, encode_public
        from fishrl.forgetful_fish.engine import _STEPS
        g = self.g
        t0 = time.perf_counter()
        self.god.append(encode_god(g))
        self.pub.append(encode_public(g))
        self.encode_s += time.perf_counter() - t0
        step = _STEPS.index(g.current_step) if g.current_step in _STEPS else -1
        self.states.append([g.turn_number, step, g.players["p1"].life,
                            g.players["p2"].life, len(g.library),
                            len(g.players["p1"].hand), len(g.players["p2"].hand),
                            1 if g.active_player == "p1" else 0])
        self._last_version = v
        return len(self.states) - 1

    def event(self, player: str, cls: str) -> None:
        g = self.g
        si = self.snapshot()
        from fishrl.forgetful_fish.engine import _STEPS
        step = _STEPS.index(g.current_step) if g.current_step in _STEPS else -1
        top, top_ctrl = "", -1
        if g.stack:
            so = g.stack[-1]
            o = g.objects.get(so.source_instance_id)
            top = o.name if o is not None else ""
            top_ctrl = 1 if so.controller == "p1" else 0
        self.events.append([si, g.turn_number, step,
                            1 if player == "p1" else 0, cls, top, top_ctrl])


_REC: list = [None]                     # [current _Recorder or None]


def _install_recorder() -> None:
    from fishrl.forgetful_fish import ai_v1_3 as M

    def _name(g, iid):
        o = g.objects.get(iid)
        return o.name if o is not None else "?"

    orig_execute = M._execute
    def execute(g, player, action):
        r = _REC[0]
        if r is not None:
            kind = action[0] if action else "pass"
            if kind == "cast":
                cls = f"cast:{_name(g, action[1])}"
            elif kind == "activate":
                cls = f"activate:{_name(g, action[1])}"
            elif kind == "cycle":
                cls = f"cycle:{_name(g, action[1])}"
            elif kind == "land":
                cls = "land"
            else:
                cls = "pass"
            r.event(player, cls)
        return orig_execute(g, player, action)
    M._execute = execute

    orig_pending = M.resolve_pending
    def resolve_pending(g):
        r = _REC[0]
        if r is not None and g.pending is not None:
            r.event(g.pending.player, f"pend:{g.pending.type}")
        return orig_pending(g)
    M.resolve_pending = resolve_pending

    for fn, cls in (("choose_attackers", "attackers"), ("choose_blocks", "blocks")):
        orig = getattr(M, fn, None)
        if orig is None:
            continue
        def wrap(orig=orig, cls=cls):
            def inner(g, player, eligible):
                r = _REC[0]
                if r is not None:
                    r.event(player, cls)
                return orig(g, player, eligible)
            return inner
        setattr(M, fn, wrap())

    orig_tt = getattr(M, "choose_trigger_target", None)
    if orig_tt is not None:
        def trigger_target(g, t, legal):
            r = _REC[0]
            if r is not None:
                r.event(t.controller, "trigger_target")
            return orig_tt(g, t, legal)
        M.choose_trigger_target = trigger_target

    orig_d = getattr(M, "choose_discards", None)
    if orig_d is not None:
        def discards(g, player, excess):
            r = _REC[0]
            if r is not None:
                r.event(player, "discards")
            return orig_d(g, player, excess)
        M.choose_discards = discards


# ── one game ──────────────────────────────────────────────────────────────────

def _play_watched(seed: int, max_turns: int, max_pumps: int = 200_000):
    from fishrl.forgetful_fish import engine as E
    g = E.new_multiplayer_game(_W["decklist"], p1_name="p1", p2_name="p2", seed=seed)
    for pid in ("p1", "p2"):
        g.players[pid].is_ai = True
        g.players[pid].ai_profile = "heuristic_1_3"
    rec = _Recorder(g)
    _REC[0] = rec
    try:
        stall = 0
        for _ in range(max_pumps):
            if g.result.get("status") != "ongoing" or g.turn_number > max_turns:
                break
            p = g.pending
            if p is None:
                break                                     # engine should always pause somewhere
            marker = (id(p), len(g.log), len(g.stack))
            if p.type == "priority":
                E._ai_mod(g, p.player).take_priority(g, p.player)
            else:
                E._resolve_ai_pending(g)
            now = g.pending
            if (g.result.get("status") == "ongoing" and now is p
                    and (id(now), len(g.log), len(g.stack)) == marker):
                stall += 1
                if stall >= 3:
                    if now.type == "priority":
                        E.pass_priority(g, now.player)
                        stall = 0
                    else:
                        break                             # stuck non-priority decision: bail
            else:
                stall = 0
        rec.snapshot()                                    # terminal (or cap) state
    finally:
        _REC[0] = None
    winner = g.result.get("winner") if g.result.get("status") != "ongoing" else None
    return rec, winner, g.turn_number


def _forward(rec: _Recorder):
    import torch
    with torch.no_grad():
        god = torch.from_numpy(np.stack(rec.god))
        pub = torch.from_numpy(np.stack(rec.pub))
        priv_p = _W["heads"]["critic"].p1_winprob(god).numpy()
        pub_p = _W["heads"]["public"].p1_winprob(pub).numpy()
    return priv_p, pub_p


def _play_one(task):
    """(seed, max_turns) -> compact per-game arrays. Runs on a big-stack thread."""
    seed, max_turns = task
    out: dict = {}

    def run():
        try:
            t0 = time.perf_counter()
            rec, winner, turns = _play_watched(seed, max_turns)
            t1 = time.perf_counter()
            priv_p, pub_p = _forward(rec)
            t2 = time.perf_counter()
            states = np.asarray(rec.states, dtype=np.int16)
            probs = np.stack([priv_p, pub_p], axis=1).astype(np.float32)
            vocab: dict = {}
            ev = np.zeros((len(rec.events), 7), dtype=np.int32)
            for i, (si, turn, step, actor, cls, top, top_ctrl) in enumerate(rec.events):
                ev[i] = (si, turn, step, actor,
                         vocab.setdefault(cls, len(vocab)),
                         vocab.setdefault(f"@{top}", len(vocab)) if top else -1,
                         top_ctrl)
            out["v"] = {"seed": seed, "winner": WINNER_CODE.get(winner, -1), "turns": turns,
                        "states": states, "probs": probs, "events": ev,
                        "vocab": list(vocab.keys()),
                        "t_game": t1 - t0, "t_fwd": t2 - t1, "t_enc": rec.encode_s}
        except BaseException as e:                        # noqa: BLE001 - relayed to caller
            out["e"] = e

    old = threading.stack_size(_GAME_STACK_BYTES)
    try:
        t = threading.Thread(target=run, name="ff-watched-game")
        t.start()
        t.join()
    finally:
        threading.stack_size(old)
    if "e" in out:
        raise out["e"]
    return out["v"]


# ── merge + save ──────────────────────────────────────────────────────────────

def _merge(results: list) -> dict:
    """Concatenate per-game arrays into flat columns with a shared vocab and
    per-game offsets -- the .npz layout the analysis reads."""
    vocab: dict = {}
    g_meta, g_soff, g_eoff = [], [0], [0]
    all_states, all_probs, all_events = [], [], []
    for r in results:
        remap = np.array([vocab.setdefault(s, len(vocab)) for s in r["vocab"]],
                         dtype=np.int32)
        ev = r["events"].copy()
        if len(ev):
            ev[:, 4] = remap[ev[:, 4]]
            has_top = ev[:, 5] >= 0
            ev[has_top, 5] = remap[ev[has_top, 5]]
        all_states.append(r["states"])
        all_probs.append(r["probs"])
        all_events.append(ev)
        g_meta.append((r["seed"], r["winner"], r["turns"]))
        g_soff.append(g_soff[-1] + len(r["states"]))
        g_eoff.append(g_eoff[-1] + len(ev))
    return {
        "games": np.asarray(g_meta, dtype=np.int32),
        "state_off": np.asarray(g_soff, dtype=np.int64),
        "event_off": np.asarray(g_eoff, dtype=np.int64),
        "states": np.concatenate(all_states) if all_states else np.zeros((0, 8), np.int16),
        "probs": np.concatenate(all_probs) if all_probs else np.zeros((0, 2), np.float32),
        "events": np.concatenate(all_events) if all_events else np.zeros((0, 7), np.int32),
        "vocab": np.asarray(list(vocab.keys())),
    }


# ── analysis ──────────────────────────────────────────────────────────────────

def _wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def analyze(d: dict, out=print) -> None:
    games, states, probs, events = d["games"], d["states"], d["probs"], d["events"]
    soff, eoff, vocab = d["state_off"], d["event_off"], [str(s) for s in d["vocab"]]
    n_games = len(games)
    decided = games[:, 1] >= 0
    p1w = games[:, 1] == 1
    out(f"games={n_games}  decided={decided.sum()}  draws/caps={(~decided).sum()}"
        f"  p1 wr (decided)={p1w.sum() / max(1, decided.sum()):.3f}"
        f"  avg turns={games[:, 2].mean():.1f}")
    out(f"states={len(states):,}  events={len(events):,}  "
        f"({len(states) / max(1, n_games):.0f} / {len(events) / max(1, n_games):.0f} per game)")

    # ── per-turn accuracy / Brier: the LAST evaluated state of each turn ──────
    max_t = int(states[:, 0].max()) if len(states) else 0
    out("\nper-turn (last state of each turn, decided games only):")
    out(f"{'turn':>4} {'n':>6} {'priv_acc':>9} {'pub_acc':>8} {'life_acc':>9}"
        f" {'priv_brier':>11} {'pub_brier':>10} {'gap':>7}")
    turn_rows = []
    for t in range(1, max_t + 1):
        pv, pb, lf, y = [], [], [], []
        for gi in np.flatnonzero(decided):
            s = slice(soff[gi], soff[gi + 1])
            st, pr = states[s], probs[s]
            in_t = np.flatnonzero(st[:, 0] == t)
            if len(in_t) == 0:
                continue
            i = in_t[-1]
            pv.append(pr[i, 0]); pb.append(pr[i, 1])
            lf.append(st[i, 2] - st[i, 3])
            y.append(1.0 if games[gi, 1] == 1 else 0.0)
        if len(y) < 20:
            continue
        pv, pb, lf, y = map(np.asarray, (pv, pb, lf, y))
        pa = float(((pv > .5) == (y > .5)).mean())
        ba = float(((pb > .5) == (y > .5)).mean())
        # life-diff baseline: predict the life leader, half credit on ties
        la = float(np.where(lf != 0, ((lf > 0) == (y > .5)).astype(float), 0.5).mean())
        pbr = float(((pv - y) ** 2).mean()); bbr = float(((pb - y) ** 2).mean())
        turn_rows.append((t, len(y), pa, ba, la, pbr, bbr))
        if t <= 30 or t % 5 == 0:
            out(f"{t:>4} {len(y):>6} {pa:>9.3f} {ba:>8.3f} {la:>9.3f}"
                f" {pbr:>11.3f} {bbr:>10.3f} {bbr - pbr:>7.3f}")

    # ── calibration (all decided-game states) ─────────────────────────────────
    keep = np.zeros(len(states), dtype=bool)
    y_all = np.zeros(len(states), dtype=np.float32)
    for gi in np.flatnonzero(decided):
        keep[soff[gi]:soff[gi + 1]] = True
        y_all[soff[gi]:soff[gi + 1]] = 1.0 if games[gi, 1] == 1 else 0.0
    out("\ncalibration (decided-game states, 10 bins of predicted P(p1)):")
    out(f"{'bin':>10} {'n_priv':>9} {'priv_emp':>9} {'n_pub':>9} {'pub_emp':>8}")
    for lo in np.arange(0, 1, 0.1):
        hi = lo + 0.1
        m1 = keep & (probs[:, 0] >= lo) & (probs[:, 0] < hi)
        m2 = keep & (probs[:, 1] >= lo) & (probs[:, 1] < hi)
        out(f"{lo:.1f}-{hi:.1f} {m1.sum():>9,} "
            f"{y_all[m1].mean() if m1.any() else float('nan'):>9.3f} "
            f"{m2.sum():>9,} {y_all[m2].mean() if m2.any() else float('nan'):>8.3f}")
    for ci, name in ((0, "priv"), (1, "pub")):
        p = probs[keep, ci]; y = y_all[keep]
        out(f"overall {name}: acc={(np.round(p) == y).mean():.4f}  "
            f"brier={((p - y) ** 2).mean():.4f}")
    dis = keep & ((probs[:, 0] > .5) != (probs[:, 1] > .5))
    if dis.any():
        pr = float((np.round(probs[dis, 0]) == y_all[dis]).mean())
        out(f"priv/pub disagree on {dis.sum():,}/{keep.sum():,} states "
            f"({dis.sum() / keep.sum():.1%}); priv is right on {pr:.3f} of those")

    # ── action classes ranked by |Δpriv| ──────────────────────────────────────
    # Δ for event i = priv(next event's state) - priv(this event's state), the
    # terminal outcome closing each game; signed toward the ACTING seat.
    out("\naction classes by mean |Δpriv| toward the acting seat"
        " (decided games; Δ = value swing the action + its consequences caused):")
    sums: dict = {}
    for gi in np.flatnonzero(decided):
        es = slice(eoff[gi], eoff[gi + 1])
        ev = events[es]
        if len(ev) == 0:
            continue
        pr = probs[soff[gi]:soff[gi + 1], 0]
        yg = 1.0 if games[gi, 1] == 1 else 0.0
        cur = pr[ev[:, 0]]
        nxt = np.append(pr[ev[1:, 0]], yg)
        dp1 = nxt - cur
        for k in range(len(ev)):
            cls = vocab[ev[k, 4]]
            seat = ev[k, 3]                              # the decision-maker
            if cls == "pass" and ev[k, 5] >= 0 and abs(dp1[k]) > 1e-6:
                # a pass with a live stack that moved value: it resolved the top.
                # Credit the swing to the SPELL'S CONTROLLER, not whoever passed.
                cls = f"resolve:{vocab[ev[k, 5]][1:]}"
                if ev[k, 6] >= 0:
                    seat = ev[k, 6]
            toward = dp1[k] if seat == 1 else -dp1[k]
            s = sums.setdefault(cls, [0, 0.0, 0.0])
            s[0] += 1; s[1] += toward; s[2] += abs(dp1[k])
        # NOTE: the last event's delta absorbs the terminal step (win credit).
    rows = [(c, n, tot / n, ab / n) for c, (n, tot, ab) in sums.items() if n >= 50]
    rows.sort(key=lambda r: -r[3])
    out(f"{'class':<34} {'n':>9} {'mean Δ(actor)':>14} {'mean |Δ|':>9}")
    for c, n, mean_t, mean_a in rows[:40]:
        out(f"{c:<34} {n:>9,} {mean_t:>14.4f} {mean_a:>9.4f}")

    # ── late-game surprises ───────────────────────────────────────────────────
    flips = conf_wrong = 0
    for gi in np.flatnonzero(decided):
        pr = probs[soff[gi]:soff[gi + 1], 0]
        st = states[soff[gi]:soff[gi + 1]]
        yg = games[gi, 1] == 1
        late = st[:, 0] >= games[gi, 2] - 2
        if late.any():
            pl = pr[late]
            if ((pl > .5) != yg).any():
                flips += 1
            if ((pl > .9) & (not yg)).any() or ((pl < .1) & yg).any():
                conf_wrong += 1
    out(f"\nlate-game (last 3 turns) priv wrong-side at least once: "
        f"{flips}/{decided.sum()} games; confidently (>0.9) wrong: {conf_wrong}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="checkpoints-v2/latest.pt")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=300)
    ap.add_argument("--workers", type=int, default=0, help="0 = one per P-core")
    ap.add_argument("--pin", action="store_true", help="pin workers to P-cores")
    ap.add_argument("--out", default="critic_watch", help="output .npz basename")
    ap.add_argument("--analyze", default=None, help="skip play; analyze this .npz")
    args = ap.parse_args()

    if args.analyze:
        analyze(dict(np.load(args.analyze, allow_pickle=False)))
        return 0

    import torch
    pl = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cd = pl["config"]
    weights = {n: {k: v.cpu().numpy() for k, v in pl["models"][n].items()}
               for n in ("critic", "public")}
    print(f"[watch] {args.ckpt}: iter={pl.get('done')} elapsed={pl.get('elapsed', 0) / 3600:.0f}h "
          f"critic={cd['encoders']['critic']}/{tuple(cd.get('critic_hidden', ()))} "
          f"public={cd['encoders']['public']}", flush=True)
    del pl

    sys.setrecursionlimit(_RECURSION_LIMIT)
    tasks = [(args.seed + i, args.max_turns) for i in range(args.games)]
    workers = args.workers or len(_PCORE_LPS)
    workers = min(workers, args.games)

    results = []
    t0 = time.time()
    tg = tf = te = 0.0

    def note(r):
        nonlocal tg, tf, te
        results.append(r)
        tg += r.pop("t_game"); tf += r.pop("t_fwd"); te += r.pop("t_enc")
        n = len(results)
        if n % 50 == 0 or n == args.games:
            el = time.time() - t0
            print(f"  {n}/{args.games} | {el / n * 1000:.0f}ms/game wall ({workers}x) | "
                  f"per-game serial: play+enc {tg / n * 1000:.0f}ms "
                  f"(enc {te / n * 1000:.0f}ms) fwd {tf / n * 1000:.0f}ms | "
                  f"eta {el / n * (args.games - n) / 60:.1f}m", flush=True)

    if workers == 1:
        _init_worker(cd, weights)
        for t in tasks:
            note(_play_one(t))
    else:
        import multiprocessing as mp
        pin = mp.Value("i", 0) if args.pin else None
        chunk = max(1, len(tasks) // (workers * 16))
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                 initargs=(cd, weights, pin)) as pool:
            for r in pool.map(_play_one, tasks, chunksize=chunk):
                note(r)

    merged = _merge(results)
    path = f"{args.out}.npz"
    np.savez_compressed(path, **merged)
    print(f"\n[watch] saved {path} ({os.path.getsize(path) / 1e6:.0f} MB)\n", flush=True)
    analyze(merged)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
