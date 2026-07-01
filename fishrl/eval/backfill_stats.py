"""Back-populate stats.json from the journald history.

stats.json only started being written from the commit that added it, but the trainer and eval
service have been logging `[status ...]` / `[eval ...]` lines to journald the whole time. This
script parses those lines back into the SAME record schema the live writers use and merges them
into stats.json, deduped by iteration (existing live rows win), so the time-series is complete
back to the start of journald retention.

It is idempotent (re-running adds nothing new) and lock-safe (uses stats.merge, the same flock +
atomic write as the live appenders), so it is safe to run while the services are up.

    python -m fishrl.eval.backfill_stats --ckpt-dir checkpoints
    python -m fishrl.eval.backfill_stats --selfplay-unit fishrl-selfplay --eval-unit fishrl-eval

The journald timestamp of each line becomes the record's `wall_time`; backfilled rows carry
`"source": "journald"` so they are distinguishable from live rows.
"""
from __future__ import annotations

import argparse
import re
import subprocess
from datetime import datetime

from fishrl.train import stats as stats_io

# `[status 46.33h it=9358 (+151, 150.0/h) T=497782] pi=-0.022 V=0.583 H=0.494 kl=0.0254
#  guess=0.311 pub=0.637 | calib priv(acc=0.83,brier=0.14) pub(acc=0.76,brier=0.19) gmae=0.18`
# NB: the float class [-\d.naif]+ must cover nan AND inf ('i' included) — the trainer's
# f-strings print non-finite losses as 'nan'/'inf', and a class that can't match them makes
# the whole line silently unparseable (the row would be dropped, not just the field).
_STATUS = re.compile(
    r"\[status\s+(?P<tag>[\d.]+h|final)\s+it=(?P<it>\d+)\s+\(\+(?P<iters>\d+),\s+"
    r"(?P<iph>[\d.]+)/h\)\s+T=(?P<T>\d+)\]\s+"
    r"pi=(?P<pi>[-\d.naif]+)\s+V=(?P<V>[-\d.naif]+)\s+H=(?P<H>[-\d.naif]+)\s+kl=(?P<kl>[-\d.naif]+)\s+"
    r"guess=(?P<guess>[-\d.naif]+)\s+pub=(?P<pub>[-\d.naif]+)\s+\|\s+calib\s+"
    r"priv\(acc=(?P<pacc>[-\d.naif]+),brier=(?P<pbri>[-\d.naif]+)\)\s+"
    r"pub\(acc=(?P<uacc>[-\d.naif]+),brier=(?P<ubri>[-\d.naif]+)\)\s+gmae=(?P<gmae>[-\d.naif]+)")

# optional inline win-rate tail (older lines had it; newer say "WR via eval timer"). Captures the
# panel size + eval seconds too, so the inline panel becomes a proper eval-array row.
_STATUS_WR = re.compile(
    r"WR\s+frozen@(?P<fat>\d+)=(?P<frozen>[-\d.naif]+)\s+random=(?P<rand>[-\d.naif]+)\s+"
    r"attacker=(?P<att>[-\d.naif]+)\s+heuristic=(?P<heu>[-\d.naif]+)\s+"
    r"\(n=(?P<n>\d+),\s+eval\s+(?P<es>[\d.]+)s\)")

# Opponent-mix tail of a `[status ...]` line: `... gmae=0.18 | opp trained=0.92 | ...`. Newer
# lines carry it; older ones don't (-> opp_trained stays None).
_OPP = re.compile(r"opp trained=(?P<opp>[\d.]+)")

# `[league it=12755] games=1216 trained=0.92 (self=912 past=205 heuristic=89 attacker=5
#  random=5) | <per-opponent win-rate table>` -- the per-category counts behind opp_trained.
# Divided by the GRAND total (league games + same-window scenario games, see _SCENARIO) into
# the same opp_* shares the website stores and the live writer emits.
_LEAGUE = re.compile(
    r"\[league\s+it=(?P<it>\d+)\]\s+games=(?P<games>\d+)\s+trained=[\d.]+\s+"
    r"\(self=(?P<self>\d+)\s+past=(?P<past>\d+)\s+heuristic=(?P<heu>\d+)\s+"
    r"attacker=(?P<att>\d+)\s+random=(?P<rand>\d+)\)")

# Scenario-curriculum games this window, in either era's format:
#   new: `[scenario it=9358] games=37 (survive_lethal=20 ...) | wr ...`
#   old: `[scenario debug it=9358] games=37 mix={'board_presence': 20, ...} (judge ...)`
# Only the total matters: the live writer normalizes opp_* by league+scenario games and
# emits opp_scenario = scen_total/grand, so the backfill must renormalize the same way.
_SCENARIO = re.compile(r"\[scenario(?:\s+debug)?\s+it=(?P<it>\d+)\]\s+games=(?P<games>\d+)")

# `[eval it=2230 @9.20h n=100 w=6 took=87.8s] WR frozen@2034=0.540 random=0.940
#  attacker=0.750 heuristic=0.070  *** NEW BEST ...`
_EVAL = re.compile(
    r"\[eval\s+it=(?P<it>\d+)\s+@(?P<eh>[\d.]+)h\s+n=(?P<n>\d+)\s+w=(?P<w>\d+)\s+"
    r"took=(?P<took>[\d.]+)s\]\s+WR\s+frozen@(?P<fat>\d+)=(?P<frozen>[-\d.naif]+)\s+"
    r"random=(?P<rand>[-\d.naif]+)\s+attacker=(?P<att>[-\d.naif]+)\s+heuristic=(?P<heu>[-\d.naif]+)")


def _f(s):
    """Parse a logged float; 'nan'/'inf' -> None so the JSON stays valid (matches the live path)."""
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return v if v == v and v not in (float("inf"), float("-inf")) else None


def _ts(line: str):
    """Epoch seconds from a leading short-iso journald timestamp (`2026-06-19T10:17:48+10:00`)."""
    try:
        return datetime.fromisoformat(line.split(" ", 1)[0]).timestamp()
    except (ValueError, IndexError):
        return None


def _journal(unit: str) -> list:
    out = subprocess.run(["journalctl", "-u", unit, "--no-pager", "-o", "short-iso"],
                         capture_output=True, text=True, check=True)
    return out.stdout.splitlines()


_OPP_KEYS = ("opp_self", "opp_past", "opp_heuristic", "opp_attacker", "opp_random",
             "opp_scenario")


def parse_scenarios(lines) -> dict:
    """Map iteration -> total scenario-curriculum games from `[scenario ...]` lines
    (both the old `[scenario debug it=...]` and the new `[scenario it=...]` formats)."""
    out = {}
    for line in lines:
        m = _SCENARIO.search(line)
        if m:
            out[int(m.group("it"))] = int(m.group("games"))
    return out


def parse_league(lines, scenarios=None) -> dict:
    """Map iteration -> opponent-mix shares from `[league ...]` lines.

    Counts are divided by the GRAND total for the window — league games plus any
    same-iteration scenario games from `scenarios` (parse_scenarios) — with
    opp_scenario = scen/grand, and opp_trained recomputed as (self+past)/grand,
    matching the live writer's normalization. With no scenario line for the window
    (pre-curriculum history), grand == league games and the shares are unchanged."""
    scenarios = scenarios or {}
    out = {}
    for line in lines:
        m = _LEAGUE.search(line)
        if not m:
            continue
        it = int(m.group("it"))
        scen = scenarios.get(it)
        grand = (int(m.group("games")) + (scen or 0)) or 1
        out[it] = {
            "opp_trained": (int(m.group("self")) + int(m.group("past"))) / grand,
            "opp_self": int(m.group("self")) / grand,
            "opp_past": int(m.group("past")) / grand,
            "opp_heuristic": int(m.group("heu")) / grand,
            "opp_attacker": int(m.group("att")) / grand,
            "opp_random": int(m.group("rand")) / grand,
            "opp_scenario": scen / grand if scen is not None else None,
        }
    return out


def parse_status(lines, league=None) -> tuple:
    """Parse `[status ...]` lines into (reports, inline_evals). Reports are PURE trainer metrics
    (plus the opponent mix); where a line carried an inline win-rate panel, that panel is emitted
    as a separate eval-array row (so win-rates live only in "evals", same as the live schema).
    `league` maps it -> opp_* shares from parse_league() (grand-total-normalized, incl. the
    recomputed opp_trained); rows with no league line keep the status line's own opp_trained
    (league-normalized, but grand == league whenever no scenario games ran) and None shares."""
    league = league or {}
    reports, inline_evals = [], []
    for line in lines:
        m = _STATUS.search(line)
        if not m:
            continue
        tag = m.group("tag")
        it = int(m.group("it"))
        elapsed_h = None if tag == "final" else float(tag[:-1])
        ts = _ts(line)
        o = _OPP.search(line)
        rec = {
            "it": it, "elapsed_h": elapsed_h, "wall_time": ts,
            "iters": int(m.group("iters")), "iters_per_h": _f(m.group("iph")),
            "transitions": int(m.group("T")),
            "policy_loss": _f(m.group("pi")), "critic_loss": _f(m.group("V")),
            "entropy": _f(m.group("H")), "approx_kl": _f(m.group("kl")),
            "guesser_loss": _f(m.group("guess")), "public_loss": _f(m.group("pub")),
            "priv_acc": _f(m.group("pacc")), "pub_acc": _f(m.group("uacc")),
            "priv_brier": _f(m.group("pbri")), "pub_brier": _f(m.group("ubri")),
            "gmae": _f(m.group("gmae")),
            "opp_trained": _f(o.group("opp")) if o else None,
        }
        rec.update(league.get(it, {k: None for k in _OPP_KEYS}))
        rec["source"] = "journald"
        reports.append(rec)
        w = _STATUS_WR.search(line)
        if w:                                            # older inline-panel era -> an eval row
            inline_evals.append({
                "it": it, "frozen_at": int(w.group("fat")), "elapsed_h": elapsed_h,
                "wall_time": ts, "n": int(w.group("n")), "workers": 1,
                "took_s": _f(w.group("es")), "frozen": _f(w.group("frozen")),
                "random": _f(w.group("rand")), "attacker": _f(w.group("att")),
                "heuristic": _f(w.group("heu")), "new_best": False,
                "source": "journald-inline",
            })
    return reports, inline_evals


def parse_evals(lines) -> list:
    recs = []
    for line in lines:
        m = _EVAL.search(line)
        if not m:
            continue
        recs.append({
            "it": int(m.group("it")), "frozen_at": int(m.group("fat")),
            "elapsed_h": float(m.group("eh")), "wall_time": _ts(line),
            "n": int(m.group("n")), "workers": int(m.group("w")), "took_s": _f(m.group("took")),
            "frozen": _f(m.group("frozen")), "random": _f(m.group("rand")),
            "attacker": _f(m.group("att")), "heuristic": _f(m.group("heu")),
            "new_best": "NEW BEST" in line, "source": "journald",
        })
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(description="Back-populate stats.json from journald logs.")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--selfplay-unit", default="fishrl-selfplay")
    ap.add_argument("--eval-unit", default="fishrl-eval")
    args = ap.parse_args()

    selfplay = _journal(args.selfplay_unit)
    reports, inline_evals = parse_status(
        selfplay, parse_league(selfplay, parse_scenarios(selfplay)))
    evals = parse_evals(_journal(args.eval_unit)) + inline_evals
    print(f"[backfill] parsed {len(reports)} status lines "
          f"({len(inline_evals)} with inline win-rates), {len(evals)} eval rows total")
    # Regenerate the backfill-owned rows (so parser fixes re-apply) and stamp a source on any
    # live rows written before the source field existed; never touch other live rows.
    added = stats_io.merge(
        args.ckpt_dir, reports=reports, evals=evals,
        replace_sources={"journald", "journald-inline"},
        live_source={"reports": "live", "evals": "eval"},
        fill=("opp_trained", *_OPP_KEYS))     # back-populate the opponent mix onto live rows
    print(f"[backfill] merged into {stats_io.stats_path(args.ckpt_dir)}: "
          f"reports {added['reports']['total']} (+{added['reports']['added']} backfilled, "
          f"{added['reports']['filled']} enriched), "
          f"evals {added['evals']['total']} (+{added['evals']['added']} backfilled); "
          f"live rows preserved")


if __name__ == "__main__":
    main()
