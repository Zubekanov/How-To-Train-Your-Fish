"""Backfill parsing of the opponent mix from journald `[status]` / `[league]` lines."""
from fishrl.eval.backfill_stats import parse_league, parse_status

_STATUS = (
    "2026-06-22T10:00:00+10:00 [status 68.82h it=12755 (+152, 151.9/h) T=450421] "
    "pi=-0.022 V=0.583 H=0.488 kl=0.0215 guess=0.311 pub=0.630 | "
    "calib priv(acc=0.78,brier=0.15) pub(acc=0.63,brier=0.22) gmae=0.18 | "
    "opp trained=0.92 | WR via eval timer")
_LEAGUE = (
    "2026-06-22T10:00:00+10:00 [league it=12755] games=1216 trained=0.92 "
    "(self=912 past=205 heuristic=89 attacker=5 random=5) | heuristic=0.22(2057)")
# An old line from before the opponent mix was logged at all.
_OLD = (
    "2026-06-01T10:00:00+10:00 [status 1.0h it=10 (+5, 5.0/h) T=100] "
    "pi=-0.02 V=0.5 H=0.5 kl=0.02 guess=0.3 pub=0.6 | "
    "calib priv(acc=0.8,brier=0.1) pub(acc=0.7,brier=0.2) gmae=0.2 | WR via eval timer")


def test_parse_league_shares_sum_to_one():
    s = parse_league([_LEAGUE])[12755]
    assert abs(s["opp_self"] - 912 / 1216) < 1e-9
    assert abs(s["opp_past"] - 205 / 1216) < 1e-9
    assert abs(s["opp_heuristic"] - 89 / 1216) < 1e-9
    assert abs(s["opp_attacker"] - 5 / 1216) < 1e-9
    assert abs(s["opp_random"] - 5 / 1216) < 1e-9
    assert abs(sum(s.values()) - 1.0) < 1e-9
    assert abs((s["opp_self"] + s["opp_past"]) - 0.92) < 5e-3  # == opp_trained


def test_parse_status_merges_league_mix():
    r = parse_status([_STATUS], parse_league([_LEAGUE]))[0][0]
    assert r["it"] == 12755 and r["source"] == "journald"
    assert r["opp_trained"] == 0.92
    assert abs(r["opp_self"] - 912 / 1216) < 1e-9
    assert abs(r["opp_past"] - 205 / 1216) < 1e-9


def test_status_without_league_keeps_trained_only():
    r = parse_status([_STATUS], {})[0][0]            # no [league] line available
    assert r["opp_trained"] == 0.92                  # still read from the status line
    assert all(r[k] is None for k in
               ("opp_self", "opp_past", "opp_heuristic", "opp_attacker", "opp_random"))


def test_old_status_has_no_opp_fields():
    r = parse_status([_OLD], {})[0][0]
    assert r["opp_trained"] is None
    assert r["opp_self"] is None
