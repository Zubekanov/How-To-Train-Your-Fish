"""Backfill parsing of the opponent mix from journald `[status]` / `[league]` /
`[scenario]` lines, and the float-token edge cases ('nan'/'inf')."""
from fishrl.eval.backfill_stats import parse_evals, parse_league, parse_scenarios, parse_status

_STATUS = (
    "2026-06-22T10:00:00+10:00 [status 68.82h it=12755 (+152, 151.9/h) T=450421] "
    "pi=-0.022 V=0.583 H=0.488 kl=0.0215 guess=0.311 pub=0.630 | "
    "calib priv(acc=0.78,brier=0.15) pub(acc=0.63,brier=0.22) gmae=0.18 | "
    "opp trained=0.92 | WR via eval timer")
_LEAGUE = (
    "2026-06-22T10:00:00+10:00 [league it=12755] games=1216 trained=0.92 "
    "(self=912 past=205 heuristic=89 attacker=5 random=5) | heuristic=0.22(2057)")
# Scenario-curriculum totals for the same window, in both logged formats.
_SCENARIO_NEW = (
    "2026-06-22T10:00:00+10:00 [scenario it=12755] games=304 (survive_lethal=200 "
    "board_presence=104) | wr survive_lethal=0.31(200) board_presence=0.55(104)")
_SCENARIO_OLD = (
    "2026-06-22T10:00:00+10:00 [scenario debug it=12755] games=304 "
    "mix={'board_presence': 104, 'survive_lethal': 200} (judge progress on "
    "vs-heuristic WR, not this)")
# An old line from before the opponent mix was logged at all.
_OLD = (
    "2026-06-01T10:00:00+10:00 [status 1.0h it=10 (+5, 5.0/h) T=100] "
    "pi=-0.02 V=0.5 H=0.5 kl=0.02 guess=0.3 pub=0.6 | "
    "calib priv(acc=0.8,brier=0.1) pub(acc=0.7,brier=0.2) gmae=0.2 | WR via eval timer")
# A blown-up window: the trainer's f-strings print non-finite floats as nan/inf/-inf.
_INF = (
    "2026-06-23T10:00:00+10:00 [status 70.0h it=12900 (+145, 145.0/h) T=460000] "
    "pi=-0.020 V=inf H=0.490 kl=nan guess=0.310 pub=-inf | "
    "calib priv(acc=0.78,brier=0.15) pub(acc=0.63,brier=0.22) gmae=0.18 | "
    "opp trained=0.92 | WR via eval timer")
_EVAL_INF = (
    "2026-06-23T10:00:00+10:00 [eval it=12900 @70.0h n=100 w=6 took=87.8s] "
    "WR frozen@12755=inf random=0.940 attacker=0.750 heuristic=0.070")
# The newest status format: clip fraction + brier gap + the window game telemetry.
_STATUS_V3 = (
    "2026-07-02T10:00:00+10:00 [status 312.0h it=58500 (+180, 180.0/h) T=470000] "
    "pi=-0.020 V=0.583 H=0.488 kl=0.0215 clip=0.08 guess=0.311 pub=0.630 | "
    "calib priv(acc=0.78,brier=0.15) pub(acc=0.63,brier=0.22) gap=0.07 gmae=0.18 | "
    "opp trained=0.92 | games=128 len=38.2 slen=14.1 trunc=0.05 draw=0.01 "
    "fatk_p1=1 fatk_p2=0 seat_p1=0.52 fdec=0.24 | wall collect=0.91 | WR via eval timer")


def test_parse_league_shares_sum_to_one():
    s = parse_league([_LEAGUE])[12755]
    assert abs(s["opp_self"] - 912 / 1216) < 1e-9
    assert abs(s["opp_past"] - 205 / 1216) < 1e-9
    assert abs(s["opp_heuristic"] - 89 / 1216) < 1e-9
    assert s["opp_heuristic11"] is None              # pre-split line: v1.1 didn't exist
    assert abs(s["opp_attacker"] - 5 / 1216) < 1e-9
    assert abs(s["opp_random"] - 5 / 1216) < 1e-9
    assert s["opp_scenario"] is None                 # no scenario line -> unknown, not 0
    assert abs(sum(v for v in s.values() if v is not None) - 1.0 - s["opp_trained"]) < 1e-9
    assert abs(s["opp_trained"] - (912 + 205) / 1216) < 1e-9


# July 2026: the h11 token after the paren block splits the merged heuristic
# count into v1.0 (paren count minus h11) and v1.1 (h11).
_LEAGUE_H11 = (
    "2026-07-02T21:00:00+10:00 [league it=60900] games=1200 trained=0.87 "
    "(self=820 past=245 heuristic=110 attacker=15 random=10) h11=30 | "
    "heuristic_1_1=0.09(180) heuristic=0.14(2300)")


def test_parse_league_splits_heuristic_versions():
    s = parse_league([_LEAGUE_H11])[60900]
    assert abs(s["opp_heuristic"] - 80 / 1200) < 1e-9     # merged 110 minus v1.1's 30
    assert abs(s["opp_heuristic11"] - 30 / 1200) < 1e-9
    shares = ("opp_self", "opp_past", "opp_heuristic", "opp_heuristic11",
              "opp_attacker", "opp_random")
    assert abs(sum(s[k] for k in shares) - 1.0) < 1e-9


def test_parse_scenarios_both_formats():
    assert parse_scenarios([_SCENARIO_NEW]) == {12755: 304}
    assert parse_scenarios([_SCENARIO_OLD]) == {12755: 304}


def test_parse_league_renormalizes_by_grand_total_with_scenarios():
    """When the window also ran scenario games, the live writer divides every opp_* share
    by the GRAND total (league + scenario) and emits opp_scenario — the backfill must
    match, or the same iteration gets incomparable shares from the two writers."""
    grand = 1216 + 304
    for scen_line in (_SCENARIO_NEW, _SCENARIO_OLD):
        s = parse_league([_LEAGUE], parse_scenarios([scen_line]))[12755]
        assert abs(s["opp_self"] - 912 / grand) < 1e-9
        assert abs(s["opp_past"] - 205 / grand) < 1e-9
        assert abs(s["opp_heuristic"] - 89 / grand) < 1e-9
        assert abs(s["opp_scenario"] - 304 / grand) < 1e-9
        assert abs(s["opp_trained"] - (912 + 205) / grand) < 1e-9
        assert abs(sum(v for v in s.values() if v is not None) - 1.0 - s["opp_trained"]) < 1e-9


def test_parse_status_v3_game_telemetry_and_old_lines_still_parse():
    # New tokens (clip=, gap=, the games segment) parse into the report row...
    r = parse_status([_STATUS_V3])[0][0]
    assert r["it"] == 58500
    assert r["clip_frac"] == 0.08 and r["brier_gap"] == 0.07
    assert r["games"] == 128 and r["dec_per_game"] == 38.2 and r["scen_dec_per_game"] == 14.1
    assert r["trunc_rate"] == 0.05 and r["draw_rate"] == 0.01
    assert r["freeatk_p1"] == 1 and r["freeatk_p2"] == 0
    assert r["mirror_p1_wr"] == 0.52 and r["forced_dec_frac"] == 0.24
    assert r["collect_frac"] == 0.91
    # ...and both older formats still parse, with the new fields absent/None.
    for old_line in (_STATUS, _OLD):
        r = parse_status([old_line])[0][0]
        assert r["clip_frac"] is None and r["brier_gap"] is None
        assert "games" not in r


def test_parse_status_merges_league_mix():
    r = parse_status([_STATUS], parse_league([_LEAGUE]))[0][0]
    assert r["it"] == 12755 and r["source"] == "journald"
    # league counts win over the 2-dp status value (same number, full precision)
    assert abs(r["opp_trained"] - (912 + 205) / 1216) < 1e-9
    assert abs(r["opp_self"] - 912 / 1216) < 1e-9
    assert abs(r["opp_past"] - 205 / 1216) < 1e-9
    assert r["opp_scenario"] is None


def test_parse_status_scenario_window_matches_live_writer():
    league = parse_league([_LEAGUE], parse_scenarios([_SCENARIO_NEW]))
    r = parse_status([_STATUS], league)[0][0]
    grand = 1216 + 304
    assert abs(r["opp_self"] - 912 / grand) < 1e-9
    assert abs(r["opp_scenario"] - 304 / grand) < 1e-9
    assert abs(r["opp_trained"] - (912 + 205) / grand) < 1e-9


def test_status_without_league_keeps_trained_only():
    r = parse_status([_STATUS], {})[0][0]            # no [league] line available
    assert r["opp_trained"] == 0.92                  # still read from the status line
    assert all(r[k] is None for k in
               ("opp_self", "opp_past", "opp_heuristic", "opp_attacker", "opp_random",
                "opp_scenario"))


def test_old_status_has_no_opp_fields():
    r = parse_status([_OLD], {})[0][0]
    assert r["opp_trained"] is None
    assert r["opp_self"] is None


def test_status_with_inf_and_nan_still_parses():
    """The float class must MATCH 'inf'/-inf/nan (a miss drops the whole row); the
    values themselves become None, same as the live writer's non-finite -> null."""
    rows, _ = parse_status([_INF], {})
    assert len(rows) == 1
    r = rows[0]
    assert r["it"] == 12900
    assert r["critic_loss"] is None                  # inf
    assert r["approx_kl"] is None                    # nan
    assert r["public_loss"] is None                  # -inf
    assert r["policy_loss"] == -0.020                # finite neighbours unharmed


def test_eval_with_inf_still_parses():
    recs = parse_evals([_EVAL_INF])
    assert len(recs) == 1
    assert recs[0]["frozen"] is None
    assert recs[0]["random"] == 0.94
