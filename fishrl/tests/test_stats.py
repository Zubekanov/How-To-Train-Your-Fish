"""Tests for the shared stats.json time-series writer."""
import json
import os

from fishrl.train import stats as stats_io


def test_append_accumulates_into_two_arrays(tmp_path):
    d = str(tmp_path)
    stats_io.append_report(d, {"it": 1, "policy_loss": 0.1})
    stats_io.append_eval(d, {"it": 1, "heuristic": 0.4})
    stats_io.append_report(d, {"it": 2, "policy_loss": 0.2})
    with open(stats_io.stats_path(d)) as f:
        data = json.load(f)
    assert data["schema"] == stats_io.SCHEMA
    assert [r["it"] for r in data["reports"]] == [1, 2]
    assert [r["it"] for r in data["evals"]] == [1]
    assert data["evals"][0]["heuristic"] == 0.4


def test_corrupt_file_is_reset_not_fatal(tmp_path):
    d = str(tmp_path)
    with open(stats_io.stats_path(d), "w") as f:
        f.write("{ this is not json")
    stats_io.append_report(d, {"it": 7})        # must not raise; starts a fresh skeleton
    with open(stats_io.stats_path(d)) as f:
        data = json.load(f)
    assert [r["it"] for r in data["reports"]] == [7]
    assert data["evals"] == []


def test_merge_fill_enriches_existing_rows_without_overwriting(tmp_path):
    d = str(tmp_path)
    stats_io.append_report(d, {"it": 1, "opp_trained": None, "source": "live"})
    stats_io.append_report(d, {"it": 2, "opp_trained": 0.5, "source": "live"})  # already set
    added = stats_io.merge(
        d, reports=[{"it": 1, "opp_trained": 0.9, "source": "journald"},
                    {"it": 2, "opp_trained": 0.1, "source": "journald"},
                    {"it": 3, "opp_trained": 0.7, "source": "journald"}],
        replace_sources={"journald"}, fill=("opp_trained",))
    with open(stats_io.stats_path(d)) as f:
        rows = {r["it"]: r for r in json.load(f)["reports"]}
    assert rows[1]["opp_trained"] == 0.9          # None -> filled from journald
    assert rows[1]["source"] == "live"            # row identity preserved
    assert rows[2]["opp_trained"] == 0.5          # already set -> untouched
    assert rows[3]["opp_trained"] == 0.7          # genuinely new row added
    assert added["reports"] == {"added": 1, "filled": 1, "total": 3}


def test_no_leftover_tmp(tmp_path):
    d = str(tmp_path)
    stats_io.append_report(d, {"it": 1})
    assert not os.path.exists(stats_io.stats_path(d) + ".tmp")
