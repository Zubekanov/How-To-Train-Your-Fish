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


def test_merge_evals_dedupe_on_it_and_wall_time(tmp_path):
    """Evals dedupe on (it, wall_time): a re-eval of the same iteration at a later time
    is a distinct datapoint and must be kept; an exact (it, wall_time) match is a true
    duplicate and is dropped (existing wins)."""
    d = str(tmp_path)
    stats_io.append_eval(d, {"it": 5, "wall_time": 100.0, "heuristic": 0.4, "source": "eval"})
    added = stats_io.merge(
        d, evals=[{"it": 5, "wall_time": 100.0, "heuristic": 0.9, "source": "journald"},  # dup
                  {"it": 5, "wall_time": 200.0, "heuristic": 0.5, "source": "journald"}])  # re-eval
    with open(stats_io.stats_path(d)) as f:
        evals = json.load(f)["evals"]
    assert added["evals"]["added"] == 1
    assert [(e["it"], e["wall_time"], e["heuristic"]) for e in evals] == [
        (5, 100.0, 0.4),                             # existing row won over the duplicate
        (5, 200.0, 0.5),                             # same it, new time -> kept
    ]


def test_merge_drops_duplicates_within_incoming_batch(tmp_path):
    d = str(tmp_path)
    added = stats_io.merge(
        d,
        reports=[{"it": 1, "policy_loss": 0.1}, {"it": 1, "policy_loss": 0.2}],
        evals=[{"it": 3, "wall_time": 50.0}, {"it": 3, "wall_time": 50.0},
               {"it": 3, "wall_time": 60.0}])
    with open(stats_io.stats_path(d)) as f:
        data = json.load(f)
    assert added["reports"]["added"] == 1            # second it=1 report dropped in-batch
    assert data["reports"][0]["policy_loss"] == 0.1  # first occurrence wins
    assert added["evals"]["added"] == 2              # exact (it, wall_time) dup dropped
    assert [(e["it"], e["wall_time"]) for e in data["evals"]] == [(3, 50.0), (3, 60.0)]


def test_no_leftover_tmp(tmp_path):
    d = str(tmp_path)
    stats_io.append_report(d, {"it": 1})
    assert not os.path.exists(stats_io.stats_path(d) + ".tmp")
