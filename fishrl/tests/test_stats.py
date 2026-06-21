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


def test_no_leftover_tmp(tmp_path):
    d = str(tmp_path)
    stats_io.append_report(d, {"it": 1})
    assert not os.path.exists(stats_io.stats_path(d) + ".tmp")


def test_add_annotation_is_idempotent(tmp_path):
    d = str(tmp_path)
    assert stats_io.add_annotation(d, 9508, "added heuristic to the training pool") is True
    assert stats_io.add_annotation(d, 9508, "added heuristic to the training pool") is False
    with open(stats_io.stats_path(d)) as f:
        ann = json.load(f)["annotations"]
    assert len(ann) == 1
    assert ann[0]["it"] == 9508 and ann[0]["kind"] == "n.b."


def test_annotation_survives_merge(tmp_path):
    d = str(tmp_path)
    stats_io.add_annotation(d, 9508, "x")
    stats_io.merge(d, reports=[{"it": 1, "source": "journald"}])   # merge must not drop it
    with open(stats_io.stats_path(d)) as f:
        data = json.load(f)
    assert len(data["annotations"]) == 1 and len(data["reports"]) == 1
