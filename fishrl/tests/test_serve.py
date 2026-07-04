"""fishrl.serve: idempotent range queries, summary shape, and SSE push.

The server is the website's sync source, so the property that matters most is
that /api/<kind>?since_it=N is a pure function of the files: a missed poll
self-heals, and [0, latest] equals the union of any split. SSE is checked with
a raw socket (replay on connect + a live event when stats.json is replaced)."""
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest

from fishrl.serve.__main__ import Store, serve


@pytest.fixture()
def site(tmp_path):
    d = str(tmp_path)
    with open(os.path.join(d, "stats.json"), "w") as f:
        json.dump({"reports": [{"it": 10, "wall_time": time.time()},
                               {"it": 20, "wall_time": time.time()}],
                   "evals": [{"it": 10, "heuristic": 0.5, "wall_time": time.time()}]}, f)
    with open(os.path.join(d, "ticks.json"), "w") as f:
        json.dump({"schema": 1, "ticks": [{"it": i} for i in (18, 19, 20)]}, f)
    with open(os.path.join(d, "owner.json"), "w") as f:
        json.dump({"host": "pc", "state": "active", "generation": 3}, f)
    httpd = serve(d, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield d, httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


def _get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as r:
        assert r.headers["Access-Control-Allow-Origin"] == "*"
        return json.loads(r.read())


def test_range_queries_idempotent(site):
    _, port = site
    full = [r["it"] for r in _get(port, "/api/reports")["reports"]]
    assert full == [10, 20]
    assert [r["it"] for r in _get(port, "/api/reports?since_it=0")["reports"]] == full
    lo = [r["it"] for r in _get(port, "/api/reports")["reports"] if r["it"] <= 10]
    hi = [r["it"] for r in _get(port, "/api/reports?since_it=10")["reports"]]
    assert lo + hi == full                           # union of a split == the whole
    assert [t["it"] for t in _get(port, "/api/ticks?since_it=18")["ticks"]] == [19, 20]
    assert _get(port, "/api/evals")["count"] == 1
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(port, "/api/nope")
    assert e.value.code == 404


def test_summary_shape(site):
    _, port = site
    s = _get(port, "/api/summary")
    assert s["counts"] == {"reports": 2, "evals": 1, "ticks": 3}
    assert s["last_report"]["it"] == 20 and s["last_tick"]["it"] == 20
    assert s["owner"]["state"] == "active"
    assert s["staleness_s"] is not None and s["staleness_s"] < 60


def test_sse_replay_and_live_event(site):
    d, port = site
    s = socket.create_connection(("127.0.0.1", port), timeout=15)
    try:
        s.sendall(b"GET /api/stream?since_it=15 HTTP/1.1\r\nHost: x\r\n\r\n")
        got = b""
        deadline = time.time() + 10
        while time.time() < deadline and got.count(b"event:") < 4:
            got += s.recv(65536)
        events = sorted(line.split(": ")[1] for line in got.decode().splitlines()
                        if line.startswith("event:"))
        assert events == ["report", "tick", "tick", "tick"]     # it>15: r20 + t18,19,20

        with open(os.path.join(d, "stats.json"), "w") as f:     # live append -> push
            json.dump({"reports": [{"it": 10}, {"it": 20}, {"it": 30}], "evals": []}, f)
        deadline = time.time() + 10
        while time.time() < deadline and b'"it": 30' not in got:
            got += s.recv(65536)
        assert b'"it": 30' in got
    finally:
        s.close()


def test_store_survives_missing_and_garbage_files(tmp_path):
    st = Store(str(tmp_path))
    assert st.stats() == {"reports": [], "evals": []} and st.ticks() == []
    with open(os.path.join(str(tmp_path), "stats.json"), "w") as f:
        f.write("{half a json")
    assert st.stats() == {"reports": [], "evals": []}            # corrupt read = no data
    assert st.summary()["counts"]["reports"] == 0


def test_refuses_wildcard_bind(tmp_path):
    with pytest.raises(SystemExit):
        serve(str(tmp_path), "0.0.0.0", 0)
