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

from fishrl.serve.__main__ import Actions, Store, serve
from fishrl.train.locks import hold_lockfile, unlock


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

        with open(os.path.join(d, "stats.json")) as f:          # live append -> push
            cur = json.load(f)
        cur["reports"].append({"it": 30, "wall_time": time.time()})
        with open(os.path.join(d, "stats.json"), "w") as f:
            json.dump(cur, f)
        deadline = time.time() + 10
        while time.time() < deadline and b'"it": 30' not in got:
            got += s.recv(65536)
        assert b'"it": 30' in got
    finally:
        s.close()


def test_sse_keeps_emitting_from_a_full_ring(site):
    """Regression: ticks.json is a fixed-cap RING -- once full its LENGTH stops
    changing, and the old count-based stream cursor then never emitted another
    tick (dashboards froze on every tick-driven panel while the append-only
    arrays kept flowing). New rows must flow by row identity, without
    replaying the rows that merely rotated."""
    d, port = site
    s = socket.create_connection(("127.0.0.1", port), timeout=15)
    try:
        s.sendall(b"GET /api/stream HTTP/1.1\r\nHost: x\r\n\r\n")
        time.sleep(1.5)                                  # stream takes its baseline
        with open(os.path.join(d, "ticks.json"), "w") as f:   # same length, window +1
            json.dump({"schema": 1, "ticks": [{"it": i} for i in (19, 20, 21)]}, f)
        got = b""
        deadline = time.time() + 10
        while time.time() < deadline and b'"it": 21' not in got:
            got += s.recv(65536)
        assert b'"it": 21' in got
        assert got.count(b"event: tick") == 1            # the NEW row only, no replay
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


def test_wait_lan_ip_polls_past_late_dhcp():
    # --bind auto at boot can race DHCP: loopback first, real address later.
    from fishrl.serve.__main__ import wait_lan_ip
    ips = iter(["127.0.0.1", "127.0.0.1", "192.168.4.35"])
    assert wait_lan_ip(timeout=60.0, poll=0.0,
                       ip_fn=lambda: next(ips), sleep_fn=lambda _s: None) == "192.168.4.35"


def test_wait_lan_ip_gives_up_to_loopback():
    from fishrl.serve.__main__ import wait_lan_ip
    calls = {"n": 0}

    def always_lo():
        calls["n"] += 1
        return "127.0.0.1"

    assert wait_lan_ip(timeout=0.0, poll=0.0,
                       ip_fn=always_lo, sleep_fn=lambda _s: None) == "127.0.0.1"
    assert calls["n"] == 1                       # expired deadline = no spin


# ── peer redirect (relay handoff telemetry) ───────────────────────────────────

def _raw_get(port, path):
    """GET without following redirects (urllib auto-follows 307s)."""
    import http.client
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        c.request("GET", path)
        r = c.getresponse()
        return r.status, dict(r.getheaders()), r.read()
    finally:
        c.close()


def _stamp_peer(d, url="http://192.0.2.7:8765/"):
    with open(os.path.join(d, "peer.json"), "w") as f:
        json.dump({"host": "gaming-pc", "url": url}, f)
    return url


def test_peer_redirect_follows_the_lineage(site, monkeypatch):
    """Released dir + reachable peer -> the data endpoints 307 to the SAME
    path there (query intact), so the website polls one stable address and
    follows the lineage. /api/summary stays local and advertises the peer."""
    import fishrl.serve.__main__ as sm
    d, port = site
    url = _stamp_peer(d)
    monkeypatch.setattr(sm, "peer_alive", lambda _u: True)

    for path in ("/api/reports?since_it=5", "/api/evals", "/api/ticks", "/api/stream"):
        status, headers, _ = _raw_get(port, path)
        assert status == 307
        assert headers["Location"] == url.rstrip("/") + path
        assert headers["Access-Control-Allow-Origin"] == "*"   # EventSource can follow

    s = _get(port, "/api/summary")                       # never redirected
    assert s["peer"]["host"] == "gaming-pc" and s["peer_alive"] is True


def test_peer_redirect_degrades_when_peer_dead_or_trainer_live(site, monkeypatch):
    import fishrl.serve.__main__ as sm
    d, port = site
    _stamp_peer(d)

    monkeypatch.setattr(sm, "peer_alive", lambda _u: False)    # peer asleep/offline
    assert [r["it"] for r in _get(port, "/api/reports")["reports"]] == [10, 20]
    assert _get(port, "/api/summary")["peer_alive"] is False

    monkeypatch.setattr(sm, "peer_alive", lambda _u: True)
    h = hold_lockfile(os.path.join(d, "trainer.lock"))         # WE are the live trainer
    try:
        assert _get(port, "/api/reports")["count"] == 2        # never redirect away
    finally:
        unlock(h)
        h.close()


def test_peer_redirect_disabled_by_flag(site, monkeypatch):
    import fishrl.serve.__main__ as sm
    d, port = site
    _stamp_peer(d)
    monkeypatch.setattr(sm, "peer_alive", lambda _u: True)
    monkeypatch.setattr(sm.Handler, "redirect", False)         # --no-redirect
    assert _get(port, "/api/reports")["count"] == 2


def test_peer_alive_probe_and_cache(site):
    """The real probe: this test's own server is a live peer; a closed port is
    not. Both verdicts cache (TTL) so a dead peer can't tax every request."""
    import fishrl.serve.__main__ as sm
    _, port = site
    sm._peer_probe_cache.clear()
    live, dead = f"http://127.0.0.1:{port}/", "http://127.0.0.1:1/"
    assert sm.peer_alive(live) is True
    assert sm.peer_alive(dead) is False
    assert set(sm._peer_probe_cache) == {live, dead}           # both verdicts cached
    assert sm.peer_alive(dead) is False                        # served from cache


# ── control plane ─────────────────────────────────────────────────────────────

def test_actions_registry_per_platform(tmp_path):
    d = str(tmp_path)
    nt = Actions(d, "fishrl-selfplay", "fishrl-eval.service", os_name="nt")
    assert [a["id"] for a in nt.list()] == ["stop_session", "run_eval_local"]
    px = Actions(d, "fishrl-selfplay", "fishrl-eval.service", os_name="posix")
    assert [a["id"] for a in px.list()] == ["stop_trainer", "start_trainer", "run_eval"]


def test_stop_session_validates_state_and_writes_stop_file(tmp_path):
    d = str(tmp_path)
    a = Actions(d, "u", "e", os_name="nt")
    ok, msg = a.run("stop_session")                      # no trainer -> refused
    assert ok is False and "no trainer" in msg
    assert not os.path.exists(os.path.join(d, "STOP"))

    h = hold_lockfile(os.path.join(d, "trainer.lock"))   # "trainer running"
    try:
        assert a.list()[0]["enabled"] is True
        ok, msg = a.run("stop_session")
        assert ok is True
        assert os.path.exists(os.path.join(d, "STOP"))
    finally:
        unlock(h)
        h.close()
    ok, _ = a.run("bogus")
    assert ok is False


def test_actions_endpoints_disabled_by_default(site):
    _, port = site
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(port, "/api/actions")
    assert e.value.code == 403                           # read-only unless --allow-actions


def test_action_post_roundtrip(tmp_path):
    import threading
    d = str(tmp_path)
    httpd = serve(d, "127.0.0.1", 0, actions=Actions(d, "u", "e", os_name="nt"))
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        assert _get(port, "/api/actions")["actions"][0]["enabled"] is False
        assert _get(port, "/api/summary")["trainer_live"] is False

        h = hold_lockfile(os.path.join(d, "trainer.lock"))
        try:
            assert _get(port, "/api/summary")["trainer_live"] is True
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/action",
                data=json.dumps({"action": "stop_session"}).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=10) as r:
                assert json.loads(r.read())["ok"] is True
            assert os.path.exists(os.path.join(d, "STOP"))
            # GET on the action path must never mutate
            with pytest.raises(urllib.error.HTTPError) as e:
                _get(port, "/api/action")
            assert e.value.code == 404
        finally:
            unlock(h)
            h.close()
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_dashboard_page_served(site):
    _, port = site
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=10) as r:
        html = r.read().decode()
    for marker in ('id="wr"', 'id="sys"', 'id="thr"', 'id="mix"',
                   "/api/stream", "EventSource"):
        assert marker in html
