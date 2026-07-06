"""Relay ownership state machine: every interrupted handoff must resolve to at
most one owner (possibly zero -- recoverable), never two. These are the exact
transitions transfer export/import and the trainer startup drive."""
from fishrl.train import ownership as o


def test_unstamped_dir_autoclaims(tmp_path):
    d = str(tmp_path)
    assert o.read(d) is None
    assert o.check(d, "pc") is None                  # pre-relay back-compat path
    s = o.read(d)
    assert s == {"host": "pc", "state": "active", "generation": 0, "ts": s["ts"]}


def test_active_owner_ok_other_host_refused(tmp_path):
    d = str(tmp_path)
    o.claim(d, "odroid")
    assert o.check(d, "odroid") is None
    r = o.check(d, "pc")
    assert r is not None and "owned by 'odroid'" in r and "--claim" in r


def test_release_refuses_everyone_and_bumps_generation(tmp_path):
    d = str(tmp_path)
    o.claim(d, "odroid")
    s = o.release(d)
    assert s["state"] == "released" and s["generation"] == 1
    for host in ("odroid", "pc"):                    # even the old owner must not resume
        r = o.check(d, host)
        assert r is not None and "exported" in r


def test_full_relay_round_trip(tmp_path):
    """ODROID exports (gen 0->1), PC imports+claims at 1, PC exports (1->2),
    ODROID imports+claims at 2 -- the generation walks the handoffs."""
    src, dst = str(tmp_path / "odroid"), str(tmp_path / "pc")
    o.claim(src, "odroid")
    gen1 = o.release(src)["generation"]              # export on the odroid
    o.claim(dst, "pc", generation=gen1)              # import on the pc
    assert o.check(dst, "pc") is None and o.check(src, "odroid") is not None
    gen2 = o.release(dst)["generation"]              # export back
    assert gen2 == gen1 + 1
    o.claim(src, "odroid", generation=gen2)          # import home
    assert o.check(src, "odroid") is None
    assert o.read(src)["generation"] == 2


def test_interrupted_handoff_leaves_nobody_then_claim_recovers(tmp_path):
    d = str(tmp_path)
    o.claim(d, "odroid")
    o.release(d)                                     # export cut; zip lost in transit
    assert o.check(d, "odroid") is not None          # nobody owns: fail toward idle
    o.claim(d, "odroid")                             # the explicit human recovery
    assert o.check(d, "odroid") is None
    assert o.read(d)["generation"] == 1              # claim asserts a turn, never bumps


def test_claim_preserves_generation_by_default(tmp_path):
    d = str(tmp_path)
    o.claim(d, "pc", generation=7)
    o.release(d)
    s = o.claim(d, "pc")
    assert s["generation"] == 8                      # released bumped it; claim keeps it


def test_peer_stamp_round_trip_and_claim_clears(tmp_path):
    """peer.json is the telemetry redirect breadcrumb the relay leaves on the
    released side; taking the turn (claim, i.e. any import) must clear it --
    an active owner IS the telemetry source, pointing away would loop."""
    d = str(tmp_path)
    assert o.read_peer(d) is None
    o.write_peer(d, "pc", "http://192.168.4.99:8765/")
    p = o.read_peer(d)
    assert p["host"] == "pc" and p["url"] == "http://192.168.4.99:8765/" and p["ts"]
    o.claim(d, "odroid")
    assert o.read_peer(d) is None
    o.clear_peer(d)                                  # idempotent on an absent file
