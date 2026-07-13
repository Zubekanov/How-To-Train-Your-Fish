"""LAN-only telemetry webserver over a fishrl checkpoint dir.

    python -m fishrl.serve --ckpt-dir checkpoints                 # binds the LAN IP, :8765
    python -m fishrl.serve --ckpt-dir checkpoints --bind 127.0.0.1 --port 9000

A SIBLING process to the trainer, never embedded in it:
it only ever open/read/closes files the trainer replaces atomically
(stats.json, ticks.json, best.json, owner.json, latest.pt mtime), so it can
be killed, restarted, or wedged without any effect on training, and it
survives trainer restarts across relay handoffs. Stdlib only; no torch.

Two channels, deliberately distinct:

  * durable pull -- ``/api/reports|evals|ticks?since_it=N``: pure idempotent
    range queries keyed on ``it``. A missed poll self-heals on the next one
    and the whole downstream DB rebuilds from ``since_it=0`` (or no param).
    The training machine is the source of truth; the puller keeps a replica.
  * live push -- ``/api/stream``: server-sent events, one event per new
    report/eval/tick row (+ best.pt rolls), driven by a 1s file-mtime poll.
    ``Last-Event-ID`` (or ``?since_it=N``) replays missed rows on reconnect.

Endpoints: /api/summary (latest of everything + staleness + owner), the three
range queries, /api/stream, /api/archives + /archives/<file> (the permanent
per-10k-iteration checkpoints -- see below), and / (a minimal human status
page). Responses set ``Access-Control-Allow-Origin: *`` so a browser frontend
served from another LAN host (the website) can fetch/stream directly.

Archives: ``archive_########.pt`` files are written every 10k iterations and
never pruned. They are PER-HOST -- the relay handoff zip does not carry them,
so each machine holds the ones written during its own sessions -- which is why
the archive routes serve the LOCAL set and are exempt from the peer redirect
(redirecting would hide this host's archives, not find more).

Peer redirect: after a relay handoff the lineage's live telemetry lives on the
OTHER machine, and the relay leaves ``peer.json`` (the new owner's dashboard
URL) behind -- see fishrl.train.ownership. When no trainer is live here and
the peer answers a cached reachability probe, the data endpoints
(/api/reports|evals|ticks|stream) answer 307 to the same path there, so the
website keeps polling ONE stable address (this server) and follows the
lineage wherever it trains. An unreachable peer (PC asleep) degrades to
serving the local rows -- stale but valid; the idempotent since_it queries
self-heal once the lineage returns. /api/summary is NEVER redirected: it
reports this machine's state and advertises the peer (+ liveness) so a
client can also re-point itself explicitly. ``--no-redirect`` disables.

Binding: NEVER 0.0.0.0. ``--bind auto`` (default) resolves the machine's
primary LAN IPv4; pass an explicit address to override.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from fishrl.serve.page import PAGE
from fishrl.train.locks import is_locked

STREAM_POLL_S = 1.0
HEARTBEAT_S = 15.0
TRAINER_LOCK = "trainer.lock"                          # fishrl.train.ownership's name
STOP_FILE = "STOP"                                     # train_loop's stop-file protocol
PEER = "peer.json"                                     # fishrl.train.ownership's name
PEER_PROBE_TIMEOUT_S = 2.0
PEER_PROBE_TTL_S = 15.0
# The ONLY files the download route will serve: the permanent archives. Anchored
# full-match on the basename = no traversal, no latest.pt/best.pt exposure.
ARCHIVE_RE = re.compile(r"^archive_(\d{8})\.pt$")

_peer_probe_cache: dict = {}                           # url -> (monotonic ts, alive)


def peer_alive(url: str) -> bool:
    """Cached reachability probe of a peer serve instance. The cache (15s TTL,
    hits and misses alike) keeps a dead peer from costing every request a
    2-second timeout, and a live one from being probed per-request."""
    now = time.monotonic()
    hit = _peer_probe_cache.get(url)
    if hit is not None and now - hit[0] < PEER_PROBE_TTL_S:
        return hit[1]
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/api/summary",
                                    timeout=PEER_PROBE_TIMEOUT_S) as r:
            alive = 200 <= r.status < 300
    except Exception:                                  # noqa: BLE001 -- any failure = not alive
        alive = False
    _peer_probe_cache[url] = (now, alive)
    return alive


def _read_json(path):
    """Open/read/close against atomically-replaced files -- a mid-swap miss or
    half-state is impossible; any failure just means 'no data this poll'."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, ValueError, OSError):
        return None


class Store:
    """Read-side view of one checkpoint dir. Stateless beyond the dir path --
    every accessor re-reads the file, so the server never holds stale caches
    (cadences are slow; these are small JSON files)."""

    def __init__(self, ckpt_dir: str):
        self.dir = ckpt_dir

    def stats(self) -> dict:
        d = _read_json(os.path.join(self.dir, "stats.json")) or {}
        return {"reports": d.get("reports", []), "evals": d.get("evals", [])}

    def ticks(self) -> list:
        d = _read_json(os.path.join(self.dir, "ticks.json")) or {}
        return d.get("ticks", [])

    def rows(self, kind: str) -> list:
        if kind == "ticks":
            return self.ticks()
        return self.stats().get(kind, [])

    def peer(self) -> dict | None:
        return _read_json(os.path.join(self.dir, PEER))

    def archives(self) -> list:
        """The permanent per-10k-iteration checkpoints in THIS host's ckpt dir,
        oldest first: [{it, file, bytes, mtime}]."""
        out = []
        for p in sorted(glob.glob(os.path.join(self.dir, "archive_*.pt"))):
            name = os.path.basename(p)
            m = ARCHIVE_RE.match(name)
            if m is None:
                continue
            try:
                st = os.stat(p)
            except OSError:
                continue
            out.append({"it": int(m.group(1)), "file": name,
                        "bytes": st.st_size, "mtime": st.st_mtime})
        return out

    def since(self, kind: str, since_it) -> list:
        rows = self.rows(kind)
        if since_it is None:
            return rows
        return [r for r in rows if (r.get("it") or 0) > since_it]

    def mtimes(self) -> tuple:
        out = []
        for name in ("stats.json", "ticks.json", "best.json"):
            try:
                out.append(os.stat(os.path.join(self.dir, name)).st_mtime)
            except OSError:
                out.append(None)
        return tuple(out)

    def summary(self) -> dict:
        stats = self.stats()
        ticks = self.ticks()
        best = _read_json(os.path.join(self.dir, "best.json"))
        owner = _read_json(os.path.join(self.dir, "owner.json"))
        try:
            latest_mtime = os.stat(os.path.join(self.dir, "latest.pt")).st_mtime
        except OSError:
            latest_mtime = None
        newest = max([r.get("wall_time") or 0 for r in
                      (stats["reports"][-1:] + stats["evals"][-1:] + ticks[-1:])] + [0])
        peer = self.peer()
        return {
            "ckpt_dir": os.path.abspath(self.dir),
            "host": socket.gethostname(),
            "server_time": time.time(),
            "trainer_live": is_locked(os.path.join(self.dir, TRAINER_LOCK)),
            "last_report": stats["reports"][-1] if stats["reports"] else None,
            "last_eval": stats["evals"][-1] if stats["evals"] else None,
            "last_tick": ticks[-1] if ticks else None,
            "counts": {"reports": len(stats["reports"]), "evals": len(stats["evals"]),
                       "ticks": len(ticks)},
            "best": best,
            "owner": owner,
            "peer": peer,
            "peer_alive": (peer_alive(peer["url"])
                           if peer and peer.get("url") else None),
            "latest_pt_mtime": latest_mtime,
            "staleness_s": (time.time() - newest) if newest else None,
        }


class Actions:
    """The opt-in control plane (--allow-actions). POST-only, server-side state
    validation, platform-specific registry:

      * Windows (a relay/console session): "End session & hand back" drops the
        STOP file -- the trainer checkpoints and exits at the iteration
        boundary, and if fishrl.relay launched it, the handback runs itself.
      * POSIX (the systemd box): stop/start the trainer unit and kick the eval
        oneshot, via ``sudo -n systemctl`` (needs the NOPASSWD rule the relay
        already relies on). A raw STOP file is wrong here: Restart=always
        would just resurrect the trainer.

    Nothing destructive is exposed -- no --fresh, no claim/force."""

    def __init__(self, ckpt_dir: str, unit: str, eval_unit: str,
                 os_name: str = os.name):
        self.dir, self.unit, self.eval_unit, self.os_name = ckpt_dir, unit, eval_unit, os_name

    def _trainer_live(self) -> bool:
        return is_locked(os.path.join(self.dir, TRAINER_LOCK))

    def list(self) -> list:
        live = self._trainer_live()
        if self.os_name == "nt":
            return [{"id": "stop_session", "label": "End session & hand back",
                     "danger": True, "enabled": live,
                     "reason": None if live else "no trainer is running here"},
                    {"id": "run_eval_local", "label": "Run eval panel now",
                     "danger": False, "enabled": True, "reason": None}]
        return [
            {"id": "stop_trainer", "label": f"Stop trainer ({self.unit})",
             "danger": True, "enabled": live,
             "reason": None if live else "trainer is not running"},
            {"id": "start_trainer", "label": "Start trainer",
             "danger": False, "enabled": not live,
             "reason": None if not live else "trainer is already running"},
            {"id": "run_eval", "label": "Run eval panel now",
             "danger": False, "enabled": True, "reason": None},
        ]

    def _systemctl(self, verb: str, unit: str) -> tuple:
        r = subprocess.run(["sudo", "-n", "systemctl", verb, unit],
                           capture_output=True, text=True, timeout=180)
        if r.returncode != 0:
            return False, (f"systemctl {verb} {unit} failed: "
                           f"{(r.stderr or r.stdout).strip() or r.returncode} "
                           f"(is the NOPASSWD sudoers rule in place?)")
        return True, f"systemctl {verb} {unit}: ok"

    def run(self, action_id: str) -> tuple:
        """(ok, message). Re-validates state at execution time -- the button the
        client rendered may be stale."""
        by_id = {a["id"]: a for a in self.list()}
        a = by_id.get(action_id)
        if a is None:
            return False, f"unknown action '{action_id}'"
        if not a["enabled"]:
            return False, f"refused: {a['reason']}"
        if action_id == "stop_session":
            with open(os.path.join(self.dir, STOP_FILE), "w") as f:
                f.write("stop requested via fishrl.serve\n")
            return True, ("STOP written -- the trainer will checkpoint and exit at the "
                          "iteration boundary (a relay session then hands back "
                          "automatically)")
        if action_id == "run_eval_local":
            import sys
            subprocess.Popen([sys.executable, "-m", "fishrl.eval.parallel_panel",
                              "--ckpt-dir", self.dir, "--reserve-cores", "12"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True, ("eval panel started (a one-shot subprocess; its row appears "
                          "in the win-rate chart when it finishes, ~30s-2min)")
        if action_id == "stop_trainer":
            return self._systemctl("stop", self.unit)
        if action_id == "start_trainer":
            return self._systemctl("start", self.unit)
        if action_id == "run_eval":
            return self._systemctl("start", self.eval_unit)
        return False, f"unhandled action '{action_id}'"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    store: Store = None                                  # set by serve()
    actions: Actions = None                              # None = read-only (the default)
    redirect: bool = True                                # follow peer.json (--no-redirect off)

    def log_message(self, fmt, *args):                   # quiet: no per-request spam
        pass

    # -- plumbing ------------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

    def _since_it(self, q: dict):
        try:
            return int(q["since_it"][0])
        except (KeyError, ValueError, IndexError):
            return None

    def _peer_target(self) -> str | None:
        """Where the lineage's live telemetry actually is, when it isn't here:
        the peer.json breadcrumb the relay leaves on the released side of a
        handoff. Redirect only when no trainer is live locally (never redirect
        away from live data) and the peer answers the cached probe (a
        sleeping/offline peer degrades to serving local rows, not errors)."""
        if not self.redirect:
            return None
        peer = self.store.peer()
        url = (peer or {}).get("url")
        if not url:
            return None
        if is_locked(os.path.join(self.store.dir, TRAINER_LOCK)):
            return None
        if not peer_alive(url):
            return None
        return url.rstrip("/")

    def _redirect_to(self, location: str) -> None:
        self.send_response(307)                          # method+body preserved; GET here
        self.send_header("Location", location)
        self.send_header("Access-Control-Allow-Origin", "*")   # EventSource/fetch follow
        self.send_header("Cache-Control", "no-store")          # cross-origin only w/ CORS
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send_archive(self, name: str) -> None:
        """Stream one archive checkpoint. `name` must full-match ARCHIVE_RE (the
        route's whole safety story: no traversal, nothing but archives). Archives
        are written atomically and never rewritten, so a plain streamed read is
        always a complete, immutable file -- safe to cache on the client too."""
        if ARCHIVE_RE.match(name) is None:
            self._json({"error": f"not an archive name: {name!r} "
                                 f"(want archive_########.pt; see /api/archives)"},
                       code=404)
            return
        path = os.path.join(self.store.dir, name)
        try:
            size = os.path.getsize(path)
            f = open(path, "rb")
        except OSError:
            self._json({"error": f"{name} not on this host (archives are per-host; "
                                 f"see /api/archives here and on the peer)"}, code=404)
            return
        with f:
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(size))
            self.send_header("Content-Disposition", f'attachment; filename="{name}"')
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                self.wfile.write(chunk)

    # -- routes ----------------------------------------------------------------
    def do_GET(self):                                    # noqa: N802 (http.server API)
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/" or u.path == "/index.html":
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif u.path == "/api/summary":
                self._json(self.store.summary())
            elif u.path == "/api/actions":
                if self.actions is None:
                    self._json({"error": "actions disabled (--allow-actions)"}, code=403)
                else:
                    self._json({"actions": self.actions.list()})
            elif u.path == "/api/archives":
                # LOCAL archives only, never peer-redirected: each host holds the
                # archives written during ITS sessions (the relay zip skips them).
                self._json({"archives": self.store.archives(),
                            "host": socket.gethostname(),
                            "download": "/archives/<file>"})
            elif u.path.startswith("/archives/"):
                self._send_archive(u.path[len("/archives/"):])
            elif u.path in ("/api/reports", "/api/evals", "/api/ticks", "/api/stream"):
                target = self._peer_target()
                if target is not None:                   # live data is on the peer
                    self._redirect_to(target + self.path)
                elif u.path == "/api/stream":
                    self._stream(self._since_it(q))
                else:
                    kind = u.path.rsplit("/", 1)[1]
                    rows = self.store.since(kind, self._since_it(q))
                    self._json({kind: rows, "count": len(rows)})
            else:
                self._json({"error": f"unknown path {u.path}"}, code=404)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass                                         # client went away; not our problem

    def do_POST(self):                                   # noqa: N802 (http.server API)
        """POST-only control plane: mutations can never be triggered by a stray
        GET (browser prefetch, a crawler, a curious click on a link)."""
        u = urlparse(self.path)
        try:
            if u.path != "/api/action":
                self._json({"error": f"unknown path {u.path}"}, code=404)
                return
            if self.actions is None:
                self._json({"error": "actions disabled (start with --allow-actions)"},
                           code=403)
                return
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                action_id = body["action"]
            except (ValueError, KeyError):
                self._json({"error": "body must be JSON: {\"action\": \"<id>\"}"}, code=400)
                return
            ok, message = self.actions.run(action_id)
            print(f"[serve] action '{action_id}' from {self.client_address[0]}: "
                  f"{'ok' if ok else 'REFUSED'} -- {message}", flush=True)
            self._json({"ok": ok, "message": message}, code=200 if ok else 409)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    # -- SSE ---------------------------------------------------------------------
    def _stream(self, since_it) -> None:
        """Push new rows as they appear. Cursor = the IDENTITY (it, wall_time) of
        the last row seen per array, NOT a row count: ticks.json is a RING capped
        at a fixed length, so once full its length never changes and a count
        cursor emits nothing forever (the bug that froze the tick-driven panels).
        On change we scan from the end for the last-seen row and emit what
        follows; identity not found (history rewritten by a merge, or the client
        lagged a whole ring turnover) resyncs to the end -- the reconnect replay
        (``Last-Event-ID``/``since_it``) is the catch-up path for lost spans."""
        replay_from = since_it
        if replay_from is None:
            hdr = self.headers.get("Last-Event-ID")
            if hdr is not None:
                try:
                    replay_from = int(hdr)
                except ValueError:
                    replay_from = None

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        def emit(kind: str, row: dict) -> None:
            it = row.get("it")
            msg = f"event: {kind}\n"
            if it is not None:
                msg += f"id: {it}\n"
            msg += f"data: {json.dumps(row)}\n\n"
            self.wfile.write(msg.encode("utf-8"))

        kinds = ("reports", "evals", "ticks")
        event_name = {"reports": "report", "evals": "eval", "ticks": "tick"}

        def ident(row: dict) -> tuple:
            return (row.get("it"), row.get("wall_time"))

        # Snapshot mtimes BEFORE reading any rows: a write that lands between the
        # snapshot and the reads below then registers as a change on the first poll,
        # and the cursors (set from what was actually read) keep it duplicate-free.
        # The other order silently absorbs such a write into the baseline.
        last_mtimes = self.store.mtimes()
        best_mtime = last_mtimes[2]
        last_id: dict = {}                               # kind -> ident of last row seen
        if replay_from is not None:                      # catch-up, then tail
            for k in kinds:
                rows = self.store.rows(k)
                for row in rows:
                    if (row.get("it") or 0) > replay_from:
                        emit(event_name[k], row)
                last_id[k] = ident(rows[-1]) if rows else None
        else:
            for k in kinds:
                rows = self.store.rows(k)
                last_id[k] = ident(rows[-1]) if rows else None
        last_beat = time.time()
        self.wfile.flush()
        while True:
            time.sleep(STREAM_POLL_S)
            mtimes = self.store.mtimes()
            if mtimes != last_mtimes:
                last_mtimes = mtimes
                for k in kinds:
                    rows = self.store.rows(k)
                    start = len(rows)                    # ident missing -> resync to end
                    if last_id[k] is None:
                        start = 0                        # was empty: everything is new
                    else:
                        for i in range(len(rows) - 1, -1, -1):
                            if ident(rows[i]) == last_id[k]:
                                start = i + 1
                                break
                    for row in rows[start:]:
                        emit(event_name[k], row)
                    if rows:
                        last_id[k] = ident(rows[-1])
                if mtimes[2] != best_mtime:              # best.json rolled
                    best_mtime = mtimes[2]
                    best = _read_json(os.path.join(self.store.dir, "best.json"))
                    if best is not None:
                        emit("best", best)
                self.wfile.flush()
            if time.time() - last_beat >= HEARTBEAT_S:
                self.wfile.write(b": ping\n\n")          # keeps proxies/clients alive
                self.wfile.flush()
                last_beat = time.time()


def lan_ip() -> str:
    """The primary LAN IPv4. UDP 'connect' assigns the outbound interface without
    sending a packet; falls back to hostname resolution, then loopback."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("192.0.2.1", 9))                  # TEST-NET-1: never actually sent
            return s.getsockname()[0]
        finally:
            s.close()
    except OSError:
        pass
    try:
        ip = socket.gethostbyname(socket.gethostname())
        if not ip.startswith("127."):
            return ip
    except OSError:
        pass
    return "127.0.0.1"


def wait_lan_ip(timeout: float = 60.0, poll: float = 2.0,
                ip_fn=lan_ip, sleep_fn=time.sleep) -> str:
    """lan_ip(), but tolerant of starting before DHCP has assigned an address
    (systemd's network-online.target can fire seconds early on netplan/
    NetworkManager split-stack boxes). Polls until a non-loopback IP appears,
    then falls back to loopback for real off-network hosts."""
    deadline = time.monotonic() + timeout
    ip = ip_fn()
    while ip.startswith("127.") and time.monotonic() < deadline:
        print(f"[serve] no LAN address yet (got {ip}); waiting...", flush=True)
        sleep_fn(poll)
        ip = ip_fn()
    return ip


class _Server(ThreadingHTTPServer):
    """ThreadingHTTPServer that stays quiet when a client simply goes away.

    We speak HTTP/1.1, so connections are keep-alive: after a response the handler
    blocks in ``handle_one_request`` reading the NEXT request line. When the peer
    disappears (browser tab closed, or the dashboard's SSE feed dropped as a session
    hands back) that read raises ConnectionAbortedError / ConnectionResetError, OUTSIDE
    any handler method -- so the per-write guards in the SSE endpoints can't catch it,
    and socketserver's default ``handle_error`` dumps a full traceback for what is a
    routine disconnect. It looks like a crash in the hand-back logs; it isn't.

    Swallow exactly that family (all are ConnectionError subclasses); every other
    exception still prints, so real handler bugs stay loud.
    """

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], ConnectionError):
            return
        super().handle_error(request, client_address)


def serve(ckpt_dir: str, bind: str, port: int,
          actions: Actions | None = None, redirect: bool = True) -> ThreadingHTTPServer:
    addr = wait_lan_ip() if bind == "auto" else bind
    if addr == "0.0.0.0":                                # LAN-only by policy
        raise SystemExit("[serve] refusing to bind 0.0.0.0; pass a concrete interface IP")
    Handler.store = Store(ckpt_dir)
    Handler.actions = actions
    Handler.redirect = redirect
    httpd = _Server((addr, port), Handler, bind_and_activate=False)
    if os.name == "nt":
        # http.server's SO_REUSEADDR means something different on Windows: it
        # permits a SECOND server to bind the same port outright (not just
        # TIME_WAIT reuse), and the OLDEST binding keeps winning connections --
        # observed as a stale instance shadowing dashboard updates for days.
        # Disable it so a duplicate start dies loudly instead. POSIX keeps the
        # default: there SO_REUSEADDR is the sane TIME_WAIT behaviour systemd
        # restarts rely on, and duplicate binds still fail.
        httpd.allow_reuse_address = False
    try:
        httpd.server_bind()
        httpd.server_activate()
    except OSError as e:
        httpd.server_close()
        raise SystemExit(f"[serve] cannot bind {addr}:{port} ({e}); is another "
                         f"fishrl.serve already running? Stop it (or use "
                         f"deploy\\fishrl-serve.ps1, which replaces it) and retry")
    httpd.daemon_threads = True                          # SSE threads die with the server
    return httpd


def main() -> None:
    ap = argparse.ArgumentParser(description="LAN telemetry server over a fishrl run.")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--bind", default="auto",
                    help="interface IP to bind ('auto' = primary LAN IPv4; never 0.0.0.0)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--allow-actions", action="store_true",
                    help="mount the control plane (stop/start buttons on the dashboard); "
                         "off = strictly read-only, the default")
    ap.add_argument("--no-redirect", action="store_true",
                    help="never 307 data endpoints at a relay peer (peer.json); "
                         "always serve the local rows")
    ap.add_argument("--unit", default="fishrl-selfplay",
                    help="systemd trainer unit the POSIX actions manage")
    ap.add_argument("--eval-unit", default="fishrl-eval.service",
                    help="systemd oneshot the 'Run eval panel now' action starts")
    args = ap.parse_args()

    actions = (Actions(args.ckpt_dir, args.unit, args.eval_unit)
               if args.allow_actions else None)
    httpd = serve(args.ckpt_dir, args.bind, args.port, actions,
                  redirect=not args.no_redirect)
    host, port = httpd.server_address[:2]
    print(f"[serve] http://{host}:{port}/  (ckpt-dir: {os.path.abspath(args.ckpt_dir)}; "
          f"{'ACTIONS ENABLED' if actions else 'read-only'}; Ctrl-C to stop)", flush=True)
    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
