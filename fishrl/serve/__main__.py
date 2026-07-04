"""LAN-only telemetry webserver over a fishrl checkpoint dir.

    python -m fishrl.serve --ckpt-dir checkpoints                 # binds the LAN IP, :8765
    python -m fishrl.serve --ckpt-dir checkpoints --bind 127.0.0.1 --port 9000

A SIBLING process to the trainer (like fishrl.monitor), never embedded in it:
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
range queries, /api/stream, and / (a minimal human status page). Responses set
``Access-Control-Allow-Origin: *`` so a browser frontend served from another
LAN host (the website) can fetch/stream directly.

Binding: NEVER 0.0.0.0. ``--bind auto`` (default) resolves the machine's
primary LAN IPv4; pass an explicit address to override.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

STREAM_POLL_S = 1.0
HEARTBEAT_S = 15.0


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
        return {
            "ckpt_dir": os.path.abspath(self.dir),
            "host": socket.gethostname(),
            "server_time": time.time(),
            "last_report": stats["reports"][-1] if stats["reports"] else None,
            "last_eval": stats["evals"][-1] if stats["evals"] else None,
            "last_tick": ticks[-1] if ticks else None,
            "counts": {"reports": len(stats["reports"]), "evals": len(stats["evals"]),
                       "ticks": len(ticks)},
            "best": best,
            "owner": owner,
            "latest_pt_mtime": latest_mtime,
            "staleness_s": (time.time() - newest) if newest else None,
        }


INDEX_HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>fishrl serve</title>
<style>body{font-family:Consolas,monospace;background:#101418;color:#d8dee6;
margin:2em;max-width:60em}a{color:#4fc3f7}code{color:#9ccc65}
pre{background:#181e24;padding:1em;overflow-x:auto}</style></head><body>
<h2>fishrl telemetry server</h2>
<p>Range queries (idempotent, keyed on <code>it</code>; omit <code>since_it</code> for everything):</p>
<ul>
<li><a href="/api/summary">/api/summary</a></li>
<li><a href="/api/reports">/api/reports</a>, <a href="/api/evals">/api/evals</a>,
    <a href="/api/ticks">/api/ticks</a> &mdash; <code>?since_it=N</code></li>
<li><code>/api/stream</code> &mdash; server-sent events
    (<code>report</code>/<code>eval</code>/<code>tick</code>/<code>best</code>)</li>
</ul>
<pre id="s">loading summary…</pre>
<script>
const pre=document.getElementById('s');
async function refresh(){try{const r=await fetch('/api/summary');
pre.textContent=JSON.stringify(await r.json(),null,2);}catch(e){pre.textContent=''+e;}}
refresh();setInterval(refresh,5000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    store: Store = None                                  # set by serve()

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

    # -- routes ----------------------------------------------------------------
    def do_GET(self):                                    # noqa: N802 (http.server API)
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/" or u.path == "/index.html":
                self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif u.path == "/api/summary":
                self._json(self.store.summary())
            elif u.path in ("/api/reports", "/api/evals", "/api/ticks"):
                kind = u.path.rsplit("/", 1)[1]
                rows = self.store.since(kind, self._since_it(q))
                self._json({kind: rows, "count": len(rows)})
            elif u.path == "/api/stream":
                self._stream(self._since_it(q))
            else:
                self._json({"error": f"unknown path {u.path}"}, code=404)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass                                         # client went away; not our problem

    # -- SSE ---------------------------------------------------------------------
    def _stream(self, since_it) -> None:
        """Push new rows as they appear. Cursor = per-array row COUNT (rows only
        ever append; on the rare shrink -- an offline stats merge rewrote history --
        the cursor resets to the new end rather than replaying the whole file).
        ``Last-Event-ID``/``since_it`` seed the cursors so a reconnecting client
        first gets everything it missed."""
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
        # Snapshot mtimes BEFORE reading any rows: a write that lands between the
        # snapshot and the reads below then registers as a change on the first poll,
        # and the cursors (set from what was actually read) keep it duplicate-free.
        # The other order silently absorbs such a write into the baseline.
        last_mtimes = self.store.mtimes()
        best_mtime = last_mtimes[2]
        if replay_from is not None:                      # catch-up, then tail
            cursor = {}
            for k in kinds:
                rows = self.store.rows(k)
                for row in rows:
                    if (row.get("it") or 0) > replay_from:
                        emit(event_name[k], row)
                cursor[k] = len(rows)
        else:
            cursor = {k: len(self.store.rows(k)) for k in kinds}
        last_beat = time.time()
        self.wfile.flush()
        while True:
            time.sleep(STREAM_POLL_S)
            mtimes = self.store.mtimes()
            if mtimes != last_mtimes:
                last_mtimes = mtimes
                for k in kinds:
                    rows = self.store.rows(k)
                    if len(rows) < cursor[k]:            # history rewritten (merge): skip ahead
                        cursor[k] = len(rows)
                        continue
                    for row in rows[cursor[k]:]:
                        emit(event_name[k], row)
                    cursor[k] = len(rows)
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


def serve(ckpt_dir: str, bind: str, port: int) -> ThreadingHTTPServer:
    addr = lan_ip() if bind == "auto" else bind
    if addr == "0.0.0.0":                                # LAN-only by policy
        raise SystemExit("[serve] refusing to bind 0.0.0.0; pass a concrete interface IP")
    Handler.store = Store(ckpt_dir)
    httpd = ThreadingHTTPServer((addr, port), Handler)
    httpd.daemon_threads = True                          # SSE threads die with the server
    return httpd


def main() -> None:
    ap = argparse.ArgumentParser(description="LAN telemetry server over a fishrl run.")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--bind", default="auto",
                    help="interface IP to bind ('auto' = primary LAN IPv4; never 0.0.0.0)")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    httpd = serve(args.ckpt_dir, args.bind, args.port)
    host, port = httpd.server_address[:2]
    print(f"[serve] http://{host}:{port}/  (ckpt-dir: {os.path.abspath(args.ckpt_dir)}; "
          f"read-only; Ctrl-C to stop)", flush=True)
    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
