"""The dashboard served at / -- one self-contained HTML string.

Inline CSS/JS + hand-rolled SVG charts only: the page must render on a LAN
with no internet (no CDNs), against the JSON APIs and the SSE stream of the
sibling __main__. Panels: win-rates (with
new-best stars), system utilization (%cpu/%gpu/%ram from the ticks -- the
losses panel it replaced was a flat line: a 5000-it mean over the 2000-it
tick ring), throughput, opponent mix, plus header/staleness
and the (opt-in) actions bar, whose buttons come from /api/actions and are
disabled server-side truth, not client guesswork.
"""

PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>fishrl dashboard</title>
<style>
:root{--bg:#101418;--panel:#171d23;--fg:#d8dee6;--dim:#8a94a0;--grid:#2a3138}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font-family:Consolas,'Cascadia Mono',monospace;font-size:14px}
header{display:flex;flex-wrap:wrap;gap:.6em 1.4em;align-items:center;
       padding:.7em 1em;border-bottom:1px solid var(--grid)}
header b{color:#4fc3f7}
.chip{padding:.15em .6em;border-radius:3px;font-weight:bold;color:#000}
.ok{background:#9ccc65}.warn{background:#ffb74d}.bad{background:#e57373}
.idle{background:#8a94a0}
#actions{display:flex;gap:.6em;margin-left:auto}
#actions button{font:inherit;padding:.35em .9em;border-radius:3px;cursor:pointer;
  border:1px solid var(--grid);background:var(--panel);color:var(--fg)}
#actions button.danger{border-color:#e57373;color:#e57373}
#actions button:disabled{opacity:.4;cursor:not-allowed}
#toast{position:fixed;bottom:1em;left:50%;transform:translateX(-50%);
  background:#263238;color:var(--fg);padding:.6em 1.2em;border-radius:4px;
  border:1px solid var(--grid);display:none;max-width:80%;z-index:9}
main{display:grid;grid-template-columns:1fr 1fr;gap:10px;padding:10px}
@media(max-width:900px){main{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--grid);border-radius:4px;padding:6px}
.panel h3{margin:.2em .4em .4em;font-size:13px;font-weight:normal;color:var(--dim)}
svg{width:100%;height:auto;display:block}
svg text{font-family:inherit;font-size:11px;fill:var(--dim)}
.legend{display:flex;flex-wrap:wrap;gap:.3em .9em;padding:.2em .4em;font-size:12px}
</style></head><body>
<header>
  <span>fishrl <b id="host">…</b></span>
  <span id="gameswrap" title="">games <b id="games">—</b></span>
  <span id="gphwrap" title=""><b id="gph">—</b> games/h</span>
  <span id="gph1kwrap" title="">recent <b id="gph1k">—</b> g/h</span>
  <span>it <b id="it">—</b></span>
  <span><b id="elapsed">—</b> h</span>
  <span>best <b id="best">—</b></span>
  <span id="turn" class="chip idle">turn: …</span>
  <span id="live" class="chip idle">trainer: …</span>
  <a id="peer" class="chip idle" style="display:none;text-decoration:none" href="#">peer</a>
  <span id="stale" class="chip warn">no data yet</span>
  <span id="actions"></span>
</header>
<main>
  <div class="panel"><h3>win-rates vs iteration (5000-it running mean; dots = raw evals)</h3>
    <svg id="wr" viewBox="0 0 600 260"></svg><div class="legend" id="wrL"></div></div>
  <div class="panel"><h3>system utilization % (250-it running mean; dots = raw ticks)</h3>
    <svg id="sys" viewBox="0 0 600 260"></svg><div class="legend" id="sysL"></div></div>
  <div class="panel"><h3>throughput (reports)</h3>
    <svg id="thr" viewBox="0 0 600 260"></svg><div class="legend" id="thrL"></div></div>
  <div class="panel"><h3>opponent mix (reports)</h3>
    <svg id="mix" viewBox="0 0 600 260"></svg><div class="legend" id="mixL"></div></div>
</main>
<div id="toast"></div>
<script>
"use strict";
const WR = {heuristic:"#4fc3f7", heuristic11:"#b39ddb", heuristic12:"#f8bbd0",
            heuristic13:"#ce93d8",
            random:"#9ccc65", attacker:"#ffb74d", frozen:"#e57373",
            // self-play seat balance (from --seat-diag-games): a LEARNED asymmetry if
            // it strays from 0.5. Same 0-1 / 0.5-reference axis as the win-rates.
            seat_p1_wr:"#80cbc4"};
const SYS = {cpu:"#4fc3f7", gpu:"#9ccc65", ram:"#b39ddb"};   // utilization % per tick
const SYS_SMOOTH = 250;   // system-panel running-mean window (ticks span ~2000 it)
const MIX = {opp_self:"#4fc3f7", opp_past:"#b39ddb", opp_heuristic:"#e57373",
             opp_heuristic11:"#ef9a9a", opp_heuristic12:"#f8bbd0",
             opp_heuristic13:"#ce93d8",
             opp_attacker:"#ffb74d", opp_random:"#9ccc65",
             opp_scenario:"#80cbc4"};
const WR_SMOOTH = 5000;   // win-rate running-mean window, in iterations
const RATE_ITERS = 1000;  // "recent" throughput window, in iterations
const S = {reports:new Map(), evals:new Map(), ticks:new Map(), best:null, summary:null};
let dataViaPeer = null;   // where the charts' data comes from (set on first poll)
const key = {reports:r=>r.it, ticks:r=>r.it, evals:r=>r.it+":"+(r.wall_time||0)};
const $ = id => document.getElementById(id);
let dirty = false;

function rows(kind){ return [...S[kind].values()].sort((a,b)=>(a.it||0)-(b.it||0)); }
function add(kind, r){ if(r && r.it != null){ S[kind].set(key[kind](r), r); dirty = true; } }

// ── hand-rolled SVG line/stack chart ─────────────────────────────────────────
function chart(svg, legendEl, series, opts){
  opts = opts || {};
  const W = 600, H = 260, L = 44, R = opts.right ? 44 : 8, T = 8, B = 22;
  const pts = series.flatMap(s => s.pts);
  svg.innerHTML = ""; legendEl.innerHTML = "";
  if (!pts.length) { svg.innerHTML =
    '<text x="300" y="130" text-anchor="middle">no data yet</text>'; return; }
  const xs = pts.map(p=>p[0]);
  let x0 = Math.min(...xs), x1 = Math.max(...xs); if (x0===x1) x1 = x0+1;
  const main = series.filter(s=>!s.right);
  function range(list){
    const ys = list.flatMap(s=>s.pts.map(p=>p[1]));
    let a = opts.y0 != null ? opts.y0 : Math.min(...ys);
    let b = opts.y1 != null ? opts.y1 : Math.max(...ys);
    if (a===b){ a-=1; b+=1; } return [a,b];
  }
  const [ya,yb] = range(main.length?main:series);
  const X = v => L + (v-x0)/(x1-x0)*(W-L-R);
  const Y = (v,a,b) => T + (1-(v-a)/(b-a))*(H-T-B);
  let out = [];
  if (opts.yStep){                                // grid at fixed data intervals
    const k0 = Math.ceil((ya-1e-9)/opts.yStep), k1 = Math.floor((yb+1e-9)/opts.yStep);
    for (let k=k0;k<=k1;k++){
      const gv = k*opts.yStep, gy = Y(gv,ya,yb), major = k%2===0;   // label every 2nd
      out.push(`<line x1="${L}" y1="${gy}" x2="${W-R}" y2="${gy}" stroke="#2a3138" stroke-width="${major?0.8:0.4}"/>`);
      if (major) out.push(`<text x="${L-4}" y="${gy+4}" text-anchor="end">${fmt(gv)}</text>`);
    }
  } else for (let i=0;i<=4;i++){                  // grid + axis labels (quarters)
    const gy = T + i*(H-T-B)/4, gv = yb - i*(yb-ya)/4;
    out.push(`<line x1="${L}" y1="${gy}" x2="${W-R}" y2="${gy}" stroke="#2a3138" stroke-width="0.5"/>`);
    out.push(`<text x="${L-4}" y="${gy+4}" text-anchor="end">${fmt(gv)}</text>`);
  }
  for (let i=0;i<=4;i++){
    const gx = L + i*(W-L-R)/4, gv = x0 + i*(x1-x0)/4;
    out.push(`<text x="${gx}" y="${H-6}" text-anchor="middle">${Math.round(gv)}</text>`);
  }
  if (opts.hline != null){
    out.push(`<line x1="${L}" y1="${Y(opts.hline,ya,yb)}" x2="${W-R}" y2="${Y(opts.hline,ya,yb)}" stroke="#3a444e" stroke-width="1"/>`);
  }
  if (opts.stack){                                // stacked areas (series order = stack order)
    const its = rows("reports").map(r=>r.it);
    let base = new Map(its.map(i=>[i,0]));
    for (const s of series){
      let up = [], down = [];
      for (const [x,y] of s.pts){
        const b0 = base.get(x)||0, b1 = b0 + y;
        up.push(`${X(x)},${Y(Math.min(b1,yb),ya,yb)}`); down.push(`${X(x)},${Y(b0,ya,yb)}`);
        base.set(x, b1);
      }
      out.push(`<polygon points="${up.join(" ")} ${down.reverse().join(" ")}" fill="${s.color}" opacity="0.75"/>`);
    }
  } else {
    for (const s of series){
      const [a,b] = s.right ? range([s]) : [ya,yb];
      if (s.dots){                                  // faint raw datapoints behind the lines
        for (const p of s.pts)
          out.push(`<circle cx="${X(p[0])}" cy="${Y(p[1],a,b)}" r="${s.r||1.8}" fill="${s.color}" opacity="0.28"/>`);
        continue;
      }
      const line = s.pts.map(p=>`${X(p[0])},${Y(p[1],a,b)}`).join(" ");
      out.push(`<polyline points="${line}" fill="none" stroke="${s.color}" stroke-width="${s.w||1.4}" opacity="${s.right?0.8:1}"/>`);
      if (s.right){
        out.push(`<text x="${W-4}" y="${T+10}" text-anchor="end" fill="${s.color}">${fmt(b)}</text>`);
        out.push(`<text x="${W-4}" y="${H-B}" text-anchor="end" fill="${s.color}">${fmt(a)}</text>`);
      }
      for (const st of s.stars||[])
        out.push(`<text x="${X(st[0])}" y="${Y(st[1],ya,yb)+5}" text-anchor="middle" fill="#ffd54f" font-size="16">★</text>`);
    }
  }
  svg.innerHTML = out.join("");
  legendEl.innerHTML = series.filter(s => s.name).map(s =>
    `<span style="color:${s.color}">■ ${s.name}</span>`).join("");
}
function fmt(v){
  if (!isFinite(v)) return "";
  const a = Math.abs(v);
  if (a>=100000) return (v/1000).toFixed(0)+"k";
  if (Number.isInteger(v)) return v.toString();   // 0/10/20 grid labels, not "10.00"
  if (a>=100) return v.toFixed(0);
  if (a>=1) return v.toFixed(2);
  return v.toFixed(3);
}

// ── panels ───────────────────────────────────────────────────────────────────
function smooth(pts, win){
  // centered running mean over an iteration window (partial windows at the
  // edges). Two-pointer sliding window -- pts arrive sorted by iteration, so
  // this stays O(n) even for the ~2000-point tick series.
  const out = new Array(pts.length);
  let lo = 0, hi = 0, sum = 0;
  for (let i=0;i<pts.length;i++){
    const a = pts[i][0] - win/2, b = pts[i][0] + win/2;
    while (hi < pts.length && pts[hi][0] <= b){ sum += pts[hi][1]; hi++; }
    while (pts[lo][0] < a){ sum -= pts[lo][1]; lo++; }
    out[i] = [pts[i][0], sum/(hi-lo)];
  }
  return out;
}

function render(){
  dirty = false;
  const ev = rows("evals"), rp = rows("reports"), tk = rows("ticks");
  const wrSeries = [];
  for (const [k,c] of Object.entries(WR)){
    const raw = ev.filter(r=>r[k]!=null).map(r=>[r.it, r[k]]);
    if (!raw.length) continue;
    wrSeries.push({name:"", color:c, dots:true, pts:raw});      // faint raw points
    wrSeries.push({name:k, color:c, pts:smooth(raw, WR_SMOOTH),
      stars: k==="heuristic" ? ev.filter(r=>r.new_best&&r[k]!=null).map(r=>[r.it,r[k]]) : []});
  }
  chart($("wr"), $("wrL"), wrSeries, {y0:0, y1:1, hline:0.5, yStep:0.05});
  const sysSeries = [];
  for (const [k,c] of Object.entries(SYS)){
    const raw = tk.filter(r=>r[k]!=null).map(r=>[r.it, r[k]]);
    if (!raw.length) continue;                            // e.g. no gpu on the ODROID
    sysSeries.push({name:"", color:c, dots:true, r:1.2, pts:raw});  // faint raw ticks
    sysSeries.push({name:k+" %", color:c, pts:smooth(raw, SYS_SMOOTH)});
  }
  chart($("sys"), $("sysL"), sysSeries, {y0:0, y1:100, yStep:10});
  // games/h per report window = games x iters_per_h / iters (window hours =
  // iters / iters_per_h). The it/h series moved to the header tooltip when
  // iteration SIZE became a regime knob (8 -> 16 -> 32 games/iter, 2026-08-07):
  // games/h is the cross-regime throughput, it/h just tracks the knob.
  chart($("thr"), $("thrL"), [
      {name:"games/h", color:"#4fc3f7",
       pts: rp.filter(r=>r.games!=null&&r.iters>0&&r.iters_per_h!=null)
              .map(r=>[r.it, r.games*r.iters_per_h/r.iters])},
      {name:"transitions", color:"#9ccc65", right:true,
       pts: rp.filter(r=>r.transitions!=null).map(r=>[r.it,r.transitions])},
    ].filter(s=>s.pts.length), {});
  chart($("mix"), $("mixL"), Object.entries(MIX).map(([k,c])=>({
      name:k.slice(4), color:c,
      pts: rp.filter(r=>r[k]!=null).map(r=>[r.it, r[k]]),
    })).filter(s=>s.pts.length), {stack:true, y0:0, y1:1});
}

function rate1k(){
  // Throughput over (up to) the last RATE_ITERS iterations ON ONE DEVICE, so the
  // number is directly comparable between the server and the PC regardless of
  // report cadence. Primary source: report rows (elapsed_h excludes downtime),
  // restricted to the trailing run of rows sharing the newest row's host tag --
  // merged relay histories interleave hosts. Fallback for a session too young
  // for two reports: tick wall-clock (live device only). `games` (games/h over
  // the same window, summed from the report rows) is null on the tick fallback.
  const rp = rows("reports");
  if (rp.length >= 2){
    const last = rp[rp.length-1], host = last.host;
    let i = rp.length - 1;
    while (i > 0 && rp[i-1].host === host) i--;
    const tail = rp.slice(i).filter(r => r.elapsed_h != null && r.it != null);
    if (tail.length >= 2 && tail[tail.length-1].it - tail[0].it > 0){
      let base = tail[0];
      for (const r of tail){ if (r.it <= last.it - RATE_ITERS) base = r; else break; }
      const dit = last.it - base.it, dh = last.elapsed_h - base.elapsed_h;
      let dg = 0;
      for (const r of tail) if (r.it > base.it && r.games != null) dg += r.games;
      if (dit > 0 && dh > 0)
        return {rate: dit/dh, games: dg > 0 ? dg/dh : null, span: dit,
                host: host || "?", src: "reports"};
    }
  }
  const tk = rows("ticks");
  if (tk.length >= 2){
    const last = tk[tk.length-1];
    let base = tk[0];
    for (const t of tk){ if (t.it <= last.it - RATE_ITERS) base = t; else break; }
    const dit = last.it - base.it, dh = (last.wall_time - base.wall_time)/3600;
    if (dit > 0 && dh > 0) return {rate: dit/dh, games: null, span: dit,
                                   host: "this device", src: "ticks"};
  }
  return null;
}

// Lifetime games. Report rows carry exact per-window game counts; iterations in
// windows no report covers (session ends, crashes) are estimated at 8 games/iter
// -- correct for essentially the whole gap, which predates the 2026-08-07 move
// to bigger iterations.
const GAMES_PER_ITER_LEGACY = 8;
function totalGames(lastIt){
  const rp = rows("reports");
  if (!rp.length || lastIt == null) return null;
  let g = 0, covered = 0;
  for (const r of rp){
    if (r.games != null) { g += r.games; covered += r.iters || 0; }
  }
  const gap = Math.max(0, lastIt - covered);
  return {games: g + gap*GAMES_PER_ITER_LEGACY, exact: g, gapIters: gap};
}
function fmtBig(v){
  if (v == null) return "—";
  if (v >= 1e6) return (v/1e6).toFixed(2)+"M";
  if (v >= 1e4) return (v/1e3).toFixed(1)+"k";
  return String(Math.round(v));
}

function header(s){
  if (!s) return;
  const dev = (s.last_report||{}).device;
  $("host").textContent = s.host + " " + s.ckpt_dir.split(/[\\/]/).pop() +
    (dev ? " [" + dev + "]" : "");
  const lr = s.last_report||{}, lt = s.last_tick||{};
  const lastIt = lt.it!=null?lt.it:(lr.it!=null?lr.it:null);
  $("it").textContent = lastIt!=null?lastIt:"—";
  $("elapsed").textContent = lr.elapsed_h!=null?lr.elapsed_h.toFixed(1):"—";
  // Games are the headline: iteration SIZE is a regime knob (8/16/32 games/iter),
  // so it/h moved into the tooltips and games carry the throughput story.
  const tg = totalGames(lastIt);
  $("games").textContent = tg ? fmtBig(tg.games) : "—";
  $("gameswrap").title = tg ?
    `${tg.games.toLocaleString()} lifetime games (${tg.exact.toLocaleString()} report-counted + ${tg.gapIters} uncovered it x ${GAMES_PER_ITER_LEGACY})` : "";
  const gph = (lr.games!=null && lr.iters>0 && lr.iters_per_h!=null)
    ? lr.games*lr.iters_per_h/lr.iters : null;
  $("gph").textContent = fmtBig(gph);
  $("gphwrap").title = lr.iters_per_h!=null ?
    `last report window; ${lr.iters_per_h.toFixed(1)} it/h x ${(lr.games&&lr.iters?(lr.games/lr.iters).toFixed(1):"?")} games/iter` : "";
  const rk = rate1k();
  $("gph1k").textContent = rk && rk.games!=null ? fmtBig(rk.games) : "—";
  $("gph1kwrap").title = rk ?
    `over the last ${rk.span} it on ${rk.host} (${rk.src}); ${rk.rate.toFixed(1)} it/h` : "";
  // best_score (maximin over anchors) on new best.json; legacy files only carry
  // the old v1.0-keyed rate.
  $("best").textContent = s.best
    ? `${(s.best.best_score ?? s.best.heuristic).toFixed(2)}@${s.best.it}` : "—";
  // Ownership chip is relay-era: on a local-only PC run it's always "this host, active"
  // -- redundant with the header host. Show it ONLY when noteworthy (released, or owned by
  // another host), so the normal case stays uncluttered.
  const o = s.owner, turnEl = $("turn");
  if (o && o.state === "active" && o.host === s.host) {
    turnEl.style.display = "none";
  } else {
    turnEl.style.display = "";
    set("turn", o?`turn: ${o.state==="active"?o.host:"released g"+o.generation}`:"turn: unclaimed",
        o&&o.state==="active"?"ok":"warn");
  }
  set("live", s.trainer_live?"trainer: RUNNING":"trainer: stopped",
      s.trainer_live?"ok":"idle");
  // Relay peer: the lineage's telemetry lives elsewhere (peer.json). The chip
  // links to that dashboard; charts here already stream from it via the
  // server's 307 redirect when the peer is reachable.
  const p = s.peer, pe = $("peer");
  if (p && p.url && !s.trainer_live){
    pe.style.display = "inline-block";
    pe.href = p.url;
    pe.textContent = "training on " + (p.host || "peer") + " ↗";
    pe.className = "chip " + (s.peer_alive ? "ok" : "idle");
    pe.title = s.peer_alive
      ? "charts stream live from " + p.url + " (redirect)"
      : p.host + " holds the lineage but is unreachable; showing local history";
  } else pe.style.display = "none";
  // If where the data comes from flipped (handoff mid-view), reload so the
  // SSE stream and history re-follow (or stop following) the redirect.
  const viaPeer = !!(p && p.url && !s.trainer_live && s.peer_alive);
  if (dataViaPeer === null) dataViaPeer = viaPeer;
  else if (viaPeer !== dataViaPeer) location.reload();
  if (s.staleness_s==null) set("stale","no data yet","warn");
  else {
    const m = s.staleness_s/60;
    set("stale", m<1?"live":`last data ${m.toFixed(0)} min ago`, m<90?"ok":(m<180?"warn":"bad"));
  }
}
function set(id, txt, cls){ const e=$(id); e.textContent=txt; e.className="chip "+cls; }

// ── actions bar (server-driven; absent unless --allow-actions) ───────────────
async function refreshActions(){
  try {
    const r = await tfetch("/api/actions");
    if (!r.ok){ $("actions").innerHTML=""; return; }
    const acts = (await r.json()).actions;
    $("actions").innerHTML = "";
    for (const a of acts){
      const b = document.createElement("button");
      b.textContent = a.label;
      if (a.danger) b.className = "danger";
      b.disabled = !a.enabled;
      b.title = a.enabled ? "" : (a.reason||"");
      b.onclick = async () => {
        if (a.countdown){
          // countdown actions (e.g. Sleep display) skip the confirm dialog: the
          // 3-2-1 on the button IS the grace period, and a second click cancels.
          if (b.dataset.counting){ delete b.dataset.counting; b.textContent = a.label; return; }
          b.dataset.counting = "1";
          for (let s = a.countdown; s > 0; s--){
            b.textContent = a.label + " in " + s + "…";
            await new Promise(r => setTimeout(r, 1000));
            if (!b.dataset.counting){ return; }          // cancelled mid-count
          }
          delete b.dataset.counting;
          b.textContent = a.label;
          b.disabled = true;
        } else {
          if (!confirm(a.label + " — are you sure?")) return;
          b.disabled = true;
        }
        try {
          const rsp = await fetch("/api/action", {method:"POST",
            headers:{"Content-Type":"application/json"},
            body:JSON.stringify({action:a.id})});
          const j = await rsp.json();
          toast(j.message || rsp.statusText);
          // Full-session teardown: the launcher will close this dashboard too.
          // Flag it so the server vanishing renders as "session ended", not as
          // an error the page retries forever.
          if (a.id === "stop_session" && j.ok) sessionEnding = true;
        } catch(e){ toast(""+e); }
        setTimeout(()=>{ refreshActions(); poll(); }, 800);
      };
      $("actions").appendChild(b);
    }
  } catch(e){ /* server gone; next poll retries */ }
}
let toastT = null;
function toast(msg){
  const t = $("toast"); t.textContent = msg; t.style.display = "block";
  clearTimeout(toastT); toastT = setTimeout(()=>t.style.display="none", 6000);
}

// ── data flow: SSE first (no gap), then range fetch, then poll header ────────
const es = new EventSource("/api/stream");
for (const [evt,kind] of [["report","reports"],["eval","evals"],["tick","ticks"]])
  es.addEventListener(evt, e => add(kind, JSON.parse(e.data)));
es.addEventListener("best", e => { if(S.summary) S.summary.best = JSON.parse(e.data); });
async function loadAll(){
  for (const kind of ["reports","evals","ticks"]){
    try {
      const d = await (await fetch("/api/"+kind)).json();
      for (const r of d[kind]) add(kind, r);
    } catch(e){}
  }
}
let sessionEnding = false;
// fetch with a deadline: browser fetch has NO default timeout, so one hung
// request (a dead address in the host's DNS set, a mid-restart socket) would
// leave poll() pending forever while its interval piles more hung requests
// onto the per-origin connection limit -- the "charts update but the header is
// dead" failure. An aborted poll rejects, the catch paints the state, and the
// next tick retries on a fresh connection.
function tfetch(url, opts){
  const c = new AbortController();
  const t = setTimeout(() => c.abort(), 4000);
  return fetch(url, {...(opts||{}), signal: c.signal}).finally(() => clearTimeout(t));
}
async function poll(){
  try { S.summary = await (await tfetch("/api/summary")).json(); header(S.summary); }
  catch(e){
    if (sessionEnding){
      // "End training session" tears down the whole session, this server
      // included -- its disappearance is the expected ending, not an outage.
      set("stale","session ended — dashboard closed","idle");
      set("live","trainer: stopped","idle");
      es.close(); clearInterval(pollT); clearInterval(actT);
      $("actions").innerHTML = "";
      return;
    }
    set("stale","server unreachable","bad");
  }
}
loadAll(); poll(); refreshActions();
const pollT = setInterval(poll, 5000);
const actT = setInterval(refreshActions, 15000);
setInterval(()=>{ if(dirty) render(); }, 700);
setTimeout(render, 500);
</script></body></html>
"""
