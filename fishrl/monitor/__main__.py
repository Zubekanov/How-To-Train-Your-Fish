"""Live training monitor — a desktop window over stats.json.

    python -m fishrl.monitor --ckpt-dir checkpoints
    python -m fishrl.train ... --gui         # trainer launches this as a subprocess

Read-only companion to the trainer/eval service: it polls ``<ckpt-dir>/
stats.json`` (the ``reports``/``evals`` time-series), ``best.json`` and the
``latest.pt`` mtime, and renders win-rates, losses, throughput and the
opponent mix in a tkinter window with embedded matplotlib charts. It never
locks, never writes, and never imports torch — reads are open/read/close
against atomically-replaced files, so a crash or a kill here cannot affect a
live trainer.

matplotlib is the one (gui-only) extra dependency: ``pip install matplotlib``
or ``pip install -e .[gui]``.
"""
from __future__ import annotations

import argparse
import json
import os
import time

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except ImportError:
    raise SystemExit(
        "fishrl.monitor needs matplotlib (gui-only dependency):\n"
        "    pip install matplotlib\n"
        "or: pip install -e .[gui]")

import tkinter as tk

BG = "#101418"
FG = "#d8dee6"
GRID = "#2a3138"
SERIES = {"heuristic": "#4fc3f7", "heuristic11": "#b39ddb", "heuristic12": "#f8bbd0",
          "random": "#9ccc65", "attacker": "#ffb74d", "frozen": "#e57373"}
LOSSES = {"policy_loss": "#4fc3f7", "critic_loss": "#e57373",
          "guesser_loss": "#9ccc65", "public_loss": "#ffb74d",
          "entropy": "#b0bec5", "approx_kl": "#f06292"}
MIX = ("opp_trained", "opp_self", "opp_past", "opp_heuristic", "opp_attacker",
       "opp_random", "opp_scenario")


def _read_json(path):
    """Fast open/read/close; writers replace atomically so a read never sees a
    half-written file — a mid-swap FileNotFoundError just means 'try next poll'."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, ValueError, OSError):
        return None


def _series(rows, key, x_key="it"):
    xs, ys = [], []
    for r in rows:
        v = r.get(key)
        if v is None or r.get(x_key) is None:
            continue
        xs.append(r[x_key])
        ys.append(v)
    return xs, ys


class Monitor:
    def __init__(self, root: tk.Tk, ckpt_dir: str, refresh_s: float):
        self.root = root
        self.ckpt_dir = ckpt_dir
        self.refresh_ms = max(1000, int(refresh_s * 1000))
        self.last_sig = None

        root.title(f"fishrl monitor — {os.path.abspath(ckpt_dir)}")
        root.configure(bg=BG)
        root.geometry("1180x820")

        self.header = tk.Label(root, bg=BG, fg=FG, anchor="w", justify="left",
                               font=("Consolas", 11), padx=10, pady=6)
        self.header.pack(fill="x")
        self.badge = tk.Label(root, bg=BG, fg="#000000", font=("Consolas", 10, "bold"),
                              padx=8, pady=2)
        self.badge.place(relx=1.0, y=8, x=-12, anchor="ne")

        self.fig = Figure(figsize=(11.6, 7.4), dpi=100, facecolor=BG)
        self.axes = self.fig.subplots(2, 2)
        self.fig.subplots_adjust(left=0.06, right=0.985, top=0.95, bottom=0.06,
                                 hspace=0.32, wspace=0.22)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.poll()

    # -- data ----------------------------------------------------------------
    def snapshot(self):
        stats = _read_json(os.path.join(self.ckpt_dir, "stats.json")) or {}
        best = _read_json(os.path.join(self.ckpt_dir, "best.json"))
        try:
            latest_mtime = os.stat(os.path.join(self.ckpt_dir, "latest.pt")).st_mtime
        except OSError:
            latest_mtime = None
        return {"reports": stats.get("reports", []), "evals": stats.get("evals", []),
                "best": best, "latest_mtime": latest_mtime}

    # -- drawing ---------------------------------------------------------------
    def _style(self, ax, title):
        ax.set_facecolor(BG)
        ax.set_title(title, color=FG, fontsize=10, loc="left")
        ax.tick_params(colors=FG, labelsize=8)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.grid(color=GRID, linewidth=0.5, alpha=0.6)

    def redraw(self, snap):
        reports, evals = snap["reports"], snap["evals"]
        (ax_wr, ax_loss), (ax_thr, ax_mix) = self.axes
        for ax in (ax_wr, ax_loss, ax_thr, ax_mix):
            ax.clear()

        self._style(ax_wr, "win-rates vs iteration (evals)")
        for key, color in SERIES.items():
            xs, ys = _series(evals, key)
            if xs:
                ax_wr.plot(xs, ys, color=color, linewidth=1.4, label=key)
        stars = [(r["it"], r.get("heuristic")) for r in evals
                 if r.get("new_best") and r.get("it") is not None]
        if stars:
            ax_wr.scatter([s[0] for s in stars], [s[1] for s in stars],
                          marker="*", s=90, color="#ffd54f", zorder=5, label="new best")
        ax_wr.axhline(0.5, color=GRID, linewidth=0.8)
        ax_wr.set_ylim(-0.02, 1.02)
        if evals:
            ax_wr.legend(loc="lower right", fontsize=7, facecolor=BG,
                         edgecolor=GRID, labelcolor=FG)

        self._style(ax_loss, "losses (reports)")
        for key, color in LOSSES.items():
            xs, ys = _series(reports, key)
            if xs:
                lw = 0.9 if key in ("entropy", "approx_kl") else 1.4
                ax_loss.plot(xs, ys, color=color, linewidth=lw, label=key)
        if reports:
            ax_loss.legend(loc="upper right", fontsize=7, facecolor=BG,
                           edgecolor=GRID, labelcolor=FG, ncols=2)

        self._style(ax_thr, "throughput")
        xs, ys = _series(reports, "iters_per_h")
        if xs:
            ax_thr.plot(xs, ys, color="#4fc3f7", linewidth=1.4, label="iters/h")
        xs2, ys2 = _series(reports, "transitions")
        if xs2:
            ax2 = ax_thr.twinx()
            ax2.plot(xs2, ys2, color="#9ccc65", linewidth=1.0, alpha=0.8)
            ax2.tick_params(colors="#9ccc65", labelsize=7)
            for s in ax2.spines.values():
                s.set_color(GRID)
        if xs:
            ax_thr.legend(loc="upper left", fontsize=7, facecolor=BG,
                          edgecolor=GRID, labelcolor=FG)

        self._style(ax_mix, "opponent mix (reports)")
        mix_keys = [k for k in MIX if any(r.get(k) is not None for r in reports)]
        if mix_keys:
            rows = [r for r in reports
                    if r.get("it") is not None
                    and all(r.get(k) is not None for k in mix_keys)]
            if rows:
                xs = [r["it"] for r in rows]
                stacks = [[r[k] for r in rows] for k in mix_keys]
                ax_mix.stackplot(xs, stacks, labels=[k[4:] for k in mix_keys], alpha=0.85)
                ax_mix.set_ylim(0, 1)
                ax_mix.legend(loc="upper left", fontsize=7, facecolor=BG,
                              edgecolor=GRID, labelcolor=FG, ncols=3)

        self.canvas.draw_idle()

    def update_header(self, snap):
        reports, evals, best = snap["reports"], snap["evals"], snap["best"]
        last_r = reports[-1] if reports else {}
        last_e = evals[-1] if evals else {}
        it = last_r.get("it", last_e.get("it", "—"))
        elapsed = last_r.get("elapsed_h", last_e.get("elapsed_h"))
        iph = last_r.get("iters_per_h")
        best_txt = "—"
        if best:
            best_txt = f"heuristic {best.get('heuristic', 0):.2f} @ it {best.get('it', '?')}"
        self.header.config(text=(
            f"it {it}   elapsed {elapsed:.1f} h   {iph:.1f} it/h   best: {best_txt}"
            if isinstance(elapsed, (int, float)) and isinstance(iph, (int, float))
            else f"it {it}   best: {best_txt}   (waiting for first report…)"))

        # staleness: age of the newest datapoint vs the report cadence
        wall = max([r.get("wall_time", 0) for r in reports[-1:]] +
                   [e.get("wall_time", 0) for e in evals[-1:]] + [0])
        if wall:
            age_min = (time.time() - wall) / 60.0
            color = "#9ccc65" if age_min < 90 else ("#ffb74d" if age_min < 180 else "#e57373")
            self.badge.config(text=f"last datapoint {age_min:.0f} min ago", bg=color)
        else:
            self.badge.config(text="no data yet", bg="#ffb74d")

    # -- loop ------------------------------------------------------------------
    def poll(self):
        try:
            snap = self.snapshot()
            sig = (len(snap["reports"]), len(snap["evals"]),
                   snap["latest_mtime"], bool(snap["best"]))
            self.update_header(snap)
            if sig != self.last_sig:                    # redraw only on new data
                self.last_sig = sig
                self.redraw(snap)
        except Exception as e:                          # never let a draw glitch kill the window
            self.header.config(text=f"monitor error: {e!r}")
        self.root.after(self.refresh_ms, self.poll)


def main() -> None:
    ap = argparse.ArgumentParser(description="Live tkinter monitor over a fishrl run.")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--refresh", type=float, default=5.0, help="poll interval, seconds")
    args = ap.parse_args()

    if os.name == "nt":                                 # crisp text on high-DPI displays
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

    root = tk.Tk()
    Monitor(root, args.ckpt_dir, args.refresh)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
