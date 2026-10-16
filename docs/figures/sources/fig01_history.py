"""01 · TRAINING HISTORY — overview: three lineages on a calendar, then the final one by iteration."""
import sys, math; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fig import *
from datetime import date

body = header("01", "TRAINING HISTORY", "Three lineages, one warm start. The final agent is v3, which ran until the cloud credit ran out")

D0, D1 = date(2026, 6, 17), date(2026, 9, 2)
X0, X1 = 640, 3045
def dx(m, d): return X0 + (date(2026, m, d) - D0).days / (D1 - D0).days * (X1 - X0)

AX = 360
body += text(175, 310, "THREE LINEAGES · calendar, to scale", 34, 800, NAVY)
body += line([(X0, AX), (X1, AX)], "#c6d3e5", 2, arrow=False)
for m, d, lab in [(6, 17, "17 Jun"), (7, 1, "1 Jul"), (7, 16, "16 Jul"), (8, 1, "1 Aug"), (8, 17, "17 Aug"), (9, 2, "2 Sep")]:
    x = dx(m, d)
    body += f'<path d="M{x:.0f},{AX} V{AX + 12}" stroke="{GREY}" stroke-width="2"/>\n' + text(x, AX - 12, lab, 24, 600, GREY, "middle")

BH = 46
def dot(x, y, color, filled=True, r=14):
    return f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{r}" fill="{color if filled else "white"}" stroke="{color}" stroke-width="4"/>\n'
def star(cx, cy, r, color):
    pts = []
    for i in range(10):
        rr = r if i % 2 == 0 else r * 0.45
        a = -math.pi / 2 + i * math.pi / 5
        pts.append(f"{cx + rr * math.cos(a):.1f},{cy + rr * math.sin(a):.1f}")
    return f'<polygon points="{" ".join(pts)}" fill="{GOLD}" stroke="{color}" stroke-width="3"/>\n'

# v1
Y1 = AX + 70
x0, x1 = dx(6, 17), dx(7, 16)
body += text(175, Y1 + 14, "v1", 40, 800, BLUE) + text(175 + 70, Y1 + 14, "flat network", 28, 600, BLUE)
body += text(175, Y1 + 46, "fresh start", 22, 400, GREY)
body += rect(x0, Y1, x1 - x0, BH, BLUE, "none", 0, 10) + dot(x0, Y1 + BH / 2, BLUE)
body += text(x1 + 30, Y1 + 14, "622 hours", 32, 800, BLUE)
body += text(x1 + 30, Y1 + 46, "plateaued: the network could not use what it was shown", 25, 400, INK)

# v2
Y2 = Y1 + 190
x0, x1 = dx(7, 16), dx(8, 17)
body += text(175, Y2 + 14, "v2", 40, 800, PURPLE) + text(175 + 70, Y2 + 14, "entity network", 28, 600, PURPLE)
body += text(175, Y2 + 46, "fresh start, nothing kept from v1", 22, 400, GREY)
body += rect(x0, Y2, x1 - x0, BH, PURPLE, "none", 0, 10) + dot(x0, Y2 + BH / 2, PURPLE)
body += text(x1 + 30, Y2 + 14, "382 hours", 32, 800, PURPLE)
body += text(x1 + 30, Y2 + 46, "stopped: its privileged critic and learned", 25, 400, INK)
body += text(x1 + 30, Y2 + 76, "belief net were measured as paying nothing", 25, 400, INK)

# warm start
Y3 = Y2 + 250
xw = dx(8, 17)
body += line([(xw, Y2 + BH + 6), (xw, Y3 - 18)], GREEN, 6)
body += text(xw - 24, Y2 + BH + 110, "warm start: the policy network's weights carried over,", 25, 700, GREEN, "end")
body += text(xw - 24, Y2 + BH + 142, "the critic and the belief model were rebuilt", 25, 400, INK, "end")

# v3
x0, x1 = dx(8, 17), dx(9, 2)
body += text(175, Y3 + 14, "v3", 40, 800, GREEN) + text(175 + 70, Y3 + 14, "final lineage", 28, 600, GREEN)
body += text(175, Y3 + 46, "warm start from v2", 22, 400, GREY)
body += rect(x0, Y3, x1 - x0, BH, GREEN, "none", 0, 10) + dot(x0, Y3 + BH / 2, GREEN)
body += star(dx(9, 1) - 22, Y3 + BH / 2, 24, NAVY)
body += f'<rect x="{x1 - 8:.0f}" y="{Y3 - 10}" width="16" height="{BH + 20}" fill="{GREEN}"/>\n'
body += text(x1, Y3 - 22, "cloud credit ran out", 24, 400, GREY, "end")
xc = dx(8, 19)
body += rect(xc, Y3 + BH + 8, x1 - xc, 8, PALE, "none", 0, 3)
body += text(xc, Y3 + BH + 40, "on a rented RTX 4090 from 19 Aug", 22, 400, GREY)
RX = x0 - 60
body += text(RX, Y3 + 14, "347 hours", 32, 800, GREEN, "end")
body += text(RX, Y3 + 48, "54.6M games · 12.6B decisions", 25, 400, INK, "end")
body += text(RX, Y3 + 80, "★ the released agent, iteration 196,263", 25, 400, INK, "end")

# ---------------------------------------------------------------- detail: v3 by iteration
SY = Y3 + 230
body += text(175, SY, "v3 BY ITERATION · win-rate vs the strongest scripted opponent", 34, 800, NAVY)
IT_END = 204225
def ix(it): return X0 + it / IT_END * (X1 - X0)
PY0, PY1 = SY + 60, SY + 560
def wy(w): return PY1 - (w - 0.15) / 0.70 * (PY1 - PY0)
body += rect(X0, PY0, X1 - X0, PY1 - PY0, "#fbfcfe", "#d6deea", 1.5, 0)
for w in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    body += f'<path d="M{X0},{wy(w):.0f} H{X1}" stroke="#e3e9f2" stroke-width="1.5"/>\n' + text(X0 - 16, wy(w) + 9, f"{w:.1f}", 24, 400, GREY, "end")
body += f'<path d="M{X0},{wy(0.5):.0f} H{X1}" stroke="{GREY}" stroke-width="2" stroke-dasharray="8,8"/>\n'
body += text(X0 + 16, wy(0.5) - 10, "even", 22, 400, GREY)
for it in range(0, 200001, 50000):
    body += f'<path d="M{ix(it):.0f},{PY1} V{PY1 + 10}" stroke="{GREY}" stroke-width="2"/>\n' + text(ix(it), PY1 + 38, "0" if it == 0 else f"{it // 1000}k", 24, 400, GREY, "middle")
body += text(X1, PY1 + 70, "iterations", 22, 400, GREY, "end")
# points
pts = [(15000, .21), (46343, .30), (107800, .530), (122700, .596), (135492, .643), (162900, .708), (184058, .724), (196263, .745)]
body += line([(ix(a), wy(b)) for a, b in pts], GREEN, 5, arrow=False)
for a, b in pts:
    body += dot(ix(a), wy(b), GREEN, True, 10)
body += text(ix(15000) + 20, wy(.21) + 9, "0.21", 24, 600, GREY)
body += star(ix(196263), wy(0.764), 26, NAVY)
body += text(ix(196263) - 36, wy(0.764) - 4, "0.764 released", 30, 800, NAVY, "end")
# four named events
EY = PY1 + 80
events = [(15000, "moved to the rented GPU", GREY), (47538, "critic swapped in place", PURPLE), (125000, "inputs widened in place (+8 pp)", BLUE), (184058, "learning rate halved,\nleague spaced out (+3 pp)", ORANGE)]
for it, lab, col in events:
    x = ix(it)
    body += f'<path d="M{x:.0f},{PY0} V{EY - 14}" stroke="{col}" stroke-width="2" stroke-dasharray="7,8" opacity="0.7"/>\n'
    body += f'<path d="M{x - 10:.0f},{EY} L{x + 10:.0f},{EY} L{x:.0f},{EY - 14} Z" fill="{col}"/>\n'
    for k, l in enumerate(lab.split("\n")):
        body += text(x, EY + 34 + k * 30, l, 24, 700, col, "middle")
H = EY + 150
body += footer(H - 40, "", "01 / 05")
open(__file__.rsplit("/", 1)[0] + "/../01-training-history.svg", "w").write(svg(H, body, "01 · Training history"))
print("ok", H)
