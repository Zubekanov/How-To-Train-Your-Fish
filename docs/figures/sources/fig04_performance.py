"""04 · PERFORMANCE — overview: the learning curve, the final ladder, and three skills that changed under the same events."""
import sys, math; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fig import *

body = header("04", "PERFORMANCE", "How strong the released agent is, how it got there, and which habits it lost along the way")

def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h
def dot(x, y, col, filled=True, r=10, sw=3):
    return f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{r}" fill="{col if filled else "white"}" stroke="{col}" stroke-width="{sw}"/>\n'
def numcircle(x, y, n, col, r=18):
    return f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{r}" fill="{col}"/>' + text(x, y + 8, str(n), 22, 800, "white", "middle")

X0, X1 = 640, 3045
IT_MAX = 210000
def ix(it): return X0 + it / IT_MAX * (X1 - X0)
rules = [(47538, "critic swapped", PURPLE), (125000, "inputs widened", BLUE), (155430, "scenario self-play", GREEN), (184058, "learning rate halved, league spaced", ORANGE)]

# ---------------------------------------------------------------- A: learning curve
AY0, AY1 = 420, 1000
body += text(175, 310, "WIN-RATE vs THE STRONGEST SCRIPTED OPPONENT, through the v3 run", 34, 800, NAVY)
def ay(w): return AY1 - (w - 0.15) / 0.70 * (AY1 - AY0)
body += rect(X0, AY0, X1 - X0, AY1 - AY0, "#fbfcfe", "#d6deea", 1.5, 0)
for w in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    body += f'<path d="M{X0},{ay(w):.0f} H{X1}" stroke="#e3e9f2" stroke-width="1.5"/>\n' + text(X0 - 16, ay(w) + 9, f"{w:.1f}", 24, 400, GREY, "end")
body += f'<path d="M{X0},{ay(0.5):.0f} H{X1}" stroke="{GREY}" stroke-width="2" stroke-dasharray="8,8"/>\n' + text(X0 + 14, ay(0.5) - 10, "even", 22, 400, GREY)
for it in range(0, 200001, 50000):
    body += f'<path d="M{ix(it):.0f},{AY1} V{AY1 + 10}" stroke="{GREY}" stroke-width="2"/>\n' + text(ix(it), AY1 + 38, "0" if it == 0 else f"{it // 1000}k", 24, 400, GREY, "middle")
body += text(X1, AY1 + 70, "iterations", 22, 400, GREY, "end")
for k, (it, lab, col) in enumerate(rules):
    x = ix(it)
    body += f'<path d="M{x:.0f},{AY0} V{AY1}" stroke="{col}" stroke-width="2.5" stroke-dasharray="7,8" opacity="0.8"/>\n'
    body += numcircle(x, AY0 - 28, k + 1, col)
pts = [(15000, .21), (46343, .30), (107800, .530), (122700, .596), (135492, .643), (162900, .708), (184058, .724), (196263, .745)]
body += line([(ix(a), ay(b)) for a, b in pts], GREEN, 5, arrow=False)
for a, b in pts:
    body += dot(ix(a), ay(b), GREEN)
body += text(ix(15000) + 22, ay(.21) + 9, "0.21", 24, 600, GREY)
body += text(ix(135492) - 20, ay(.643) - 24, "+8 pp", 26, 800, BLUE, "end")
body += text(ix(190000), ay(.70) + 9, "+3 pp", 26, 800, ORANGE, "middle")
x = ix(196263)
body += dot(x, ay(.764), NAVY, True, 14)
body += text(x - 30, ay(.764) - 30, "0.764 released", 30, 800, NAVY, "end")

# ---------------------------------------------------------------- B: ladder
BY0 = 1130
body += text(175, BY0, "THE RELEASED AGENT vs EACH OPPONENT", 34, 800, NAVY)
body += text(175, BY0 + 34, "win-rate with its 95% range", 24, 400, GREY)
BX0, BX1 = 640, 1560
def bx(w): return BX0 + (w - 0.4) / 0.6 * (BX1 - BX0)
rows = [("random play", 1.000, 50), ("always attack", .980, 100), ("heuristic v1.0", .932, 400), ("heuristic v1.1", .850, 400), ("heuristic v1.2", .777, 400), ("heuristic v1.3 (strongest)", .764, 2400), ("an earlier copy of itself", .495, 400)]
RH = 54
T = BY0 + 70; B = T + 20 + len(rows) * RH
body += rect(BX0, T, BX1 - BX0, B - T, "#fbfcfe", "#d6deea", 1.5, 0)
for w in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
    dash = ' stroke-dasharray="8,8"' if w == 0.5 else ""
    body += f'<path d="M{bx(w):.0f},{T} V{B}" stroke="{"#9fb0c8" if w == 0.5 else "#e3e9f2"}" stroke-width="{2 if w == 0.5 else 1.5}"{dash}/>\n'
    body += text(bx(w), B + 30, f"{w:.1f}", 22, 400, GREY, "middle")
body += text(bx(0.5) + 8, T + 26, "even", 20, 400, GREY)
for k, (name, p, n) in enumerate(rows):
    y = T + 50 + k * RH
    lo, hi = wilson(p, n); col = NAVY if "strongest" in name else GREEN
    body += text(BX0 - 18, y + 9, name, 25, 700 if col == NAVY else 400, INK, "end")
    body += f'<path d="M{bx(lo):.0f},{y} H{bx(hi):.0f} M{bx(lo):.0f},{y - 7} V{y + 7} M{bx(hi):.0f},{y - 7} V{y + 7}" stroke="{col}" stroke-width="3"/>\n'
    body += dot(bx(p), y, col, True, 12 if col == NAVY else 10)
    body += text(BX1 + 20, y + 9, f"{p:.2f}", 26, 800 if col == NAVY else 600, col)
body += text(175, B + 90, "Against itself it is seat-fair: 0.501 / 0.499 across 1,600 mirror games.", 24, 400, INK)

# ---------------------------------------------------------------- C: three habits, same x axis
CX0 = ix(95000)
body += text(CX0, BY0, "THREE HABITS IT LOST", 34, 800, NAVY)
body += text(CX0, BY0 + 34, "same iteration axis as the curve above · each measured over 2,000 games", 24, 400, GREY)
strips = [("throwing away a Fact-or-Fiction split (0–5)", "% of splits", [(107800, 17.4), (122700, 17.1), (135492, .3), (162900, .1), (196263, .1)], 0, 20, "%"),
          ("telegraphing a trick in the main phase", "% of casts", [(122700, 22.5), (135492, 2.0), (162900, 1.4), (184058, 2.7), (196263, 1.0)], 0, 25, "%"),
          ("losing by running out of cards", "per 2,000 games", [(135492, 127), (162900, 69), (184058, 80), (196263, 70)], 0, 140, "")]
SY0, SH, SG = BY0 + 70, 170, 24
for k, (title, unit, pts, lo, hi, suf) in enumerate(strips):
    y0 = SY0 + k * (SH + SG); y1 = y0 + SH
    def sy(v, lo=lo, hi=hi, y0=y0, y1=y1): return y1 - 24 - (v - lo) / (hi - lo) * (y1 - y0 - 76)
    body += rect(CX0, y0, X1 - CX0, SH, "#fbfcfe", "#d6deea", 1.5, 0)
    body += text(CX0 + 16, y0 + 32, title, 25, 700, NAVY)
    body += text(X1 - 16, y0 + 32, unit, 22, 400, GREY, "end")
    for it, lab, col in rules[1:]:
        body += f'<path d="M{ix(it):.0f},{y0} V{y1}" stroke="{col}" stroke-width="2" stroke-dasharray="6,8" opacity="0.6"/>\n'
    body += line([(ix(a), sy(b)) for a, b in pts], GREEN, 3, arrow=False)
    for a, b in pts:
        body += dot(ix(a), sy(b), GREEN, True, 8, 2.5)
    a, b = pts[0]; body += text(ix(a) - 16, sy(b) + 9, f"{b:g}{suf}", 24, 700, GREY, "end")
    a, b = pts[-1]; body += text(ix(a) + 18, sy(b) + 9, f"{b:g}{suf}", 24, 700, GREEN)
yb = SY0 + 3 * (SH + SG) - SG
for it in range(100000, 200001, 50000):
    body += f'<path d="M{ix(it):.0f},{yb} V{yb + 10}" stroke="{GREY}" stroke-width="2"/>\n' + text(ix(it), yb + 36, f"{it // 1000}k", 22, 400, GREY, "middle")

# ---------------------------------------------------------------- key + footer
KY = max(B + 130, yb + 90)
for k, (it, lab, col) in enumerate(rules):
    x = 175 + k * 720
    body += numcircle(x + 18, KY - 8, k + 1, col) + text(x + 48, KY, lab, 24, 600, col)
body += text(175, KY + 60, "54.6M games · 12.6B decisions · 347 hours · one rented RTX 4090", 28, 700, NAVY)
H = KY + 130
body += footer(H - 40, "", "04 / 05")
open(__file__.rsplit("/", 1)[0] + "/../04-performance.svg", "w").write(svg(H, body, "04 · Performance"))
print("ok", H)
