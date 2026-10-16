"""05 · ARCHITECTURAL MISTAKES & CHANGES — overview: what each part of the system started as, and what replaced it."""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fig import *

body = header("05", "ARCHITECTURAL MISTAKES & CHANGES", "Every clever part the project started with was measured, found to pay nothing, and replaced with something simpler")
body += ('<defs><pattern id="hatch5" width="12" height="12" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
         f'<rect width="12" height="12" fill="#f7f9fc"/><rect width="3" height="12" fill="#d6deea"/></pattern></defs>\n')

# time axis: v1 | v2 | v3 (by iteration)
V1X, V1W = 620, 340
V2X, V2W = 980, 340
V3X, V3W = 1340, 1705
IT_END = 205000
def x3(it): return V3X + it / IT_END * V3W
AX = 330
for x, w, a in [(V1X, V1W, "v1 · 622 h"), (V2X, V2W, "v2 · 382 h"), (V3X, V3W, "v3 · 347 h, by iteration")]:
    body += rect(x, AX, w, 46, "#e9eef6", "none", 0, 6) + text(x + w / 2, AX + 31, a, 24, 700, NAVY, "middle")
for it in (50000, 100000, 150000, 200000):
    body += f'<path d="M{x3(it):.0f},{AX + 46} V{AX + 58}" stroke="{GREY}" stroke-width="2"/>\n' + text(x3(it), AX + 82, f"{it // 1000}k", 20, 400, GREY, "middle")

LANE_Y0 = AX + 110
LH = 250
lanes = [("WHAT THE NETS SEE", BLUE), ("THE POLICY NET", PURPLE), ("THE CRITIC", ORANGE), ("BELIEF ABOUT THE OPPONENT", GOLD), ("OPPONENTS & CURRICULUM", GREEN), ("THE OPTIMISER", RED)]
def lane_y(i): return LANE_Y0 + i * LH
def base(i): return lane_y(i) + 125
for i, (t, col) in enumerate(lanes):
    y = lane_y(i)
    body += rect(175, y, 2870, LH - 12, "#fbfcfe" if i % 2 == 0 else "white", "#e3e9f2", 1.5, 10)
    for k, l in enumerate(wrap(t, 24, 400, 800)):
        body += text(195, y + 44 + k * 28, l, 24, 800, col)
    body += f'<path d="M{V1X},{base(i)} H{x3(IT_END)}" stroke="#c6d3e5" stroke-width="2"/>\n'
TOP, BOT = AX + 46, lane_y(6) - 12
for x, lab in [((V1X + V1W + V2X) / 2, "restart"), ((V2X + V2W + V3X) / 2, "restart (warm)")]:
    body += f'<path d="M{x:.0f},{TOP} V{BOT}" stroke="{NAVY}" stroke-width="7"/>\n'
    body += text(x, AX - 14, lab, 22, 700, NAVY, "middle")

def diamond(x, y, col, filled=True, r=14):
    return f'<polygon points="{x},{y - r} {x + r},{y} {x},{y + r} {x - r},{y}" fill="{col if filled else "white"}" stroke="{col}" stroke-width="3.5"/>\n'
def cross(x, y, col, r=13):
    return (f'<circle cx="{x}" cy="{y}" r="{r}" fill="white" stroke="{col}" stroke-width="3"/>'
            f'<path d="M{x - 6},{y - 6} L{x + 6},{y + 6} M{x + 6},{y - 6} L{x - 6},{y + 6}" stroke="{col}" stroke-width="3"/>\n')
def label(x, y, side, lines, col, anchor="middle", size=22, width=320, bold=False):
    out = []
    for ln in lines: out += wrap(ln, size, width, 700 if bold else 400)
    s = ""
    for k, l in enumerate(out):
        yy = y - 26 - (len(out) - 1 - k) * (size + 6) if side == "above" else y + 44 + k * (size + 6)
        s += text(x, yy, l, size, 700 if bold else 400, col if bold else INK, anchor)
    return s
def mark(i, x, kind, col, lines, side, anchor="middle", width=320, bold=False, dx=16):
    y = base(i)
    s = {"inplace": diamond(x, y, col), "abandon": cross(x, y, col), "note": ""}[kind]
    lx = x if kind == "note" else x + (dx if anchor == "start" else (-dx if anchor == "end" else 0))
    return s + label(lx, y, side, lines, col, anchor, width=width, bold=bold)
def before(i, txt, width=320):
    return label(V1X + 10, base(i), "above", [txt], GREY, "start", width=width)

i = 0
body += before(i, "a smooth 'cards left' number; piles, stack targets and search lists not shown", 400)
body += mark(i, V1X + 27 / 29 * V1W, "inplace", BLUE, ["deckout clock added mid-run: ignored"], "below", "end", 300)
body += mark(i, (V2X + V2W / 2), "note", BLUE, ["same clock, trained from scratch: used"], "below", "middle", 300)
body += mark(i, x3(125000), "inplace", BLUE, ["inputs widened in place: piles, targets, context"], "above", "middle", 380)
body += mark(i, x3(125000), "note", BLUE, ["+8 pp, the biggest gain of the run"], "below", "middle", 380, bold=True)

i = 1
body += before(i, "flat network, 92% of its weights in one input layer")
body += label(V2X + 12, base(i), "below", ["entity network: one shared card reader, pooled per zone"], PURPLE, "start", width=440, bold=True)
body += mark(i, V2X + V2W / 2, "abandon", PURPLE, ["attention: no gain at 8× the cost"], "above", "middle", 300)

i = 2
body += before(i, "saw both hands and the hidden library order")
body += label(V2X + 12, base(i), "below", ["its edge over an honest critic shrank to nothing"], GREY, "start", width=320)
body += mark(i, x3(47538), "inplace", ORANGE, ["swapped for a critic that sees both hands only, and does better"], "above", "start", 420)
body += mark(i, x3(78700), "abandon", ORANGE, ["heavy consistency loss stalled the run"], "below", "start", 320)
body += mark(i, x3(184058), "inplace", ORANGE, ["a light consistency loss on payment steps"], "above", "end", 300)

i = 3
body += before(i, "a learned net guessing the opponent's hand")
body += mark(i, V2X + V2W / 2, "abandon", GOLD, ["imitation learning: copies moves, still loses"], "below", "middle", 320)
body += label(V3X + 12, base(i), "above", ["replaced by plain card counting: no parameters, a third faster"], GOLD, "start", width=460, bold=True)

i = 4
body += before(i, "7 hand-built scenarios in positions no real game reaches; past selves 2 h apart", 330)
body += mark(i, x3(46343), "inplace", GREEN, ["scenarios sampled from real games, 10 → 18"], "below", "start", 360)
body += mark(i, x3(155430), "inplace", GREEN, ["scenarios played against itself, not a bot"], "above", "end", 300)
body += mark(i, x3(164300), "inplace", GREEN, ["scenario share cut: lessons learned"], "below", "start", 300)
body += mark(i, x3(184058), "inplace", GREEN, ["past selves spaced 16 h apart"], "above", "start", 230)

i = 5
body += before(i, "one learning rate for 200k iterations, no schedule", 330)
body += rect(x3(152000), base(i) - 10, x3(184058) - x3(152000), 20, "#fde3e1", "none", 0, 6)
body += text(x3(152000) + 10, base(i) + 7, "30k iterations of churn", 18, 600, RED)
body += mark(i, x3(184058), "inplace", RED, ["learning rate halved: +3 pp with the league change"], "above", "end", 340, bold=True)
body += mark(i, x3(196735), "abandon", RED, ["entropy reheat: no harm, no gain"], "below", "end", 300)

LY = lane_y(6) + 20
items = [(diamond(0, 0, NAVY), "changed in place, training continued"), (f'<path d="M0,-16 V16" stroke="{NAVY}" stroke-width="7"/>', "restarted from scratch"), (cross(0, 0, NAVY), "tried, measured, dropped")]
for k, (g, lab) in enumerate(items):
    x = 175 + k * 760
    body += f'<g transform="translate({x + 16},{LY + 16})">{g}</g>\n' + text(x + 48, LY + 24, lab, 24, 400, INK)
H = LY + 110
body += footer(H - 40, "", "05 / 05")
open(__file__.rsplit("/", 1)[0] + "/../05-architectural-changes.svg", "w").write(svg(H, body, "05 · Architectural mistakes & changes"))
print("ok", H)
