"""02 · ARCHITECTURE & INFERENCE — overview: the play-time path on top, the training-only critic below."""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fig import *

body = header("02", "ARCHITECTURE & INFERENCE", "What one decision runs through at play time, and the critic that exists only in training")
body += ('<defs><pattern id="hatch" width="10" height="10" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
         f'<rect width="10" height="10" fill="white"/><rect width="4" height="10" fill="{RED}"/></pattern></defs>\n')

def box(x, y, w, h, color, title, lines, title_size=32, line_size=25, icon_kind=None, fill=None, big=None):
    s = rect(x, y, w, h, fill or FILL.get(color, "white"), color, 3, 18)
    tx = x + 26
    if icon_kind:
        s += icon(icon_kind, x + 22, y + 18, color, 0.9); tx = x + 74
    s += text(tx, y + 48, title, title_size, 700, color)
    cy = y + 52
    if big:
        cy += 48; s += text(x + 26, cy, big, 40, 800, color)
    for ln in lines:
        for l in wrap(ln, line_size, w - 52):
            cy += line_size * 1.4
            s += text(x + 26, cy, l, line_size, 400, INK)
    return s

# ---------------------------------------------------------------- lane 1: play time
body += text(175, 310, "PLAY TIME · one decision, left to right", 34, 800, NAVY)

EX, EY, EW, EH = 175, 380, 330, 1020
body += rect(EX, EY, EW, EH, "#f7f9fc", NAVY, 3, 20)
body += icon("gear", EX + 22, EY + 20, NAVY, 1.0)
body += text(EX + 76, EY + 50, "GAME ENGINE", 32, 700, NAVY)
for i, l in enumerate(["rules, hidden information,", "and which actions are legal"]):
    body += text(EX + 26, EY + 100 + i * 32, l, 25, 400, INK)

# observation box
OX, OY, OW, OH = 540, 400, 560, 520
body += box(OX, OY, OW, OH, BLUE, "WHAT THIS SEAT CAN SEE", [], icon_kind="eye")
# proportional bar: 6,424 cards | 73 globals | 721 counting
BX, BY, BW, BH = OX + 26, OY + 90, OW - 52, 70
tot = 7218
segs = [(6424, BLUE), (73, NAVY), (721, PURPLE)]
x = BX
for n, col in segs:
    w = n / tot * BW
    body += f'<rect x="{x:.1f}" y="{BY}" width="{w:.1f}" height="{BH}" fill="{col}" stroke="white" stroke-width="2"/>\n'
    x += w
body += text(BX, BY + BH + 40, "146 card slots × 44 features", 25, 700, BLUE)
body += text(BX, BY + BH + 74, "hand, battlefield, graveyard, exile, stack, library", 22, 400, GREY)
body += text(BX, BY + BH + 122, "+ 73 game-state values", 25, 700, NAVY)
body += text(BX, BY + BH + 156, "life, lands, mana, turn, step, what is pending", 22, 400, GREY)
body += text(BX, BY + BH + 204, "+ card counting", 25, 700, PURPLE)
body += text(BX, BY + BH + 238, "expected cards in the opponent's hand, piles and", 22, 400, GREY)
body += text(BX, BY + BH + 266, "targets on the stack · pure arithmetic, no parameters", 22, 400, GREY)
body += text(OX + OW - 26, OY + OH - 26, "7,218 numbers in", 30, 800, NAVY, "end")
body += line([(EX + EW, OY + OH / 2), (OX - 6, OY + OH / 2)], BLUE, 6)

# network boxes
NY, NH = 400, 520
N1X, N1W = 1160, 500
N2X, N2W = 1690, 330
MX, MW = 1850, 0
boxes_x = [1160, 1690, 2060, 2400, 2740]
widths = [500, 340, 310, 310, 305]
body += box(1160, NY, 500, NH, BLUE, "ENTITY ENCODER", ["one small shared network reads every card slot the same way, then pools each zone", "the game-state values and card counts join after pooling"], icon_kind="net", big="→ 2,842")
body += box(1690, NY, 340, NH, BLUE, "POLICY HEAD", ["three dense layers", "768 → 768 → 384", "one score per action"], icon_kind="net", big="→ 285")
body += box(2060, NY, 310, NH, ORANGE, "MASK", ["illegal actions are blanked out", "the engine's only say in the choice"], icon_kind="mask")
body += box(2400, NY, 310, NH, GREEN, "SAMPLE", ["pick one action by its probability", "(not the best-scoring one)"], icon_kind="dice")
body += box(2740, NY, 305, NH, NAVY, "APPLY", ["the engine plays it", "multi-step choices are built one sub-action at a time"], line_size=23, icon_kind="gear")
mid = NY + NH / 2
body += line([(OX + OW, mid), (1160 - 6, mid)], NAVY, 6)
for a, b, col in [(1660, 1690, BLUE), (2030, 2060, BLUE), (2370, 2400, ORANGE), (2710, 2740, GREEN)]:
    body += line([(a, mid), (b - 6, mid)], col, 6)
# return arrow along the top
RY = 350
body += path(f"M{2740 + 152},{NY} V{RY} H{EX + EW / 2} V{EY - 6}", NAVY, 6)
body += text(2740 + 140, RY - 14, "next decision", 26, 700, NAVY, "end")
# mask entry from engine
LY = 975
body += path(f"M{EX + EW},{LY} H{2060 + 155} V{NY + NH + 6}", ORANGE, 5, dashed=True)
body += text(2060 + 135, LY - 14, "which actions are legal right now", 25, 600, ORANGE, "end")

# action strip
KY, KH = 1070, 60
KX, KW = 1160, 3045 - 1160
U = KW / 285
groups = [(2, "pass", GREY), (36, "play from hand", BLUE), (102, "use the battlefield", BLUE), (6, "pay", ORANGE), (6, "setup", GREY), (25, "text change", PURPLE), (66, "pick a target", GREEN), (42, "arrange / split", GREEN)]
body += text(KX, KY - 24, "THE 285 ACTIONS", 28, 800, NAVY)
body += text(KX + tw("THE 285 ACTIONS", 28, 800) + 18, KY - 24, "width = how many slots each kind has", 24, 400, GREY)
x = KX
for i, (n, lab, col) in enumerate(groups):
    w = n * U
    body += f'<rect x="{x:.1f}" y="{KY}" width="{w:.1f}" height="{KH}" fill="{FILL[col]}" stroke="{col}" stroke-width="2"/>\n'
    if tw(lab, 23, 600) < w - 12:
        body += text(x + w / 2, KY + KH / 2 + 8, lab, 23, 600, col, "middle")
    else:
        yy = KY + KH + 36
        body += f'<path d="M{x + w / 2:.1f},{KY + KH} V{yy - 18}" stroke="{col}" stroke-width="2"/>\n'
        anchor = "middle" if i == 0 else ("end" if i == 3 else "start")
        body += text(x + w / 2 + (0 if i == 0 else (-8 if i == 3 else 8)), yy, lab, 23, 600, col, anchor)
    x += w
for aid in (145, 242):
    hx = KX + aid * U
    body += f'<rect x="{hx:.1f}" y="{KY}" width="{U:.1f}" height="{KH}" fill="url(#hatch)" stroke="{RED}" stroke-width="2"/>\n'
body += text(3045, KY + KH + 36, "hatched = the two 'cancel' actions, never allowed (they taught stalling)", 23, 400, RED, "end")

# headline numbers lane 1
body += text(540, 1245, "3.2M parameters", 34, 800, NAVY)
body += text(540 + tw("3.2M parameters", 34, 800) + 30, 1245, "about 1 ms per decision on one CPU core · a fifth of decisions have one legal action and skip the network", 25, 400, GREY)

# ---------------------------------------------------------------- lane 2: critic
DR = 1310
body += path(f"M{EX + EW + 30},{DR} H3045", GREY, 3, dashed=True, arrow=False)
body += text(3045, DR - 14, "TRAINING ONLY · never runs at play time", 30, 800, GREY, "end")

CY = 1370; CH = 300
body += box(540, CY, 560, CH, PURPLE, "WHAT THE CRITIC SEES", ["both players' hands, nothing of the library order", "plus the same counting features"], icon_kind="eye", big="6,846 numbers in")
body += line([(EX + EW, CY + CH / 2), (540 - 6, CY + CH / 2)], PURPLE, 6)
body += box(1160, CY, 500, CH, PURPLE, "ENTITY ENCODER", ["its own copy, same design"], icon_kind="net", big="→ 2,566")
body += box(1690, CY, 340, CH, PURPLE, "VALUE HEAD", ["512 → 512 → 256"], icon_kind="net", big="→ 1")
body += box(2060, CY, 650, CH, PURPLE, "P(player 1 wins)", ["used only to weight the policy's updates: it never touches a decision"], icon_kind="chart")
body += line([(1100, CY + CH / 2), (1154, CY + CH / 2)], PURPLE, 6)
body += line([(1660, CY + CH / 2), (1684, CY + CH / 2)], PURPLE, 6)
body += line([(2030, CY + CH / 2), (2054, CY + CH / 2)], PURPLE, 6)
body += text(2740, CY + 60, "1.7M parameters", 34, 800, PURPLE)
body += text(2740, CY + 104, "4.9M trained in total", 25, 400, GREY)
body += text(2740, CY + 140, "no learned belief net, no attention,", 22, 400, GREY)
body += text(2740, CY + 170, "no search on top", 22, 400, GREY)

H = CY + CH + 110
body += footer(H - 40, "", "02 / 05")
open(__file__.rsplit("/", 1)[0] + "/../02-architecture-inference.svg", "w").write(svg(H, body, "02 · Architecture & inference"))
print("ok", H)
