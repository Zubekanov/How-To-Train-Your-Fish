"""03 · TRAINING LOOP — overview: where each iteration's games come from, and the two ways a weakness mine feeds back."""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fig import *

body = header("03", "TRAINING LOOP", "Where each iteration's games come from, and the two places a weakness mine can feed back into")

def box(x, y, w, h, color, title, lines, title_size=34, line_size=25, icon_kind=None):
    s = rect(x, y, w, h, FILL.get(color, "white"), color, 3, 18)
    tx = x + 28
    if icon_kind:
        s += icon(icon_kind, x + 24, y + 18, color, 0.9); tx = x + 76
    s += text(tx, y + 46, title, title_size, 700, color)
    cy = y + 46
    for ln in lines:
        for l in wrap(ln, line_size, w - 56):
            cy += line_size * 1.4
            s += text(x + 28, cy, l, line_size, 400, INK)
    return s

PX, PY, PW, PH = 1100, 300, 1000, 150
body += box(PX, PY, PW, PH, BLUE, "THE POLICY", ["one network plays every learner seat"], icon_kind="net")

GX, GY, GW, GH = 175, 540, 2870, 480
body += rect(GX, GY, GW, GH, "#f7f9fc", "#c6d3e5", 2, 20)
body += text(GX + 30, GY + 50, "GAMES PER ITERATION", 34, 700, NAVY)
BX, BY, BW, BH = GX + 30, GY + 130, GW - 60, 230
mirror_w = BW * 0.40
seg_w = (BW - mirror_w) / 3
body += text(BX + mirror_w / 2, BY - 24, "40% self-play", 30, 700, BLUE, "middle")
body += text(BX + mirror_w + (BW - mirror_w) / 2, BY - 24, "60% against the league · opponents the policy loses to are picked more often", 30, 700, GREEN, "middle")
body += f'<path d="M{BX},{BY - 10} H{BX + mirror_w - 6}" stroke="{BLUE}" stroke-width="4"/><path d="M{BX + mirror_w + 6},{BY - 10} H{BX + BW}" stroke="{GREEN}" stroke-width="4"/>\n'
segs = [(BLUE, mirror_w, "policy vs itself", ["both seats learn"]),
        (GREEN, seg_w, "6 scripted bots", ["random, attacker, four heuristics"]),
        (GREEN, seg_w, "8 past selves", ["frozen snapshots, ~16 h apart"]),
        (GREEN, seg_w, "18 scenarios", ["start mid-game in a known weak spot", "then play out normally"])]
x = BX
for col, w, t, lines in segs:
    body += rect(x, BY, w - 6, BH, FILL[col], col, 3, 12)
    body += text(x + 24, BY + 54, t, 32, 700, col)
    cy = BY + 62
    for l in lines:
        cy += 38; body += text(x + 24, cy, l, 25, 400, INK)
    x += w
body += text(BX + BW, BY + BH + 50, "league shares float with results; the three widths are not to scale", 22, 400, GREY, "end")

UX, UY, UW, UH = 1100, 1110, 1000, 150
body += box(UX, UY, UW, UH, BLUE, "PPO UPDATE", ["learn from the learner seats' moves"], icon_kind="loop")
body += line([(PX + PW / 2, PY + PH), (PX + PW / 2, GY)], BLUE, 6)
body += text(PX + PW / 2 + 16, PY + PH + 58, "plays", 25, 600, BLUE)
body += line([(GX + GW / 2, GY + GH), (UX + UW / 2, UY)], BLUE, 6)
body += text(GX + GW / 2 + 16, GY + GH + 58, "games", 25, 600, BLUE)
body += path(f"M{UX + UW},{UY + UH / 2} H3075 V{PY + PH / 2} H{PX + PW}", BLUE, 6)
body += text(3060, UY + UH / 2 - 22, "every iteration", 26, 700, BLUE, "end")

OY = 1350
body += text(540, OY, "EVERY FEW DAYS · A WEAKNESS MINE", 30, 700, ORANGE)
EX, EW, EH = 175, 620, 200
body += box(EX, OY + 30, EW, EH, GREY, "EVALUATE", ["an hourly panel against the scripted bots picks the best checkpoint"], 30, 24, icon_kind="chart")
MX, MW = 855, 620
body += box(MX, OY + 30, MW, EH, ORANGE, "MINE FOR WEAKNESSES", ["2,000 fresh games, the critic flags the costly decisions"], 30, 24, icon_kind="eye")
DX, DY_, DW, DH = 1535, OY + 30, 560, EH
cx, cy = DX + DW / 2, DY_ + DH / 2
body += f'<path d="M{cx},{DY_} L{DX + DW},{cy} L{cx},{DY_ + DH} L{DX},{cy} Z" fill="{FILL[ORANGE]}" stroke="{ORANGE}" stroke-width="3"/>\n'
body += text(cx, cy - 20, "Can the network", 26, 700, ORANGE, "middle") + text(cx, cy + 14, "even see the mistake?", 26, 700, ORANGE, "middle")
FX, FW = 2155, 890
F1Y = OY + 30; F2Y = OY + 290
body += box(FX, F1Y, FW, 220, PURPLE, "NO → SHOW IT MORE", ["widen the network's inputs in place, without restarting training", "the biggest single gain of the run came this way: +8 pp"], 30, 24)
body += box(FX, F2Y, FW, 190, GREEN, "YES → DRILL IT", ["add a scenario that starts in that spot to the league", "its share shrinks on its own once the lesson is learned"], 30, 24)
body += line([(EX + EW, OY + 30 + EH / 2), (MX, OY + 30 + EH / 2)], GREY, 5)
body += line([(MX + MW, OY + 30 + EH / 2), (DX, OY + 30 + EH / 2)], ORANGE, 5)
body += line([(DX + DW, cy), (FX, F1Y + 110)], PURPLE, 5)
body += text(DX + DW - 4, cy - 16, "no", 26, 700, PURPLE, "end")
body += path(f"M{cx},{DY_ + DH} V{F2Y + 95} H{FX}", GREEN, 5)
body += text(cx + 14, DY_ + DH + 40, "yes", 26, 700, GREEN)
body += path(f"M{UX},{UY + UH / 2} H{EX + EW / 2} V{OY + 30}", GREY, 5)
body += text(EX + EW / 2 + 14, UY + UH / 2 + 90, "checkpoints", 25, 600, GREY)
body += path(f"M{FX + FW},{F1Y + 110} H3125 V{PY + PH / 2 - 40} H{PX + PW}", PURPLE, 6)
body += text(3115, PY + PH / 2 - 60, "changes what the policy can see", 26, 700, PURPLE, "end")
seg3_x = BX + BW - 40
body += path(f"M{FX + FW},{F2Y + 95} H3175 V{BY - 60} H{seg3_x} V{BY}", GREEN, 6)
body += text(3040, GY - 16, "changes which games are played", 26, 700, GREEN, "end")

H = F2Y + 190 + 110
body += footer(H - 40, "", "03 / 05")
open(__file__.rsplit("/", 1)[0] + "/../03-training-loop.svg", "w").write(svg(H, body, "03 · Training loop"))
print("ok", H)
