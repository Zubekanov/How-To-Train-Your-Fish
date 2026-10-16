"""Tiny SVG builder reproducing the kaggriculture-solution figure style.

Canvas is 3200 wide, Arial, navy header bar, colour-themed rounded cards with an icon,
a bold title, a bold headline and small grey detail text.
"""
from html import escape

W = 3200
NAVY, INK, GREY = "#10294e", "#142a44", "#52627b"
BLUE, GREEN, PURPLE, ORANGE, GOLD, RED = "#2459ba", "#18805c", "#7949b7", "#b85c12", "#9d7115", "#b3261e"
TEAL, PALE = "#91e0d0", "#dce6f4"
FILL = {BLUE: "#f1f6ff", GREEN: "#eff9f3", PURPLE: "#f6f1fd", ORANGE: "#fff6e9",
        GOLD: "#fff8df", GREY: "#ffffff", NAVY: "#ffffff", RED: "#fdf0ef"}
FONT = "Arial, Helvetica, sans-serif"
PROJECT = "HOW TO TRAIN YOUR FISH"
TAGLINE = "MASKED PPO SELF-PLAY · PFSP LEAGUE · SCENARIO CURRICULUM"


def tw(s, size, weight=400):
    """Rough Arial text width."""
    k = 0.60 if weight >= 600 else 0.545
    return len(s) * size * k


def text(x, y, s, size=27, weight=400, fill=INK, anchor="start", extra=""):
    return (f'<text x="{x:.0f}" y="{y:.0f}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}" {extra}>{escape(str(s))}</text>\n')


def wrap(s, size, width, weight=400):
    words, lines, cur = s.split(), [], ""
    for w_ in words:
        t = (cur + " " + w_).strip()
        if tw(t, size, weight) > width and cur:
            lines.append(cur); cur = w_
        else:
            cur = t
    if cur: lines.append(cur)
    return lines


def rect(x, y, w, h, fill="white", stroke=NAVY, sw=2.5, rx=16, extra=""):
    return (f'<rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {extra}/>\n')


def line(pts, color=BLUE, sw=4, dashed=False, arrow=True, extra=""):
    d = "M " + " L ".join(f"{x:.0f},{y:.0f}" for x, y in pts)
    dash = ' stroke-dasharray="14,12"' if dashed else ""
    m = f' marker-end="url(#arrow-{color[1:]})"' if arrow else ""
    return (f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" '
            f'stroke-linecap="round" stroke-linejoin="round"{dash}{m} {extra}/>\n')


def path(d, color=BLUE, sw=4, dashed=False, arrow=True, fill="none"):
    dash = ' stroke-dasharray="14,12"' if dashed else ""
    m = f' marker-end="url(#arrow-{color[1:]})"' if arrow else ""
    return (f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{sw}" '
            f'stroke-linecap="round" stroke-linejoin="round"{dash}{m}/>\n')


# ---- icons (drawn at ~44px, origin = top-left of a 44x44 box) -------------------
def icon(kind, x, y, color, scale=1.0):
    g = f'<g transform="translate({x:.0f},{y:.0f}) scale({scale})" fill="white" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">'
    if kind == "db":      # cylinder
        g += ('<path d="M4,10 V34 C4,40 40,40 40,34 V10"/><ellipse cx="22" cy="10" rx="18" ry="7"/>'
              '<path d="M4,22 C4,28 40,28 40,22" fill="none"/>')
    elif kind == "net":   # small network
        g += ('<path d="M6,36 L22,6 L38,36 Z M6,36 L22,26 L38,36 M22,6 L22,26" fill="none" stroke-width="2"/>'
              '<circle cx="6" cy="36" r="4.5"/><circle cx="22" cy="6" r="4.5"/><circle cx="38" cy="36" r="4.5"/><circle cx="22" cy="26" r="4.5"/>')
    elif kind == "loop":  # circular arrow
        g += ('<path d="M34,26 A14,14 0 1,1 30,12" fill="none" stroke-width="4"/>'
              '<path d="M24,8 L31,13 L25,19" fill="none" stroke-width="4"/>')
    elif kind == "doc":
        g += ('<path d="M8,4 H26 L36,14 V40 H8 Z"/><path d="M26,4 V14 H36 M14,22 H30 M14,29 H30 M14,36 H24" fill="none"/>')
    elif kind == "gear":
        g += ('<path d="M18,4 H26 L28,10 L34,12 L39,9 L43,16 L39,20 V24 L43,28 L39,35 L34,32 L28,34 L26,40 H18 L16,34 L10,32 L5,35 L1,28 L5,24 V20 L1,16 L5,9 L10,12 L16,10 Z"/>'
              '<circle cx="22" cy="22" r="7"/>')
    elif kind == "eye":
        g += ('<path d="M3,22 C10,10 34,10 41,22 C34,34 10,34 3,22 Z"/><circle cx="22" cy="22" r="6"/>')
    elif kind == "mask":  # action mask: grid with one blocked cell
        g += ('<rect x="4" y="4" width="36" height="36" rx="4"/><path d="M4,16 H40 M4,28 H40 M16,4 V40 M28,4 V40" fill="none"/>'
              f'<rect x="17" y="17" width="10" height="10" fill="{color}" stroke="none"/>')
    elif kind == "chip":  # hardware
        g += ('<rect x="10" y="10" width="24" height="24" rx="3"/><path d="M16,10 V4 M22,10 V4 M28,10 V4 M16,40 V34 M22,40 V34 M28,40 V34 M10,16 H4 M10,22 H4 M10,28 H4 M40,16 H34 M40,22 H34 M40,28 H34" fill="none"/>')
    elif kind == "swords":  # opponents
        g += ('<path d="M6,38 L34,10 M30,6 L38,14 M6,38 L12,38 M6,38 L6,32" fill="none" stroke-width="4"/>'
              '<path d="M38,38 L10,10 M6,14 L14,6 M38,38 L32,38 M38,38 L38,32" fill="none" stroke-width="4"/>')
    elif kind == "flag":
        g += ('<path d="M8,40 V4 M8,6 H36 L30,14 L36,22 H8" />')
    elif kind == "fish":
        g += ('<path d="M4,22 C12,10 26,8 34,20 L42,12 V32 L34,24 C26,36 12,34 4,22 Z"/><circle cx="14" cy="20" r="2.5" fill="' + color + '" stroke="none"/>')
    elif kind == "clock":
        g += ('<circle cx="22" cy="22" r="18"/><path d="M22,10 V22 L30,27" fill="none" stroke-width="4"/>')
    elif kind == "chart":
        g += ('<path d="M4,40 V4 M4,40 H40" fill="none"/><path d="M10,32 L18,22 L26,26 L38,10" fill="none" stroke-width="4"/>')
    elif kind == "dice":
        g += ('<rect x="4" y="4" width="36" height="36" rx="6"/>' + "".join(
            f'<circle cx="{cx}" cy="{cy}" r="3.5" fill="{color}" stroke="none"/>' for cx, cy in [(12, 12), (32, 12), (22, 22), (12, 32), (32, 32)]))
    g += "</g>\n"
    return g


# ---- composite pieces ------------------------------------------------------------
def header(num, title, subtitle, project=PROJECT, tagline=TAGLINE):
    s = f'<g id="figure-header" font-family="{FONT}"><rect width="{W}" height="240" fill="{NAVY}"/>'
    s += text(175, 64, project, 42, 800, "white")
    s += text(3045, 64, tagline, 32, 600, TEAL, "end")
    s += text(175, 147, f"{num} · {title}", 58, 700, "white")
    s += text(175, 206, subtitle, 32, 400, PALE)
    return s + "</g>\n"


def band(x, y, w, letter, title, right=""):
    s = rect(x, y, w, 100, NAVY, "none", 0, 16)
    s += text(x + 32, y + 65, letter, 41, 800, TEAL)
    s += text(x + 120, y + 65, title, 39, 700, "white")
    if right:
        s += text(x + w - 38, y + 64, right, 26, 400, PALE, "end")
    return s


def card(x, y, w, color, title, headline=None, lines=(), small=(), icon_kind="net", h=None,
         fill=None, stroke_w=2.5, title_size=30, headline_size=33, line_size=27, small_size=25,
         dashed=False, faded=False):
    """A rounded card. Returns (svg, height)."""
    pad = 26
    inner = w - 2 * pad
    body = []
    for ln in lines:
        body += [(l, line_size, 400, INK) for l in wrap(ln, line_size, inner)]
    for ln in small:
        body += [(l, small_size, 400, GREY) for l in wrap(ln, small_size, inner)]
    hl_lines = wrap(headline, headline_size, inner, 700) if headline else []
    hh = 52 + 42 * len(hl_lines) + sum(sz * 1.42 for _, sz, _, _ in body) + (18 if body or hl_lines else 10)
    h = h or hh
    fillc = fill or FILL.get(color, "white")
    extra = 'stroke-dasharray="12,10"' if dashed else ""
    if faded:
        extra += ' opacity="0.45"'
    s = f'<g {"opacity=\"0.5\"" if faded else ""}>'
    s += rect(x, y, w, h, fillc, color, stroke_w, 16, extra='stroke-dasharray="12,10"' if dashed else "")
    s += icon(icon_kind, x + pad - 4, y + 16, color, 0.8)
    s += text(x + pad + 40, y + 44, title, title_size, 700, color)
    cy = y + 52
    for l in hl_lines:
        cy += 42
        s += text(x + pad, cy - 6, l, headline_size, 700, color)
    for l, sz, wt, col in body:
        cy += sz * 1.42
        s += text(x + pad, cy - 4, l, sz, wt, col)
    s += "</g>\n"
    return s, h


def column(x, y, w, h, color, title):
    s = rect(x, y, w, h, "#f7f9fc", "#d6deea", 1.5, 20)
    s += rect(x, y, w, 76, color, "none", 0, 16)
    s += f'<rect x="{x}" y="{y+40}" width="{w}" height="36" fill="{color}"/>\n'
    s += text(x + 28, y + 50, title, 28, 700, "white")
    return s


def takeaway(x, y, w, h, title, lines, title_color=NAVY):
    s = rect(x, y, w, h, "#f7f9fc", "#c6d3e5", 2, 20)
    s += text(x + 40, y + 53, title, 37, 700, title_color)
    cy = y + 53
    for i, (ln, col, wt) in enumerate(lines):
        cy += 46 if i == 0 else 42
        s += text(x + 40, cy, ln, 28 if i == 0 else 27, wt, col)
    return s


def footer(y, left, page):
    return text(175, y, left, 25, 400, GREY) + text(3045, y, page, 25, 400, GREY, "end")


def legend_item(x, y, color, label, dashed=False):
    return line([(x, y), (x + 75, y)], color, 4, dashed) + text(x + 98, y + 9, label, 27, 500, color)


def svg(height, body, title=None):
    height = int(round(height))
    defs = "<defs>" + "".join(
        f'<marker id="arrow-{c[1:]}" markerWidth="12" markerHeight="12" refX="9" refY="5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L10,5 L0,10 Z" fill="{c}"/></marker>'
        for c in [BLUE, GREEN, PURPLE, ORANGE, GREY, GOLD, NAVY, RED, INK]) + "</defs>"
    t = f"<title>{escape(title)}</title>" if title else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" viewBox="0 0 {W} {height}">'
            f'{t}{defs}<rect width="{W}" height="{height}" fill="white"/>'
            f'<g font-family="{FONT}" id="figure-body">{body}</g></svg>')
