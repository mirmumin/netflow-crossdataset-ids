"""Figure 2: the proposed Fusion-B, drawn in the same style as the workflow figure.

Writes fusion_b_diagram.svg / .png / .pdf next to this file.
Run: python make_fusion_b_diagram.py   (needs playwright with Chromium)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
W, H = 2060, 1000
INK = "#2b2d42"
C = {  # fill, stroke
    "blue": ("#d3e5f8", "#4a8fd6"), "green": ("#d2f0e0", "#3aa66a"),
    "orange": ("#fde3d0", "#e8833a"), "purple": ("#e3dbf6", "#8a6fd1"),
    "yellow": ("#fdf2bd", "#d9ad00"), "pink": ("#fbd6e8", "#d9559a"),
    "grey": ("#e2e2e2", "#8a8a8a"), "red": ("#fad0d0", "#e05a5a"),
    "teal": ("#cdeff3", "#2fb3c4"), "white": ("#ffffff", "#9a9a9a"),
}
out = []


def pill(x, y, w, text, color, h=44, size=21):
    f, s = C[color]
    out.append(f'<rect x="{x}" y="{y - h / 2}" width="{w}" height="{h}" rx="{h / 2}" fill="{f}" '
               f'stroke="{s}" stroke-width="3" filter="url(#sh)"/>')
    out.append(f'<text x="{x + w / 2}" y="{y + size * 0.36}" class="b" font-size="{size}" '
               f'text-anchor="middle">{text}</text>')


def group(x, y, w, h, title=None, sub=None, dash="2,7"):
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="26" fill="none" stroke="#222" '
               f'stroke-width="3" stroke-dasharray="{dash}" stroke-linecap="round"/>')
    if title:
        out.append(f'<text x="{x + w / 2}" y="{y + 38}" class="b" font-size="23" text-anchor="middle">'
                   f'{title}</text>')
    if sub:
        out.append(f'<text x="{x + w / 2}" y="{y + 64}" class="i" font-size="18" text-anchor="middle">'
                   f'{sub}</text>')


def arrow(points, dotted=False):
    d = "M" + " L".join(f"{a},{b}" for a, b in points)
    extra = ' stroke-dasharray="3,7" stroke-linecap="round"' if dotted else ""
    out.append(f'<path d="{d}" fill="none" stroke="{INK}" stroke-width="3"{extra} '
               f'marker-end="url(#ah)"/>')


def text(x, y, s, cls="i", size=18, anchor="middle"):
    out.append(f'<text x="{x}" y="{y}" class="{cls}" font-size="{size}" text-anchor="{anchor}">{s}</text>')


# ---- left: data -------------------------------------------------------------
group(30, 30, 380, 290, "SOURCE NETWORKS", "the other three, attacks labelled")
pill(55, 140, 330, "TRAINING FLOWS", "blue")
pill(55, 220, 330, "HELD-OUT ATTACKS R", "red")
text(220, 285, "R is never used to train XGBoost")

group(30, 345, 380, 205, "TARGET NETWORK", "new deployment, benign labels only")
pill(55, 450, 330, "k BENIGN FLOWS", "green")
pill(55, 515, 330, "TEST FLOWS (UNLABELLED)", "grey")

# ---- middle: the two knowledge sources ------------------------------------
pill(480, 140, 270, "XGBOOST", "orange")
text(615, 190, "trained on source flows")
pill(480, 450, 270, "ISOLATION FOREST", "purple")
text(615, 500, "fitted on k benign, 5-fold")
text(615, 522, "cross-fitted reference")
pill(830, 140, 300, "SOURCE SCORE s(x)", "yellow")
text(980, 190, "what attacks look like elsewhere")
pill(830, 450, 300, "ANOMALY RANK a(x)", "pink")
text(980, 500, "what is unusual in this network")

arrow([(385, 140), (472, 140)])
arrow([(750, 140), (822, 140)])
arrow([(385, 450), (472, 450)])
arrow([(750, 450), (822, 450)])

# ---- augmented representation ------------------------------------------
group(1200, 25, 330, 485, "AUGMENTED FLOW", "φ(x) = [ x ; s(x) ; a(x) ]")
pill(1225, 140, 280, "s(x)", "yellow")
pill(1225, 295, 280, "x : 47 NETFLOW FEATURES", "teal", size=19)
pill(1225, 450, 280, "a(x)", "pink")
arrow([(1130, 140), (1217, 140)])
arrow([(1130, 450), (1217, 450)])

# test flows run along the bottom into the augmented representation
arrow([(385, 515), (440, 515), (440, 580), (1290, 580), (1290, 518)])
text(800, 570, "every flow (context and test) is augmented the same way")

# ---- context set ----------------------------------------------------------
group(1200, 640, 330, 215, "CONTEXT SET", "m = k source attacks (100 / 1000)")
pill(1225, 745, 280, "k BENIGN → LABEL 0", "green")
pill(1225, 810, 280, "R ATTACKS → LABEL 1", "red")
arrow([(1440, 510), (1440, 632)])

# ---- TabPFN chip ------------------------------------------------------------
cx, cy, s = 1700, 300, 110
out.append(f'<text x="{cx}" y="{cy - s / 2 - 34}" class="b" font-size="23" text-anchor="middle">TABPFN</text>')
for i in range(5):
    o = -s / 2 + 15 + i * (s - 30) / 4
    out.append(f'<line x1="{cx + o}" y1="{cy - s / 2 - 14}" x2="{cx + o}" y2="{cy + s / 2 + 14}" '
               f'stroke="{INK}" stroke-width="5"/>')
    out.append(f'<line x1="{cx - s / 2 - 14}" y1="{cy + o}" x2="{cx + s / 2 + 14}" y2="{cy + o}" '
               f'stroke="{INK}" stroke-width="5"/>')
out.append(f'<rect x="{cx - s / 2}" y="{cy - s / 2}" width="{s}" height="{s}" rx="10" fill="#5b4bc4" '
           f'stroke="{INK}" stroke-width="4"/>')
out.append(f'<rect x="{cx - 32}" y="{cy - 32}" width="64" height="64" rx="6" fill="#ffd54a" '
           f'stroke="{INK}" stroke-width="3"/>')
out.append(f'<text x="{cx}" y="{cy + 11}" class="s" font-size="30" text-anchor="middle" '
           f'fill="#5b4bc4">AI</text>')
text(cx, cy + s / 2 + 52, "pretrained, no training", size=17)
text(cx, cy + s / 2 + 74, "on the target network", size=17)

arrow([(1530, 295), (1625, 295)])
text(1578, 280, "test", size=17)
arrow([(1530, 750), (1700, 750), (1700, 440)])
text(1615, 735, "context", size=17)

# ---- decision ---------------------------------------------------------------
pill(1800, 300, 240, "SCORE p(x)", "white")
arrow([(1772, 300), (1792, 300)])
pill(1800, 430, 240, "THRESHOLD τ", "blue")
text(1920, 485, "95th percentile of the")
text(1920, 507, "cross-fitted benign scores")
arrow([(1920, 322), (1920, 400)])
pill(1790, 600, 125, "ATTACK", "red", size=19)
pill(1925, 600, 125, "BENIGN", "green", size=19)
arrow([(1920, 520), (1920, 545), (1852, 545), (1852, 570)])
arrow([(1920, 545), (1987, 545), (1987, 570)])
text(1852, 650, "p ≥ τ", size=17)
text(1987, 650, "p &lt; τ", size=17)

# ---- Fusion-B-v2 note (bottom left) --------------------------------------------------------
group(30, 680, 640, 150, dash="10,8")
text(350, 720, "FUSION-B-v2 (VARIANT)", cls="b", size=21)
text(350, 754, "the features of every network are first mapped to the quantiles", size=18)
text(350, 778, "of its own benign traffic (in the target: the k benign flows),", size=18)
text(350, 802, "then XGBoost and TabPFN are applied to the aligned features", size=18)

# ---- caption ------------------------------------------------------------
out.append(f'<text x="{W / 2}" y="{H - 45}" font-size="24" text-anchor="middle" class="r">'
           f'<tspan class="b">Figure 2.</tspan> The proposed Fusion-B for benign-only cross-network '
           f'intrusion detection.</text>')

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<defs>
<marker id="ah" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="7" markerHeight="7" orient="auto">
<path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker>
<filter id="sh" x="-5%" y="-20%" width="110%" height="150%">
<feDropShadow dx="0" dy="2" stdDeviation="1.5" flood-opacity="0.18"/></filter>
<style>
text {{ font-family: "Times New Roman", "Liberation Serif", "DejaVu Serif", serif; fill: #1d1d1d; }}
.b {{ font-weight: bold; }} .i {{ font-style: italic; fill: #333; }}
.s {{ font-family: "Liberation Sans", Arial, sans-serif; font-weight: bold; }}
</style></defs>
<rect width="100%" height="100%" fill="#ffffff"/>
{chr(10).join(out)}
</svg>'''
(HERE / "fusion_b_diagram.svg").write_text(svg, encoding="utf-8")

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sys.exit("SVG written; install the Python playwright package to also write PNG and PDF.")

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": W, "height": H}, device_scale_factor=2.5)
    pg.set_content(f"<html><body style='margin:0'>{svg}</body></html>")
    pg.screenshot(path=str(HERE / "fusion_b_diagram.png"), full_page=True)
    pg.pdf(path=str(HERE / "fusion_b_diagram.pdf"), width=f"{W}px", height=f"{H}px",
           print_background=True, page_ranges="1")
    b.close()
print("written to", HERE)
