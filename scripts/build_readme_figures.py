#!/usr/bin/env python3
"""
Emit assets/pipeline.svg and assets/ladder.svg.

Both are generated rather than hand-written for the same reason the site's
charts are: label placement on a dense diagram is arithmetic, and doing it by
hand means the next edit silently overlaps two things.

Palette is shared with assets/frontier.svg and assets/routes.svg.
"""
import pathlib

INK = "#0f172a"
MUTED = "#64748b"
FAINT = "#94a3b8"
LINE = "#e2e8f0"
LINE2 = "#cbd5e1"
CARD = "#f8fafc"
WHITE = "#ffffff"
CPU = "#2563eb"
CPU_SOFT = "#eff6ff"
FAB = "#ea580c"
FAB_SOFT = "#fff7ed"
FAB_LINE = "#fed7aa"

FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "assets"


def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


# --------------------------------------------------------------------------
# pipeline.svg
# --------------------------------------------------------------------------
def pipeline():
    W, H = 1200, 640
    p = []
    add = p.append

    add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'font-family="{FONT}" role="img" '
        f'aria-label="End-to-end pipeline: Fashion-MNIST to a PYNQ-Z1, plus the accelerator branch">')
    add(f'<rect width="{W}" height="{H}" fill="{WHITE}"/>')

    add(f'<text x="60" y="46" font-size="25" font-weight="700" fill="{INK}">'
        f'From Fashion-MNIST to the board, end to end</text>')
    add(f'<text x="60" y="72" font-size="14.5" fill="{MUTED}">'
        f'Every box has a config under configs/ and a document under docs/ &#183; '
        f'the numbers are the paper&#8217;s, transcribed into benchmarks/results.csv</text>')

    # ---- the CPU delivery route: six nodes left to right -----------------
    boxes = [
        ("DATA", "Fashion-MNIST",
         ["60 000 train", "10 000 test", "frozen splits"], CPU),
        ("TEACHER", "EfficientNet-B0",
         ["94.77 % test", "4 020 358 params", "configs/teacher/02"], CPU),
        ("DISTILL", "Soft targets",
         ["exported once", "blend + KL", "configs/teacher/04"], CPU),
        ("STUDENTS", "TinyFashionCNN \u00d74",
         ["90.45 \u2013 92.21 %", "133 714 \u2013 658 490", "configs/cpu/00..05"], CPU),
        ("EXPORT", "ONNX \u2192 INT8",
         ["round-trip checked", "static, QOperator", "256 calibration imgs"], CPU),
        ("BOARD", "PYNQ-Z1",
         ["445 img/s peak", "26.60 s / 10 000", "intra_op=2, batch=256"], CPU),
    ]
    bx, bw, gap = 60, 160, 24
    by, bh = 104, 126

    for i, (kicker, title, lines, accent) in enumerate(boxes):
        x = bx + i * (bw + gap)
        add(f'<rect x="{x}" y="{by}" width="{bw}" height="{bh}" rx="10" fill="{CARD}" stroke="{LINE}"/>')
        add(f'<rect x="{x}" y="{by}" width="{bw}" height="3" rx="1.5" fill="{accent}"/>')
        add(f'<text x="{x+16}" y="{by+26}" font-size="9.5" font-weight="700" fill="{FAINT}" '
            f'letter-spacing="0.7">{esc(kicker)}</text>')
        add(f'<text x="{x+16}" y="{by+48}" font-size="14.5" font-weight="700" fill="{INK}">{esc(title)}</text>')
        for j, ln in enumerate(lines):
            fill = FAINT if ln.startswith("configs/") else MUTED
            fam = f' font-family="{MONO}"' if ln.startswith("configs/") else ""
            add(f'<text x="{x+16}" y="{by+70+j*17}" font-size="10.5" fill="{fill}"{fam}>{esc(ln)}</text>')

        # connector to the next node
        if i < len(boxes) - 1:
            cx = x + bw
            add(f'<line x1="{cx+4}" y1="{by+bh/2}" x2="{cx+gap-4}" y2="{by+bh/2}" '
                f'stroke="{LINE2}" stroke-width="1.6"/>')
            tip = cx + gap - 4
            add(f'<path d="M {tip} {by+bh/2} l -6 -4 l 0 8 z" fill="{LINE2}"/>')

    # label the row for what it is
    add(f'<text x="1140" y="{by-12}" font-size="10.5" font-weight="700" fill="{CPU}" '
        f'letter-spacing="0.6" text-anchor="end">THE ROUTE THAT CLOSED &#183; SHIPPED</text>')

    # ---- the branch into the accelerator routes --------------------------
    jx = bx + 3 * (bw + gap) + bw / 2          # under STUDENTS
    add(f'<line x1="{jx}" y1="{by+bh}" x2="{jx}" y2="288" stroke="{FAB_LINE}" '
        f'stroke-width="1.8" stroke-dasharray="5 4"/>')
    add(f'<circle cx="{jx}" cy="288" r="4" fill="{FAB}"/>')
    add(f'<text x="{jx+14}" y="282" font-size="10.5" fill="{FAB}" font-weight="600">'
        f'the same students, retargeted at the fabric</text>')

    cy, ch = 306, 236
    add(f'<rect x="60" y="{cy}" width="1080" height="{ch}" rx="12" fill="{FAB_SOFT}" stroke="{FAB_LINE}"/>')
    add(f'<text x="84" y="{cy+28}" font-size="10.5" font-weight="700" fill="{FAB}" letter-spacing="0.7">'
        f'THE ACCELERATOR BRANCH &#183; TWO MORE ANSWERS TO THE SAME QUESTION</text>')

    branches = [
        ("Brevitas QAT \u2192 1W1A", "weights and activations binarised", "BNN-PYNQ fabric",
         [("closed loop", CPU), ("83.16 % test", INK), ("154 096 img/s", INK)], True),
        ("Brevitas QAT \u2192 4W4A", "FINN-friendly head, dataflow safe", "FINN bitstream",
         [("bitstream runs", FAB), ("93.55 % on host", INK), ("not reached on board", MUTED)], False),
    ]

    colx = [96, 624]
    colw = 480
    for (top_t, top_s, bot_t, stats, closed), cx in zip(branches, colx):
        ty, thh = cy + 46, 58
        add(f'<rect x="{cx}" y="{ty}" width="{colw}" height="{thh}" rx="8" fill="{WHITE}" stroke="{FAB_LINE}"/>')
        add(f'<text x="{cx+16}" y="{ty+24}" font-size="13" font-weight="700" fill="{INK}">{esc(top_t)}</text>')
        add(f'<text x="{cx+16}" y="{ty+42}" font-size="10.5" fill="{MUTED}">{esc(top_s)}</text>')

        add(f'<line x1="{cx+colw/2}" y1="{ty+thh}" x2="{cx+colw/2}" y2="{ty+thh+22}" '
            f'stroke="{FAB_LINE}" stroke-width="1.8"/>')
        add(f'<path d="M {cx+colw/2} {ty+thh+22} l -4 -6 l 8 0 z" fill="{FAB_LINE}"/>')

        byy = ty + thh + 22
        bhh = 58
        add(f'<rect x="{cx}" y="{byy}" width="{colw}" height="{bhh}" rx="8" fill="{WHITE}" '
            f'stroke="{FAB}" stroke-width="1.4"/>')
        add(f'<text x="{cx+16}" y="{byy+24}" font-size="13" font-weight="700" fill="{INK}">{esc(bot_t)}</text>')
        # three inline stats on fixed columns, so the two branches line up
        for k, (label, col) in enumerate(stats):
            add(f'<text x="{cx+16+k*164}" y="{byy+43}" font-size="10.5" font-weight="600" '
                f'fill="{col}">{esc(label)}</text>')

    # ---- closing note ----------------------------------------------------
    add(f'<text x="60" y="{cy+ch+26}" font-size="11.5" fill="{MUTED}">'
        f'The FINN branch is written up as unfinished rather than packaged as a success: the bitstream generates, '
        f'loads and executes,</text>')
    add(f'<text x="60" y="{cy+ch+43}" font-size="11.5" fill="{MUTED}">'
        f'but its output does not match the golden reference. Root cause is isolated to the stitched-IP integration '
        f'path &#8212; docs/05.</text>')
    add('</svg>')

    (OUT / "pipeline.svg").write_text("\n".join(p) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# ladder.svg
# --------------------------------------------------------------------------
def ladder():
    W, H = 1200, 452
    p = []
    add = p.append

    add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'font-family="{FONT}" role="img" '
        f'aria-label="The four CPU tiers: parameters, test accuracy and board throughput">')
    add(f'<rect width="{W}" height="{H}" fill="{WHITE}"/>')

    add(f'<text x="60" y="46" font-size="25" font-weight="700" fill="{INK}">'
        f'The width ladder: four tiers, one board</text>')
    add(f'<text x="60" y="72" font-size="14.5" fill="{MUTED}">'
        f'Static INT8, ONNX Runtime 1.16.0 on the Cortex-A9 &#183; every accuracy is a full 10 000-image test-set '
        f'run, not a 256-image bundle</text>')

    # (model, tier label, params, accuracy or None, throughput)
    rows = [
        ("tinyfast_xxxs", "speed-first", 133714, 0.9045, 445.19, False),
        ("tinyfast_xxs", "balanced \u2014 the default", 182478, 0.9109, 375.99, True),
        ("tinyfast_xs", "middle tier", 237658, None, 310.0, False),
        ("tinyplus_kd_fulltrain", "accuracy-first", 658490, 0.9221, 186.8, False),
    ]

    # column geometry
    cx_model, w_model = 60, 220
    cx_params, w_bar_p = 306, 236          # bar track
    cx_acc = 640                            # accuracy number (left-aligned)
    cx_thru, w_bar_t = 812, 236
    right = 1140

    # header
    hy = 108
    add(f'<text x="{cx_model}" y="{hy}" font-size="9.5" font-weight="700" fill="{FAINT}" '
        f'letter-spacing="0.7">TIER</text>')
    add(f'<text x="{cx_params}" y="{hy}" font-size="9.5" font-weight="700" fill="{FAINT}" '
        f'letter-spacing="0.7">PARAMETERS &#183; WHAT IT COSTS</text>')
    add(f'<text x="{cx_acc}" y="{hy}" font-size="9.5" font-weight="700" fill="{FAINT}" '
        f'letter-spacing="0.7">TEST ACCURACY</text>')
    add(f'<text x="{cx_thru}" y="{hy}" font-size="9.5" font-weight="700" fill="{FAINT}" '
        f'letter-spacing="0.7">BOARD THROUGHPUT</text>')
    add(f'<line x1="60" y1="{hy+10}" x2="{right}" y2="{hy+10}" stroke="{LINE}"/>')

    max_params = max(r[2] for r in rows)
    max_thru = max(r[4] for r in rows)

    rh = 62
    y0 = hy + 34
    for i, (model, tier, params, acc, thru, star) in enumerate(rows):
        ry = y0 + i * rh
        # hover-ish band for the recommended row
        if star:
            add(f'<rect x="52" y="{ry-14}" width="{right-52+8}" height="{rh-8}" rx="8" '
                f'fill="{CPU_SOFT}" stroke="none"/>')
            add(f'<rect x="52" y="{ry-14}" width="3" height="{rh-8}" rx="1.5" fill="{CPU}"/>')

        label_fill = INK if star else INK
        add(f'<text x="{cx_model}" y="{ry+4}" font-size="13.5" font-weight="700" fill="{label_fill}" '
            f'font-family="{MONO}">{esc(model)}</text>')
        add(f'<text x="{cx_model}" y="{ry+21}" font-size="10.5" fill="{MUTED}">{esc(tier)}</text>')

        # parameters bar (cost) -------------------------------------------------
        bw = params / max_params * w_bar_p
        add(f'<rect x="{cx_params}" y="{ry-7}" width="{w_bar_p}" height="13" rx="6.5" fill="#f1f5f9"/>')
        add(f'<rect x="{cx_params}" y="{ry-7}" width="{bw:.1f}" height="13" rx="6.5" fill="{FAINT}"/>')
        add(f'<text x="{cx_params+w_bar_p+12}" y="{ry+4}" font-size="12" fill="{MUTED}" '
            f'font-family="{MONO}">{params:,}</text>'.replace(",", "\u00a0"))

        # accuracy number -------------------------------------------------------
        if acc is None:
            add(f'<text x="{cx_acc}" y="{ry+4}" font-size="14" font-weight="700" fill="{FAINT}">'
                f'not published</text>')
            add(f'<text x="{cx_acc}" y="{ry+21}" font-size="10.5" fill="{FAINT}">'
                f'the paper records throughput only</text>')
        else:
            add(f'<text x="{cx_acc}" y="{ry+6}" font-size="20" font-weight="700" fill="{INK}">'
                f'{acc*100:.2f} %</text>')

        # throughput bar (what you buy) ----------------------------------------
        tw = thru / max_thru * w_bar_t
        add(f'<rect x="{cx_thru}" y="{ry-7}" width="{w_bar_t}" height="13" rx="6.5" fill="#f1f5f9"/>')
        add(f'<rect x="{cx_thru}" y="{ry-7}" width="{tw:.1f}" height="13" rx="6.5" fill="{CPU}"/>')
        rate = f"{thru:,.0f}".replace(",", "\u00a0")
        add(f'<text x="{cx_thru+w_bar_t+12}" y="{ry+4}" font-size="12" font-weight="700" fill="{INK}" '
            f'font-family="{MONO}">{rate}</text>')
        add(f'<text x="{cx_thru+w_bar_t+12}" y="{ry+19}" font-size="9.5" fill="{FAINT}">img/s</text>')

    # ---- the trade, stated once ------------------------------------------
    last_y = y0 + (len(rows) - 1) * rh + 30
    add(f'<line x1="60" y1="{last_y}" x2="{right}" y2="{last_y}" stroke="{LINE}"/>')

    scale_left = rows[0][4] / rows[-1][4]
    pt = rows[-1][3] - rows[0][3]
    add(f'<text x="60" y="{last_y+28}" font-size="13" font-weight="700" fill="{INK}">'
        f'Trading {pt*100:.2f} accuracy points buys {scale_left:.2f}\u00d7 the throughput.</text>')
    add(f'<text x="60" y="{last_y+49}" font-size="11.5" fill="{MUTED}">'
        f'Read top to bottom and you are buying accuracy with latency; read bottom to top and you are giving accuracy '
        f'away for it.</text>')
    add(f'<text x="60" y="{last_y+66}" font-size="11.5" fill="{MUTED}">'
        f'The middle tier is left empty on purpose: the paper publishes no accuracy for tinyfast_xs, so none is quoted '
        f'here.</text>')
    add('</svg>')

    (OUT / "ladder.svg").write_text("\n".join(p) + "\n", encoding="utf-8")


if __name__ == "__main__":
    pipeline()
    ladder()
    for f in ("pipeline.svg", "ladder.svg"):
        print("wrote assets/" + f, (OUT / f).stat().st_size, "bytes")
