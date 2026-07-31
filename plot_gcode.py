#!/usr/bin/env python3
"""Graph a G-code toolpath with matplotlib.

Parses linear (G0/G1) and arc (G2/G3) moves, honors absolute/relative
positioning (G90/G91) and unit selection (G20/G21), and distinguishes
travel moves from extruding/cutting moves by whether the E axis advances.

Usage:
    python plot_gcode.py path/to/file.gcode          # 2D top-down (XY)
    python plot_gcode.py path/to/file.gcode --3d     # 3D view
    python plot_gcode.py path/to/file.gcode -o out.png
"""

import argparse
import math
import re
import sys

import matplotlib.pyplot as plt

# One token like "X12.3", "G1", "E-0.5"
TOKEN = re.compile(r"([A-Za-z])\s*(-?\d*\.?\d+)")


def parse_line(line):
    """Return {letter: value} for a single g-code line, comments stripped."""
    # Strip comments: ';' to end of line, and anything in parentheses.
    line = line.split(";", 1)[0]
    line = re.sub(r"\(.*?\)", "", line)
    words = {}
    for letter, value in TOKEN.findall(line):
        words[letter.upper()] = float(value)
    return words


def arc_points(x0, y0, x1, y1, i, j, clockwise, segments=64):
    """Interpolate a G2/G3 arc from (x0,y0) to (x1,y1) about center offset (i,j)."""
    cx, cy = x0 + i, y0 + j
    r = math.hypot(i, j)
    a0 = math.atan2(y0 - cy, x0 - cx)
    a1 = math.atan2(y1 - cy, x1 - cx)
    if clockwise:
        if a1 >= a0:
            a1 -= 2 * math.pi
    else:
        if a1 <= a0:
            a1 += 2 * math.pi
    pts = []
    for k in range(1, segments + 1):
        a = a0 + (a1 - a0) * k / segments
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def parse_gcode(path):
    """Return list of segments: dict(xs, ys, zs, extruding)."""
    x = y = z = e = 0.0
    absolute = True          # G90 default
    absolute_e = True
    segments = []

    with open(path) as fh:
        for raw in fh:
            w = parse_line(raw)
            if not w:
                continue

            g = None
            if "G" in w:
                g = int(w["G"])

            if g in (90, 91):
                absolute = absolute_e = (g == 90)
                continue
            if g in (20, 21):
                continue  # units are only labels for our plot
            # M82/M83 set extruder absolute/relative independently.
            if "M" in w and int(w["M"]) in (82, 83):
                absolute_e = int(w["M"]) == 82
                continue

            if g not in (0, 1, 2, 3):
                continue

            nx = w["X"] if "X" in w else (x if absolute else 0.0)
            ny = w["Y"] if "Y" in w else (y if absolute else 0.0)
            nz = w["Z"] if "Z" in w else (z if absolute else 0.0)
            if not absolute:
                nx, ny, nz = x + nx, y + ny, z + nz

            if "E" in w:
                ne = w["E"] if absolute_e else e + w["E"]
            else:
                ne = e
            extruding = ne > e + 1e-9

            if g in (2, 3):
                i = w.get("I", 0.0)
                j = w.get("J", 0.0)
                pts = arc_points(x, y, nx, ny, i, j, clockwise=(g == 2))
                xs = [x] + [p[0] for p in pts]
                ys = [y] + [p[1] for p in pts]
                zs = [z] * len(xs)  # arcs treated as planar in Z
            else:
                xs, ys, zs = [x, nx], [y, ny], [z, nz]

            segments.append({"xs": xs, "ys": ys, "zs": zs, "extruding": extruding})
            x, y, z, e = nx, ny, nz, ne

    return segments


def plot(segments, three_d=False, out=None, title=""):
    if not segments:
        print("No drawable moves found.", file=sys.stderr)
        sys.exit(1)

    fig = plt.figure(figsize=(9, 8))
    move_c, travel_c = "#1f77b4", "#d62728"

    if three_d:
        ax = fig.add_subplot(111, projection="3d")
        for s in segments:
            ax.plot(s["xs"], s["ys"], s["zs"],
                    color=move_c if s["extruding"] else travel_c,
                    lw=0.8 if s["extruding"] else 0.4,
                    alpha=1.0 if s["extruding"] else 0.35)
        ax.set_zlabel("Z")
    else:
        ax = fig.add_subplot(111)
        for s in segments:
            ax.plot(s["xs"], s["ys"],
                    color=move_c if s["extruding"] else travel_c,
                    lw=0.8 if s["extruding"] else 0.4,
                    alpha=1.0 if s["extruding"] else 0.35)
        ax.set_aspect("equal", adjustable="datalim")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    handles = [plt.Line2D([], [], color=move_c, label="extrude/cut"),
               plt.Line2D([], [], color=travel_c, label="travel", alpha=0.5)]
    ax.legend(handles=handles, loc="best")

    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved {out}")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description="Plot a G-code toolpath.")
    ap.add_argument("file", help="G-code file to plot")
    ap.add_argument("--3d", dest="three_d", action="store_true", help="3D plot")
    ap.add_argument("-o", "--out", help="save to image file instead of showing")
    args = ap.parse_args()

    segments = parse_gcode(args.file)
    n_ext = sum(s["extruding"] for s in segments)
    print(f"{len(segments)} moves ({n_ext} extruding, {len(segments) - n_ext} travel)")
    plot(segments, three_d=args.three_d, out=args.out, title=args.file)


if __name__ == "__main__":
    main()
