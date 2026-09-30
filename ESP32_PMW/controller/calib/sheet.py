#!/usr/bin/env python3
"""
The printable board, and the ruler bar that proves the print was not scaled.

`theory.md` section 14.4: a print scaled by the printer is invisible to every residual in
`calibrate.py`, so the sheet carries its own scale bar. Measure it, then pass the measured
pitch as ``--square-mm``.

    python sheet.py                  # writes assets/charuco_<board>.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from controller.calib.calibrate import HERE, SPEC, make_spec

PAGES = {"letter": (215.9, 279.4), "a4": (210.0, 297.0)}


def make_board(spec=None, path=None, page="letter"):
    """A printable sheet at true size, and what its squares actually measure."""

    spec = spec or SPEC
    path = Path(path) if path else HERE / "assets" / f"charuco_{spec.name}.pdf"
    path, actual = generate_pdf(spec, path, page=page)
    print(f"wrote {path}")
    print(f"  pitch {actual:.4f} mm ({(actual / spec.square_mm - 1) * 100:+.3f}% vs the "
          f"{spec.square_mm} mm asked for -- rasterising at a whole number of pixels per "
          f"square quantises it)")
    print(f"  board is {spec.cols * actual:.1f} x {spec.rows * actual:.1f} mm")
    # Simulated into a 1280x800 frame, all corners survive down to about 14 px of marker
    # and then fall off a cliff. A small board sets a working distance, not a preference.
    for f_px in (900, 1200, 2765):
        print(f"  at f={f_px} px the {spec.marker_mm} mm markers hit the 14 px detection "
              f"floor at {spec.marker_mm * f_px / 14:.0f} mm -- shoot closer than that")
    print("\nPrint at 100%, cut to the corner marks, measure the ruler bar, then pass the "
          "measured pitch as --square-mm.")
    return path

# ---- printing -----------------------------------------------------------------------
PAGES = {"letter": (215.9, 279.4), "a4": (210.0, 297.0)}


def generate_pdf(spec, path, page="letter", dpi=600, margin_mm=12.0):
    """Printable sheet at true size. Returns ``(path, actual_square_mm)``."""
    page_mm = PAGES[page]
    mm2px = lambda mm: int(round(mm * dpi / 25.4))   # noqa: E731

    square_px = mm2px(spec.square_mm)
    actual_square_mm = square_px * 25.4 / dpi
    board_w, board_h = spec.cols * square_px, spec.rows * square_px
    page_w, page_h = mm2px(page_mm[0]), mm2px(page_mm[1])
    if board_w > page_w - 2 * mm2px(margin_mm) or board_h > page_h - 2 * mm2px(margin_mm):
        raise ValueError(f"board {board_w}x{board_h} px does not fit {page_w}x{page_h} px")

    page_img = np.full((page_h, page_w), 255, np.uint8)
    x0, y0 = (page_w - board_w) // 2, mm2px(margin_mm) + mm2px(14.0)
    page_img[y0:y0 + board_h, x0:x0 + board_w] = spec.board.generateImage(
        (board_w, board_h), marginSize=0)

    s = dpi / 600.0                                   # text metrics tuned at 600 dpi
    def text(msg, x, y, size=1.6, thick=4):
        cv2.putText(page_img, msg, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                    size * s, 0, max(1, int(round(thick * s))), cv2.LINE_AA)

    text(f"{spec.name}  {spec.cols}x{spec.rows} squares  {actual_square_mm:.4f} mm pitch  "
         f"marker {spec.marker_mm:.3f} mm  {spec.dict_name}", x0, y0 - mm2px(9.0))

    # Crop marks outside the board's exact corners, so no ink lands on the pattern and
    # cutting to them gives a piece that fits the 25 x 75 mm slide exactly.
    gap, arm = mm2px(1.0), mm2px(4.0)
    for cx, sx_ in ((x0, -1), (x0 + board_w, 1)):
        for cy, sy_ in ((y0, -1), (y0 + board_h, 1)):
            cv2.line(page_img, (cx, cy + sy_ * gap), (cx, cy + sy_ * (gap + arm)), 0,
                     max(1, int(3 * s)))
            cv2.line(page_img, (cx + sx_ * gap, cy), (cx + sx_ * (gap + arm), cy), 0,
                     max(1, int(3 * s)))

    bar_y, bar_len = y0 + board_h + mm2px(18.0), mm2px(100.0)
    bar_x = (page_w - bar_len) // 2
    cv2.line(page_img, (bar_x, bar_y), (bar_x + bar_len, bar_y), 0, max(1, int(6 * s)))
    for mm in range(0, 101, 10):
        tick = mm2px(6.0 if mm % 50 else 10.0)
        x = bar_x + mm2px(float(mm))
        cv2.line(page_img, (x, bar_y), (x, bar_y + tick), 0, max(1, int(6 * s)))
    text("0", bar_x - mm2px(2.0), bar_y + mm2px(18.0), 1.4)
    text("100 mm", bar_x + bar_len - mm2px(12.0), bar_y + mm2px(18.0), 1.4)

    for i, line in enumerate([
            "Print at 100% / actual size -- no 'fit to page', no scaling.",
            "Then measure this bar. If it is not 100.0 mm, the print is scaled:",
            "measure one square with calipers and set SQUARE_MM in this notebook.",
            "Mount on glass or foam board. Paper curl is a systematic bias."]):
        text(line, bar_x, bar_y + mm2px(30.0 + 7.0 * i), 1.15, 3)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(page_img).save(path, format="PDF", dpi=(dpi, dpi))
    return path, actual_square_mm


if __name__ == "__main__":
    make_board(make_spec(float(sys.argv[1])) if len(sys.argv) > 1 else SPEC)
