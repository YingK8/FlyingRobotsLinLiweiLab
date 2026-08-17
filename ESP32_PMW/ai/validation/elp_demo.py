"""
Render what the `dark` appearance does to the real ELP frames, stage by stage.

`overlay.py` draws a pose against known truth, which rendered frames have and real
ones do not. This is the other case: no truth, so what is worth showing is not
"how wrong is it" but **what the segmenter looked at, what it threw away, and what
it fitted** -- the three things that are invisible in a pose trace and that a wrong
region reproduces perfectly while being wrong.

Each capture gets one strip:

    frame | valid region | silhouette | fit

and one before/after pair contrasting the shipped gating against an ungated
threshold, which is the argument for the gate in a single picture: without it the
hull spans the frame, because the room beyond the backdrop is *darker than the
robot* and reaches the image border.

Run: uv run python controller/pose/validation/elp_demo.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("POSE_APPEARANCE", "dark")

HERE = Path(__file__).resolve().parent
# Scratch may depend on the whole pipeline, so all four stages go on the path.
# (This is the one direction the layering allows to be unrestricted: ai/ is not
# a stage, it is what the stages are exercised by.)
_C = HERE.parents[1] / "controller"
sys.path[:0] = [
    str(HERE),
    str(HERE.parent / "validation"),
    str(_C / "pose"),
    str(_C / "calib"),
    str(_C / "camera"),
]

import segment as segmod  # noqa: E402

FRAMES = _C / "pose" / "assets" / "captures" / "elp"
RESULTS = HERE.parents[2] / "results" / "elp" / "demo"

LABEL_H = 26
PAD = 6
BG = (24, 24, 28)


def _bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()


def _label(img, text, sub=""):
    """
    A titled panel. The caption carries the numbers, so the strip reads alone.
    """

    out = np.full((img.shape[0] + LABEL_H, img.shape[1], 3), BG, np.uint8)
    out[LABEL_H:] = _bgr(img)
    cv2.putText(
        out,
        text,
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (235, 235, 240),
        1,
        cv2.LINE_AA,
    )
    if sub:
        (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.putText(
            out,
            sub,
            (14 + w, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (150, 200, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def _row(panels, scale=0.5):
    panels = [
        cv2.resize(p, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        for p in panels
    ]
    h = max(p.shape[0] for p in panels)
    strip = np.full(
        (h, sum(p.shape[1] for p in panels) + PAD * (len(panels) - 1), 3), BG, np.uint8
    )
    x = 0
    for p in panels:
        strip[: p.shape[0], x : x + p.shape[1]] = p
        x += p.shape[1] + PAD
    return strip


def ungated(gray):
    """
    What a plain inverted threshold does: the comparison the gate exists to win.

        Runs the identical pipeline with the region removed, so the only difference
        between this and the shipped result is the gating itself.
    """

    return segmod.segment(
        cv2.bitwise_not(gray), appearance="bright", thresh=segmod.DARK_THRESH
    )


def stages(name, scale=0.5):
    gray = cv2.imread(str(FRAMES / name), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(FRAMES / name)

    region = segmod.valid_region(gray)
    ch, lvl = segmod.score_channel(gray, "dark", region=region)
    sil = (
        cv2.threshold(ch, lvl, 255, cv2.THRESH_BINARY)[1]
        if ch is not None
        else np.zeros_like(gray)
    )
    seg = segmod.segment(gray, appearance="dark")

    # Region panel: the frame under its own mask, so what survived is readable
    # rather than a white silhouette that could be hiding anything.
    reg_vis = _bgr(gray)
    if region is not None:
        segmod.shade_rejected(reg_vis, region)
    frac = 0.0 if region is None else 100.0 * region.mean() / 255.0

    fit_vis = segmod.draw(gray, seg)
    caption = (
        "no detection"
        if seg is None
        else f"major {seg.ellipse[1][0]:.0f}px  rms {seg.fit_rms_px:.2f}px  "
        f"{seg.n_points} pts  {seg.t_ms:.1f}ms"
    )

    panels = [
        _label(gray, "1. frame", f"{gray.shape[1]}x{gray.shape[0]} mono"),
        _label(reg_vis, "2. valid region", f"{frac:.0f}% kept, rest ignored"),
        _label(sil, "3. silhouette", f"level {lvl}"),
        _label(fit_vis, "4. fit", caption),
    ]
    return _row(panels, scale), seg


def comparison(name, scale=0.5):
    gray = cv2.imread(str(FRAMES / name), cv2.IMREAD_GRAYSCALE)
    gated = segmod.segment(gray, appearance="dark")
    plain = ungated(gray)

    def cap(s):
        return "no detection" if s is None else f"major {s.ellipse[1][0]:.0f}px"

    return (
        _row(
            [
                _label(
                    segmod.draw(gray, plain, rejected=False),
                    "ungated threshold",
                    cap(plain),
                ),
                _label(segmod.draw(gray, gated), "region + spread gated", cap(gated)),
            ],
            scale,
        ),
        gated,
        plain,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in FRAMES.glob("*.jpeg"))
    if not names:
        print(f"no frames in {FRAMES}")
        return 1

    print(
        f"appearance={segmod.APPEARANCE}  level={segmod.DARK_THRESH}  "
        f"spread={segmod.DARK_MAX_SPREAD}  "
        f"region={'background' if segmod.load_background() is not None else 'backdrop finder'}"
    )

    written = []
    for name in names:
        strip, seg = stages(name, args.scale)
        p = out / f"stages_{Path(name).stem}.png"
        cv2.imwrite(str(p), strip)
        written.append(p)
        print(
            f"  {name:<18} "
            + (
                "no detection"
                if seg is None
                else f"c=({seg.ellipse[0][0]:.0f},{seg.ellipse[0][1]:.0f}) "
                f"major={seg.ellipse[1][0]:.0f} minor={seg.ellipse[1][1]:.0f} "
                f"rms={seg.fit_rms_px:.2f}px  {seg.t_ms:.1f}ms"
            )
        )

        cmp_img, gated, plain = comparison(name, args.scale)
        pc = out / f"gated_vs_ungated_{Path(name).stem}.png"
        cv2.imwrite(str(pc), cmp_img)
        written.append(pc)
        if gated is not None and plain is not None:
            print(
                f"  {'':<18} ungated major {plain.ellipse[1][0]:.0f}px vs "
                f"gated {gated.ellipse[1][0]:.0f}px "
                f"({plain.ellipse[1][0] / gated.ellipse[1][0]:.1f}x too big)"
            )

    print("\nwrote:")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
