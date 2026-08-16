"""The `dark` appearance against the real ELP frames, not rendered ones.

Every other test in this package scores the estimator on synthetic images, where
the background is whatever `validation/backgrounds.py` chose to draw. That is the
wrong instrument for this appearance: what makes the mono rig hard is not the
robot, it is the *rest of the room* -- bronze coils, wires, a support box, and a
dark ambient beyond the white backdrop that is **darker than the robot itself**.
None of that is rendered, so none of it is tested anywhere else.

So this scores against the only two real frames the repo has, at
`pose/assets/captures/elp/`, and against truth read off them by hand. Hand-read
truth is weak -- it is why the tolerances here are tens of pixels rather than the
fractions of a pixel the synthetic tests use -- but it is measuring the right
thing, and a coarse measurement of the right thing beats a precise measurement of
the wrong one.

The two frames are chosen, not arbitrary. Both view the rim obliquely, but the
clutter they present is opposite: `output.jpeg` has no competing blob at all (the
robot is 16877 px and the next is 108), so it tests whether the rim survives the
gating; `output-top.jpeg` has six strays of 386-4231 px inside the region, so it
tests whether they are kept out of the hull. A change that fixes one and breaks
the other is common enough that testing only one is worse than useless.

**Only the centre and the major axis are asserted.** Hand-read truth cannot pin
the minor axis: the silhouette is the rim *union* the body and blades, so how
much of the mound above the disc enters it depends on the axial weighting, and
the eye cannot separate the two on a JPEG. The major axis is the outermost
extent, which the eye can read, and it is also what the depth estimate rests on.
The minor axis is reported for inspection and bounded only loosely, because a
gross failure will show up there while a subtle one is beyond this instrument --
that is what the synthetic tests, with analytic truth, are for.

Run: uv run python controller/pose/test/test_elp_captures.py
"""

from __future__ import annotations

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
sys.path[:0] = [str(HERE), str(HERE.parent / "validation"),
                str(_C / "pose"), str(_C / "calib"), str(_C / "camera")]

import segment as segmod  # noqa: E402

FRAMES = _C / "pose" / "assets" / "captures" / "elp"

# Truth read by eye off the two frames: rim centre and major axis in pixels.
# `ratio` is a loose sanity band only -- see the module docstring.
CASES = {
    "output.jpeg":     dict(centre=(655, 398), major=405, ratio=(0.05, 0.95),
                            view="no clutter"),
    "output-top.jpeg": dict(centre=(600, 320), major=325, ratio=(0.05, 0.95),
                            view="6 strays"),
}

CENTRE_TOL_PX = 45
MAJOR_TOL_FRAC = 0.25
BORDER_PX = 2


def check(name, case, verbose=True):
    """Returns (ok, message). Loads mono, exactly as `CameraSource` delivers it."""
    path = FRAMES / name
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return False, f"{name}: could not read {path}"
    h, w = gray.shape

    seg = segmod.segment(gray, appearance="dark")
    if seg is None:
        return False, f"{name} ({case['view']}): no detection"

    (cx, cy), (major, minor), _ = seg.ellipse
    ratio = minor / major if major > 0 else 0.0
    d = float(np.hypot(cx - case["centre"][0], cy - case["centre"][1]))
    rel = abs(major - case["major"]) / case["major"]

    # The hull must not reach the frame edge. This is the cheapest detector of the
    # exact failure this appearance is prone to: the dark ambient outside the
    # backdrop touches the border, so if it has entered the hull, the hull does too.
    pts = np.asarray(seg.contour).reshape(-1, 2)
    touches = bool((pts[:, 0] <= BORDER_PX).any() or (pts[:, 0] >= w - 1 - BORDER_PX).any()
                   or (pts[:, 1] <= BORDER_PX).any() or (pts[:, 1] >= h - 1 - BORDER_PX).any())

    fails = []
    if d > CENTRE_TOL_PX:
        fails.append(f"centre off by {d:.0f} px (max {CENTRE_TOL_PX})")
    if rel > MAJOR_TOL_FRAC:
        fails.append(f"major {major:.0f} vs {case['major']} ({rel:+.0%}, max {MAJOR_TOL_FRAC:.0%})")
    lo, hi = case["ratio"]
    if not (lo <= ratio <= hi):
        fails.append(f"ratio {ratio:.2f} outside [{lo:.2f}, {hi:.2f}]")
    if touches:
        fails.append("hull touches the frame border")

    if verbose:
        print(f"  {name:<18} {case['view']:<8} c=({cx:6.1f},{cy:6.1f}) d={d:5.1f}px  "
              f"major={major:6.1f} ({rel:+6.1%})  ratio={ratio:.2f}  "
              f"rms={seg.fit_rms_px:5.2f}px  n={seg.n_points:4d}  {seg.t_ms:5.2f} ms")

    return (not fails), (f"{name} ({case['view']}): " + "; ".join(fails) if fails else "")


def test_captures():
    """Both real frames segment to the right ellipse."""
    print(f"\nappearance={segmod.APPEARANCE}  DARK_THRESH={segmod.DARK_THRESH}  "
          f"background={'yes' if segmod.load_background() is not None else 'no (backdrop finder)'}")
    bad = []
    for name, case in CASES.items():
        ok, msg = check(name, case)
        if not ok:
            bad.append(msg)
    assert not bad, "\n    " + "\n    ".join(bad)


def test_valid_region_excludes_clutter():
    """The valid region reaches no frame corner.

    All four corners are outside the backdrop in both frames, so a region that
    includes one has failed regardless of what the ellipse fit then does with it.
    Checked separately from the fit because a good pose from a bad region is luck.
    """
    for name in CASES:
        gray = cv2.imread(str(FRAMES / name), cv2.IMREAD_GRAYSCALE)
        region = segmod.valid_region(gray)
        assert region is not None, f"{name}: no valid region found"
        h, w = region.shape
        corners = {"tl": region[0, 0], "tr": region[0, w - 1],
                   "bl": region[h - 1, 0], "br": region[h - 1, w - 1]}
        hit = [k for k, v in corners.items() if v]
        assert not hit, f"{name}: valid region includes frame corner(s) {hit}"


def test_refuses_without_a_region():
    """A frame with no backdrop returns None rather than hulling the whole image."""
    empty = np.full((240, 320), 20, np.uint8)      # uniformly dark: no bright region
    assert segmod.valid_region(empty) is None
    assert segmod.segment(empty, appearance="dark") is None


def main():
    fails = 0
    for fn in (test_captures, test_valid_region_excludes_clutter, test_refuses_without_a_region):
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            fails += 1
            print(f"FAIL {fn.__name__}: {e}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
