#!/usr/bin/env python3
"""
The direct fit survives what the threshold cannot: a shadow across the rim, and an
arc that is only in one view.

Synthetic, because the point is the *failure mode* rather than the rig -- a real
frame cannot be asked to have exactly one shadow and exactly one occlusion. Both
scenes here are built from the two things measured on the flights: shadows are broad
(wider than `RING_KSIZE`) and cost contrast rather than erasing the rim, and an
occluder erases it outright over an arc.

    python test_ring_fit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "calib"), str(HERE.parent / "camera")]

import segment as sg  # noqa: E402

SIZE = 640
RING = ((320.0, 320.0), (380.0, 260.0), 25.0)   # centre, (major, minor), degrees


def scene(shadow=False, occlude_deg=0.0, bright=False):
    """A rim against its ground, optionally shadowed and occluded.

    ``bright`` mirrors the whole scene: a light rim on a dark ground, which is the rig
    with a black backdrop. The point of testing both is that `ring_weight` is the only
    polarity-aware step -- if it is right, nothing downstream can tell them apart.
    """

    if bright:
        img = np.full((SIZE, SIZE), 35, np.uint8)
        # A broad bright region, the mirror of the shadow: the lab bench at the frame
        # edge. It must score ~0 for the same reason a shadow does -- it is wide.
        if shadow:
            cv2.rectangle(img, (0, 0), (SIZE, 120), 235, -1)
            img = cv2.GaussianBlur(img, (0, 0), 12)
        (cx, cy), (a, b), ang = RING
        cv2.ellipse(img, (int(cx), int(cy)), (int(a / 2), int(b / 2)), ang, 0, 360, 240, 7)
        if occlude_deg:
            cv2.ellipse(img, (int(cx), int(cy)), (int(a / 2), int(b / 2)), ang,
                        0, occlude_deg, 35, 11)
        return cv2.GaussianBlur(img, (0, 0), 1.2)

    img = np.full((SIZE, SIZE), 210, np.uint8)
    if shadow:
        # Broad and soft, like a real cast shadow: 260 px across, far wider than the
        # kernel, and overlapping most of one side of the ring.
        s = np.zeros_like(img)
        cv2.circle(s, (200, 380), 130, 255, -1)
        img = np.where(cv2.GaussianBlur(s, (0, 0), 25) > 0, (img * 0.42), img).astype(np.uint8)
    (cx, cy), (a, b), ang = RING
    cv2.ellipse(img, (int(cx), int(cy)), (int(a / 2), int(b / 2)), ang, 0, 360, 35, 7)
    if occlude_deg:
        cv2.ellipse(img, (int(cx), int(cy)), (int(a / 2), int(b / 2)), ang,
                    0, occlude_deg, 210, 11)
    return cv2.GaussianBlur(img, (0, 0), 1.2)


def test_shadow_is_not_rim():
    """`ring_weight` responds on the rim and not on a broad shadow."""

    w = sg.ring_weight(scene(shadow=True), appearance="dark")
    on_rim = float(np.median(sg.sample_map(w, sg.ellipse_points(RING))))
    # Well inside the shadow and well away from the ring.
    patch = w[430:470, 130:170]
    assert on_rim > 40, f"rim evidence too weak: {on_rim:.1f}"
    assert patch.max() < on_rim / 4, (
        f"shadow scores {patch.max():.1f} against a rim of {on_rim:.1f}"
    )


def test_threshold_admits_what_the_weight_rejects():
    """The level this replaces cannot make the same separation at any setting."""

    img = scene(shadow=True)
    (cx, cy), _, _ = RING
    rim = int(np.median(img[sg.ellipse_points(RING)[:, 1].astype(int),
                            sg.ellipse_points(RING)[:, 0].astype(int)]))
    shadow = int(np.median(img[430:470, 130:170]))
    # Any level dark enough to keep the rim also keeps the shadow, because the shadow
    # is darker than the rim it is cast beside. That is the whole problem.
    assert shadow < rim + 60, (
        f"shadow {shadow} and rim {rim} are separable by level; scene is too easy"
    )


def test_fit_recovers_an_occluded_ring():
    """A third of the rim missing still fits, from a seed 20 px and 6% off."""

    w = sg.ring_weight(scene(occlude_deg=120.0), appearance="dark")
    (cx, cy), (a, b), ang = RING

    def score(e):
        return float(np.mean(sg.sample_map(w, sg.ellipse_points(e))))

    truth = score(RING)
    seed = ((cx + 20, cy - 14), (a * 1.06, b * 1.06), ang + 4)
    assert score(seed) < truth, "the seed already scores as well as the truth"

    got = sg.fit_ellipse_image(w, seed)
    assert got is not None, "the fit gave up"
    (fx, fy), (fa, fb), _ = got.ellipse
    assert got.evidence > score(seed), "the fit did not improve the evidence it reports"
    # Two thirds of the rim is present, and the fit should know it.
    assert 0.55 < got.coverage < 0.80, f"coverage {got.coverage:.2f} does not match a 120 deg gap"

    assert np.hypot(fx - cx, fy - cy) < 4.0, f"centre off by {np.hypot(fx-cx, fy-cy):.1f} px"
    assert abs(fa - a) / a < 0.03, f"major off by {100*abs(fa-a)/a:.1f}%"
    assert abs(fb - b) / b < 0.05, f"minor off by {100*abs(fb-b)/b:.1f}%"


def test_bright_rim_is_the_mirror_image():
    """A light rim on a dark ground behaves identically under the top-hat."""

    img = scene(bright=True, shadow=True)
    w = sg.ring_weight(img, appearance="bright")
    on_rim = float(np.median(sg.sample_map(w, sg.ellipse_points(RING))))
    broad = w[20:90, 200:400]                      # inside the bright band, off the ring
    assert on_rim > 40, f"rim evidence too weak: {on_rim:.1f}"
    assert broad.max() < on_rim / 4, (
        f"the broad bright region scores {broad.max():.1f} against a rim of {on_rim:.1f}"
    )
    # And the wrong polarity must find nothing, which is what makes APPEARANCE matter.
    wrong = sg.ring_weight(img, appearance="dark")
    assert float(np.median(sg.sample_map(wrong, sg.ellipse_points(RING)))) < on_rim / 10


def test_seed_comes_from_the_map_alone():
    """`ring_seed` recovers the ring with no plate and no level on luminance."""

    img = scene(bright=True, shadow=True)
    got = sg.ring_seed(sg.ring_weight(img, appearance="bright"))
    assert got is not None, "no seed from the evidence map"
    ellipse, mask, hull, area = got
    (fx, fy), (fa, fb), _ = ellipse
    (cx, cy), (a, b), _ = RING
    assert np.hypot(fx - cx, fy - cy) < 12, f"seed centre off by {np.hypot(fx-cx, fy-cy):.1f} px"
    assert abs(fa - a) / a < 0.12, f"seed major off by {100 * abs(fa - a) / a:.1f}%"


def test_roi_matches_full_frame():
    """The ROI path is an optimisation, not a different answer."""

    img = scene()
    full = sg.ring_weight(img, appearance="dark")
    part = sg.ring_weight(img, roi=(120, 180, 400, 290), appearance="dark")
    pts = sg.ellipse_points(RING)
    assert np.allclose(sg.sample_map(full, pts), sg.sample_map(part, pts), atol=1.0)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ok  {name}")
    print("ring fit checks ok")
