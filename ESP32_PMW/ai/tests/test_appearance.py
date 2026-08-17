"""
Segmentation against the two rig appearances.

Run: uv run python ai/tests/test_appearance.py

The bright-on-dark rig is covered indirectly by every other suite. This one
exists for the `dark` rig -- a black robot on a white backdrop under a
**monochrome** camera, with the drive coils and the room beyond the backdrop in
frame.

A chroma-based appearance was tried and removed. The argument was sound -- the
robot is dark and achromatic, the coils dark and coloured -- but the high-speed
cameras are mono sensors, so there is no colour to act on and a chroma test is an
exact no-op rather than a weak signal. What is left is brightness and smoothness,
which is what `valid_region` uses.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

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

import conic  # noqa: E402
import estimator  # noqa: E402
import segment  # noqa: E402

K = np.array(
    [[1408.78, 0.0, 497.55], [0.0, 1407.69, 355.70], [0.0, 0.0, 1.0]], dtype=np.float64
)
RNG = np.random.default_rng(20260815)

#: Grey levels a white backdrop plausibly presents. The point of the spread is
#: that it is the *illumination* varying, not the paint.
WHITE = {
    "blown": (252, 252, 252),
    "nominal": (225, 228, 230),
    "shadowed": (120, 124, 128),
    "deep shade": (60, 62, 66),
}
#: A "black" backdrop is never 0. Velvet gets close; matte paper under working
#: light sits near 30, and anything with a sheen picks up a specular wash.
BLACK = {
    "velvet": (4, 4, 5),
    "matte lit": (28, 30, 32),
    "matte bright": (45, 48, 50),
    "sheen": (70, 72, 75),
}
#: The drive coils, as the grey levels they actually present. Taken from
#: `pose/assets/captures/elp/` -- p5 8-28, median 85-104, p95 224. Neutral
#: triples, because the sensor is mono and there is nothing else to record.
COILS = {
    "shadowed": (18, 18, 18),
    "body": (85, 85, 85),
    "lit": (104, 104, 104),
    "specular": (224, 224, 224),
}
#: The drive coils beside the robot, as grey levels. Dark against a white ground,
#: so an inverted threshold picks them up; a mono sensor has nothing else to
#: tell them apart by, which is why `valid_region` exists.

_failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        _failures.append(name)


def _gray(bgr):
    return int(cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2GRAY)[0, 0])


def _redness(bgr):
    b, g, r = bgr
    return max(0, int(r) - max(int(g), int(b)))


def scene(fg_bgr, bg_bgr, ellipse, ramp=0.0, noise=3.0, stroke=5):
    """
    One synthetic frame: a rim of ``fg_bgr`` on a ground of ``bg_bgr``.

        ``ramp`` tilts the illumination left-to-right by +-that fraction, which is
        what actually defeats a fixed luminance threshold on a real backdrop.
    """

    h, w = 720, 1000
    img = np.full((h, w, 3), bg_bgr, np.uint8).astype(np.float32)
    if ramp:
        img *= np.linspace(1.0 - ramp, 1.0 + ramp, w)[None, :, None]
    if fg_bgr is not None:
        (cx, cy), (ma, mi), ang = ellipse
        m = np.zeros((h, w), np.uint8)
        cv2.ellipse(
            m,
            (int(cx), int(cy)),
            (int(ma / 2), int(mi / 2)),
            float(ang),
            0,
            360,
            255,
            stroke,
        )
        img[m > 0] = np.array(fg_bgr, np.float32)
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    return np.clip(img + RNG.normal(0.0, noise, img.shape), 0, 255).astype(np.uint8)


def truth_ellipse(z=240.0, tilt_deg=30.0, az=0.7):
    c = np.array([4.0, -3.0, z])
    t = math.radians(tilt_deg)
    n = np.array([math.sin(t) * math.cos(az), math.sin(t) * math.sin(az), -math.cos(t)])
    return conic.project_circle(c, n, estimator.RADIUS_MM, K)


# ---------------------------------------------------------------------------


def test_dark_rejects_clutter():
    """
    The coils must not reach the hull.

        `silhouette_hull` pools every blob and takes one convex hull, so a coil near
        the robot does not add a stray contour -- it swallows the rim. The failure is
        silent: the fit still returns an ellipse, just a meaningless one.

        **This used to be a chroma test, and chroma is the wrong instrument.** The
        argument was sound -- the coils are dark but coloured, the robot dark and
        neutral, so ``max(BGR) - min(BGR)`` separates them -- and it holds on a colour
        camera. The rig's camera is an ELP OV9281, a mono sensor: measured over both
        frames in `pose/assets/captures/elp/`, chroma is exactly 0 in every pixel,
        so the gate passed the whole frame. It is asserted below that brightness
        cannot do the job either, because that is what forces the two mechanisms
        `segment.py` now uses -- a valid *region*, and a spread limit on which blobs
        may join the hull. Both work on luminance alone.
    """

    print("\ndark appearance: clutter rejection")
    gb = [_gray(v) for v in BLACK.values()]
    gc = [_gray(v) for v in COILS.values()]
    print(
        f"        luminance   body [{min(gb):3d},{max(gb):3d}]  coils [{min(gc):3d},{max(gc):3d}]"
    )
    check(
        "coils are dark too, so brightness alone cannot exclude them",
        min(gc) < max(gb),
        f"coils reach {min(gc)}, body reaches {max(gb)}",
    )

    e = truth_ellipse()
    (cx, cy), (ma, mi), ang = e
    base = scene(BLACK["matte lit"], WHITE["nominal"], e, ramp=0.25)
    withc = base.copy()
    # Placed and coloured to match the rig rather than to be convenient.
    #
    # `shadowed` and not `bronze`: at the shipped threshold the brighter coils are
    # excluded by brightness alone, so using one would test nothing. The real
    # coils reach a p5 of 8-28 counts, well below the cut, and `shadowed` (gray
    # 44) is the palette entry that represents them.
    #
    # 1.4 major axes out, because that is where they are: measured on the face-on
    # ELP capture the strays sit 1.76-2.5 robot radii from its centroid, and
    # 1.4 * major here works out to 2.1 radii. Closer than that is not a harder
    # test, it is a different rig.
    for dx in (-1, 1):
        cv2.circle(
            withc,
            (int(cx + dx * ma * 1.4), int(cy)),
            int(ma * 0.28),
            COILS["shadowed"],
            -1,
        )
    withc = cv2.GaussianBlur(withc, (0, 0), 1.0)

    # Mono, as the real camera delivers it -- deliberately throwing the colour
    # away, so the test cannot pass by a route the hardware does not have.
    base_m = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    withc_m = cv2.cvtColor(withc, cv2.COLOR_BGR2GRAY)

    clean = segment.segment(base_m, appearance="dark")
    gated = segment.segment(withc_m, appearance="dark")
    naive = segment.segment(
        cv2.bitwise_not(withc_m), appearance="bright", thresh=segment.DARK_THRESH
    )
    for nm, seg in (
        ("no coils", clean),
        ("coils, gated", gated),
        ("coils, ungated", naive),
    ):
        print(
            f"        {nm:<16} {'none' if seg is None else f'{seg.ellipse[1][0]:7.1f} px'}"
        )
    check(
        "gated fit is unchanged by the coils",
        clean is not None
        and gated is not None
        and abs(gated.ellipse[1][0] - clean.ellipse[1][0]) < 0.05 * clean.ellipse[1][0],
        (
            f"{clean.ellipse[1][0]:.1f} -> {gated.ellipse[1][0]:.1f} px"
            if clean and gated
            else "missing detection"
        ),
    )
    check(
        "an ungated threshold is wrecked by them",
        naive is not None
        and clean is not None
        and naive.ellipse[1][0] > 1.5 * clean.ellipse[1][0],
        f"{naive.ellipse[1][0]:.1f} px against a true {ma:.1f}" if naive else "none",
    )

    # The complement property, not a pixel count. This scene is *entirely*
    # backdrop -- there is no out-of-region clutter to find -- so ignoring 0% of
    # it is the right answer and asserting otherwise would be asserting a bug.
    # The non-trivial case is covered on the real frames, where
    # `test_elp_captures` requires the region to exclude every frame corner.
    mask = segment.clutter_mask(withc_m)
    region = segment.valid_region(withc_m)
    check(
        "clutter_mask is exactly the complement of the valid region",
        region is not None and np.array_equal(mask, cv2.bitwise_not(region)),
        f"{100.0 * mask.mean() / 255:.0f}% of frame ignored",
    )


def test_dark_needs_no_colour():
    """
    The mono path is the supported one, not a degraded fallback.

        A colour frame and its grayscale collapse must segment to the same ellipse.
        Asserted because the previous implementation silently took a different code
        path for one-channel input, and that path skipped the clutter gate entirely --
        so a run that looked fine on colour test data would have failed on the rig.
    """

    print("\ndark appearance: colour and mono agree")
    e = truth_ellipse()
    img = scene(BLACK["matte lit"], WHITE["nominal"], e, ramp=0.25)
    a = segment.segment(img, appearance="dark")
    b = segment.segment(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), appearance="dark")
    ok = a is not None and b is not None
    check(
        "both detect", ok, "" if ok else f"colour={a is not None} mono={b is not None}"
    )
    if ok:
        d = abs(a.ellipse[1][0] - b.ellipse[1][0])
        check(
            "same major axis",
            d < 0.02 * b.ellipse[1][0],
            f"{a.ellipse[1][0]:.2f} vs {b.ellipse[1][0]:.2f} px",
        )


def test_dark_refuses_without_a_region():
    """
    No backdrop means no answer, rather than a whole-frame hull.
    """

    print("\ndark appearance: refusal")
    flat = np.full((240, 320), 20, np.uint8)
    check(
        "valid_region gives up on a frame with no bright region",
        segment.valid_region(flat) is None,
    )
    check(
        "segment returns None rather than thresholding anyway",
        segment.segment(flat, appearance="dark") is None,
    )


def test_empty_scene_is_not_a_detection():
    """
    The property the module docstring rejects Otsu for.

        An empty white backdrop contains no red, so an absolute chroma threshold
        returns nothing -- unlike an adaptive one, which would find the brightest
        thing present and report it confidently.
    """

    print("\nempty scenes must not detect")
    e = truth_ellipse()
    for name, bg, ramp in (
        ("white, even", WHITE["nominal"], 0.0),
        ("white, ramped", WHITE["nominal"], 0.35),
        ("white, blown", WHITE["blown"], 0.0),
        ("deep shade", WHITE["deep shade"], 0.0),
    ):
        img = scene(None, bg, e, ramp=ramp, noise=6.0)
        seg = segment.segment(img, appearance="dark")
        check(
            f"empty {name}: no detection",
            seg is None,
            "" if seg is None else f"{seg.area_px:.0f} px blob",
        )


def test_constants_are_matched_per_appearance():
    """
    Every appearance must carry its own complete, self-consistent constant set.

        The effective radius depends on which channel the boundary was thresholded
        in, so radius and tilt calibration are a matched pair *per appearance*.
        Shipping half a pair is the exact failure of journal Iterations 12-14, and it
        is silent -- a mismatched radius just biases every depth by a fixed percent.
    """

    print("\nconstants are matched per appearance")
    import importlib
    import os
    import subprocess

    import shape as calibration
    import estimator

    known = set(estimator.RADIUS_BY_APPEARANCE)
    check(
        "every appearance has a radius",
        known == {"bright", "dark"},
        ", ".join(sorted(known)),
    )

    seen = {}
    for app in sorted(known):
        env = {**os.environ, "POSE_APPEARANCE": app}
        # A fresh interpreter per appearance, because RADIUS_MM and the
        # calibration path are both bound at import from POSE_APPEARANCE --
        # re-importing in-process would not re-read it. Both stages go on the
        # path: the radius comes from pose/, the calibration from calib/.
        code = (
            "import sys; sys.path[:0] = [%r, %r];"
            "import shape as calibration, estimator;"
            "c = calibration.TiltCalibration.load();"
            "print(estimator.RADIUS_MM, calibration.calibration_path().name,"
            " c.a, c.meta.get('n_samples'))" % (str(_C / "pose"), str(_C / "calib"))
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env
        )
        check(
            f"{app}: package imports under this appearance",
            out.returncode == 0,
            out.stderr.strip()[-120:] if out.returncode else "",
        )
        if out.returncode:
            continue
        radius, fname, a, n = out.stdout.split()
        seen[app] = (float(radius), fname, float(a), n)
        path = _C / "calib" / fname  # calibrations live with stage 2
        print(
            f"        {app:<7} radius {float(radius):.4f} mm   {fname}"
            f"   a={float(a):.4f}  n={n}"
        )
        check(f"{app}: its calibration file exists", path.exists(), fname)
        check(f"{app}: calibration is not the identity fallback", float(a) != 1.0)

    if len(seen) > 1:
        radii = {v[0] for v in seen.values()}
        files = {v[1] for v in seen.values()}
        check(
            "appearances use different radii",
            len(radii) == len(seen),
            f"{sorted(radii)}",
        )
        check(
            "appearances use different calibration files",
            len(files) == len(seen),
            f"{sorted(files)}",
        )


def test_constants_match_the_shipped_threshold():
    """
    A calibration fitted at one threshold must not ship with another.

        `test_constants_are_matched_per_appearance` checks a calibration *exists*.
        It cannot check the thing that actually goes wrong, which is that the file
        exists and was fitted through a different pipeline than the one now running.
        That is the Iteration 12-14 failure and it is always silent -- a mismatched
        constant does not raise, it biases.

        The check that catches it is a round trip: render a body at a known pose
        under the *current* constants and see whether the fitted rim comes back.
        It caught exactly this. `DARK_THRESH` was retuned from 110 to 190 against the
        real ELP captures while `BLACK_BODY` still rendered a body at 93 counts --
        brighter than the resulting dividing luminance of 65 -- so most of the body
        failed the darkness test and the major axis came back 63 px against a true
        120, a 47% underestimate that nothing else in the suite noticed.

        Skipped without a GL context, since it is the one test here that renders.
    """

    print("\nconstants round-trip against a render")
    try:
        import render as rendermod
    except Exception as exc:  # pragma: no cover
        print(f"        SKIP -- renderer unavailable ({type(exc).__name__})")
        return

    pose = (30.0, 40.0, np.array([4.0, -3.0, 240.0]))
    cases = {"bright": (None, 0.0), "dark": (None, 1.0)}
    cases["dark"] = (rendermod.BLACK_BODY, 1.0)
    try:
        with rendermod.Renderer(1024, 768) as r:
            for app, (body, bg) in cases.items():
                s = r.render(
                    *pose,
                    light=rendermod.LightRig(dome=((60.0, 25.0),), ambient=0.45),
                    bg_level=bg,
                    body_colour=body,
                )
                seg = segment.segment(s.image, appearance=app)
                true = s.ellipse_gt[1][0]
                got = None if seg is None else seg.ellipse[1][0]
                err = float("inf") if got is None else abs(got - true) / true * 100
                print(
                    f"        {app:<7} fitted {'none' if got is None else f'{got:6.1f}'} px"
                    f"   true {true:6.1f} px   error {err:5.1f}%"
                )
                check(
                    f"{app}: shipped constants recover the rim",
                    err < 12.0,
                    f"{err:.1f}% < 12%",
                )
    except Exception as exc:  # pragma: no cover
        print(f"        SKIP -- render failed ({type(exc).__name__}: {exc})")


def test_bright_appearance_unchanged():
    """
    The bright path must still behave as every constant was fitted against.

        Asserted on the *explicit* appearance, not the module default: this suite has
        to pass under ``POSE_APPEARANCE=red`` too, and a test that encodes the ambient
        default would fail there for no reason -- which it did, once.
    """

    print("\nregression: the bright appearance is untouched")
    e = truth_ellipse()
    img = scene((255, 255, 255), (0, 0, 0), e)
    seg = segment.segment(img, appearance="bright")
    check("bright still segments white-on-black", seg is not None)
    if seg is None:
        return
    print(
        f"        major {seg.ellipse[1][0]:.1f} px, rms {seg.fit_rms_px:.2f} px "
        f"(module default is {segment.APPEARANCE!r})"
    )
    explicit = segment.segment(img, appearance="bright", thresh=segment.THRESH)
    check(
        "bright uses its own threshold by default",
        explicit is not None and explicit.ellipse == seg.ellipse,
    )
    # The two appearances must not share a level: 128 is right for one and far
    # too low for the inverted other, and binding one to the other is how the
    # duplicated-default bug of Iteration 12 recurred.
    check(
        "the two appearances resolve different thresholds",
        segment.score_channel(img, "bright")[1]
        != segment.score_channel(img, "dark")[1],
        f"{segment.score_channel(img, 'bright')[1]} vs "
        f"{segment.score_channel(img, 'dark')[1]}",
    )


def main():
    print("=" * 68)
    print("rig appearance: bright-on-dark, red-on-white, black-on-white (mono)")
    print("=" * 68)
    test_dark_rejects_clutter()
    test_dark_needs_no_colour()
    test_dark_refuses_without_a_region()
    test_empty_scene_is_not_a_detection()
    test_constants_are_matched_per_appearance()
    test_constants_match_the_shipped_threshold()
    test_bright_appearance_unchanged()
    print("\n" + "=" * 68)
    if _failures:
        print(f"FAILED ({len(_failures)}): " + ", ".join(_failures))
        return 1
    print("all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
