"""Segmentation against the two rig appearances.

Run: uv run python controller/pose/test_appearance.py

The bright-on-dark rig is covered indirectly by every other suite. This one
exists for the red robot, and for two claims in particular:

    **1. Brightness cannot separate a red robot from its backdrop reliably --
    not on white in either polarity, and not on black once the backdrop has any
    sheen. Chroma can.**

    **2. Chroma is background-invariant.** ``R - max(G, B)`` responds to the
    paint, and every neutral surface has R = G = B however it is lit, so black,
    grey and white all read ~0-4. One threshold and one calibration serve all of
    them -- measured at 0.21 % spread in fitted rim size across seven backdrops.

Neither is a preference; both are measurements over plausible surface patches
spanning specular highlight to deep shade. If a future change reintroduces a
luminance threshold for this rig, the overlap assertion below is what catches it.

Scenes are drawn analytically rather than rendered, deliberately: the renderer
produces a white mesh on a dark ground and cannot express this rig at all yet, so
a test that depended on it could not exist.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

import conic  # noqa: E402
import estimator  # noqa: E402
import segment  # noqa: E402

K = np.array(
    [[1408.78, 0.0, 497.55], [0.0, 1407.69, 355.70], [0.0, 0.0, 1.0]], dtype=np.float64
)
RNG = np.random.default_rng(20260815)

#: BGR patches a red robot and a white backdrop plausibly present. The point of
#: the spread is that it is the *illumination* varying, not the paint.
RED = {"bright": (55, 45, 190), "mid": (40, 32, 130),
       "shadowed": (22, 18, 70), "specular": (150, 145, 235)}
WHITE = {"blown": (252, 252, 252), "nominal": (225, 228, 230),
         "shadowed": (120, 124, 128), "deep shade": (60, 62, 66)}
#: A "black" backdrop is never 0. Velvet gets close; matte paper under working
#: light sits near 30, and anything with a sheen picks up a specular wash.
BLACK = {"velvet": (4, 4, 5), "matte lit": (28, 30, 32),
         "matte bright": (45, 48, 50), "sheen": (70, 72, 75)}
#: The bronze/orange drive coils beside the robot. Dark against a white ground,
#: so an inverted threshold picks them up; strongly coloured, so chroma does not.
COILS = {"bright orange": (30, 110, 225), "orange": (30, 90, 180),
         "bronze": (40, 85, 140), "dark bronze": (25, 55, 95),
         "shadowed": (18, 38, 66), "copper sheen": (90, 150, 215)}

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
    """One synthetic frame: a rim of ``fg_bgr`` on a ground of ``bg_bgr``.

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
        cv2.ellipse(m, (int(cx), int(cy)), (int(ma / 2), int(mi / 2)),
                    float(ang), 0, 360, 255, stroke)
        img[m > 0] = np.array(fg_bgr, np.float32)
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    return np.clip(img + RNG.normal(0.0, noise, img.shape), 0, 255).astype(np.uint8)


def truth_ellipse(z=240.0, tilt_deg=30.0, az=0.7):
    c = np.array([4.0, -3.0, z])
    t = math.radians(tilt_deg)
    n = np.array([math.sin(t) * math.cos(az), math.sin(t) * math.sin(az), -math.cos(t)])
    return conic.project_circle(c, n, estimator.RADIUS_MM, K)


# ---------------------------------------------------------------------------


def test_channel_separation():
    """The measurement the red appearance exists because of."""
    print("\nchannel separation, red robot vs white ground")
    gr, gw = [_gray(v) for v in RED.values()], [_gray(v) for v in WHITE.values()]
    rr, rw = [_redness(v) for v in RED.values()], [_redness(v) for v in WHITE.values()]
    print(f"        luminance   robot [{min(gr):3d},{max(gr):3d}]  ground [{min(gw):3d},{max(gw):3d}]")
    print(f"        R-max(G,B)  robot [{min(rr):3d},{max(rr):3d}]  ground [{min(rw):3d},{max(rw):3d}]")

    check("luminance ranges OVERLAP (no threshold works, either polarity)",
          min(gr) < max(gw) and min(gw) < max(gr),
          f"gap {min(gr) - max(gw):+d} counts")
    check("chroma ranges are disjoint", min(rr) > max(rw),
          f"margin {min(rr) - max(rw):+d} counts")
    check("the shipped threshold sits inside that margin",
          max(rw) < segment.REDNESS_THRESH < min(rr),
          f"{max(rw)} < {segment.REDNESS_THRESH} < {min(rr)}")


def test_background_invariance():
    """The claim that makes this appearance worth having.

    ``R - max(G, B)`` responds to the *paint*, not the illumination, and every
    neutral surface -- black, grey or white -- has R = G = B by definition. So the
    same channel, the same threshold and the same calibration should serve any
    neutral backdrop, and swapping the backdrop should not move the fitted rim.
    """
    print("\nbackground invariance (red appearance)")
    e = truth_ellipse()
    grounds = ([(f"black {k}", v) for k, v in BLACK.items()]
               + [(f"white {k}", v) for k, v in WHITE.items() if k != "deep shade"])
    majors = {}
    for name, bg in grounds:
        img = scene(RED["mid"], bg, e, ramp=0.25)
        seg = segment.segment(img, appearance="red")
        check(f"{name}: detected", seg is not None)
        if seg is not None:
            majors[name] = seg.ellipse[1][0]
    if len(majors) > 1:
        v = np.array(list(majors.values()))
        spread = (v.max() - v.min()) / v.mean() * 100
        print(f"        major axis {v.min():.2f}-{v.max():.2f} px over "
              f"{len(v)} backdrops ({spread:.2f}% spread)")
        check("one calibration serves every neutral backdrop", spread < 1.0,
              f"{spread:.2f}% < 1%")


def test_luminance_is_fragile_on_black():
    """Red on black *nearly* works on brightness, and that is the trap.

    The polarity is right and the robot is brighter than a dark backdrop, so a
    lowered threshold detects it -- until the backdrop picks up a sheen, at which
    point the whole frame passes the threshold. The chroma channel is indifferent.
    """
    print("\nred on black: brightness vs chroma")
    e = truth_ellipse()
    for name, bg in BLACK.items():
        img = scene(RED["mid"], bg, e, ramp=0.25)
        lum = segment.segment(img, appearance="bright", thresh=40)
        chroma = segment.segment(img, appearance="red")
        print(f"        {name:<14} brightness@40 {'ok' if lum else 'none':>4}"
              f"   chroma {'ok' if chroma else 'none':>4}")
        check(f"black {name}: chroma detects", chroma is not None)
    lit = segment.segment(scene(RED["mid"], BLACK["sheen"], e, ramp=0.25),
                          appearance="bright", thresh=40)
    check("brightness fails on a sheeny black backdrop (chroma does not)",
          lit is None)
    default = segment.segment(scene(RED["mid"], BLACK["velvet"], e),
                              appearance="bright")
    check("the shipped brightness threshold (128) misses red entirely",
          default is None)


def test_dark_rejects_clutter():
    """The coils must not reach the hull.

    `silhouette_hull` pools every blob and takes one convex hull, so a coil near
    the robot does not add a stray contour -- it swallows the rim. The failure is
    silent: the fit still returns an ellipse, just a meaningless one.

    **This used to be a chroma test, and chroma is the wrong instrument.** The
    argument was sound -- the coils are dark but coloured, the robot dark and
    neutral, so ``max(BGR) - min(BGR)`` separates them -- and it holds on a colour
    camera. The rig's camera is an ELP OV9281, a mono sensor: measured over both
    frames in `vision/drone_orientation/elp/`, chroma is exactly 0 in every pixel,
    so the gate passed the whole frame. It is asserted below that brightness
    cannot do the job either, because that is what forces the two mechanisms
    `segment.py` now uses -- a valid *region*, and a spread limit on which blobs
    may join the hull. Both work on luminance alone.
    """
    print("\ndark appearance: clutter rejection")
    gb = [_gray(v) for v in BLACK.values()]
    gc = [_gray(v) for v in COILS.values()]
    print(f"        luminance   body [{min(gb):3d},{max(gb):3d}]  coils [{min(gc):3d},{max(gc):3d}]")
    check("coils are dark too, so brightness alone cannot exclude them",
          min(gc) < max(gb), f"coils reach {min(gc)}, body reaches {max(gb)}")

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
        cv2.circle(withc, (int(cx + dx * ma * 1.4), int(cy)), int(ma * 0.28),
                   COILS["shadowed"], -1)
    withc = cv2.GaussianBlur(withc, (0, 0), 1.0)

    # Mono, as the real camera delivers it -- deliberately throwing the colour
    # away, so the test cannot pass by a route the hardware does not have.
    base_m = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    withc_m = cv2.cvtColor(withc, cv2.COLOR_BGR2GRAY)

    clean = segment.segment(base_m, appearance="dark")
    gated = segment.segment(withc_m, appearance="dark")
    naive = segment.segment(cv2.bitwise_not(withc_m),
                            appearance="bright", thresh=segment.DARK_THRESH)
    for nm, seg in (("no coils", clean), ("coils, gated", gated),
                    ("coils, ungated", naive)):
        print(f"        {nm:<16} {'none' if seg is None else f'{seg.ellipse[1][0]:7.1f} px'}")
    check("gated fit is unchanged by the coils",
          clean is not None and gated is not None
          and abs(gated.ellipse[1][0] - clean.ellipse[1][0]) < 0.05 * clean.ellipse[1][0],
          f"{clean.ellipse[1][0]:.1f} -> {gated.ellipse[1][0]:.1f} px"
          if clean and gated else "missing detection")
    check("an ungated threshold is wrecked by them",
          naive is not None and clean is not None
          and naive.ellipse[1][0] > 1.5 * clean.ellipse[1][0],
          f"{naive.ellipse[1][0]:.1f} px against a true {ma:.1f}" if naive else "none")

    # The complement property, not a pixel count. This scene is *entirely*
    # backdrop -- there is no out-of-region clutter to find -- so ignoring 0% of
    # it is the right answer and asserting otherwise would be asserting a bug.
    # The non-trivial case is covered on the real frames, where
    # `test_elp_captures` requires the region to exclude every frame corner.
    mask = segment.clutter_mask(withc_m)
    region = segment.valid_region(withc_m)
    check("clutter_mask is exactly the complement of the valid region",
          region is not None and np.array_equal(mask, cv2.bitwise_not(region)),
          f"{100.0 * mask.mean() / 255:.0f}% of frame ignored")


def test_dark_needs_no_colour():
    """The mono path is the supported one, not a degraded fallback.

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
    check("both detect", ok, "" if ok else f"colour={a is not None} mono={b is not None}")
    if ok:
        d = abs(a.ellipse[1][0] - b.ellipse[1][0])
        check("same major axis", d < 0.02 * b.ellipse[1][0],
              f"{a.ellipse[1][0]:.2f} vs {b.ellipse[1][0]:.2f} px")


def test_dark_refuses_without_a_region():
    """No backdrop means no answer, rather than a whole-frame hull."""
    print("\ndark appearance: refusal")
    flat = np.full((240, 320), 20, np.uint8)
    check("valid_region gives up on a frame with no bright region",
          segment.valid_region(flat) is None)
    check("segment returns None rather than thresholding anyway",
          segment.segment(flat, appearance="dark") is None)


def test_red_detects_where_bright_fails():
    print("\nsegmentation across illumination")
    e = truth_ellipse()
    true_major = e[1][0]
    cases = [
        ("even light", RED["mid"], WHITE["nominal"], 0.0),
        ("+-35% ramp", RED["mid"], WHITE["nominal"], 0.35),
        ("robot in shade", RED["shadowed"], WHITE["nominal"], 0.0),
        ("blown-out ground", RED["specular"], WHITE["blown"], 0.0),
    ]
    majors = []
    for name, fg, bg, ramp in cases:
        img = scene(fg, bg, e, ramp=ramp)
        a = segment.segment(img, appearance="bright")
        b = segment.segment(img, appearance="red")
        print(f"        {name:<18} bright={'none' if a is None else 'DETECTED'}"
              f"   red={'none' if b is None else f'{b.ellipse[1][0]:6.1f} px'}")
        check(f"{name}: red appearance detects", b is not None)
        check(f"{name}: bright appearance does not", a is None)
        if b is not None:
            majors.append(b.ellipse[1][0])

    # The absolute size is offset by the drawn stroke width, so what matters is
    # that illumination does not move it: a chroma threshold cuts the same place
    # on the edge however the scene is lit.
    if majors:
        spread = (max(majors) - min(majors)) / np.mean(majors) * 100
        print(f"        major axis across all four: {min(majors):.1f}-{max(majors):.1f} px "
              f"({spread:.1f}% spread), stroke-inflated from {true_major:.1f} px true")
        check("major axis is stable across illumination", spread < 4.0,
              f"{spread:.1f}% < 4%")


def test_empty_scene_is_not_a_detection():
    """The property the module docstring rejects Otsu for.

    An empty white backdrop contains no red, so an absolute chroma threshold
    returns nothing -- unlike an adaptive one, which would find the brightest
    thing present and report it confidently.
    """
    print("\nempty scenes must not detect")
    e = truth_ellipse()
    for name, bg, ramp in (("white, even", WHITE["nominal"], 0.0),
                           ("white, ramped", WHITE["nominal"], 0.35),
                           ("white, blown", WHITE["blown"], 0.0),
                           ("deep shade", WHITE["deep shade"], 0.0)):
        img = scene(None, bg, e, ramp=ramp, noise=6.0)
        seg = segment.segment(img, appearance="red")
        check(f"empty {name}: no detection", seg is None,
              "" if seg is None else f"{seg.area_px:.0f} px blob")


def test_grayscale_input_is_refused():
    """A grayscale frame cannot carry chroma, so say so rather than guess."""
    print("\ninput validation")
    img = scene(RED["mid"], WHITE["nominal"], truth_ellipse())
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        segment.segment(gray, appearance="red")
        check("grayscale input raises a clear error", False, "no exception")
    except ValueError as exc:
        check("grayscale input raises a clear error", "colour" in str(exc).lower(),
              f"{str(exc)[:60]}...")


def test_constants_are_matched_per_appearance():
    """Every appearance must carry its own complete, self-consistent constant set.

    The effective radius depends on which channel the boundary was thresholded
    in, so radius and tilt calibration are a matched pair *per appearance*.
    Shipping half a pair is the exact failure of journal Iterations 12-14, and it
    is silent -- a mismatched radius just biases every depth by a fixed percent.
    """
    print("\nconstants are matched per appearance")
    import importlib
    import os
    import subprocess

    import calibration
    import estimator

    known = set(estimator.RADIUS_BY_APPEARANCE)
    check("every appearance has a radius", known >= {"bright", "red"},
          ", ".join(sorted(known)))

    seen = {}
    for app in sorted(known):
        env = {**os.environ, "POSE_APPEARANCE": app}
        code = (
            "import sys; sys.path.insert(0, %r);"
            "import calibration, estimator;"
            "c = calibration.TiltCalibration.load();"
            "print(estimator.RADIUS_MM, calibration.calibration_path().name,"
            " c.a, c.meta.get('n_samples'))" % str(PKG)
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, env=env)
        check(f"{app}: package imports under this appearance", out.returncode == 0,
              out.stderr.strip()[-120:] if out.returncode else "")
        if out.returncode:
            continue
        radius, fname, a, n = out.stdout.split()
        seen[app] = (float(radius), fname, float(a), n)
        path = PKG / fname
        print(f"        {app:<7} radius {float(radius):.4f} mm   {fname}"
              f"   a={float(a):.4f}  n={n}")
        check(f"{app}: its calibration file exists", path.exists(), fname)
        check(f"{app}: calibration is not the identity fallback", float(a) != 1.0)

    if len(seen) > 1:
        radii = {v[0] for v in seen.values()}
        files = {v[1] for v in seen.values()}
        check("appearances use different radii", len(radii) == len(seen),
              f"{sorted(radii)}")
        check("appearances use different calibration files",
              len(files) == len(seen), f"{sorted(files)}")


def test_bright_appearance_unchanged():
    """The bright path must still behave as every constant was fitted against.

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
    print(f"        major {seg.ellipse[1][0]:.1f} px, rms {seg.fit_rms_px:.2f} px "
          f"(module default is {segment.APPEARANCE!r})")
    # The luminance path must not have picked up the chroma threshold.
    explicit = segment.segment(img, appearance="bright", thresh=segment.THRESH)
    check("bright uses the luminance threshold by default",
          explicit is not None and explicit.ellipse == seg.ellipse)
    check("the two appearances resolve different thresholds",
          segment.score_channel(img, "bright")[1] != segment.score_channel(img, "red")[1],
          f"{segment.score_channel(img, 'bright')[1]} vs "
          f"{segment.score_channel(img, 'red')[1]}")


def main():
    print("=" * 68)
    print("rig appearance: bright-on-dark, red-on-white, black-on-white (mono)")
    print("=" * 68)
    test_channel_separation()
    test_background_invariance()
    test_luminance_is_fragile_on_black()
    test_dark_rejects_clutter()
    test_dark_needs_no_colour()
    test_dark_refuses_without_a_region()
    test_red_detects_where_bright_fails()
    test_empty_scene_is_not_a_detection()
    test_grayscale_input_is_refused()
    test_constants_are_matched_per_appearance()
    test_bright_appearance_unchanged()
    print("\n" + "=" * 68)
    if _failures:
        print(f"FAILED ({len(_failures)}): " + ", ".join(_failures))
        return 1
    print("all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
