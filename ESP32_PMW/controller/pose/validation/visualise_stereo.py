"""Build the stereo diagnostics page.

    uv run python controller/pose/validation/visualise_stereo.py

Renders a small batch of stereo pairs, runs the estimator over them, picks a
typical case and a hard one, and writes a single self-contained HTML file with
the camera frames, the back-projection geometry in 3-D, and the measured error.

**Every number shown is measured on the pair shown.**  The obvious shortcut --
pick interesting rows out of `stereo_results.csv` and re-render those poses --
does not work, because those rows record the pose but not the lighting, exposure
or noise draw that went with it.  A re-render would produce a *different* image
with a *different* error, captioned with the old row's numbers, and nothing in
the page would reveal the mismatch.  So the batch is regenerated through the
same functions the sweep used, with a fixed seed, and the cases are chosen from
its own measurements.  The CSV is consulted only to say where a case sits in the
population of 900.

Same self-containment contract as `visualise.py`: nothing is fetched at view
time, every image is a data URI, and the page opens from disk with no network.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import math

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import conic  # noqa: E402
import gallery  # noqa: E402
import overlay as overlaymod  # noqa: E402
import render as rendermod  # noqa: E402
import render_stereo as rs  # noqa: E402
import scene3d  # noqa: E402
import stereo as stereomod  # noqa: E402
import sweep_stereo  # noqa: E402
import measurements  # noqa: E402
from page_stereo import (_b64_jpeg, _b64_png, _finding,  # noqa: E402
                         _mtable, render_page)
from rig import StereoRig  # noqa: E402

RESULTS = HERE.parents[2] / "results" / "pose_validation"
DEFAULT_OUT = RESULTS / "stereo_diagnostics.html"
DEFAULT_CSV = RESULTS / "stereo_results.csv"

# The configuration the sweep recommends: both cameras 45 degrees above
# horizontal, 90 degrees apart in bearing. 0.352 mm worst axis, 93% detection.
ELEV = (45.0, 45.0)
AZIM = (0.0, 90.0)
RANGE_MM = 250.0
WIDTH, HEIGHT = 500, 375
CALIB_WIDTH = 1024.0

TILE = 420  # per-frame overlay size, a little larger than gallery's 340
GATE_MM = 0.5

# Amber, matching scene3d.COL_EST -- but in BGR, because overlay.py draws with
# cv2. Getting this backwards yields a blue "estimate" and a very confusing page.
COL_REPROJ_BGR = (10, 147, 200)
# Green for the axes: distinct from the fitted ellipse (blue), the truth (white)
# and the reprojection (amber), all of which are already on the frame.
COL_MAJOR_BGR = (90, 220, 90)


class _ViewPose:
    """A per-view stand-in that `overlay.draw` will accept.

    `overlay.draw` is camera-frame throughout: it assumes `pose.xyz_mm` is
    already in the coordinates of the camera it is drawing into.  A
    `stereo.StereoPose` is in the **world** frame, so handing it over directly
    would put the orientation arrows and the residual in the wrong place -- and
    plausibly so, which is worse than obviously so.

    This carries that view's own ellipse and the world pose mapped through
    `rig.Camera.to_camera`.  `overlay.py` itself is untouched: it is tested and
    shared with the monocular page.
    """

    def __init__(self, pose, cam, seg):
        self.xyz_mm, self.normal = cam.to_camera(pose.xyz_mm, pose.normal)
        self.ellipse = seg.ellipse
        self.area_px = seg.area_px
        self.fit_rms_px = seg.fit_rms_px
        self.theta_deg = pose.theta_deg
        self.phi_deg = pose.phi_deg
        self.psi_deg = seg.ellipse[2]
        self.ambiguity_margin_deg = pose.ambiguity_margin_deg
        # No alternative branches: the whole point of stereo is that the branch
        # was decided by geometry, so drawing a rejected one would misrepresent
        # what happened. `overlay.draw` skips the layer when this is empty.
        self.extra = {}


def build_batch(rig, n, seed, subframes=5, on_renderer=None):
    """Render and estimate a batch, reusing the sweep's own sampling.

    Same functions, same call order, same seed semantics as
    `sweep_stereo.run`, so the cases are drawn from the distribution the
    published numbers describe rather than from a fresh ad-hoc one.
    """
    rng = np.random.default_rng(seed)
    tilt, az, centres = sweep_stereo.sample_poses(rng, n)
    lights = sweep_stereo.lighting(rng, n)
    exps = sweep_stereo.exposures(rng, n, subframes=subframes)
    alphas = rng.choice([0.8, 0.9, 1.0], n)
    bgs = rng.choice([0.0, 0.1, 0.2], n)

    est = stereomod.StereoPoseEstimator(rig)
    out = []
    extra = None
    with rs.StereoRenderer(rig, WIDTH, HEIGHT) as r:
        for i in range(n):
            s = r.render_pair(float(tilt[i]), float(az[i]), centres[i],
                              alpha=float(alphas[i]), light=lights[i],
                              bg_level=float(bgs[i]), exposure=exps[i])
            pose = est.update([v.image for v in s.views], t=float(i))
            if pose is None or pose.n_views < len(rig.cameras):
                continue
            d = pose.xyz_mm - s.center_world
            out.append({
                "i": i,
                "sample": s,
                "pose": pose,
                "d": d,
                "pos_mm": float(np.linalg.norm(d)),
                "normal_deg": stereomod.line_angle_deg(pose.normal, s.normal_world),
            })
            if (i + 1) % 20 == 0:
                print(f"    {i + 1}/{n} rendered", flush=True)

        # Anything needing the renderer must run before this block exits: there
        # is one GL context per process (pyglet/Cocoa), so it cannot be reopened
        # afterwards. Hence the callback rather than returning the renderer.
        if on_renderer is not None:
            extra = on_renderer(r._renderer, out)
    return out, extra


def pick_cases(batch):
    """The case nearest the median position error, and one near p95.

    Deliberately not "best and worst": the best case flatters and the single
    worst is usually a segmentation failure that says nothing about the
    geometry. The median is what the estimator normally does, and p95 is what it
    does when things go badly but not catastrophically -- which is the regime a
    controller actually has to survive.
    """
    errs = np.array([b["pos_mm"] for b in batch])
    order = np.argsort(errs)
    typical = batch[int(order[len(order) // 2])]
    hard = batch[int(order[min(len(order) - 1, int(0.95 * len(order)))])]
    return [("typical", typical), ("hard", hard)]


def _frame_image(sample_view, pose, cam, seg, radius_mm):
    """One annotated camera frame, JPEG-encoded.

    Three ellipses, and the third is the one that matters. `overlay.draw`
    supplies the fitted (solid, blue) and the true (dashed, white) for this
    view. On top of those goes the **joint 3-D estimate reprojected into this
    view** -- the same pose drawn into both images, which is the claim the
    stereo solve makes and which no monocular overlay can express.
    """
    img = overlaymod.draw(
        sample_view.image, _ViewPose(pose, cam, seg), sample_view,
        K=cam.K, radius_mm=radius_mm,
        layers=("fit", "truth", "normal", "text"),
        crop=2.4, size=TILE, legend=False,
    )

    c_cam, n_cam = cam.to_camera(pose.xyz_mm, pose.normal)
    try:
        reproj = conic.project_circle(c_cam, n_cam, radius_mm, cam.K)
        truth_e = conic.project_circle(sample_view.center_mm, sample_view.normal,
                                       radius_mm, cam.K)
    except Exception:
        return gallery._encode(img)

    # `overlay.draw` cropped and resized, so the ellipse must follow. Cropping
    # is a translation and resizing a scale, and `_crop` returns the shifted K
    # -- but not to us, so the transform is recovered from the size ratio the
    # same way `_crop`/`_resize` computed it.
    scale = img.shape[1] / (2.4 * max(seg.ellipse[1]))
    (cx, cy), (major, minor), ang = reproj
    (ox, oy) = seg.ellipse[0]
    moved = (
        (img.shape[1] / 2 + (cx - ox) * scale, img.shape[0] / 2 + (cy - oy) * scale),
        (major * scale, minor * scale),
        ang,
    )
    cv2.ellipse(img, tuple(int(round(v)) for v in moved[0]),
                tuple(max(1, int(round(v / 2))) for v in moved[1]),
                moved[2], 0, 360, COL_REPROJ_BGR, 1, cv2.LINE_AA)

    # The two axes, drawn because their reliability differs by 25x and that is
    # invisible unless you show it: the major axis holds to ~0.5% at every tilt,
    # the minor axis scatter reaches 26% past 60 degrees. Depth is measured from
    # the major axis, tilt from the ratio -- which is the whole reason position
    # survives tilts that destroy orientation.
    def _axes(ell, colour, thickness, dashed):
        (ex, ey), (mj, mn), a = conic.normalise_ellipse(ell)
        cx_, cy_ = (img.shape[1] / 2 + (ex - ox) * scale,
                    img.shape[0] / 2 + (ey - oy) * scale)
        th = math.radians(a)
        for half, (ux, uy), lab in ((mj * scale / 2, (math.cos(th), math.sin(th)), "major"),
                                    (mn * scale / 2, (-math.sin(th), math.cos(th)), "minor")):
            p0 = (int(cx_ - half * ux), int(cy_ - half * uy))
            p1 = (int(cx_ + half * ux), int(cy_ + half * uy))
            if dashed:
                n = max(2, int(2 * half / 6))
                for j in range(0, n, 2):
                    q0 = (int(p0[0] + (p1[0] - p0[0]) * j / n),
                          int(p0[1] + (p1[1] - p0[1]) * j / n))
                    q1 = (int(p0[0] + (p1[0] - p0[0]) * (j + 1) / n),
                          int(p0[1] + (p1[1] - p0[1]) * (j + 1) / n))
                    cv2.line(img, q0, q1, colour, thickness, cv2.LINE_AA)
            else:
                cv2.line(img, p0, p1, colour, thickness, cv2.LINE_AA)

    _axes(seg.ellipse, COL_MAJOR_BGR, 1, False)
    _axes(truth_e, (255, 255, 255), 1, True)
    overlaymod._text(img, (10, img.shape[0] - 24), "axes: fitted solid, true dashed",
                     COL_MAJOR_BGR)
    overlaymod._text(img, (10, img.shape[0] - 10), "joint estimate reprojected",
                     COL_REPROJ_BGR)
    return gallery._encode(img)


def _ordinal(n):
    """1st, 2nd, 3rd, 4th -- including the 11th/12th/13th exceptions."""
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _percentile_of(csv_path, value, column="stereo_pos_mm"):
    """Where this case sits in the swept population. ``None`` if no CSV.

    Restricted to rows from **this** rig geometry. The sweep spans six
    configurations whose accuracies differ by 4x, so comparing against the
    pooled population made an entirely ordinary case read as 25th percentile --
    a statement about which rigs were in the CSV, not about the case.
    """
    try:
        import pandas as pd

        df = pd.read_csv(csv_path, comment="#")
        same = (df["detected"] == 1)
        for col, want in (("elev_a_deg", ELEV[0]), ("elev_b_deg", ELEV[1]),
                          ("azim_a_deg", AZIM[0]), ("azim_b_deg", AZIM[1])):
            if col in df.columns:
                same &= np.isclose(df[col], want)
        v = df.loc[same, column].dropna().to_numpy()
        return (float((v < value).mean() * 100.0), int(len(v))) if len(v) else None
    except Exception:
        return None


def _fmt(v, spec="{:.2f}"):
    return None if v is None else spec.format(v)


def build_measurements(batch, rig, radius_mm, renderer):
    """The findings section: tables and one chart, all computed in this run.

    Called from inside the renderer's context, because two of the six findings
    need to render sliced copies of the mesh and there is one GL context per
    process.
    """
    out = []

    # --- what causes the contamination ------------------------------------
    att = measurements.mesh_attribution(renderer)
    hdr = ["mesh"] + [f"{t:g}°" for t in att["tilts"]]
    rows = [(r["name"],
             [None if c is None else f"{c[0]:+.1f} / {c[1]:+.1f}" for c in r["cells"]],
             r["name"] == "rim ring only")
            for r in att["rows"]]
    out.append(_finding(
        "Which part of the robot spoils the outline",
        "The robot is not a flat ring — a rod sticks out one side of it and the body "
        "the other. Here the same poses are re-rendered with those parts deleted from "
        "the 3-D model, one at a time. Numbers are how far the outline bulges past a "
        "perfect ellipse, in pixels, on each of the two sides. If deleting a part "
        "makes the bulge vanish, that part caused it.",
        _mtable(hdr, rows)
        + "<p class=\"capt\">Removing the <b>rod</b> makes the second column collapse "
          "past 55&deg;. Removing the <b>body</b> changes nothing at any tilt. The "
          "<b>ring on its own</b> reproduces the first column exactly. So below about "
          "55&deg; the error is not the rod or the magnet at all &mdash; it is that the "
          "ring itself has thickness, and a thick ring seen at an angle is wider than "
          "the flat circle the maths assumes.</p>"))

    # --- and its functional form -------------------------------------------
    fit = measurements.wall_model_fit(renderer)
    if fit:
        mean_h = sum(f["implied_h_mm"] for f in fit) / len(fit)
        out.append(_finding(
            "The wall model",
            f"If the extra width really is the ring’s own thickness, then it "
            f"should grow in a specific way as the robot tilts — in proportion to "
            f"sin(tilt) — and dividing one by the other should give a flat line at the "
            f"ring’s half-thickness. Measured on the ring alone: {mean_h:.3f} mm.",
            f'<figure class="scene"><img src="{_b64_png(measurements.wall_model_chart(fit))}" '
            f'alt="implied wall half-height against tilt, flat if the model holds" '
            f'loading="lazy"></figure>'
            "<p class=\"capt\">Flat means the explanation is right. It also means the "
            "correction can be worked out from the shape of the robot rather than fitted "
            "to data &mdash; one number, the ring&rsquo;s thickness, instead of a curve "
            "with arbitrary coefficients. The point below 20&deg; drops off because the "
            "effect there is smaller than one pixel at this resolution.</p>"))

    # --- the centre, which is a position error rather than an orientation one
    cb = measurements.centre_bias(renderer)
    if cb:
        out.append(_finding(
            "The outline's centre is displaced too",
            "the parts that stick out grow the outline on one side, so its centre "
            "moves — and sideways position is read straight off that centre. "
            "Displacement is split along the outline's own long and short axes.",
            _mtable(["tilt", "along long axis", "along short axis",
                     "as distance", "with one-sided re-weighting"],
                    [(f'{r["tilt"]:.0f}°',
                      [f'{r["d_major_px"]:+.2f} px', f'{r["d_minor_px"]:+.2f} px',
                       f'{r["plain_mm"]:.3f} mm', f'{r["one_sided_mm"]:.3f} mm'],
                      False) for r in cb])
            + "<p class=\"capt\">Almost all of it is along the <b>short</b> axis — the "
              "direction things stick out in — while the long axis stays flat. The sign "
              "<b>flips</b> near 57°: below that the rim&rsquo;s own thickness pushes the "
              "centre one way, above it the rod pulls it the other, so no single "
              "tilt-based correction can undo both. At 70° this displacement alone is "
              "0.87&nbsp;mm, past the 0.5&nbsp;mm target. Re-weighting the outline only "
              "partly helps, and not at all below 55°, because the weights are judged "
              "against an ellipse that has already absorbed the shift. <b>This is not yet "
              "corrected anywhere in the shipped estimator.</b></p>"))

    # --- channel reliability ------------------------------------------------
    ch = measurements.channel_reliability(batch, rig, radius_mm)
    if ch:
        out.append(_finding(
            "Which channels survive",
            "How wrong each of the ellipse’s measurements is, compared with what "
            "geometry says it should be, grouped by how side-on that camera’s view "
            "of the robot is. This is the table everything else follows from.",
            _mtable(["how side-on", "samples", "long axis, length", "long axis, angle",
                     "short axis, length"],
                    [(r["band"], [str(r["n"]),
                                  f'{r["major_pct"]:+.2f}% ± {r["major_sd"]:.2f}',
                                  f'{r["psi_deg"]:.2f}°',
                                  f'{r["minor_pct"]:+.2f}% ± {r["minor_sd"]:.2f}'], False)
                     for r in ch])
            + "<p class=\"capt\">The <b>long</b> axis is measured to well under a percent "
              "however side-on the view, and its angle to a fraction of a degree. The "
              "<b>short</b> axis is an order of magnitude worse, because that is the "
              "direction the protruding parts widen. Distance is read from the long axis "
              "and tilt from the ratio of the two &mdash; which is exactly why distance "
              "stays good in situations that ruin tilt.</p>"))

    # --- the tilt models ----------------------------------------------------
    cal = measurements.calibration_models()
    if cal:
        rows = []
        for r in cal["rows"]:
            label = r["name"]
            if r["k"] is not None:
                label += f'  (k={r["k"]:.4f}, floor {r["floor"]:.2f}°)'
            rows.append((label,
                         [None if c is None else f"{c[0]:.2f} / {c[1]:+.2f}"
                          for c in r["cells"]],
                         r["name"].startswith("cylinder, fit 20")))
        out.append(_finding(
            "Correcting for it",
            f'median |tilt error| / bias, degrees, on the {cal["n_test"]}-pose held-out '
            f'split. Every model fitted on train only.',
            _mtable(["model"] + cal["bands"], rows)
            + "<p class=\"capt\">The one-parameter physical model beats the "
              "two-parameter fitted curve &mdash; but only when fitted where it applies. "
              "Over the full range the mast drags k upward and the result is worse than "
              "applying no correction at all.</p>"))

    # --- hull weighting -----------------------------------------------------
    w = measurements.weighting_schemes(batch, rig, radius_mm)
    if w:
        out.append(_finding(
            "Re-weighting the outline",
            "Three ways of fitting the ellipse to the same outlines. Each cell is the "
            "typical tilt error and the systematic part of it, in degrees.",
            _mtable(["scheme"] + w["bands"],
                    [(r["name"], [None if c is None else f"{c[0]:.2f} / {c[1]:+.2f}"
                                  for c in r["cells"]],
                      r["name"].startswith("C")) for r in w["rows"]])
            + "<p class=\"capt\"><b>B</b> pays less attention to the parts of the outline "
              "nearest the ends of the <i>short</i> axis, which is where the spoiling "
              "happens. It loses &mdash; because those same points are the only ones that "
              "reveal how long the short axis is, so it discards the very measurement it "
              "was trying to clean. <b>C</b> instead pays less attention to points that "
              "stick <i>outward</i>. That works because the outline can only ever bulge "
              "outward, never inward, so &ldquo;outward&rdquo; is a reliable sign of "
              "trouble while &ldquo;near the short axis&rdquo; is not.</p>"))

    # --- the major-axis channel ---------------------------------------------
    ab = measurements.major_channel_ab(batch, rig)
    if ab:
        out.append(_finding(
            "A way to get tilt without the short axis at all",
            "The short axis is the unreliable one, so here is a route that never uses "
            "it. The <i>angle</i> of the long axis tells each camera which way the robot "
            "leans, though not how far; two cameras each supplying that is enough to pin "
            "the tilt down completely. Numbers are typical tilt error in degrees. Bands "
            "with fewer than five samples in this batch are left blank rather than "
            "filled in from an earlier run.",
            _mtable(["configuration"] + ab["bands"],
                    [(r["name"], [_fmt(c, "{:.2f}°") for c in r["cells"]], False)
                     for r in ab["rows"]])
            + "<p class=\"capt\"><b>Off by default</b>, and this table shows why: in the "
              "bands the report&rsquo;s batch actually populates &mdash; which are the "
              "bands the default rig operates in &mdash; the channel makes the normal "
              "worse. A dedicated A/B over a wider pose range (90 frames, "
              "<code>tilt_max=45°</code>) found it a 2.25× <i>gain</i> at 55–70° and a "
              "1.65× gain overall, so the geometry is sound; what is wrong is the blend "
              "weight. Its σ was derived from the precision of ψ in a <i>single</i> view, "
              "but the normal comes from a cross product of two and is worse than either, "
              "so the channel claims 0.35° against the ratio channel&rsquo;s 0.45° and "
              "outvotes it exactly where the ratio channel is still better. Fitting that σ "
              "from measured normal error is the outstanding work. Enable with "
              "<code>use_major_channel=True</code>.</p>"))

    return "".join(out)


def build_payload(n_batch, seed, csv_path):
    rig = StereoRig.from_spherical(
        elev_deg=ELEV, azim_deg=AZIM, range_mm=RANGE_MM).scaled(WIDTH / CALIB_WIDTH)
    radius_mm = stereomod.RADIUS_MM

    print(f"rendering {n_batch} pairs at {WIDTH}x{HEIGHT} ...", flush=True)

    def _measure(renderer, so_far):
        print("  measuring ...", flush=True)
        return build_measurements(so_far, rig, radius_mm, renderer)

    batch, measured = build_batch(rig, n_batch, seed, on_renderer=_measure)
    if len(batch) < 4:
        raise SystemExit(f"only {len(batch)} usable pairs; cannot pick cases")
    print(f"  {len(batch)}/{n_batch} usable", flush=True)

    cases = []
    for label, b in pick_cases(batch):
        pose, s = b["pose"], b["sample"]
        ellipses = [seg.ellipse if seg is not None else None for seg in pose.per_view]

        frames = []
        for k, (cam, view, seg) in enumerate(zip(rig.cameras, s.views, pose.per_view)):
            inter = np.logical_and(seg.mask > 0, view.mask).sum()
            union = np.logical_or(seg.mask > 0, view.mask).sum()
            frames.append({
                "name": cam.name or str(k),
                "img": _b64_jpeg(_frame_image(view, pose, cam, seg, radius_mm)),
                "iou": float(inter / union) if union else 0.0,
                "rms": float(seg.fit_rms_px),
            })

        scenes = []
        for zoom, cap in (
            (False, "The whole setup seen from outside. Each camera sends out a "
                    "fan of lines through the outline it measured; the two fans "
                    "meet where the robot is. That intersection IS the answer — "
                    "one camera alone gives you a fan, not a point."),
            (True, "Close-up of the robot itself, at true size. Black is where it "
                   "really was, amber is where the estimate put it. The dots are "
                   "the rim as each camera saw it, carried back into 3-D: how far "
                   "they sit off the circle is how much the two disagree."),
        ):
            png = scene3d.render(
                rig, s.center_world, s.normal_world, pose.xyz_mm, pose.normal,
                radius_mm, ellipses=ellipses, image_size=(WIDTH, HEIGHT), zoom=zoom,
                title=None,
            )
            scenes.append({"img": _b64_png(png), "caption": cap,
                           "alt": ("3-D view of the two cameras, their "
                                   "back-projection cones and the estimated disk"
                                   if not zoom else
                                   "close 3-D view of the ground-truth and "
                                   "estimated disk and rotation axis")})

        # The panel that answers "why does tilt hurt": where on the rim the
        # silhouette leaves the ellipse. Measured against the TRUE rim, so it
        # shows the physical departure rather than what the fit could not absorb.
        dev_views = []
        for cam, view, seg in zip(rig.cameras, s.views, pose.per_view):
            try:
                te = conic.normalise_ellipse(conic.project_circle(
                    view.center_mm, view.normal, radius_mm, cam.K))
            except Exception:
                te = None
            dev_views.append((cam.name or "?", seg.contour if seg else None, te))
        scenes.append({
            "img": _b64_png(scene3d.deviation_panel(dev_views)),
            "caption": ("How far the robot's measured outline sits outside a "
                        "perfect ellipse, walking once around the rim. Zero means "
                        "the outline is exactly elliptical there. The bumps near "
                        "±90° are the parts of the robot that stick out sideways "
                        "(the rim's own thickness, and past ~55° the rod); the ends "
                        "of the long axis (0°, 180°) stay flat, which is why "
                        "distance stays accurate when tilt does not."),
            "alt": "plot of outward deviation of hull points against position around the rim",
        })

        pct = _percentile_of(csv_path, b["pos_mm"])
        where = (f"{_ordinal(round(pct[0]))} percentile of {pct[1]} swept poses "
                 f"at this rig geometry"
                 if pct else "percentile unavailable (no sweep CSV)")
        d = b["d"]
        cases.append({
            "title": f"{label.capitalize()} case",
            "subtitle": (
                f"tilt {s.tilt_deg:.1f}° lean at bearing {s.azimuth_deg:.0f}°, "
                f"centre ({s.center_world[0]:+.1f}, {s.center_world[1]:+.1f}, "
                f"{s.center_world[2]:+.1f}) mm — {where}"
            ),
            "frames": frames,
            "scenes": scenes,
            "dx_mm": float(d[0]), "dy_mm": float(d[1]), "dz_mm": float(d[2]),
            "pos_mm": b["pos_mm"], "normal_deg": b["normal_deg"],
            "discrepancy_mm": float(pose.discrepancy_mm),
            "margin": float(pose.margin),
            "refine_rms_px": float(pose.refine_rms_px),
            "worst_axis_mm": float(np.abs(d).max()),
            "gate_mm": GATE_MM,
        })

    sig = rig.predicted_sigma_mm(
        stereomod.SIGMA_LAT_MM / (WIDTH / CALIB_WIDTH),
        stereomod.SIGMA_DEPTH_MM / (WIDTH / CALIB_WIDTH),
    )
    rig_rows = [
        ("elevation", f"{ELEV[0]:+.0f}° / {ELEV[1]:+.0f}° above horizontal"),
        ("bearing", f"{AZIM[0]:.0f}° / {AZIM[1]:.0f}° about world +z"),
        ("optical axes", f"{rig.axis_separation_deg():.1f}° apart (as lines)"),
        ("baseline", f"{rig.baseline_mm():.0f} mm"),
        ("rotor tilt seen", " / ".join(f"{t:.0f}°" for t in rig.tilt_seen_deg())),
        ("resolution", f"{WIDTH}×{HEIGHT}"),
        ("predicted fused σ", f"{sig.min():.3f} / {sig[1]:.3f} / {sig.max():.3f} mm "
                                    f"(worst axis {sig.max():.3f})"),
        ("gate", f"{GATE_MM} mm on every axis"),
    ]

    return {
        "title": "Stereo pose diagnostics",
        "lede": ("Two cameras, one 5-DOF pose. Each case shows the stereo pair the "
                 "estimator saw, the back-projection geometry that produced the "
                 "answer, and that answer against ground truth."),
        "rig_rows": rig_rows,
        "cases": cases,
        "measurements": measured or "",
        "provenance": (
            f"Generated {time.strftime('%Y-%m-%d %H:%M')} from {len(batch)} rendered "
            f"pairs (seed {seed}), radius {radius_mm:.4f} mm, mesh rim "
            f"{rendermod.RIM_RADIUS_MM:.4f} mm. Self-contained: every image is "
            f"embedded, nothing is fetched when the page opens."
        ),
    }


def _strip_images(p):
    """The payload minus the data URIs, for a readable JSON sidecar."""
    out = json.loads(json.dumps(p, default=float))
    for c in out["cases"]:
        for f in c["frames"]:
            f["img"] = f"<{len(f['img'])} B data uri>"
        for s in c["scenes"]:
            s["img"] = f"<{len(s['img'])} B data uri>"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--batch", type=int, default=60,
                    help="pairs to render before picking cases")
    ap.add_argument("--seed", type=int, default=20260808)
    ap.add_argument("--csv", default=str(DEFAULT_CSV),
                    help="sweep results, used only for the percentile line")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--payload-only", action="store_true",
                    help="write the JSON and stop, for iterating on the HTML")
    args = ap.parse_args(argv)

    payload = build_payload(args.batch, args.seed, Path(args.csv))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(_strip_images(payload), indent=2))
    print(f"wrote {out.with_suffix('.json')}")

    if args.payload_only:
        return 0

    out.write_text(render_page(payload))
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
