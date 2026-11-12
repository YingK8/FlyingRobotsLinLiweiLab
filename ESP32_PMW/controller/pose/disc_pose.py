#!/usr/bin/env python3
"""Stereo attitude of a bare rotor-on-a-mast, through the existing pose pipeline.

`stereo.StereoPoseEstimator` is built for a robot with a bright rim. The tilt-sweep
robots have none -- a propeller and a mast -- and on their footage the pipeline solves 0
of 400 frames. Everything after segmentation is exactly what is wanted, though: undistort,
`conic.backproject_ellipse`, the two-view branch resolution in `fuse`, and a normal in
the WORLD frame, so the tilt it reports is from the rig's vertical and not from wherever a
camera happens to point. Only the segmentation needs replacing, so only it is.

WHAT IS SEGMENTED, AND WHY THE FIRST ATTEMPT WAS WRONG
------------------------------------------------------
The spinning rotor blurs into a filled grey disc. At the pipeline's default threshold the
segmenter takes the whole scene; at 160 it takes a **specular highlight near the hub**,
1-2% of the frame, whose ellipse has nothing to do with the rotor's -- measured 2026-09-02,
that gave "tilts" of 49-67 deg at every frequency that were just the highlight's shape.
The disc itself sits around 100-140 grey against a background that never exceeds ~100.

So: threshold at `DISC_THRESH`, keep the largest component (disc + mast + the guy wires,
which are all one blob at that level), then **open with an 11 px disc**. The mast is ~6 px
wide and the wires thinner, so the opening deletes them; the rotor is 60+ px and
survives. The repo's own `segment.fit_ellipse` is then fitted to that disc's hull, and
the rest of the pipeline is untouched.

THE SECOND ATTEMPT WAS HALF RIGHT, LITERALLY
--------------------------------------------
The disc is lit from one side. Its shadowed half reads 65-100 grey, under `DISC_THRESH`,
so the plain threshold kept the lit half and the ellipse was a half-moon in every view
of every robot (seen 2026-09-02 on the demo videos, once there were demo videos). The
bright component is now only a *seed*, grown into everything above `DISC_LOW` that
touches it, and roundish enclosed holes are filled so drone 3's thin bright rim round a
dark interior comes out as a disc instead of being opened away. Every number used in the
earlier stereo passes was fitted to the lit half and has been regenerated.

THE BACKGROUND IS SUBTRACTED, FROM A PLATE THE ROBOT IS NOT IN (pose/theory.md 20)
--------------------------------------------------------------
Take 205012 opens with a hand and a sheet across camera B and a bright patch on the foam
that wins the threshold when the robot is out of view. `build_plate` estimates each
camera's static background from the take itself and `segment_disc` works on the
difference. The plate is a LOW percentile over time, not an average -- the reasoning and
the numbers are at `PLATE_FRAMES` -- and a blob over `MAX_AREA_PX` is refused, because a
hand is transient and no plate removes it.

THE RIM RADIUS IS WRONG FOR THIS ROTOR, AND THAT DOES NOT MATTER FOR THE ANGLE
-------------------------------------------------------------------------------
`radius_mm` is the rim's 10.24 mm. `conic.backproject` uses it only to scale the centre's
distance: the normal came out identical to six decimals at 10.2, 25 and 60 mm (verified
2026-09-02). So the attitude is right and the position is not, and `do_refine` is forced
off, because the joint refinement DOES fit against the radius and would pull the normal
toward a circle of the wrong size.

THE MAST IS THE BODY AXIS, AND IT TELLS THE BLADES FROM THE ROD
---------------------------------------------------------------
Below ~80 Hz the shutter freezes the blades, and a frozen blade is a bright elongated
component just like the mast. What separates them is not brightness or size but
direction: the mast is perpendicular to the rotor plane, a blade lies in it. So
`find_mast` scores every thin bright component by how perpendicular it is to the disc's
projected major axis and takes the best. Triangulated across the pair (`mast_direction`)
it is an independent measurement of the same axis the disc normal estimates, from a
feature that does not strobe.

    uv run python controller/pose/disc_pose.py                   # self-check
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from controller.pose import conic
from controller.pose import segment as segmod
from controller.pose import stereo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

#: Grey level that takes the blurred disc but not the foam blocks below it. From the
#: 2026-09-01 drone-1 stills: the background's brightest pixel is ~100, the disc's dim
#: half ~110-140, its bright crescent >200. Re-check with `--sheet` if lighting changes.
DISC_THRESH = 110
#: The disc is lit from one side, and its shadowed half reads 65-100 -- under
#: `DISC_THRESH`, so a plain threshold kept the bright half only and the ellipse fitted a
#: half-moon (every robot, both views, seen 2026-09-02 on the demo videos). Pixels above
#: this level that CONNECT to the bright seed are kept; the foam blocks below sit at 60-90
#: and touch the disc in some views, and 50 leaked into them on drones 1, 2 and 3 while 65
#: did not. Measured on 14 frames across the three robots, spinning and at rest.
DISC_LOW = 65
#: Holes enclosed by the grown mask are filled before opening -- drone 3's rim is a thin
#: bright ring round a dark interior, and the opening deletes the ring and leaves nothing.
#: Only roundish holes: the frozen blades, mast and guy wires at rest enclose a tall thin
#: sliver (42x113, 51x133 px) that must not be filled, while disc interiors ran 78x42 to
#: 89x62. ponytail: bbox aspect is a thin margin (1.9 vs 2.7); a hole inside the seed's
#: fitted ellipse would be the principled test if this ever misfires.
HOLE_MAX_ASPECT = 2.2
#: Thresholds on the plate-SUBTRACTED frame (`build_plate`). The backdrop and blocks go
#: to ~0-10 there, so the low level can drop well under `DISC_LOW`; the disc's dim half
#: read 37-63 (p10 of disc pixels, 205012 and 210758, both views) and its crescent >170.
DIFF_THRESH, DIFF_LOW = 90, 35
#: Largest thing that is a rotor. Drone 3's disc, the biggest, reached 21912 px at 640x400;
#: a hand or a sheet in front of the lens is 100000+ (205012 frame 9219) and is nothing.
MAX_AREA_PX = 40000
#: Background plate: per-pixel PLATE_PCT-th percentile over PLATE_FRAMES frames spread
#: across the take, after dropping the brightest PLATE_DROP_FRAC of them by mean grey.
#: A percentile that low, not a mean or median: the robot never leaves the frame -- it
#: sits on its mast for the whole take and spins for ~70% of it -- so the mean and the
#: median at a disc pixel ARE the spinning disc (152-202 grey on 205012; subtracting
#: either left the disc at 0-2 above background). Between frozen blades at rest the
#: backdrop shows through, so the 10th percentile is the backdrop (19-38) everywhere but
#: the hub. Dropping the brightest frames keeps a hand or a sheet across the lens (mean
#: grey 108-150 against 37) out of the plate; it does nothing on a clean take.
PLATE_FRAMES, PLATE_PCT, PLATE_DROP_FRAC = 300, 10, 0.2
#: Opening kernel, px. Must exceed the mast's width (~6 px) and stay well under the disc's
#: minor axis (~50 px at 640x400). 11 deletes mast, wires and bead; 15 starts eating the
#: disc's thin edge-on end.
OPEN_PX = 11
#: The opening erases everything thinner than `OPEN_PX`, and a BLADE TIP is thinner than
#: that as often as a guy wire is. Measured on drone 1, 16 views across the take: 12-35 %
#: of the above-threshold pixels never reached the mask, and the erased set is the mast,
#: the wires, the bead -- and a visible white blade tip (2026-09-04, reported from the
#: overlay video as "the white propellers are not all detected").
#:
#: So the opening picks the disc BODY, and the pre-opening mask is then given back inside
#: the body's own ellipse scaled by this, keeping only what still connects to the body.
#: The window is the point: mast and wires leave the disc along its NORMAL, which is the
#: thin direction of that ellipse, so they exit the window within a few px however far
#: they run, while a blade tip leaves along the disc plane and stays inside it. It can
#: therefore only extend the hull where a blade sticks out, which is the defect.
#:
#: 1.25 recovers 167-1240 px per view with the hull fit rms staying in its 1-3.4 px band.
#: Do NOT instead keep the components the opening severed: measured on the same frames,
#: that grabs something off-rotor and takes the minor axis 86 -> 114 px and the rms
#: 0.8 -> 4.5. ponytail: blade tips past 1.25 radii (a rotor at rest, seen edge-on) are
#: still cut; pushing the scale out reaches the wires, so that case wants the mast pick
#: to mask them out first, not a bigger window.
RECOVER_SCALE = 1.25
#: The residual component holding the mast also holds the bead and the guy wire above it
#: (measured 585-1318 px during holds, more at rest with the frozen blades attached).
MAST_MIN_PX, MAST_MAX_PX = 40, 4000
MAST_MIN_ELONG = 3.0
#: Sign reference for the triangulated mast: world -y, camera A's up (`stereo_rig.json`
#: says world = camera A, OpenCV axes, +y down). The mast points from hub to bead, up.
MAST_UP = (0.0, -1.0, 0.0)


def _kernel():
    import cv2
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_PX, OPEN_PX))


def build_plate(video_path, n_frames=PLATE_FRAMES, pct=PLATE_PCT, drop_frac=PLATE_DROP_FRAC):
    """Background plate for one camera's video, uint8, same size as its frames.

    See `PLATE_FRAMES`: a low per-pixel percentile over frames spread across the take,
    the brightest fraction of frames dropped first.
    """

    import cv2

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        raise FileNotFoundError(f"no frames in {video_path}")
    frames = []
    for k in np.linspace(0, total - 1, min(n_frames, total)).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
        ok, f = cap.read()
        if ok:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f)
    cap.release()
    F = np.array(frames)
    bright = F.mean(axis=(1, 2))
    keep = bright <= np.percentile(bright, 100 * (1 - drop_frac))
    return np.percentile(F[keep], pct, axis=0).astype(np.uint8)


def plates_for_flight(flight_dir, tags="AB"):
    """``{tag: plate}`` for a recording, built once and cached as ``plate_<tag>.png``
    beside its videos."""

    import cv2

    flight_dir = Path(flight_dir)
    out = {}
    for tag in tags:
        cache = flight_dir / f"plate_{tag}.png"
        if cache.exists():
            out[tag] = cv2.imread(str(cache), cv2.IMREAD_GRAYSCALE)
            continue
        vids = sorted(flight_dir.glob(f"{tag}/*.mp4")) or sorted(flight_dir.glob(f"{tag}.mp4"))
        if not vids:
            continue
        out[tag] = build_plate(vids[0])
        cv2.imwrite(str(cache), out[tag])
    return out


def _subtract(gray, plate):
    import cv2
    return gray if plate is None else cv2.subtract(gray, plate)


def _fill_round_holes(mask, max_aspect=HOLE_MAX_ASPECT):
    """Fill enclosed background regions whose bounding box is no taller than wide by
    more than ``max_aspect`` (either way). See `HOLE_MAX_ASPECT`."""

    import cv2

    inv = (1 - mask).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, 4)
    h, w = mask.shape
    out = mask.copy()
    for k in range(1, n):
        x, y, bw, bh, _ = st[k]
        if x == 0 or y == 0 or x + bw == w or y + bh == h:
            continue                       # touches the border: background, not a hole
        if max(bw, bh) <= max_aspect * min(bw, bh):
            out[lab == k] = 1
    return out


def _recover_blades(body, grown, scale=RECOVER_SCALE):
    """Give back the blade tips the opening thinned away. See `RECOVER_SCALE`.

    ``body`` is the opened disc, ``grown`` the mask before the opening. Returns the part
    of ``grown`` that lies inside ``body``'s own ellipse scaled by ``scale`` and still
    connects to ``body`` -- so the mast and the wires, which leave along the disc's
    normal, are outside the window and stay out.
    """

    import cv2

    hull, _ = segmod.silhouette_hull(body)
    if hull is None or len(hull) < 6:
        return body
    fit = segmod.fit_ellipse(hull)
    if fit is None:
        return body
    (cx, cy), (ma, mi), ang = fit[0]
    win = np.zeros_like(body)
    cv2.ellipse(win, ((cx, cy), (ma * scale, mi * scale), ang), 1, -1)
    n, lab, _, _ = cv2.connectedComponentsWithStats((grown & win).astype(np.uint8), 8)
    if n < 2:
        return body
    ids = [int(v) for v in np.unique(lab[body > 0]) if v > 0]
    return np.isin(lab, ids).astype(np.uint8) if ids else body


def segment_disc(gray, thresh=None, min_area=800, low=None, plate=None, max_area=MAX_AREA_PX):
    """`segment.Segmentation` for the rotor disc, or None.

    Same shape as `segment.segment` returns, so `StereoPoseEstimator` can consume it
    unchanged. ``mask`` is the opened disc, ``contour`` its convex hull, ``ellipse`` the
    repo's direct fit to that hull.

    Hysteresis: the largest component above ``thresh`` is the seed, and every component
    above ``low`` that touches it is kept, so the shadowed half of the disc comes with the
    lit half (`DISC_LOW`). Roundish enclosed holes are then filled (`HOLE_MAX_ASPECT`),
    the thin parts are opened away to leave the disc body, and the blade tips that opening
    thinned away are given back by `_recover_blades` -- the mast and the wires are not.

    With ``plate`` (`build_plate`) the frame is background-subtracted first and the
    thresholds default to `DIFF_THRESH` / `DIFF_LOW`; without one, to `DISC_THRESH` /
    `DISC_LOW`. A component over ``max_area`` is not a rotor (`MAX_AREA_PX`) and returns
    None rather than a confident ellipse round a hand.
    """

    import cv2

    t0 = time.perf_counter()
    if thresh is None:
        thresh = DIFF_THRESH if plate is not None else DISC_THRESH
    if low is None:
        low = DIFF_LOW if plate is not None else DISC_LOW
    gray = _subtract(gray, plate)
    m = (gray > thresh).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        return None
    seed = lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if low is not None and low < thresh:
        _, lab_lo, _, _ = cv2.connectedComponentsWithStats((gray > low).astype(np.uint8), 8)
        ids = np.unique(lab_lo[seed])
        rob = np.isin(lab_lo, ids[ids > 0]).astype(np.uint8)
        rob = _fill_round_holes(rob)
    else:
        rob = seed.astype(np.uint8)
    disc = cv2.morphologyEx(rob, cv2.MORPH_OPEN, _kernel())
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(disc, 8)
    if n2 < 2:
        return None
    k = 1 + int(np.argmax(st2[1:, cv2.CC_STAT_AREA]))
    if not (min_area <= int(st2[k, cv2.CC_STAT_AREA]) <= max_area):
        return None
    body = (lab2 == k).astype(np.uint8)
    disc = _recover_blades(body, rob)
    hull, _ = segmod.silhouette_hull(disc)
    if hull is None or len(hull) < 6:
        # Recovery can only add pixels, and a mask it makes exactly convex gives
        # `silhouette_hull` too few contour points to fit. Fall back to the opened body,
        # which is what this returned before the recovery existed -- never lose the frame.
        disc = body
        hull, _ = segmod.silhouette_hull(disc)
    if hull is None or len(hull) < 6:
        return None
    area = int(disc.sum())
    fit = segmod.fit_ellipse(hull)
    if fit is None:
        return None
    ellipse, rms = fit
    return segmod.Segmentation(
        mask=disc, ellipse=ellipse, contour=np.asarray(hull, dtype=np.float64),
        area_px=float(area), n_points=len(hull), fit_rms_px=rms, threshold=thresh,
        t_ms=(time.perf_counter() - t0) * 1e3)


def find_mast(gray, seg, thresh=None, clear_px=5, plate=None, attach_px=50):
    """The mast as an image line: ``(centroid, unit_dir_up, elong, perp, area)`` or None.

    Taken from what the opening REMOVED: the robot blob at the disc threshold, minus the
    disc dilated a few px so its edge cannot leak in. What is left is the mast, the bead,
    the guy wires and any frozen blade tips. Because the disc is removed geometrically
    rather than by a brighter threshold, the mast can never merge into it -- which is how
    the first version of this lost it in every frame above 100 Hz, where the rod runs
    straight into the bright crescent at any level that separates it from the background.

    ``unit_dir_up`` points away from the disc. ``perp`` is 1 - |cos| against the disc's
    projected major axis: the rod reads near 1, a blade near 0, and that is the whole
    discrimination.
    """

    import cv2

    (cx, cy), (MA, ma), ang = seg.ellipse
    dax = np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))])
    if thresh is None:
        thresh = DIFF_THRESH if plate is not None else DISC_THRESH
    gray = _subtract(gray, plate)
    m = (gray > thresh).astype(np.uint8)
    # Every bright component that comes within `attach_px` of the disc, not only the
    # disc's own. The rod is often its own component at this level -- the hub between
    # it and the disc is dim -- and restricting to the disc's component lost the mast in
    # 30-50% of hold frames on drone 1 (2026-09-02) with the rod and bead plainly
    # visible. `attach_px` is the reach of that adjacency. 12 px served drone 1, whose
    # rod is bright and touches the disc at the threshold; drone 3's rod is DARK, only
    # its bead and wires pass, and the bead sits 30-50 px above the rim -- at 12 px the
    # rod was found in 0% of drone 3's rest frames, at 50 px in 100% with 1.3 deg of
    # direction scatter, and drone 1 was unchanged (100%, 2.5 deg) (2026-09-02). A wire
    # segment that far from the disc is still excluded by the distance and direction
    # gates below, which is what the 12 -> 50 sweep showed.
    n0, lab0, _, _ = cv2.connectedComponentsWithStats(m, 8)
    reach_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * attach_px + 1, 2 * attach_px + 1))
    ids = np.unique(lab0[cv2.dilate(seg.mask, reach_k) > 0])
    rob = np.isin(lab0, ids[ids > 0]).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * clear_px + 1, 2 * clear_px + 1))
    thin = cv2.bitwise_and(rob, cv2.bitwise_not(cv2.dilate(seg.mask, k)))
    # Direction is hub -> component centroid, NOT the component's own principal axis.
    # Camera B looks nearly along the mast, so in its view the rod is a stub a few px
    # long topped by the bead: a line fit there has nothing to fit (elongation 1.2-1.9,
    # measured), while the hub-to-bead vector is still exactly the mast's projection.
    # The bead sits on the rod at a fixed height, so this holds in every view, and it is
    # what makes the two cameras' lines triangulable (`mast_direction`) at all.
    #
    # The mast is the residual that REACHES HIGHEST above the hub. Not the one most
    # perpendicular to the disc axis, which was the first criterion: with the rotor
    # stopped there is no disc, the "ellipse" is the hub and a blade or two (96x38 px,
    # measured on a rest frame), and an axis read off that is a blade's, so "perpendicular
    # to it" scored the mast 0.08 and lost every rest frame -- exactly the frames the
    # datum is built from. Height needs no disc: a blade lies in the rotor plane, which
    # is near horizontal in both views, and cannot climb to the bead. ``perp`` is still
    # returned as a diagnostic.
    n, lab, st, _ = cv2.connectedComponentsWithStats(thin, 8)
    centre = np.array([cx, cy])
    best = None
    for k in range(1, n):
        a = int(st[k, cv2.CC_STAT_AREA])
        if a < MAST_MIN_PX or a > MAST_MAX_PX:
            continue
        ys, xs = np.nonzero(lab == k)
        pts = np.c_[xs, ys].astype(np.float64)
        c = pts.mean(0)
        r = c - centre
        dist = float(np.linalg.norm(r))
        # Too close is the disc's own edge leaking through; far is fine (the wire above
        # the bead is part of the same component and pulls the centroid up the mast).
        if dist < 0.2 * MA or dist > 3.0 * MA:
            continue
        d = r / dist
        if d[1] > -0.2:          # must point up in the image; blades and edge do not
            continue
        reach = float(cy - ys.min())          # px climbed above the hub
        _, sv, _ = np.linalg.svd(pts - c, full_matrices=False)
        elong = float(sv[0] / max(sv[1], 1e-9))
        perp = 1.0 - abs(float(d @ dax))
        score = reach * min(a, 800) / 800.0
        if best is None or score > best[0]:
            best = (score, c, d, elong, perp, a)
    if best is None:
        return None
    _, c, d, elong, perp, a = best
    return c, d, elong, perp, a


def mast_plane(line, cam):
    """Unit world normal of the plane through the lens centre and the mast's image line.

    The mast lies in this plane whatever the other camera says, so one view alone is a
    constraint: a normal estimated elsewhere can be projected into the plane
    (`control/tilt_report.lean_table`). ``line`` is ``(centroid, unit_dir)`` in pixels;
    the plane normal is ``K^T l`` for the homogeneous line ``l``, rotated to world.
    """

    c, d = line
    p1 = np.array([c[0], c[1], 1.0])
    p2 = np.array([c[0] + d[0], c[1] + d[1], 1.0])
    n_w = cam.R @ (cam.K.T @ np.cross(p1, p2))
    return n_w / np.linalg.norm(n_w)


def mast_direction(lines, cams, up=(0.0, 0.0, 1.0)):
    """World unit vector of the mast from its image line in two or more cameras.

    An image line and the lens centre span a plane; the mast lies in every such plane, so
    its direction is the intersection -- the cross product of two plane normals. Sign is
    set by ``up``. Each plane normal is ``K^T l`` rotated to world, ``l`` the homogeneous
    image line ``(a, b, c)`` through the centroid along the fitted direction.
    """

    normals = [mast_plane(line, cam) for line, cam in zip(lines, cams)]
    if len(normals) < 2:
        return None
    v = np.cross(normals[0], normals[1])
    nv = np.linalg.norm(v)
    if nv < 1e-9:
        return None
    v /= nv
    return -v if float(v @ np.asarray(up)) < 0 else v


#: Per-frame scatter of each sensor about its hold mean, drone 1 at 40-140 Hz with the
#: plate segmenter (2026-09-02): disc 2.3-3.4 deg median, mast 1.9-4.0. The mast gets
#: the smaller sigma because it does not strobe. `fused_axis` blends the two at these
#: weights (`stereo.blend_normals`): one axis, two readings, inverse-variance. Their
#: systematic 5 deg disagreement (same at every frequency) is NOT resolved by this and
#: is reported as `agree_deg`; whichever sensor carries it, the blend sits 2 deg from the
#: mast. `pose/theory.md` 20.4.
SIGMA_DISC_DEG, SIGMA_MAST_DEG = 3.0, 2.0
#: A frame whose disc and mast disagree by more than this is one of them being wrong --
#: frozen blades at 20 Hz, a blade outscoring the mast -- and is refused rather than
#: reported as a jittery axis. Holds run 5-7 deg at p90; 20 Hz runs 29.
AGREE_MAX_DEG = 15.0
#: Edge-on deweighting of the disc. When a view's ellipse is a sliver the opening erodes
#: it and the hub and mast are much of what is left. Measured on drone 1 (13587 frames
#: with a mast): disc-vs-mast 5.3-5.9 deg median with the thinner view's minor/major at
#: 0.25-0.50, 15.5 under 0.25. The disc's sigma is divided by ``q``: 1 above
#: `RATIO_EDGE_OK`, linear down to `RATIO_EDGE_MIN`, floor 0.05 (the mast carries the
#: frame). A view looking straight down the axis needs nothing here: `stereo.fuse`
#: weights each view by sin^2 of the tilt it sees, and a face-on ellipse's orientation is
#: meaningless while its normal is still the axis, which is that camera's own.
RATIO_EDGE_MIN, RATIO_EDGE_OK = 0.12, 0.25


def disc_quality(ratio_min):
    """``q`` in (0.05, 1]: the disc's weight against the mast this frame."""

    if ratio_min is None or not np.isfinite(ratio_min):
        return 1.0
    q = (ratio_min - RATIO_EDGE_MIN) / (RATIO_EDGE_OK - RATIO_EDGE_MIN)
    return float(min(1.0, max(0.05, q)))


def _angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))


def fused_axis(n, mast=None, planes=(), ratio_min=None, ref=None):
    """One rotor axis from the disc normal and whatever the rod gave this frame.

    ``(axis, code)`` with code 1 = disc blended with the triangulated mast, 2 = disc
    projected into one view's mast plane, 0 = disc alone; or ``None`` when the disc and
    the rod disagree by more than `AGREE_MAX_DEG` on a disc worth believing. ``ref`` fixes
    the sign (the mast when there is one, else ``ref``); the ellipse alone cannot tell
    rotor-up from rotor-down. The same code serves the offline report and the live loop,
    which is the point of it being here. `pose/theory.md` 20.4.
    """

    n = np.asarray(n, float)
    n = n / np.linalg.norm(n)
    q = disc_quality(ratio_min)
    m = None if mast is None or not np.all(np.isfinite(mast)) else np.asarray(mast, float)
    sign_ref = m if m is not None else ref
    if sign_ref is not None and float(n @ sign_ref) < 0:
        n = -n
    if m is not None:
        m = m / np.linalg.norm(m)
        if q > 0.5 and _angle_deg(n, m) > AGREE_MAX_DEG:
            return None
        out, _ = stereo.blend_normals(n, SIGMA_DISC_DEG / q, m, SIGMA_MAST_DEG, reference=m)
        return out, 1
    planes = [p for p in planes if np.all(np.isfinite(p))]
    if len(planes) == 1:
        p = np.asarray(planes[0], float)
        n_in = n - float(n @ p) * p
        if np.linalg.norm(n_in) < 1e-9 or (q > 0.5 and _angle_deg(n, n_in) > AGREE_MAX_DEG):
            return None
        return n_in / np.linalg.norm(n_in), 2
    return n, 0


def triangulate_centre(centres_px, cams):
    """World point nearest both centre rays, and the rays' gap in mm.

    Each view's (distorted) pixel is undistorted through its own ``K``/``dist`` into a
    ray from that camera's centre; the answer is the midpoint of the closest approach.
    No radius enters, which is what makes it right for a rotor whose rim radius the
    pipeline does not know (header). The gap is an honest stereo residual.
    """

    import cv2

    if len(centres_px) < 2:
        return None, float("nan")
    origins, dirs = [], []
    for (u, v), cam in zip(centres_px, cams):
        pt = np.array([[[float(u), float(v)]]], np.float64)
        dist = cam.dist if cam.dist is not None else np.zeros(5)
        xy = cv2.undistortPoints(pt, cam.K, dist).reshape(2)
        d = cam.R @ np.array([xy[0], xy[1], 1.0])
        origins.append(cam.position)
        dirs.append(d / np.linalg.norm(d))
    o1, o2, d1, d2 = origins[0], origins[1], dirs[0], dirs[1]
    w = o1 - o2
    a, b, c = float(d1 @ d1), float(d1 @ d2), float(d2 @ d2)
    d, e = float(d1 @ w), float(d2 @ w)
    den = a * c - b * b
    if den < 1e-12:
        return None, float("nan")
    s = (b * e - c * d) / den
    t = (a * e - b * d) / den
    p1, p2 = o1 + s * d1, o2 + t * d2
    return 0.5 * (p1 + p2), float(np.linalg.norm(p1 - p2))


def live_estimator(rig_path=None):
    """``DiscStereoEstimator`` on the measured rig with a `LowPlate` per camera, for
    `live_viz.stereo_frames(est=...)`. Offline runs use `plates_for_flight` instead."""

    from controller.calib import rig as rigmod

    rig = rigmod.StereoRig.load(rig_path or rigmod.DEFAULT_PATH)
    return DiscStereoEstimator(rig, backgrounds={c.name: LowPlate() for c in rig.cameras})


def _running_plate_base():
    from controller.pose import background as bgmod
    return bgmod.RunningPlate


class LowPlate(_running_plate_base()):
    """`background.RunningPlate` walking to a low percentile instead of the median.

    The running median "cannot see a robot that never moves" (its own docstring), and on
    the mast rig the robot never moves: a median plate absorbs the spinning disc, exactly
    as the whole-take median did (`PLATE_FRAMES`). A quantile step -- up by ``pct`` of a
    step when the frame is above the plate, down by ``1 - pct`` when below -- converges
    to the ``pct``-th percentile, which between frozen blades and at the disc's dark
    edge is the backdrop. Same O(1) memory and cost.
    """

    def __init__(self, step=1.0, pct=PLATE_PCT / 100.0, warmup=None):
        from controller.pose import background as bgmod
        super().__init__(step=step, warmup=bgmod.WARMUP_FRAMES if warmup is None else warmup)
        self.pct = float(pct)

    def update(self, gray):
        f = gray.astype(np.float32)
        if self.bg is None or self.bg.shape != f.shape:
            self.bg = f.copy()
        else:
            self.bg += self.step * np.where(f > self.bg, self.pct, -(1.0 - self.pct))
        self.n += 1
        return self.bg.astype(np.uint8) if self.ready else None


class DiscStereoEstimator(stereo.StereoPoseEstimator):
    """`StereoPoseEstimator` with the rim segmenter swapped for the disc one.

    Constructed `never_reject=True` and `do_refine=False`: the reporting gates were
    tuned on rims and reject every disc frame (`max_fit_rms_rel` is the one that fires),
    and the joint refinement fits against a radius that is wrong here. Quality rides on
    each row as `fit_rms_px` and `discrepancy_mm`, as the base class's own docstring
    prescribes for offline use.
    """

    def __init__(self, rig, **kw):
        kw.setdefault("never_reject", True)
        kw.setdefault("do_refine", False)
        kw.setdefault("direct", False)
        kw.setdefault("backgrounds", None)
        super().__init__(rig, **kw)
        self.last_mast = {}
        # Always re-arbitrate the branch pair against the prior. The base class only
        # does so when the two views' centres disagree by more than `_suspect_mm`, and
        # the mirrored branch pair agrees with itself (`stereo.match` docstring) -- and
        # here the centres are scaled by a rim radius this rotor does not have, so their
        # discrepancy says nothing anyway. With the mast as the prior (`_window_normal`)
        # the arbitration is a physical constraint, not a smoothing.
        self._suspect_mm = float("-inf")

    def mast_world(self):
        """The mast's world direction from this frame's two views, or None.

        The rotor axis IS the mast, so this is the estimator's prior for the two-fold
        branch ambiguity: the mirrored pair sits ~84 deg off it and loses at once,
        where the sliding-window median (`stereo._window_normal`) needs five frames and
        follows a wrong branch once it has settled on one. `pose/theory.md` 20.4.
        """

        lines = []
        for cam in self.rig.cameras:
            m = self.last_mast.get(cam.name)
            if m is None:
                return None
            lines.append((m[0], m[1]))
        return mast_direction(lines, self.rig.cameras, up=MAST_UP)

    def _window_normal(self, now):
        m = self.mast_world()
        return m if m is not None else super()._window_normal(now)

    def update(self, frames, **kw):
        """The base solve, then the centre re-done by triangulation (`triangulate_centre`).

        The base class places each view's centre at a depth set by `radius_mm`, which is
        the rim's and wrong for this rotor (header), so `xyz_mm` was on the right rays at
        the wrong depth. Both views' ellipse centres, undistorted, fix the point without a
        radius; the rays' gap replaces `discrepancy_mm`, and the disc's true radius follows
        from the major axis at the now-known depth (``extra["disc_radius_mm"]``).
        """

        pose = super().update(frames, **kw)
        if pose is None:
            return pose
        used = pose.extra.get("views_used", [])
        segs = pose.per_view
        if len(used) < 2 or any(segs[i] is None for i in used):
            return pose
        cams = [self.rig.cameras[i] for i in used]
        centres = [segs[i].ellipse[0] for i in used]
        centre, gap = triangulate_centre(centres, cams)
        if centre is None:
            return pose
        _, normal_world = pose.extra["world"]
        pose.xyz_mm, _ = self.zero.apply(centre, normal_world)
        pose.discrepancy_mm = gap
        pose.extra["world"] = (centre, normal_world)
        # radius from the reference view: half the major axis, at the ray's depth
        cam, seg = cams[0], segs[used[0]]
        depth = float((cam.R.T @ (centre - cam.position))[2])
        pose.extra["disc_radius_mm"] = 0.5 * seg.ellipse[1][0] * depth / float(cam.K[0, 0])
        return pose

    def _view_candidates(self, frame, cam):
        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        plate = self.backgrounds.get(cam.name)       # `plates_for_flight`, keyed by tag
        if hasattr(plate, "update"):                 # a live `LowPlate`: fold this frame in
            plate = plate.update(gray)
        seg = segment_disc(gray, plate=plate)
        if seg is None:
            self.last_mast[cam.name] = None
            return None, [], None, None
        self.last_mast[cam.name] = find_mast(gray, seg, plate=plate)
        ellipse = seg.ellipse
        if self.undistort and cam.dist is not None and np.any(cam.dist):
            try:
                ellipse = segmod.undistort_ellipse(ellipse, cam.K, cam.dist)
            except Exception:  # noqa: BLE001
                ellipse = seg.ellipse
        ellipse = self.tilt_cal.apply(ellipse)
        return (seg,
                conic.backproject_ellipse(ellipse, cam.K, self.radius_mm,
                                          verify_tol=self.verify_tol),
                ellipse, None)


def mast_pass(flight_dir, out_csv, rig=None, up=(0.0, -1.0, 0.0), min_perp=-1.0,
              progress=True, plates=None):
    """Every frame's mast line in each view, and the triangulated world direction.

    Written as its own pass rather than folded into the stereo estimator's CSV because
    `live_viz.from_recording` owns that file's columns. Joined to it by ``frame``.

    ``up`` defaults to world -y: the rig's world frame is camera A (`stereo_rig.json`),
    OpenCV axes, +y down, so -y is the nearest thing to vertical it has. Only the SIGN
    of the mast comes from it; the datum proper is set later from the rest frames.

    ``min_perp`` is kept as an optional gate but defaults off: `find_mast` no longer
    scores on it, and at rest -- the frames the datum comes from -- there is no disc for
    it to be measured against. The gate that is applied is `find_mast`'s own: the
    direction must climb in the image.
    """

    import csv

    import cv2

    from controller.camera import record

    flight_dir = Path(flight_dir)
    if rig is None:
        from controller.calib.rig import StereoRig
        rig = StereoRig.load()
    stamps, _ = record.read_index(flight_dir)
    videos = sorted(flight_dir.glob("*/*.mp4"))
    caps = [cv2.VideoCapture(str(v)) for v in videos]
    tags = [v.parent.name for v in videos]
    if plates is None:
        plates = plates_for_flight(flight_dir, tags)
    cams, scale = None, None
    n, out = 0, []
    cols = ["frame", "t"]
    for tg in tags:
        cols += [f"{tg}_cx", f"{tg}_cy", f"{tg}_dx", f"{tg}_dy", f"{tg}_perp", f"{tg}_area",
                 f"{tg}_disc_cx", f"{tg}_disc_cy", f"{tg}_disc_major", f"{tg}_disc_minor",
                 f"{tg}_disc_deg", f"{tg}_disc_rms", f"{tg}_px", f"{tg}_py", f"{tg}_pz"]
    cols += ["mast_x", "mast_y", "mast_z"]
    fh = open(out_csv, "w", newline="")
    w = csv.writer(fh)
    w.writerow(cols)
    while True:
        got = [c.read() for c in caps]
        if not all(ok for ok, _ in got) or n >= len(stamps):
            break
        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f for _, f in got]
        if cams is None:
            # The rig is calibrated at its capture resolution; these frames may be a
            # rescaled mode. `Camera.scaled` does what the estimator does internally.
            scale = grays[0].shape[1] / (2.0 * rig.cameras[0].K[0, 2])
            cams = [c.scaled(scale) for c in rig.cameras]
            if progress:
                print(f"mast_pass: intrinsics x{scale:.3f}")
        row = [n, float(np.mean(stamps[n]))]
        lines, ok = [], True
        for i, (tg, g) in enumerate(zip(tags, grays)):
            seg = segment_disc(g, plate=plates.get(tg))
            m = find_mast(g, seg, plate=plates.get(tg)) if seg is not None else None
            nan = float("nan")
            if seg is None:
                row += [nan] * 15
                ok = False
                continue
            (cx, cy), (MA, ma), ang = seg.ellipse
            disc_cols = [cx, cy, MA, ma, ang, seg.fit_rms_px]
            if m is None:
                # the disc still goes out: the report deweights it by its axis ratio
                row += [nan] * 6 + disc_cols + [nan] * 3
                ok = False
                continue
            c, d, elong, perp, a = m
            row += [c[0], c[1], d[0], d[1], perp, a] + disc_cols + list(mast_plane((c, d), cams[i]))
            lines.append((c, d))
            ok = ok and perp >= min_perp
        v = mast_direction(lines, cams, up=up) if ok and len(lines) == 2 else None
        row += list(v) if v is not None else [float("nan")] * 3
        w.writerow([f"{x:.4f}" if isinstance(x, float) else x for x in row])
        n += 1
        if n % 500 == 0:
            fh.flush()
        if progress and n % 5000 == 0:
            print(f"  {n} frames", flush=True)
    fh.close()
    for c in caps:
        c.release()
    if progress:
        print(f"mast_pass: {n} frames -> {out_csv}")
    return Path(out_csv)


def _self_check():
    """Synthetic disc + mast: the disc must segment clean of the mast, and the mast must
    be found and read as perpendicular to the disc, not as one of the blades."""

    import cv2

    img = np.zeros((400, 640), np.uint8)
    cv2.ellipse(img, (320, 230), (70, 35), 20, 0, 360, 130, -1)      # dim blurred disc
    # shadowed half: under DISC_THRESH, above DISC_LOW. Must still be part of the disc.
    half = np.zeros_like(img)
    cv2.ellipse(half, (320, 230), (70, 35), 20, 90, 270, 1, -1)
    img[half > 0] = 80
    cv2.ellipse(img, (330, 225), (30, 12), 20, 0, 360, 230, -1)      # bright crescent
    # mast: thin bright rod, perpendicular to the disc's major axis (20 deg -> 110 deg)
    d = np.array([math.cos(math.radians(110)), math.sin(math.radians(110))])
    p0 = (320, 230)
    p1 = tuple(np.round(np.array(p0) - d * 120).astype(int))
    cv2.line(img, p0, p1, 235, 5)
    # a frozen blade: same brightness, lying ALONG the disc axis
    b = np.array([math.cos(math.radians(20)), math.sin(math.radians(20))])
    cv2.line(img, tuple(np.round(np.array(p0) + b * 40).astype(int)),
             tuple(np.round(np.array(p0) + b * 95).astype(int)), 235, 5)
    cv2.line(img, (300, 40), (340, 5), 200, 1)                        # a guy wire
    cv2.rectangle(img, (0, 300), (640, 400), 90, -1)                   # foam blocks

    seg = segment_disc(img)
    assert seg is not None
    (cx, cy), (MA, ma), ang = seg.ellipse
    assert abs(cx - 320) < 6 and abs(cy - 230) < 6, (cx, cy)
    assert abs(((ang - 20) + 90) % 180 - 90) < 6, ang
    assert 0.35 < ma / MA < 0.65, (MA, ma)        # 35/70 = 0.5; mast must not inflate it
    m = find_mast(img, seg)
    assert m is not None, "mast not found"
    c, dm, elong, perp, a = m
    got = math.degrees(math.atan2(-dm[1], dm[0]))
    assert abs(((got - 70) + 90) % 180 - 90) < 6, f"mast read as {got:.1f}, blade?"
    assert perp > 0.9, perp
    # A blade tip thinner than OPEN_PX is erased by the opening and given back by
    # `_recover_blades`; the mast, equally thin, must stay out. Reported 2026-09-04 as
    # "the white propellers are not all detected" -- see `RECOVER_SCALE`.
    tipped = img.copy()
    u = np.array([math.cos(math.radians(20)), math.sin(math.radians(20))])
    ctr = np.array([320.0, 230.0])
    cv2.line(tipped, tuple(np.round(ctr - u * 58).astype(int)),
             tuple(np.round(ctr - u * 80).astype(int)), 150, 7)      # 7 px < OPEN_PX
    seg_t = segment_disc(tipped)
    assert seg_t is not None and seg_t.area_px > seg.area_px + 100, \
        f"thin blade tip erased by the opening ({seg_t.area_px} vs {seg.area_px})"
    (_, (MA_t, ma_t), _) = seg_t.ellipse
    assert 0.35 < ma_t / MA_t < 0.65, (MA_t, ma_t)   # the mast must not have come back too
    assert segment_disc(np.zeros((400, 640), np.uint8)) is None
    # A static bright block bigger than the disc wins the plain threshold; with it in the
    # plate the disc wins again. And a hand-sized blob is refused, not fitted.
    blk = img.copy()
    cv2.rectangle(blk, (20, 20), (180, 150), 180, -1)
    assert abs(segment_disc(blk).ellipse[0][0] - 100) < 10, "block should win without a plate"
    plate = np.zeros_like(img)
    cv2.rectangle(plate, (20, 20), (180, 150), 180, -1)
    plate[img > 0] = 0
    (cx, cy), _, _ = segment_disc(blk, plate=plate).ellipse
    assert abs(cx - 320) < 6 and abs(cy - 230) < 6, (cx, cy)
    hand = np.full((400, 640), 200, np.uint8)
    hand[:, 400:] = 0
    assert segment_disc(hand) is None, "a 160000 px blob is not a rotor"
    # Two cameras, a known point: triangulation must land on it whatever radius anyone
    # believes, and the ray gap must be ~0 for consistent pixels.
    from controller.calib.rig import Camera
    K = np.array([[500.0, 0, 320.0], [0, 500.0, 200.0], [0, 0, 1.0]])
    T1 = np.eye(4)
    R2 = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])   # looks along -x
    T2 = np.eye(4); T2[:3, :3] = R2; T2[:3, 3] = [200.0, 0.0, 200.0]
    cams = [Camera(K=K, dist=None, T_world_cam=T1, name="A"),
            Camera(K=K, dist=None, T_world_cam=T2, name="B")]
    X = np.array([10.0, -5.0, 180.0])
    px = []
    for cam in cams:
        xc = cam.R.T @ (X - cam.position)
        uv = K @ (xc / xc[2])
        px.append((uv[0], uv[1]))
    Xt, gap = triangulate_centre(px, cams)
    assert np.linalg.norm(Xt - X) < 1e-6 and gap < 1e-6, (Xt, gap)
    # fused_axis: blends toward the mast, projects into one plane, refuses a disagreement
    v = np.array([0.1, -0.95, 0.2]); v /= np.linalg.norm(v)
    n10 = v * np.cos(np.radians(10)) + np.cross(v, [1, 0, 0]) / np.linalg.norm(np.cross(v, [1, 0, 0])) * np.sin(np.radians(10))
    ax, code = fused_axis(-n10, mast=v)                    # wrong sign, 10 deg off
    assert code == 1 and _angle_deg(ax, v) < 4.0, (code, _angle_deg(ax, v))
    pl = np.cross(v, [1.0, 0, 0]); pl /= np.linalg.norm(pl)
    ax, code = fused_axis(v * np.cos(np.radians(10)) + pl * np.sin(np.radians(10)), planes=[pl], ref=v)
    assert code == 2 and _angle_deg(ax, v) < 1e-6
    assert fused_axis(np.cross(v, [0, 0, 1.0]), mast=v) is None, "30 deg off must be refused"
    assert fused_axis(v, ratio_min=0.1, mast=np.cross(v, [0, 0, 1.0])) is not None, "edge-on disc defers to mast"
    # LowPlate walks to a low percentile: a pixel that is 30 for 90% of frames and 200
    # for 10% must settle near 30, where the median-walking plate would too but a mean
    # would not; a pixel at 200 for 70% of frames must still read ~30 (the robot case).
    lp = LowPlate(step=2.0, pct=0.1, warmup=1)
    rng = np.random.default_rng(0)
    for _ in range(400):
        f = np.full((4, 4), 30, np.uint8)
        f[0, 0] = 200 if rng.random() < 0.7 else 30
        out = lp.update(f)
    assert out[0, 0] < 60 and abs(int(out[1, 1]) - 30) <= 2, out
    print("disc_pose: self-check passed (disc clean of mast, mast chosen over blade, plate, "
          "area cap, triangulation, fused axis, low plate)")


if __name__ == "__main__":
    _self_check()
