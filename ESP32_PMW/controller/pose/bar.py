#!/usr/bin/env python3
"""Segment the straight bright bar -- the tether wire and the rod the robot rides.

`segment.py` fits the robot's rim *ellipse*; this fits the *line* beside it. They share
the bright-mask step (`segment.threshold_mask`, which is where the rig's two appearances
are handled) and nothing else, because a conic fitted to a straight bar is degenerate.

WHAT IS IN FRAME, ON 2026-09-01_201024
--------------------------------------
Top to bottom: a hairpin wire loop entering from the top edge, a white bead where it
terminates, and a rod continuing from the bead down to the robot. Measured on that take,
the dominant axis sits 18-22 deg off image vertical in camera A.

**The wire and the rod are two bars, not one.** They are collinear only when the hairpin
hangs closed. On frame 0, with the loop splayed, the wire above the bead runs
(298,153)-(343,16) and the rod below it runs (259,246)-(305,115): parallel to within
1.1 deg and 39 px apart. A single "longest line" answer silently returns whichever
happened to win, so `find_bars` returns a ranked list and `find_bar` takes the top one.
Use `below=` to ask for the rod specifically -- see the bottom of this docstring.

WHY HOUGH, AND NOT THE ELONGATED-BLOB AXIS
------------------------------------------
The obvious method is to threshold, take the most elongated connected component and use
its principal axis. It fails here for a structural reason, not a tuning one: at the
brightness that keeps the wire, the wire, the bead, the rod and the robot are frequently
ONE component (measured: blob areas 816-1114 px spanning y=0 to the bead, with the rod
merging into the robot's blob at 2340-2487 px). A principal axis over that is dragged by
the robot's blur, which is the widest and brightest thing in the mask and is not part of
the bar. Hough votes only for collinear evidence, so the robot -- bright but not straight
-- contributes almost nothing.

Hough's own axis resolution is quantised (`THETA_STEP`), so the winner is refitted by
total least squares (`cv2.fitLine`) over the pixels that actually support it. That is
what makes the reported angle continuous rather than 0.5 deg granular.

WHAT THIS IS AND IS NOT
-----------------------
This is a 2D image measurement: an axis in pixels, per camera, per frame. It is NOT a
world-frame vertical and must not be read as one. Turning it into an angle in millimetres
needs the rig, and the same 1/sin(theta) conditioning `attitude.py` documents applies --
`control/theory.md` and `attitude.py` are where the tilt argument lives, not here.

    uv run python controller/pose/bar.py                          # self-check
    uv run python controller/pose/bar.py results/flights/<take>   # track + overlay
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
# Pipeline layering: a stage sees only the stages before it. pose is stage 3 of 4.

from controller.pose import segment

#: Hough angular step. 0.5 deg: finer buys nothing once `refit` runs, coarser starts
#: merging the hairpin's two legs, which sit 8-9 deg apart when the loop is splayed.
THETA_STEP = math.pi / 360
#: A bar must be at least this long to be one. The rod below the bead measures 130-140 px
#: on this take and the hairpin legs 120-180, so 50 keeps both with room to spare while
#: rejecting the background's edge fragments.
MIN_LEN_PX = 50
#: Votes for a Hough segment. Low, because the winner is chosen by pixel support below
#: rather than by vote count -- this only has to avoid missing the bar.
HOUGH_VOTES = 40
#: Bridged over the bead, which interrupts the wire for ~10 px without bending it.
MAX_GAP_PX = 12
#: Pixels within this distance of the axis count as supporting it. The wire is 3-4 px
#: wide and the rod ~12, so 4 covers the rod's near edge without reaching the hairpin's
#: other leg (39 px away at its widest).
BAND_PX = 4.0
#: Beyond this from image vertical it is not the bar. The robot's blur and the
#: background's edges land at 60-87 deg, so this is a wide margin, not a tight gate.
MAX_TILT_DEG = 45.0
#: Two Hough segments are the same bar within both of these. 3 deg is well under the
#: 8-9 deg between the hairpin's legs; 6 px is over the wire's own width and under the
#: 39 px between wire and rod.
MERGE_ANGLE_DEG = 3.0
MERGE_OFFSET_PX = 6.0
#: Radius of the disk that decides thin from fat. A bar is THIN, and that -- not
#: brightness, and not straightness -- is what separates it from the robot where the two
#: overlap, which they do on every frame because the rod ends at the rotor.
#: Measured on 2026-09-01_201024: wire 3-4 px wide, rod ~12, robot blur 40-60 px across.
#: A disk of radius 8 (17 px) passes through neither wire nor rod and sits inside the
#: blur, so opening with it leaves exactly the fat structure behind.
#: Without this gate a rod ending inside the blur fits 1.4 deg off and overruns its true
#: end by 30 px -- both measured, in `_self_check`.
FAT_RADIUS_PX = 8


@dataclass
class Bar:
    """One straight bar: its axis, its extent, and how much evidence stands behind it.

    ``p0`` is always the upper endpoint (smaller y), so ``tilt_deg`` has a stable sign
    without the caller having to normalise the direction it got back.
    """

    p0: tuple[float, float]
    p1: tuple[float, float]
    tilt_deg: float       # from image vertical; + leans right going DOWN the frame
    support_px: int       # mask pixels within BAND_PX of the axis, over the bar's extent
    length_px: float

    @property
    def midpoint(self):
        return (0.5 * (self.p0[0] + self.p1[0]), 0.5 * (self.p0[1] + self.p1[1]))

    def x_at(self, y):
        """Where the axis crosses row ``y``. Vertical-ish by construction, so this is safe."""

        (x0, y0), (x1, y1) = self.p0, self.p1
        if y1 == y0:
            return 0.5 * (x0 + x1)
        return x0 + (x1 - x0) * (y - y0) / (y1 - y0)


def _tilt_deg(dx, dy):
    """Angle from image vertical, in (-90, 90]. Direction-agnostic."""

    a = math.degrees(math.atan2(dx, dy))
    return (a + 90.0) % 180.0 - 90.0


def _line_params(x0, y0, x1, y1):
    """``(theta, rho)`` of the infinite line, for comparing two segments."""

    theta = math.atan2(y1 - y0, x1 - x0)
    # Direction is arbitrary, so fold to a half-turn before comparing.
    theta %= math.pi
    rho = x0 * math.sin(theta) - y0 * math.cos(theta)
    return theta, rho


def thin_points(mask, radius=FAT_RADIUS_PX):
    """Mask pixels belonging to a thin structure, as an ``(N, 2)`` array of ``(x, y)``.

    The gate that keeps the robot out of the bar's fit. See `FAT_RADIUS_PX`.

    Opening, not `cv2.distanceTransform`. The distance transform is the obvious way to
    ask "how thick is the mask here", and it fails on exactly this shape: a filled blob's
    distance is large in the middle but goes to zero at its rim, so the robot's whole
    OUTLINE passes a thinness threshold and the tips of the blur still drag the fit
    (measured: 1.4 deg off, unchanged by the gate). Opening asks the structural question
    instead -- "does the disk fit inside here at all" -- and answers it for the blob as a
    whole. Dilating the result removes the fringe the opening erodes off the blob's tips,
    which is where the surviving pull came from.
    """

    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    fat = cv2.dilate(cv2.morphologyEx(mask, cv2.MORPH_OPEN, disk), disk)
    ys, xs = np.nonzero(cv2.bitwise_and(mask, cv2.bitwise_not(fat)))
    return np.column_stack([xs, ys]).astype(np.float64)


def _support(mask_pts, p0, p1, band=BAND_PX):
    """``(count, projections)`` for mask points within ``band`` of the segment's axis.

    Extent comes from the supporting pixels rather than from Hough's endpoints, which
    stop wherever the vote ran out -- routinely short of the bar's real end.
    """

    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    n = math.hypot(dx, dy)
    if n == 0:
        return 0, None
    ux, uy = dx / n, dy / n
    rx = mask_pts[:, 0] - x0
    ry = mask_pts[:, 1] - y0
    perp = np.abs(rx * uy - ry * ux)
    keep = perp <= band
    if not keep.any():
        return 0, None
    return int(keep.sum()), (rx[keep] * ux + ry[keep] * uy, keep)


def _refit(mask_pts, keep, p0, p1):
    """Total-least-squares axis through the supporting pixels, as endpoints.

    `cv2.fitLine` with `DIST_L2` is plain TLS -- the right estimator here because the
    supporting set has already been restricted to a 4 px band, so there are no outliers
    left for a robust norm to earn its cost against.
    """

    pts = mask_pts[keep].astype(np.float32)
    if len(pts) < 2:
        return p0, p1
    vx, vy, cx, cy = (float(v) for v in
                      cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel())
    t = (pts[:, 0] - cx) * vx + (pts[:, 1] - cy) * vy
    a, b = float(t.min()), float(t.max())
    q0 = (cx + vx * a, cy + vy * a)
    q1 = (cx + vx * b, cy + vy * b)
    return (q0, q1) if q0[1] <= q1[1] else (q1, q0)


def find_bars(frame, mask=None, thresh=None, max_tilt_deg=MAX_TILT_DEG,
              min_len_px=MIN_LEN_PX, band=BAND_PX, appearance=None, background=None):
    """Every straight near-vertical bar in the frame, best-supported first.

    ``mask`` overrides the bright mask, for a caller that already has one or wants a
    different level. Otherwise `segment.threshold_mask` builds it, which is what keeps
    the rig's bright/dark appearance handling in one place.
    """

    if mask is None:
        mask = segment.threshold_mask(frame, thresh=thresh, appearance=appearance,
                                      background=background)
    if mask is None or not mask.any():
        return []

    segs = cv2.HoughLinesP(mask, 1, THETA_STEP, threshold=HOUGH_VOTES,
                           minLineLength=int(min_len_px), maxLineGap=MAX_GAP_PX)
    if segs is None:
        return []

    # Hough runs on the full mask -- the bar's own pixels are what vote -- but the fit
    # and the extent use only the thin ones, so the robot cannot drag either.
    mask_pts = thin_points(mask)
    if len(mask_pts) < 2:
        return []

    # Collapse the near-duplicates Hough returns for one bar (both edges of the wire,
    # and several partial spans of it) before scoring, so support is not counted twice
    # for the same structure.
    clusters = []
    for x0, y0, x1, y1 in segs.reshape(-1, 4).astype(float):
        if abs(_tilt_deg(x1 - x0, y1 - y0)) > max_tilt_deg:
            continue
        theta, rho = _line_params(x0, y0, x1, y1)
        for c in clusters:
            dth = abs(math.degrees(math.atan2(math.sin(theta - c["theta"]),
                                              math.cos(theta - c["theta"]))))
            if min(dth, 180.0 - dth) <= MERGE_ANGLE_DEG and \
                    abs(rho - c["rho"]) <= MERGE_OFFSET_PX:
                if math.hypot(x1 - x0, y1 - y0) > c["len"]:
                    c.update(theta=theta, rho=rho, p0=(x0, y0), p1=(x1, y1),
                             len=math.hypot(x1 - x0, y1 - y0))
                break
        else:
            clusters.append(dict(theta=theta, rho=rho, p0=(x0, y0), p1=(x1, y1),
                                 len=math.hypot(x1 - x0, y1 - y0)))

    bars = []
    for c in clusters:
        n, got = _support(mask_pts, c["p0"], c["p1"], band)
        if not n or got is None:
            continue
        q0, q1 = _refit(mask_pts, got[1], c["p0"], c["p1"])
        length = math.hypot(q1[0] - q0[0], q1[1] - q0[1])
        if length < min_len_px:
            continue
        bars.append(Bar(p0=q0, p1=q1,
                        tilt_deg=_tilt_deg(q1[0] - q0[0], q1[1] - q0[1]),
                        support_px=n, length_px=length))
    bars.sort(key=lambda b: b.support_px, reverse=True)
    return bars


def find_bar(frame, below=None, **kw):
    """The single best bar, or None.

    ``below`` keeps only bars whose lower endpoint is under that image row, which is how
    the rod is asked for rather than the tether above the bead: pass the bead's y. With
    the hairpin closed the two coincide and either answer is the same line; with it
    splayed they are 39 px apart and only this distinguishes them.
    """

    bars = find_bars(frame, **kw)
    if below is not None:
        bars = [b for b in bars if b.p1[1] > below]
    return bars[0] if bars else None


def draw(frame, bars, colour=(0, 255, 0), best=(0, 220, 255)):
    """Overlay: the best bar highlighted, the rest in ``colour``."""

    out = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
    for i, b in enumerate(reversed(list(bars))):
        c = best if i == len(bars) - 1 else colour
        cv2.line(out, tuple(int(round(v)) for v in b.p0),
                 tuple(int(round(v)) for v in b.p1), c, 1, cv2.LINE_AA)
    if bars:
        b = bars[0]
        cv2.putText(out, f"{b.tilt_deg:+.1f} deg  {b.length_px:.0f}px  n={b.support_px}",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, best, 1, cv2.LINE_AA)
    return out


def track(flight_dir, tag="A", csv_out=None, overlay=None, max_frames=None, **kw):
    """Run the segmentation over one camera of a flight. Returns a dict of arrays.

    Frames where no bar is found are kept as nan rather than dropped, so the row index
    stays the video's frame index and `control/sync.py` can line a row up with the
    serial log without a second mapping.
    """

    import csv as _csv

    flight_dir = Path(flight_dir)
    video = flight_dir / tag / f"{tag}.mp4"
    if not video.exists():
        raise FileNotFoundError(video)
    cap = cv2.VideoCapture(str(video))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rows, found = [], 0
    try:
        i = 0
        while max_frames is None or i < max_frames:
            ok, f = cap.read()
            if not ok:
                break
            g = f[:, :, 0] if f.ndim == 3 else f
            bars = find_bars(g, **kw)
            if bars:
                b = bars[0]
                found += 1
                rows.append([i, f"{b.tilt_deg:.4f}", f"{b.p0[0]:.2f}", f"{b.p0[1]:.2f}",
                             f"{b.p1[0]:.2f}", f"{b.p1[1]:.2f}", f"{b.length_px:.2f}",
                             b.support_px, len(bars)])
            else:
                rows.append([i, "", "", "", "", "", "", "", 0])
            i += 1
    finally:
        cap.release()

    if csv_out:
        with open(csv_out, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["frame", "tilt_deg", "x0", "y0", "x1", "y1",
                        "length_px", "support_px", "n_bars"])
            w.writerows(rows)

    if overlay:
        _write_overlay(video, rows, overlay, max_frames, **kw)

    tilt = np.array([float(r[1]) if r[1] else np.nan for r in rows])
    print(f"{video}: {found}/{len(rows)} frames with a bar "
          f"({100.0 * found / max(len(rows), 1):.1f}%)"
          + (f", of {n_total} in the file" if len(rows) != n_total else ""))
    if found:
        print(f"  tilt {np.nanmedian(tilt):+.2f} deg median, "
              f"{np.nanpercentile(tilt, 5):+.2f} to {np.nanpercentile(tilt, 95):+.2f} "
              f"(5-95%), sd {np.nanstd(tilt):.2f}")
    return {"frame": np.array([r[0] for r in rows]), "tilt_deg": tilt,
            "support_px": np.array([r[7] if r[7] != "" else 0 for r in rows], float)}


def _write_overlay(video, rows, out_path, max_frames, **kw):
    cap = cv2.VideoCapture(str(video))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"avc1"), 30.0, (w, h), True)
    try:
        for i, row in enumerate(rows):
            ok, f = cap.read()
            if not ok:
                break
            g = f[:, :, 0] if f.ndim == 3 else f
            vw.write(draw(g, find_bars(g, **kw)))
    finally:
        cap.release()
        vw.release()
    print(f"  overlay -> {out_path}")


def _self_check():
    """A synthetic bar at a known angle must come back at that angle.

    Two bars, because one is the case that hides the failure this module exists to avoid:
    a single line is found by almost any method, and it is the second parallel structure
    (the rod beside the tether) that separates a real fit from a lucky one.
    """

    img = np.zeros((400, 640), np.uint8)
    tilt = -20.0
    # Unit vector down the bar, and the perpendicular the second bar is offset along.
    # Offsetting in x at a different y instead would leave the two COLLINEAR, which is
    # the degenerate case this check is meant to avoid -- and it looks identical on the
    # page. Measured separation on 2026-09-01_201024 is 39 px.
    d = (math.sin(math.radians(tilt)), math.cos(math.radians(tilt)))
    nrm = (d[1], -d[0])
    # Tether: from the top edge down to where the bead would be. It STOPS there, as the
    # real one does -- overlapping the two in y would make `below` untestable.
    a0 = (300.0, 10.0)
    a1 = (a0[0] + d[0] * 190.0, a0[1] + d[1] * 190.0)
    cv2.line(img, tuple(int(v) for v in a0), tuple(int(v) for v in a1), 255, 3)
    # Rod: parallel, 39 px off the tether's axis, starting near the bead and running down.
    b0 = (a0[0] + nrm[0] * 39.0 + d[0] * 130.0, a0[1] + nrm[1] * 39.0 + d[1] * 130.0)
    b1 = (b0[0] + d[0] * 200.0, b0[1] + d[1] * 200.0)
    cv2.line(img, tuple(int(v) for v in b0), tuple(int(v) for v in b1), 255, 5)
    # A bright non-linear blob where the robot sits: it must not bend the fit.
    cv2.ellipse(img, (200, 330), (70, 22), 8, 0, 360, 255, -1)

    bars = find_bars(img, mask=(img > 127).astype(np.uint8) * 255)
    assert len(bars) >= 2, f"expected both bars, got {len(bars)}"
    # Swept over tilts -20/-5/0/+12 deg: 0.29 deg worst without the blob, 0.74 with it,
    # and the 0.74 is always the rod -- the bar whose end lies inside the blur. The gate
    # takes that from 1.4 deg; it does not take it to zero, and a check that asserted
    # zero would be asserting something this method does not deliver.
    for b in bars[:2]:
        assert abs(b.tilt_deg - tilt) < 1.0, b         # leans left going down -> negative
    # The tether never touches the blob, so it carries the method's own floor.
    tether = min(bars[:2], key=lambda b: b.p1[1])
    assert abs(tether.tilt_deg - tilt) < 0.3, tether
    # The rod's extent stops SHORT of its true end rather than running past it. How short
    # is set by the robot, not by a tuning constant: the blob reaches 24 px above its own
    # centre here, and the gate dilates that by FAT_RADIUS_PX, so the last ~32 px of rod
    # goes with it. Conservative in the direction that matters -- the reported bar is all
    # bar, never part robot. Before the gate this ran 30 px PAST the true end, into the
    # blur, which is the error that actually corrupts a measurement.
    rod = max(bars[:2], key=lambda b: b.p1[1])
    assert -35.0 < rod.p1[1] - b1[1] <= 2.0, (rod.p1, b1)
    # The blob is 140 px wide and brighter than either bar; a principal-axis method
    # returns its axis instead. Nothing here may come back near horizontal.
    assert all(abs(b.tilt_deg) < MAX_TILT_DEG for b in bars), bars

    # `below` must separate the two, which is the whole reason it exists: the tether ends
    # at y=189, the rod runs past it, so a cut between them returns the rod alone.
    only = find_bar(img, below=220.0, mask=(img > 127).astype(np.uint8) * 255)
    assert only is not None and only.p1[1] > 220.0, only
    assert abs(only.p0[0] - rod.p0[0]) < 2.0, (only.p0, rod.p0)   # it is the rod, not the tether

    # x_at must agree with the endpoints it was built from.
    b = bars[0]
    assert abs(b.x_at(b.p0[1]) - b.p0[0]) < 1e-6, b
    assert abs(b.x_at(b.p1[1]) - b.p1[0]) < 1e-6, b

    # An empty frame yields nothing rather than a confident answer on noise.
    assert find_bars(np.zeros((400, 640), np.uint8),
                     mask=np.zeros((400, 640), np.uint8)) == []

    print(f"bar: self-check passed ({len(bars)} bars, "
          f"tilt {bars[0].tilt_deg:+.2f} / {bars[1].tilt_deg:+.2f} deg vs {tilt:+.2f} true)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        d = Path(sys.argv[1])
        for tag in ("A", "B"):
            if (d / tag / f"{tag}.mp4").exists():
                track(d, tag, csv_out=d / f"bar_{tag}.csv",
                      overlay=d / f"bar_{tag}.mp4")
    else:
        _self_check()
