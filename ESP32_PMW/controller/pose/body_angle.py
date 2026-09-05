#!/usr/bin/env python3
"""Body attitude of a bare rotor-on-a-mast, straight from the silhouette.

Why this exists beside `estimator.py` rather than inside it
-----------------------------------------------------------
The stereo pose pipeline segments a **bright rim circle**. The tilt-sweep robots have no
rim -- they are a propeller and a mast -- so `live_viz.from_recording` solves 0 of 400
frames on `2026-09-01_210758` and reports every one as lost. That is the rim extractor
correctly finding no rim, not a tuning failure, and no threshold recovers it.

What those frames *do* have is a rotor disc, brightly lit against a dark scene, occupying
1.0-1.9% of the pixels at every frequency in the sweep. A fixed threshold plus the
largest connected component segments it in essentially every frame, and the disc's
second-moment ellipse is the same primitive `estimator.py` fits to the rim:

    tilt from the camera axis    theta = arccos(minor / major)
    azimuth of that tilt         the major axis angle

A circle seen off-axis projects to an ellipse foreshortened along the tilt direction.
That is the whole measurement, and it needs no rig, no calibration and no datum -- which
is the point, because none of those exist for these robots.

READ THE CHANGE, NOT THE ABSOLUTE ANGLE
---------------------------------------
`tilt_deg` measured on the drone-1 sweep sits at **49-67 deg at every frequency**. That is
not body lean: the cameras view this rig from near the rotor's edge, so the disc is
strongly foreshortened before the robot does anything. The fixed rig geometry dominates
the absolute number, and the experiment's signal is the few degrees it MOVES when the
30% command lands. Everything downstream is therefore referred to a pre-drop baseline;
an absolute `tilt_deg` from this module is not an attitude and should not be quoted as one.

FROZEN BLADES SHOW UP AS SCATTER, NOT AS A PER-FRAME FLAG
---------------------------------------------------------
Below ~60 Hz the blades do not blur into a disc: the exposure freezes them mid-rotation
and the "ellipse" is the arrangement of blades at whatever phase the shutter caught. An
earlier version of this file flagged that per frame with `minor/major`, on the reasoning
that a filled disc is rounder. **That does not work**, twice over. Measured over 25 frames
per point on drone 1, the ratio ran 0.40-0.66 at every frequency with no separation
between 20 Hz and 160. And synthetically, four blades at 90 deg spacing are rotationally
symmetric, so their second moments are indistinguishable from a filled disc -- a frozen
rotor can read perfectly round.

What does separate them is frame-to-frame SCATTER: a frozen rotor presents a different
blob every frame, a blurred one presents the same disc. Angle scatter over the same
samples was 14.5 deg at 20 Hz against 5.6-9.3 across 60-160. So quality is reported per
point by `alignment` as `angle_sd`, and the low-frequency points are to be read with it
in hand -- there is no honest per-frame flag, and pretending otherwise put a confident
number on blade phase.

This is a 2-D, per-camera measurement. It is not fused across the stereo pair: with no
rim there is no correspondence to fuse, and each camera's tilt is therefore measured
about ITS OWN optical axis. Two cameras 90 deg apart in azimuth will disagree, correctly,
and that disagreement is signal about tilt direction rather than error.

    uv run python controller/pose/body_angle.py                      # self-check
    uv run python controller/pose/body_angle.py <flight_dir> <log>   # analyse a take
"""

from __future__ import annotations

import csv
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

#: Grey level separating the robot from the scene. Chosen from the 2026-09-01 sweep:
#: the robot is 1.0-1.9% of pixels at every frequency and the background never reaches
#: this, so one constant works across the whole take. 200 and 230 erode the disc; 160
#: keeps it whole. Re-check with `--sheet` if the lighting is ever changed.
THRESH = 160
#: Below this many pixels the largest component is a reflection, not the robot. The real
#: disc runs 2700-3700 px at 640x400.
MIN_AREA_PX = 400
#: `duty[%]: A=` stepping off 100 is the only timestamp the 30% drop has -- the schedule
#: emits no label for it, because `addCarrierDutyCycleTask` is not a labelled boundary.
DUTY_RE = re.compile(r"duty\[%\]:\s*A=([\d.]+)")
FREQ_RE = re.compile(r"freq=([\d.]+)")
LABEL_RE = re.compile(r"label=([A-Z0-9_]+)")
POINT_RE = re.compile(r"^FREQ_(\d+)HZ$")


# ---- one frame ----------------------------------------------------------------------
def segment(gray, thresh=THRESH, min_area=MIN_AREA_PX):
    """Largest bright component, or None. ``(mask, area_px)``.

    Largest-component rather than every bright pixel: the guy wires above the robot are
    bright too, and a thin wire dragged into the second moments rotates the ellipse by
    tens of degrees. They are a separate component and are dropped by construction.
    """

    import cv2

    m = (gray > thresh).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[k, cv2.CC_STAT_AREA])
    if area < min_area:
        return None
    return (lab == k).astype(np.uint8), area


def disc_ellipse(mask):
    """Second-moment ellipse of a mask: ``(cx, cy, major, minor, angle_deg)``.

    Moments, not `cv2.fitEllipse`: fitEllipse fits the CONTOUR, so a disc with a mast
    sticking out of it is pulled toward the mast. The second moments of the filled region
    weight by area, and the mast is ~100 px against the disc's ~3000.

    ``angle_deg`` is the major axis in image coordinates, wrapped to (-90, +90]: an
    unsigned line direction, because an ellipse's axis has no head or tail.
    """

    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return None
    x, y = xs.astype(np.float64), ys.astype(np.float64)
    cx, cy = x.mean(), y.mean()
    cov = np.cov(np.vstack([x - cx, y - cy]))
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    # 2 sigma each way is the ellipse that matches a uniform disc's extent.
    major, minor = (2.0 * math.sqrt(max(v, 0.0)) for v in vals)
    ang = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
    return cx, cy, major, minor, (ang + 90.0) % 180.0 - 90.0


def tilt_from_ratio(major, minor):
    """Tilt away from the camera's optical axis, degrees, from disc foreshortening.

    A circle of radius r tilted by theta projects to an ellipse with the major axis still
    2r and the minor axis 2r*cos(theta). So theta = arccos(minor/major), 0 = face-on.

    Degenerate by construction at theta -> 0: d(minor/major)/d(theta) = -sin(theta)
    vanishes there, so a face-on disc gives up its tilt slowly and noisily. That is the
    same 1/sin(theta) ill-conditioning `control/attitude.py` prices at 9.5 deg for a
    near-upright robot, and it is why this is used on a DELIBERATELY tilted rotor.
    """

    if major <= 0:
        return float("nan")
    return math.degrees(math.acos(min(1.0, max(0.0, minor / major))))


def measure(gray, thresh=THRESH):
    """Everything one frame yields, or None when nothing segments."""

    got = segment(gray, thresh)
    if got is None:
        return None
    mask, area = got
    el = disc_ellipse(mask)
    if el is None:
        return None
    cx, cy, major, minor, ang = el
    ratio = minor / major if major > 0 else float("nan")
    return {"cx": cx, "cy": cy, "area_px": area, "major": major, "minor": minor,
            "ratio": ratio, "disc_angle_deg": ang,
            "tilt_deg": tilt_from_ratio(major, minor)}


# ---- the run's timeline -------------------------------------------------------------
def timeline(log_path):
    """``[{freq, t_start, t_drop, t_end}]``, one per frequency point that completed.

    ``t_drop`` is when `duty[%]: A` first reads 30 inside the point -- the instant the
    30% command reached the coil, taken from the firmware's own telemetry because the
    schedule emits no label for it. A point with no such line never got its drop and is
    dropped from the analysis rather than reported with a guessed instant.
    """

    from controller.control import sync

    entries, _ = sync.read_log(log_path)
    marks, duty = [], []
    for t, d, text, _ in entries:
        if d != "<-":
            continue
        m = LABEL_RE.search(text)
        if m:
            marks.append((t, m.group(1)))
        u = DUTY_RE.search(text)
        if u:
            f = FREQ_RE.search(text)
            duty.append((t, float(u.group(1)), float(f.group(1)) if f else float("nan")))

    ends = {n[len("DOWN_"):]: t for t, n in marks if n.startswith("DOWN_")}
    out = []
    for i, (t0, name) in enumerate(marks):
        m = POINT_RE.match(name)
        if not m:
            continue
        hz = int(m.group(1))
        t_end = ends.get(f"{hz:03d}HZ")
        if t_end is None or t_end <= t0:
            continue
        # A host-driven run (`control/tilt_servo.py`) drops to a fraction of a trimmed
        # duty, never to exactly 30, so it stamps `label=DROP_<f>HZ` itself; that label
        # wins when present. The schedule firmware's runs have only the telemetry.
        drops = [t for t, n in marks if n == f"DROP_{hz:03d}HZ" and t0 <= t <= t_end]
        if not drops:
            drops = [t for t, a, _ in duty if t0 <= t <= t_end and abs(a - 30.0) < 0.01]
        if not drops:
            continue
        out.append({"freq": hz, "t_start": t0, "t_drop": drops[0], "t_end": t_end})
    return out


# ---- a whole take -------------------------------------------------------------------
def analyse(flight_dir, log_path=None, out_csv=None, cams=None, thresh=THRESH,
            progress=True):
    """Measure every frame of every camera. Returns ``(rows, points, stats)``.

    Frames outside a frequency point (the coils-off gaps) are measured too and carry
    ``freq`` -1: they are the only look at the robot at rest, and they are what the
    segmentation rate should be judged on, since a rate computed only over the holds
    would hide a segmenter that fails whenever the rotor stops.

    ``log_path=None`` measures a take that has no `sweep.log` at all -- a hand-driven
    take, where nothing printed a label and the timeline has to come out of the data.
    Every row then carries ``freq`` -1, which is the same thing the coils-off gaps carry:
    "no schedule said what this frame is". It is not a degraded mode of the schedule
    path, it is the absence of a schedule.
    """

    import cv2

    from controller.camera import record

    flight_dir = Path(flight_dir)
    pts = timeline(Path(log_path)) if log_path else []
    stamps, _ = record.read_index(flight_dir)
    if stamps is None:
        raise SystemExit(f"{flight_dir} has no frames.csv, so no frame has a time.")
    videos = sorted(flight_dir.glob("*/*.mp4"))
    if cams:
        videos = [v for v in videos if v.parent.name in cams]

    rows, stats = [], {}
    for vi, vid in enumerate(videos):
        tag = vid.parent.name
        cap = cv2.VideoCapture(str(vid))
        n, seen, solved = 0, 0, 0
        while cap.isOpened():
            ok, img = cap.read()
            if not ok:
                break
            if n >= len(stamps):
                break
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            t = float(stamps[n][vi]) if stamps.shape[1] > vi else float(stamps[n][0])
            hz, rel = -1, float("nan")
            for p in pts:
                if p["t_start"] <= t <= p["t_end"]:
                    hz, rel = p["freq"], t - p["t_drop"]
                    break
            m = measure(g, thresh)
            seen += 1
            if m is not None:
                solved += 1
                rows.append({"cam": tag, "frame": n, "t": t, "freq_hz": hz,
                             "t_rel_drop_s": rel, **m})
            n += 1
            if progress and seen % 10000 == 0:
                print(f"  {tag}: {seen} frames, {100*solved/seen:.2f}% segmented",
                      flush=True)
        cap.release()
        stats[tag] = (solved, seen)
        print(f"{tag}: {solved}/{seen} segmented = {100*solved/max(seen,1):.2f}%")

    if out_csv:
        out_csv = Path(out_csv)
        cols = ["cam", "frame", "t", "freq_hz", "t_rel_drop_s", "cx", "cy", "area_px",
                "major", "minor", "ratio", "disc_angle_deg", "tilt_deg"]
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, cols)
            w.writeheader()
            w.writerows(rows)
        print(f"{len(rows)} rows -> {out_csv}")
    return rows, pts, stats


# ---- what the drop did --------------------------------------------------------------
def _median_filter(y, n):
    """Running median, edge-padded. Robust to the blade-phase spikes a mean chases."""

    if n < 3:
        return y
    n |= 1
    pad = np.pad(y, n // 2, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, n)
    return np.median(win, axis=1)


def alignment(rows, pts, cam, pre_s=1.0, resp_s=0.3, settle_s=2.0, smooth_s=0.25,
              fps=208.0):
    """Per frequency: attitude either side of the drop, and how fast it moved.

    Everything is referred to the pre-drop baseline, for the reason in the module
    docstring: the absolute tilt is dominated by where the cameras sit.

    ``rate_deg_s`` is the slope of a straight-line fit over the first ``resp_s`` after the
    drop, NOT the largest instantaneous derivative. The first version of this took
    ``max|d(tilt)/dt|`` and reported 250-1000 deg/s at every frequency including the ones
    with no response at all -- that was the noise floor being differentiated, not the
    robot. With 3-15 deg of frame-to-frame scatter at 208 fps, a per-frame derivative is
    meaningless however it is smoothed; a fit over a window long enough to contain the
    response is not.

    ``tau_s`` is when the smoothed track first covers 63% of the total change -- a
    first-order time constant if the response is one, and a blank if the change never
    gets there, which is itself the useful answer.
    """

    n_sm = max(3, int(round(smooth_s * fps)))
    out = []
    for p in pts:
        r = sorted((x for x in rows if x["cam"] == cam and x["freq_hz"] == p["freq"]),
                   key=lambda x: x["t"])
        if len(r) < n_sm * 2:
            continue
        t = np.array([x["t_rel_drop_s"] for x in r])
        y = np.array([x["tilt_deg"] for x in r])
        ang = np.array([x["disc_angle_deg"] for x in r])
        ys = _median_filter(y, n_sm)

        pre_w = (t >= -pre_s) & (t < 0)
        post_w = (t > settle_s) & (t <= settle_s + 2.0)
        base = float(np.nanmean(y[pre_w])) if pre_w.any() else float("nan")
        post = float(np.nanmean(y[post_w])) if post_w.any() else float("nan")
        d = post - base

        resp = (t >= 0) & (t <= resp_s)
        rate = float("nan")
        if resp.sum() >= 5:
            rate = float(np.polyfit(t[resp], ys[resp], 1)[0])

        tau = float("nan")
        if np.isfinite(d) and abs(d) > 1e-6:
            after = t >= 0
            hit = np.nonzero(after & (np.abs(ys - base) >= 0.63 * abs(d)))[0]
            if len(hit):
                tau = float(t[hit[0]])

        out.append({
            "freq": p["freq"], "n": len(r),
            # Frame-to-frame scatter of the disc axis before the drop: the honest quality
            # number, high where blades are frozen rather than blurred, and high again at
            # 160 Hz where the rotor itself is not holding steady.
            "angle_sd": float(np.nanstd(ang[pre_w])) if pre_w.any() else float("nan"),
            "tilt_pre": base, "tilt_post": post, "d_tilt": d,
            "rate_deg_s": rate, "tau_s": tau,
        })
    return out


def oscillation(rows, pts, cam, lo_s=0.2, hi_s=5.0, band=(0.3, 20.0)):
    """Dominant post-drop wobble per point: ``(freq, peak_hz, amp_deg, prominence)``.

    A peak here means nothing on its own -- any noisy track has one. It means something
    when the TWO CAMERAS AGREE, because they look down different optical axes and share
    no noise: a common peak is the body moving, a disagreement is two spectra of nothing.
    Measured on drone 1, 140 Hz gives 3.18 vs 3.24 Hz and 100 Hz gives 4.10 vs 4.08,
    while 40 Hz gives 3.09 vs 12.15. Compare the pair before believing either.
    """

    out = []
    for p in pts:
        r = sorted((x for x in rows if x["cam"] == cam and x["freq_hz"] == p["freq"]),
                   key=lambda x: x["t"])
        t = np.array([x["t_rel_drop_s"] for x in r])
        y = np.array([x["tilt_deg"] for x in r])
        m = (t >= lo_s) & (t <= hi_s)
        if m.sum() < 200:
            continue
        yy = _median_filter(y[m], 11)
        yy = yy - yy.mean()
        fs = 1.0 / float(np.median(np.diff(t[m])))
        Y = np.abs(np.fft.rfft(yy * np.hanning(len(yy))))
        f = np.fft.rfftfreq(len(yy), 1.0 / fs)
        sel = (f > band[0]) & (f < band[1])
        k = int(np.argmax(Y[sel]))
        out.append({"freq": p["freq"], "peak_hz": float(f[sel][k]),
                    "amp_deg": float(2 * Y[sel][k] / len(yy)),
                    "prominence": float(Y[sel][k] / Y[sel].mean())})
    return out


def plot(rows, pts, out_dir, cams=("A", "B"), pre_s=2.0, post_s=5.0):
    """Three figures: the per-point tracks, the summary against frequency, a montage."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    freqs = [p["freq"] for p in pts]
    col = {"A": "#1f77b4", "B": "#d62728"}

    # ---- 1. one panel per frequency, everything referred to the pre-drop baseline ----
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True, sharey=True)
    for ax, p in zip(axes.ravel(), pts):
        for cam in cams:
            r = sorted((x for x in rows if x["cam"] == cam and x["freq_hz"] == p["freq"]),
                       key=lambda x: x["t"])
            if not r:
                continue
            t = np.array([x["t_rel_drop_s"] for x in r])
            y = np.array([x["tilt_deg"] for x in r])
            base = np.nanmean(y[(t >= -1.0) & (t < 0)])
            ys = _median_filter(y - base, 52)
            m = (t >= -pre_s) & (t <= post_s)
            ax.plot(t[m], (y - base)[m], color=col[cam], alpha=0.13, lw=0.4)
            ax.plot(t[m], ys[m], color=col[cam], lw=1.8, label=f"cam {cam}")
        ax.axvline(0, color="k", lw=1.4, ls="--")
        ax.axvspan(-1.0, 0, color="0.85", zorder=0)
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_title(f"{p['freq']} Hz", fontsize=11)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8, loc="lower left")
    axes[0, 0].set_ylim(-25, 25)
    for ax in axes[1]:
        ax.set_xlabel("time from 30% duty drop (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("tilt change from baseline (deg)")
    fig.suptitle("Drone 1 - body tilt response to the 30% duty drop on channel 0\n"
                 "dashed line = drop instant (from telemetry duty[%]:A -> 30); "
                 "grey = baseline window; faint = per frame, bold = 0.25 s median",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_dir / "body_angle_panels.png", dpi=130)
    plt.close(fig)

    # ---- 2. summary against frequency ----------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.4))
    for cam in cams:
        a = alignment(rows, pts, cam)
        o = oscillation(rows, pts, cam)
        f = [x["freq"] for x in a]
        axes[0].plot(f, [x["d_tilt"] for x in a], "o-", color=col[cam], label=f"cam {cam}")
        axes[1].plot([x["freq"] for x in o], [x["amp_deg"] for x in o], "o-",
                     color=col[cam], label=f"cam {cam}")
        axes[2].plot([x["freq"] for x in o], [x["peak_hz"] for x in o], "o-",
                     color=col[cam], label=f"cam {cam}")
        axes[3].plot(f, [x["angle_sd"] for x in a], "o-", color=col[cam], label=f"cam {cam}")
    axes[0].set_ylabel("tilt change at the drop (deg)")
    axes[1].set_ylabel("post-drop wobble amplitude (deg)")
    # No "alignment rate" panel. The per-frame derivative of this track is the noise
    # floor being differentiated -- it read 250-1000 deg/s at every point including the
    # ones with no response. What the drop actually excites is a wobble, so that is what
    # is plotted; `alignment()` still returns `rate_deg_s` from a window fit for anyone
    # who wants it, with the same caveat.
    axes[2].set_ylabel("wobble frequency (Hz)")
    axes[2].annotate("believe it only where\nthe two cameras agree", (0.03, 0.86),
                     xycoords="axes fraction", fontsize=8, color="#333")
    axes[3].set_ylabel("pre-drop disc-axis scatter (deg)")
    axes[3].axhspan(10, 40, color="#ffcccc", zorder=0)
    axes[3].annotate("blade phase / unsteady:\nread these with care", (0.03, 0.86),
                     xycoords="axes fraction", fontsize=8, color="#a00")
    for ax, ttl in zip(axes, ("Effect", "Wobble excited", "Wobble frequency",
                              "Measurement quality")):
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_xlabel("drive frequency (Hz)")
        ax.set_title(ttl)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Drone 1 - effect of the channel-0 30% duty drop against drive frequency",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_dir / "body_angle_summary.png", dpi=130)
    plt.close(fig)
    return out_dir


def montage(flight_dir, pts, out_path, cam="A", before_s=-0.5, after_s=2.0):
    """Before/after the drop, one column per frequency: the effect you can see."""

    import cv2

    from controller.camera import record

    flight_dir = Path(flight_dir)
    stamps, _ = record.read_index(flight_dir)
    ft = [float(r[0]) for r in stamps]
    vid = flight_dir / cam / f"{cam}.mp4"
    want = {}
    for p in pts:
        for tag, off in (("before", before_s), ("after", after_s)):
            k = int(np.argmin([abs(x - (p["t_drop"] + off)) for x in ft]))
            want[k] = (p["freq"], tag)
    cap, n, got = cv2.VideoCapture(str(vid)), 0, {}
    while cap.isOpened() and want:
        ok, im = cap.read()
        if not ok:
            break
        if n in want:
            got[want.pop(n)] = im
        n += 1
    cap.release()
    cols = []
    for p in pts:
        tiles = []
        for tag in ("before", "after"):
            im = got.get((p["freq"], tag))
            if im is None:
                continue
            im = im.copy()
            cv2.putText(im, f"{p['freq']}Hz {tag}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(cv2.resize(im, None, fx=0.55, fy=0.55))
        if tiles:
            cols.append(np.vstack(tiles))
    if cols:
        cv2.imwrite(str(out_path), np.hstack(cols))
    return out_path


def _self_check():
    """Synthetic tilted discs: does arccos(minor/major) return the tilt that made them?

    Asserts the measurement, not the plumbing. A segmenter that finds a blob and an
    ellipse fit that returns numbers are both easy; returning the RIGHT angle is the
    thing that breaks, and it breaks silently.
    """

    import cv2

    for truth in (0, 15, 30, 45, 60):
        img = np.zeros((400, 640), np.uint8)
        a = 90
        b = int(round(a * math.cos(math.radians(truth))))
        cv2.ellipse(img, (320, 200), (a, b), 0, 0, 360, 255, -1)
        m = measure(img)
        assert m is not None, truth
        got = m["tilt_deg"]
        assert abs(got - truth) < 3.0, f"tilt {truth} -> {got:.1f}"

    # A rotated disc must report the rotation in the axis angle, tilt unchanged.
    img = np.zeros((400, 640), np.uint8)
    cv2.ellipse(img, (320, 200), (90, 45), 30, 0, 360, 255, -1)
    m = measure(img)
    assert abs(m["tilt_deg"] - 60.0) < 3.0, m["tilt_deg"]
    assert abs(m["disc_angle_deg"] - 30.0) < 3.0, m["disc_angle_deg"]

    # A bright thin wire elsewhere must not be picked up: largest component only.
    img2 = img.copy()
    cv2.line(img2, (10, 10), (630, 30), 255, 2)
    m2 = measure(img2)
    assert abs(m2["disc_angle_deg"] - m["disc_angle_deg"]) < 1.0, "wire captured the fit"

    # Empty frame: refuse, never invent.
    assert measure(np.zeros((400, 640), np.uint8)) is None

    # Four frozen blades at 90 deg are rotationally symmetric and read as a ROUND disc.
    # Pinned here because it is the reason there is no per-frame "blurred" flag: any
    # such flag based on minor/major would call this frozen rotor a clean disc.
    spiky = np.zeros((400, 640), np.uint8)
    for k in range(4):
        th = math.radians(k * 90 + 10)
        cv2.line(spiky, (320, 200),
                 (int(320 + 90 * math.cos(th)), int(200 + 90 * math.sin(th))), 255, 7)
    ms = measure(spiky)
    assert ms is not None and ms["ratio"] > 0.9, ms["ratio"]

    print("body_angle: self-check passed (tilt 0-60 deg within 3, wire rejected, "
          "symmetric frozen blades confirmed indistinguishable from a disc)")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        analyse(sys.argv[1], sys.argv[2],
                out_csv=sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        _self_check()
