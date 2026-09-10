#!/usr/bin/env python3
"""How fast the robot re-aligns after two coils are cut, per drive frequency.

    uv run python controller/control/alignment_rate.py                  # self-check
    uv run python controller/control/alignment_rate.py <take_dir>       # one take
    uv run python controller/control/alignment_rate.py <take> --log <sweep.log>

Input is the `axis.csv` + `tilt_A/B.csv` layout that `pose/disc_axis.py` writes, plus the
`sweep.log` `tilt_sweep.py` records alongside it. Output is one row per kill and one summary
row per frequency, in `results/alignment_rate/<stamp>/`.

WHAT IS BEING MEASURED
----------------------
The schedule spins to a frequency, holds level, then zeroes coils A and C. The field it
leaves is weaker and asymmetric, so the robot leans to a new equilibrium. The rate of that
lean is the number. Measured on `2026-09-08_205108`: about 7 deg over 3 s at 20 Hz.

THE DATUM IS THE COILS-OFF REST ATTITUDE
----------------------------------------
"Tilt from what?" The world frame here is camera A -- `stereo_rig.json` says so -- so a raw
angle is measured from wherever camera A points and means nothing physical. What the take
does contain is the robot at rest: after each frequency the carrier is cut and it hangs for
10 s with nothing driving it. That attitude is the zero. Every row below is an angle from it.

Normalise each axis row BEFORE averaging them into the datum. Averaging unnormalised rows
and then calling `arccos` reads the shortening of the mean vector as tilt.

A take with no `sweep.log` has no rest window to use, so `--no-log` falls back to the mean
axis over the whole take. That measures the transient's SHAPE correctly and its absolute
attitude not at all; the summary marks which datum it used.

DO NOT DIFFERENTIATE THE TRACE
------------------------------
The first version of this analysis, in `pose/body_angle.py`, took `max|d(tilt)/dt|` and
reported 250-1000 deg/s at every frequency INCLUDING the ones with no response at all: it
was differentiating the noise floor, which on this pipeline is 2 deg of second difference
per frame. The rate here is a 10-90% crossing of a smoothed trajectory over a step whose
size is measured separately, and the exponential fit reports a sigma that grows when the
record ends before the robot settles.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from controller.control import sync

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "results" / "alignment_rate"

LABEL_RE = re.compile(r"label=([A-Z0-9_]+)")
#: `KILL_020HZ_3` from the swept schedule; `KILL_020HZ` from a single-shot one.
KILL_RE = re.compile(r"^KILL_(\d+)HZ(?:_(\d+))?$")
FREQ_RE = re.compile(r"^FREQ_(\d+)HZ$")
DOWN_RE = re.compile(r"^DOWN_(\d+)HZ$")

#: Rest window after a point's DOWN label: 4 s of down-ramp, then the carrier is cut for
#: 10 s. Trimmed both ends so the spin-down and the next ramp stay out. Same numbers as
#: `tilt_report.REST_FROM_S/REST_TO_S`, for the same reason.
REST_FROM_S, REST_TO_S = 5.0, 13.5
#: Datum quality: over ~2 deg of RMS scatter and the robot was swinging on its wire.
DATUM_MAX_SPREAD_DEG = 5.0

PRE_S = 1.0            #: baseline window before the kill
POST_FROM_S = 4.0      #: settled window after it -- the transient is ~3 s
POST_TO_S = 6.0
SMOOTH_S = 0.25        #: box smoother for the crossing search
#: Below this the step is not a response and no rate is reported. The pre-kill scatter on
#: the shipped takes is ~1.1 deg, so 2 deg is a signal that clears its own noise.
MIN_AMP_DEG = 2.0

#: A jump this large (deg) defines the alignment event, and its gradient is the rate.
#: Anchored in ABSOLUTE degrees, not in a fraction of the total swing, which is the whole
#: point: the 10-90% band is measured off `amp`, so a slow drift after the jump inflates
#: the denominator (30 and 50 Hz read far too low) and a swing that never resolves a
#: crossing reports no rate at all (60 and 70 Hz came back empty). 10 deg clears the ~1 deg
#: per-frame scatter by 10x and is a fraction of every measured swing (40-53 deg).
JUMP_DEG = 10.0

#: How long after departure the first swing is allowed to take, when the rate is measured as
#: departure-to-peak. Wide enough for the slowest measured swing (~210 ms at 110 Hz) and well
#: short of the return (the trace is back near baseline by 500 ms), so the bounded maximum is
#: the top of the FIRST swing and never the second.
SWING_MAX_S = 0.40

#: Fraction of the ramp trimmed from EACH end before the gradient is fitted. A turning point
#: is by definition where the curve has flattened, so the samples nearest one are the least
#: representative of the ramp between them and they bias the slope low at both ends. Fitting
#: the central 1 - 2*BUFFER_FRAC removes that bias. Kept modest because the ramp is short --
#: ~150 ms against ~19.6 ms sampling is 8 samples, so 0.2 a side leaves about 5 to fit.
BUFFER_FRAC = 0.2


# ---------------------------------------------------------------- reading the take


def _csv(path):
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if not next(iter(r)).startswith("#")]
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def load(take_dir, which=None):
    """``(t, axis_unit, quality_deg)`` from a take's solved axis.

    ``which`` picks the estimator: ``"minor"`` reads `axis_minor.csv` (the rotor axis from
    the minor axis treated as a projected normal, two planes intersected -- directions only),
    ``"conic"`` reads `axis.csv` (conic backprojection, which uses the minor axis LENGTH).
    Default prefers the minor-axis file when it exists.

    `disc_axis` 20.6: the minor-axis construction cannot be biased by an over-long minor
    axis, its radius is stable to 0.2%, and on the overlay footage its vector tracks the
    rotor visibly better. The conic file stays the fallback and keeps the 56 older takes
    readable.
    """

    take = Path(take_dir)
    minor = take / "axis_minor.csv"
    use_minor = (which == "minor") or (which is None and minor.exists())
    a = _csv(minor if use_minor else take / "axis.csv")
    n = np.c_[a["nx"], a["ny"], a["nz"]]
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    q = a.get("vs_conic_deg", a.get("agree_deg", np.zeros(len(n))))
    return a["t"], n, q


def timeline(log_path):
    """Points from a `sweep.log`: ``[{freq, t_start, kills: [t, ...], t_end}]``.

    Only points that reached their `DOWN_` label are returned -- an interrupted point has
    no rest window and therefore no datum.
    """

    entries, _ = sync.read_log(log_path)
    pts, cur = [], None
    for t, _dirn, text, _dev in entries:
        m = LABEL_RE.search(text)
        if not m:
            continue
        name = m.group(1)
        if (f := FREQ_RE.match(name)):
            cur = {"freq": float(f.group(1)), "t_start": t, "kills": [], "t_end": None}
            pts.append(cur)
        elif (k := KILL_RE.match(name)) and cur is not None:
            cur["kills"].append(t)
        elif (d := DOWN_RE.match(name)) and cur is not None:
            cur["t_end"] = t
            cur = None
    return [p for p in pts if p["t_end"] is not None and p["kills"]]


def find_kills(t, tilt, min_step=MIN_AMP_DEG, window_s=3.0):
    """Kill instants from the data alone, for a take with no `sweep.log`.

    A step detector, not a derivative: the difference of the means either side of a
    candidate instant. A 7 deg step over 3 s is obvious to that, where a per-frame
    derivative sees only the 2 deg of frame-to-frame noise.

    Two refinements, both of which the synthetic check caught:

    * **Only steps in the same direction as the largest are kept.** A schedule that
      restores the coils between repeats produces a step back the other way, equal and
      opposite, and every one of those was being reported as a kill.
    * **The reported instant is the ONSET, not the peak of the detector.** With a
      symmetric window the difference-of-means has a broad plateau, so its argmax wanders
      by most of a second under noise -- which then drags `theta_from`'s one-second
      baseline window into the rising signal. Walking back to where the trace first leaves
      its baseline puts the instant where the physics is.
    """

    ts = _smooth(t, tilt, 0.25)
    step = np.array([np.mean(ts[(t >= x) & (t < x + window_s)])
                     - np.mean(ts[(t > x - window_s) & (t < x)])
                     if ((t > x - window_s) & (t < x)).sum() > 10
                     and ((t >= x) & (t < x + window_s)).sum() > 10 else 0.0
                     for x in t])
    sign = np.sign(step[int(np.argmax(np.abs(step)))])   # the kill direction
    mag = np.where(np.sign(step) == sign, np.abs(step), 0.0)

    out = []
    while True:
        i = int(np.argmax(mag))
        if mag[i] < min_step:
            break
        amp = mag[i]
        pre = (t > t[i] - window_s) & (t < t[i] - window_s / 2.0)
        base = np.median(ts[pre]) if pre.sum() > 5 else ts[max(i - 1, 0)]
        back = np.where((t <= t[i]) & (np.abs(ts - base) < 0.1 * amp))[0]
        out.append(float(t[back[-1]] if len(back) else t[i]))
        mag[(t > t[i] - 2 * window_s) & (t < t[i] + 2 * window_s)] = 0.0
    return sorted(out)


# ---------------------------------------------------------------- the datum


def datum(t, axis, pts=None):
    """``(up, n_used, spread_deg, kind)``. Rest windows when there are any, else the take."""

    if pts:
        m = np.zeros_like(t, dtype=bool)
        for p in pts:
            m |= (t >= p["t_end"] + REST_FROM_S) & (t <= p["t_end"] + REST_TO_S)
        kind = "rest"
    else:
        m = np.ones_like(t, dtype=bool)
        kind = "take-mean (NO rest window -- shape only, not absolute attitude)"
    if m.sum() < 50:
        raise SystemExit(f"datum: only {m.sum()} rows, need 50")
    V = _hemisphere(axis[m], axis[m][0])         # unit rows (see load()), sign-aligned
    up = V.mean(0)
    up /= np.linalg.norm(up)
    spread = float(np.degrees(np.arccos(np.clip(V @ up, -1, 1))).std())
    return up, int(m.sum()), spread, kind


def _hemisphere(axis, ref):
    """Flip each axis row into ``ref``'s hemisphere. The rotor axis is a LINE.

    An ellipse cannot tell rotor-up from rotor-down, so the sign is arbitrary and
    `stereo.orient` picks it against a fixed reference. That choice becomes unstable when the
    axis approaches perpendicular to the reference -- which is exactly what happens as the
    robot tilts further at higher drive frequency. Measured on the 60 Hz chunk: frame-to-frame
    VECTOR jumps up to 161 deg while the same jumps as LINES are only 29.9 deg. Nothing moved;
    the sign flipped.

    Left uncorrected, `arccos(axis . ref)` reads those flips as ~180-theta and reports the
    robot rotating 28 deg +- 54 at 60 Hz and 42 +- 53 at 70 Hz -- numbers that are estimator
    artefacts, not motion. Every angle in this module is therefore taken sign-invariantly.
    """

    a = np.asarray(axis, float)
    ref = np.asarray(ref, float)
    ref = ref / np.linalg.norm(ref)
    sgn = np.sign(a @ ref)
    sgn[sgn == 0] = 1.0
    return a * sgn[:, None]


def orient_continuous(axis, seed=None):
    """Resolve the axis sign ONCE and hold it by continuity. The robot cannot invert.

    An ellipse cannot tell rotor-up from rotor-down, so each frame's axis arrives with an
    arbitrary sign. `_hemisphere` handles that by folding everything into one hemisphere,
    which is safe but lossy: it caps every angle at 90 deg and silently folds anything past
    it, so a genuine large lean and an estimator error look alike.

    The rig supplies a stronger constraint -- the drone never flips upside down -- so the
    sign is physical, not arbitrary. Seed it from the first frame (or the rest datum) and
    thereafter choose whichever sign keeps the axis nearer the previous frame. The axis
    cannot move 90 deg between frames at 200 Hz, so continuity resolves it unambiguously,
    and angles beyond 90 deg stay visible instead of being folded back.
    """

    a = np.asarray(axis, float).copy()
    if not len(a):
        return a
    prev = np.asarray(seed, float) if seed is not None else a[0]
    prev = prev / np.linalg.norm(prev)
    for i in range(len(a)):
        if float(a[i] @ prev) < 0.0:
            a[i] = -a[i]
        n = np.linalg.norm(a[i])
        if n > 1e-12:
            prev = a[i] / n
    return a


def tilt_from(axis, up):
    """Angle between the axis LINE and ``up``, in [0, 90]."""

    return np.degrees(np.arccos(np.clip(np.abs(axis @ up), 0.0, 1.0)))


def angles(axis, ref):
    """``(radial_deg, azimuth_deg)`` of the axis about a reference direction.

    Radial is the polar angle from ``ref`` -- how far the rotor axis has tipped. Azimuth is
    where it tipped TO, measured in the plane perpendicular to ``ref`` and unwrapped, so a
    steadily precessing axis gives a straight ramp rather than a sawtooth.

    The pair separates two motions the single tilt scalar conflates: a cone of constant
    half-angle is CONSTANT in radial and LINEAR in azimuth, while a lean that grows and stays
    is a step in radial with azimuth fixed.
    """

    ref = np.asarray(ref, float)
    ref = ref / np.linalg.norm(ref)
    e1 = np.cross(ref, [0.0, 0.0, 1.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(ref, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ref, e1)
    a = _hemisphere(axis, ref)          # a line, so put it all in one hemisphere first
    radial = np.degrees(np.arccos(np.clip(a @ ref, -1.0, 1.0)))
    azimuth = np.degrees(np.unwrap(np.arctan2(a @ e2, a @ e1)))
    return radial, azimuth


def dft(x, t, freqs):
    """Amplitude spectrum of ``x`` sampled at the ACTUAL times ``t``.

    Not `np.fft`. The interleaved pose stream is non-uniform -- dt runs 1.5 to 20 ms -- and
    an FFT reads the grid it assumes rather than the one it got: on this rig
    `1/median(dt)` says 236 Hz against a true 190.7, which puts a 20.0 Hz line at 24.8.
    """

    x = np.asarray(x, float) - np.mean(x)
    t = np.asarray(t, float)
    w = np.hanning(len(x))
    x = x * w
    scale = 2.0 / (len(x) * np.mean(w))
    return np.abs((x[None, :] * np.exp(-2j * np.pi
                                       * np.asarray(freqs, float)[:, None] * t[None, :])
                   ).sum(1)) * scale


def deproject(azimuth, t, f_hz):
    """Remove the component of ``azimuth`` at ``f_hz`` -- least squares, not a filter.

    A precessing axis puts the SAME frequency into both angles: the radial angle oscillates
    at the precession rate and the azimuth carries it too. Subtracting the fitted sinusoid at
    that one frequency leaves whatever the azimuth is doing that is NOT precession, without
    the passband distortion a notch filter would add either side of it.
    """

    t = np.asarray(t, float)
    y = np.asarray(azimuth, float)
    A = np.c_[np.cos(2 * np.pi * f_hz * t), np.sin(2 * np.pi * f_hz * t),
              np.ones_like(t), t]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    fitted = A[:, :2] @ coef[:2]          # the periodic part only; keep mean and drift
    return y - fitted, float(np.hypot(coef[0], coef[1]))


def azimuth_from_rest(t, axis, up, t_kill, pre_s=PRE_S):
    """Lean DIRECTION about the rest datum, referenced to its pre-cut value.

    This, not the total rotation, is what the coil cut actually changes. Measured across the
    campaign, the azimuth swings 41-50 deg at EVERY frequency from 10 to 90 Hz -- MAD 0.83 deg
    at 20 Hz, 1.06 at 40 -- while the radial tilt barely moves and if anything goes slightly
    negative (the robot ends a little more upright).

    ~45 deg is what the coil geometry predicts: with A and C at 0 and 180 deg removed, the
    remaining asymmetry from B and D at 90/270 sits 45 deg away. So the number is a geometric
    constant being recovered, not a drive-dependent response, which is why it does not scale
    with frequency the way `rotation_from_baseline` appears to. That apparent growth is mostly
    geometry: a fixed azimuth swing sweeps a longer great-circle arc at a larger radial tilt.
    """

    up = np.asarray(up, float)
    up = up / np.linalg.norm(up)
    e1 = np.cross(up, [0.0, 0.0, 1.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(up, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    a = orient_continuous(axis, up)
    azi = np.degrees(np.unwrap(np.arctan2(a @ e2, a @ e1)))
    m = (t >= t_kill - pre_s) & (t < t_kill)
    if m.sum() < 20:
        return None
    # Wrap the CHANGE into +-180. `unwrap` fixes the within-record continuity but leaves each
    # repeat on an arbitrary +-360 branch, so pooling them mixed 54 deg with -306 and blew the
    # variance up by four orders of magnitude. The swing is ~45 deg, comfortably inside a half
    # turn, so wrapping cannot fold a real signal.
    return (azi - np.median(azi[m]) + 180.0) % 360.0 - 180.0


def rotation_from_baseline(t, axis, t_kill, pre_s=PRE_S):
    """Angle the rotor axis has turned THROUGH since just before the cut. ``None`` if short.

    This, not `tilt_from(axis, datum)`, is what the experiment measures. The angle from a
    datum is a scalar magnitude, so a rotation that is partly perpendicular to the datum
    direction partly cancels in it -- and worse, two estimators whose datums sit a few
    degrees apart put the motion on opposite sides and report opposite SIGNS for the same
    physical event. Measured on the 20 Hz chunk: the datum metric gave +1.76 deg from the
    conic axis and -3.09 deg from the minor-axis one, while the rotation from the pre-kill
    attitude gives +8.25 and +7.21 -- the same answer, and four times larger, because the
    scalar was cancelling most of it.

    The reference is the mean axis over the second before the cut, which is why the schedule
    holds level for 10 s first: it needs a settled attitude to rotate away from.
    """

    m = (t >= t_kill - pre_s) & (t < t_kill)
    if m.sum() < 20:
        return None
    # Sign-align the baseline window before averaging, or a flip inside it shortens the mean
    # and biases the reference; then measure every frame as a line angle.
    seed = axis[m][0]
    ref = _hemisphere(axis[m], seed).mean(0)
    n = np.linalg.norm(ref)
    if n < 1e-9:
        return None
    return np.degrees(np.arccos(np.clip(np.abs(axis @ (ref / n)), 0.0, 1.0)))


# ---------------------------------------------------------------- one transient


def rev_window(spin_hz, target_s=SMOOTH_S):
    """A smoothing window of a WHOLE number of rotor revolutions, near ``target_s``.

    The tilt trace carries a large once-per-rev component -- the rotor turns at the drive
    frequency and the pose pipeline samples at ~200 Hz, so it is resolved, not aliased away.
    Pooled over five repeats it is still +-3 deg at 10 Hz, which is twice the step being
    measured. It does not average out across repeats because the cut is itself an event the
    rotor phase can lock to.

    A boxcar whose length is an exact multiple of the period has a null at that frequency AND
    at every harmonic of it, so choosing the window this way removes the wobble exactly rather
    than attenuating it. An arbitrary 0.25 s window at 10 Hz is 2.5 revolutions and leaks
    badly -- which is what made `tau` fit at 5.3 s while `t10`-`t90` said 0.4 s.
    """

    if not (spin_hz and np.isfinite(spin_hz) and spin_hz > 0):
        return target_s
    n = max(1, int(round(target_s * spin_hz)))
    return n / float(spin_hz)


def _smooth(t, y, span_s=SMOOTH_S):
    """Box smoother with EDGE padding.

    `np.convolve(..., "same")` zero-pads, which drags the first half-window toward 0 --
    and the first half-window is exactly where the step being measured begins. That put
    `frac` far negative right after the kill and fitted tau at 8.7 s against a true 0.8.
    """

    k = max(1, int(round(span_s / max(np.median(np.diff(t)), 1e-9))))
    pad = k // 2
    yp = np.pad(np.asarray(y, float), pad, mode="edge")
    return np.convolve(yp, np.ones(k) / k, mode="valid")[:len(y)]


def relu_rate(ts, ys, t_kill, win_s):
    """Slope of a hinge pinned at ``t_kill``: ``y = c + m * max(0, t - t_kill)``.

    Returns ``(m, rms_residual_deg, n_samples)``, or ``(nan, nan, 0)``.

    The coil cut is a STEP whose instant is known -- it is the schedule's KILL label, not
    something to be found in the data -- so the elbow is not a free parameter. Pinning it is
    what makes the rate measurable at all: the jump lasts ~40 ms and the solved trace samples
    at ~19.6 ms (stride 4), so a gradient taken across the jump alone rests on two points.
    A hinge instead spends every sample in the window on two parameters, and the flat
    pre-kill segment pins the intercept, so the slope is determined by the post-kill points
    collectively rather than by any pair of them.

    Two parameters, both linear, so this is an ordinary least-squares solve and not an
    optimisation -- there is no starting guess to get wrong and no local minimum to fall into.
    """

    ts = np.asarray(ts, float)
    ys = np.asarray(ys, float)
    m = (ts >= t_kill - PRE_S) & (ts <= t_kill + win_s)
    if m.sum() < 6:
        return float("nan"), float("nan"), 0
    x = np.maximum(0.0, ts[m] - t_kill)
    if not np.any(x > 0):
        return float("nan"), float("nan"), 0
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, ys[m], rcond=None)
    resid = ys[m] - A @ coef
    return float(coef[1]), float(np.sqrt(np.mean(resid ** 2))), int(m.sum())


def relu_window(ts, ys, t_kill, amp_sign=None, sd_base=None):
    """First steep, large ramp after the cut. Returns ``(slope, lag, span)``.

    The operator's definition, and it is the whole rule: find the turning points of the
    smoothed trace after the coils drop, take the FIRST consecutive pair whose height differs
    by at least ``JUMP_DEG``, and fit a line between them. That pair IS the alignment event.

    "Large" is defined turning-point to turning-point rather than against a baseline, which
    is what makes it self-contained: it needs no departure threshold, no noise estimate, no
    swing direction supplied from elsewhere, and no window search. Every one of those was
    tried first (2026-09-10) and every one had a failure mode of its own -- a residual-scored
    window search overfits, because three samples fit a line whatever they are; a threshold
    off `baseline_sd_deg` reads the raw wobble and never fires; a direction taken from `amp`
    at 4-6 s points the wrong way once the response has swung back. A rise between two turning
    points has none of those degrees of freedom.

    The response is a lightly damped oscillation, not a step to a new equilibrium (measured
    2026-09-10, 110 Hz: flat ~50 ms, peak 65 deg at 200 ms, back to 18 deg by 500 ms), so
    "first pair" and not "largest pair" is what keeps the return swings out of the rate.
    """

    ts, ys = np.asarray(ts, float), np.asarray(ys, float)
    post = np.flatnonzero(ts >= t_kill)
    if post.size < 5:
        return float("nan"), float("nan"), float("nan")
    i0 = int(post[0])
    y, x = ys[i0:], ts[i0:]

    # Turning points: where the first difference changes sign. The cut instant itself opens
    # the list, since the trace is flat before it and the first ramp starts there.
    d = np.diff(y)
    nz = np.flatnonzero(d != 0)
    if nz.size < 2:
        return float("nan"), float("nan"), float("nan")
    # Seeded at the first sample that MOVES, not at the cut: the trace is flat through the
    # dead time (~50 ms of coil-current decay and rotor inertia), and opening the list at the
    # cut would fold that flat run into the ramp and shallow the gradient.
    turns = [int(nz[0])]
    sign = np.sign(d[nz[0]])
    for k in nz[1:]:
        sk = np.sign(d[k])
        if sk != sign:
            turns.append(int(k))
            sign = sk
    turns.append(len(y) - 1)

    for p_, q_ in zip(turns, turns[1:]):
        if q_ - p_ < 2:
            continue
        if abs(y[q_] - y[p_]) < JUMP_DEG:
            continue
        # Trim the buffer, then fit the middle. Backed off if the ramp is too short to
        # spare it -- a biased slope beats no slope, and the bias is toward under-reading.
        n_ = q_ - p_ + 1
        cut = int(round(BUFFER_FRAC * n_))
        while cut > 0 and n_ - 2 * cut < 4:
            cut -= 1
        a_, b_ = p_ + cut, q_ - cut
        slope, icept = (float(v) for v in np.polyfit(x[a_:b_ + 1], y[a_:b_ + 1], 1))

        # Dead time by extrapolation, not by "the first sample that moved". Run the fitted
        # ramp back to where it crosses the level the trace held before it, and the lag is
        # that crossing minus the cut. Reading it off the first moving sample instead makes
        # it a function of where the samples happen to fall (19.6 ms apart) and of whichever
        # noise excursion crossed first; a two-line intersection uses every sample in both.
        # The level is taken from the cut to the ramp's own start, so it spans the flat dead
        # time as well as the pre-cut baseline.
        flat = (ts >= t_kill - PRE_S) & (ts <= x[p_])
        lag = float("nan")
        if flat.sum() >= 3 and slope != 0:
            lag = float((float(np.median(ys[flat])) - icept) / slope - t_kill)
        return slope, lag, float(x[q_] - x[p_])
    return float("nan"), float("nan"), float("nan")


def transient(t, tilt, agree, t_kill, min_amp=MIN_AMP_DEG, spin_hz=None):
    """Metrics for one coil-kill step. ``None`` if the windows are not in the record.

    ``min_amp`` is the step size below which no rate is reported at all. It defaults to a
    fixed 2 deg, which is the right gate for a SINGLE repeat against this pipeline's ~1 deg
    of per-frame scatter. A POOLED curve has its own, much smaller noise, so `pool_transients`
    passes a floor derived from that instead -- a fixed 2 deg there would throw away a step
    measured at seven standard errors just because it is small.
    """

    pre_m = (t >= t_kill - PRE_S) & (t < t_kill)
    post_m = (t >= t_kill + POST_FROM_S) & (t <= t_kill + POST_TO_S)
    if pre_m.sum() < 10 or post_m.sum() < 10:
        return None
    theta_from = float(np.median(tilt[pre_m]))
    theta_to = float(np.median(tilt[post_m]))
    amp = theta_to - theta_from

    out = {"t_kill": float(t_kill), "theta_from": theta_from, "theta_to": theta_to,
           "amp_deg": amp, "baseline_sd_deg": float(tilt[pre_m].std()),
           "agree_med_deg": float(np.median(agree[(t >= t_kill) & (t <= t_kill + POST_TO_S)])),
           "t10": np.nan, "t90": np.nan,
           "rate_fit_deg_s": np.nan, "rate_jump_deg_s": np.nan,
           "jump_deg": np.nan, "jump_span_s": np.nan,
           "rate_relu_deg_s": np.nan, "relu_win_s": np.nan, "relu_lag_s": np.nan,
           "t_jump0": np.nan, "t_jump1": np.nan,
           "tau_s": np.nan, "tau_sigma_s": np.nan, "settled": False}
    if abs(amp) < min_amp:
        out["note"] = f"step {amp:+.2f} deg below the {min_amp:.2f} deg floor -- no response"
        return out

    span = rev_window(spin_hz)
    w = (t >= t_kill) & (t <= t_kill + POST_TO_S)
    ts, ys = t[w], _smooth(t[w], tilt[w], span)
    frac = (ys - theta_from) / amp                       # 0 at the kill, 1 when settled

    def cross(level):
        i = np.argmax(frac >= level)
        return float(ts[i]) if frac[i] >= level else np.nan

    out["t10"], out["t90"] = cross(0.10), cross(0.90)
    if np.isfinite(out["t10"]) and np.isfinite(out["t90"]) and out["t90"] > out["t10"]:
        # Least-squares gradient through every sample in the 10-90% band. Kept for
        # continuity, but it is anchored on `amp` and so inherits its failures -- see
        # JUMP_DEG. The chord (0.8 * amp / (t90 - t10)) was removed 2026-09-10: it read
        # the same band with two noisy samples instead of all of them, so it could only
        # ever be a worse version of the same flawed measurement.
        band = (ts >= out["t10"]) & (ts <= out["t90"])
        if band.sum() >= 4:
            out["rate_fit_deg_s"] = float(np.polyfit(ts[band], ys[band], 1)[0])

    # The alignment rate: the gradient of the FIRST rise of JUMP_DEG degrees after the kill.
    # `prog` is progress along the response in the direction the step actually went, so a
    # jump is positive whichever way the robot swings. The rise is measured from a running
    # minimum rather than from the kill, so a jump that starts after a brief settling delay
    # is still measured over its own span and not diluted by the wait.
    #
    # It gets its OWN smoothing window, one single revolution. `span` above is ~0.25 s at
    # every frequency, but a 10 deg jump at the measured 130-210 deg/s lasts 0.05-0.08 s --
    # the boxcar would be wider than the event and read a gradient several times too low.
    # One revolution is the SHORTEST window that still puts an exact null on the
    # once-per-rev wobble and all its harmonics (see `rev_window`), so it is the least
    # blunting that remains honest about the wobble.
    span_j = (1.0 / spin_hz) if (spin_hz and np.isfinite(spin_hz) and spin_hz > 0) else span
    ys_j = _smooth(t[w], tilt[w], span_j)
    # Which way the robot swung, taken from the SWING and not from `amp`. `amp` is the median
    # over POST_FROM_S..POST_TO_S (4-6 s), which assumes the response settles there. It does
    # not: the trace overshoots and comes back, so at 4-6 s it can sit near the baseline or
    # the wrong side of it, and 10/40/50 Hz reported no rate at all because the sign handed
    # to the swing search pointed away from the swing.
    early = (t >= t_kill) & (t <= t_kill + SWING_MAX_S)
    sgn = 1.0 if amp >= 0 else -1.0
    if early.sum() >= 5:
        dev = tilt[early] - theta_from
        far = dev[np.argmax(np.abs(dev))]
        if np.isfinite(far) and far != 0:
            sgn = 1.0 if far > 0 else -1.0

    # The headline rate: a hinge pinned at the known cut instant. It spans the kill, so it
    # needs the flat pre-kill samples too -- `w` above starts AT the kill and would leave the
    # intercept resting on the ramp alone.
    w2 = (t >= t_kill - PRE_S - 0.2) & (t <= t_kill + POST_TO_S)
    if w2.sum() >= 12:
        ys2 = _smooth(t[w2], tilt[w2], span_j)
        # The threshold must be the scatter of the trace the search actually reads.
        # `baseline_sd_deg` is measured on the RAW tilt, which carries the full once-per-rev
        # wobble -- several degrees at low frequency -- so using it here set a departure
        # threshold (and a peak floor of 3x it) far above anything the smoothed trace does,
        # and 10, 40 and 50 Hz reported no rate on every repeat.
        pre2 = (t[w2] >= t_kill - PRE_S) & (t[w2] < t_kill)
        sd2 = float(ys2[pre2].std()) if pre2.sum() >= 5 else out["baseline_sd_deg"]
        m_relu, lag, win = relu_window(t[w2], ys2, t_kill, sgn, sd_base=sd2)
        if np.isfinite(m_relu):
            out["rate_relu_deg_s"] = abs(m_relu)
            out["relu_lag_s"], out["relu_win_s"] = lag, win

    prog = (ys_j - theta_from) * sgn
    if prog.size:
        lo_i, lo_v = 0, prog[0]
        for i in range(prog.size):
            if prog[i] < lo_v:
                lo_i, lo_v = i, prog[i]
            if prog[i] - lo_v >= JUMP_DEG:
                j = slice(lo_i, i + 1)
                dt = ts[i] - ts[lo_i]
                if dt > 0:
                    # All samples across the jump span, not just its two endpoints.
                    g = (float(np.polyfit(ts[j], ys_j[j], 1)[0]) if (i - lo_i + 1) >= 4
                         else float((ys_j[i] - ys_j[lo_i]) / dt))
                    out["rate_jump_deg_s"] = abs(g)
                    out["jump_deg"] = float(prog[i] - lo_v)
                    out["jump_span_s"] = float(dt)
                    out["t_jump0"], out["t_jump1"] = float(ts[lo_i]), float(ts[i])
                break
        else:
            out["note"] = (f"no {JUMP_DEG:.0f} deg jump after the kill "
                           f"(largest rise {float(prog.max() - prog.min()):.1f} deg)")

    # Exponential: (theta_to - theta) decays from |amp| to 0, so log of it is linear in t.
    # Fitted only where the residual is between 15% and 85% of the step -- above that the
    # log is dominated by the pre-kill scatter, below it by wherever the tail happens to sit.
    y = 1.0 - frac
    band = (y > 0.15) & (y < 0.85)
    if band.sum() >= 10:
        x = ts[band] - t_kill
        p, cov = np.polyfit(x, np.log(y[band]), 1, cov=True)
        if p[0] < 0:
            out["tau_s"] = float(-1.0 / p[0])
            # sigma on tau from the slope's own sigma, by propagation
            out["tau_sigma_s"] = float(np.sqrt(cov[0, 0]) / p[0] ** 2)
    # Did the record reach the tail, or did the window end first? NOT by asking whether the
    # trace approached `theta_to` -- `theta_to` is measured from that same tail, so the test
    # is circular and returns True on a step that is still climbing. Ask instead whether the
    # fitted time constant is short against the window: at least three of them must fit
    # before the settled window opens. tau 0.8 s passes on a 4 s window, tau 4 s does not.
    out["settled"] = bool(np.isfinite(out["tau_s"]) and POST_FROM_S >= 3.0 * out["tau_s"])
    if not out["settled"] and np.isfinite(out["tau_sigma_s"]):
        # `amp` is also an underestimate when the step never settled, which biases tau LOW
        # (the slow synthetic case fits 1.96 s against a true 4.0). The sigma is widened
        # rather than the value corrected: the record does not contain the answer.
        out["tau_sigma_s"] *= 3.0
        out["note"] = "did not settle inside the window -- tau is an extrapolation, sigma x3"
    return out


# ---------------------------------------------------------------- the report


def report(take_dir, log_path=None, out_dir=None, use_log=True):
    take_dir = Path(take_dir)
    t, axis, agree = load(take_dir)
    t = t - t[0] if log_path is None else t

    pts = []
    if use_log:
        log_path = Path(log_path) if log_path else next(
            iter(sorted(take_dir.glob("*.log")) + sorted(take_dir.glob("../*.log"))), None)
        if log_path and Path(log_path).exists():
            pts = timeline(log_path)
    up, n_used, spread, kind = datum(t, axis, pts)
    tilt = tilt_from(axis, up)
    print(f"datum: {kind}, {n_used} rows, spread {spread:.2f} deg")
    if spread > DATUM_MAX_SPREAD_DEG:
        print(f"*** datum spread {spread:.2f} deg > {DATUM_MAX_SPREAD_DEG} -- the robot was "
              f"moving while it should have been at rest. Treat every angle below as suspect. ***")

    if not pts:
        print("no sweep.log points -- finding the kills in the data")
        pts = [{"freq": float("nan"), "t_start": t[0], "t_end": t[-1],
                "kills": find_kills(t, tilt)}]

    rows = []
    for p in pts:
        for i, tk in enumerate(p["kills"], 1):
            r = transient(t, tilt, agree, tk)
            if r is None:
                continue
            rows.append({"freq_hz": p["freq"], "repeat": i, **r})

    out = Path(out_dir or (OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")))
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "transients.csv", rows,
           ["freq_hz", "repeat", "t_kill", "theta_from", "theta_to", "amp_deg",
            "t10", "t90", "rate_jump_deg_s", "tau_s", "tau_sigma_s", "settled",
            "baseline_sd_deg", "agree_med_deg", "note"])

    summary = _summarise(rows, str(take_dir), kind)
    _write(out / "summary.csv", summary, list(summary[0]) if summary else [])
    _print_table(summary)
    print(f"\n-> {out}")
    return out


def _summarise(rows, flight_dir, kind="rest"):
    summary = []
    for f in sorted({r["freq_hz"] for r in rows}, key=lambda x: (np.isnan(x), x)):
        g = [r for r in rows if (r["freq_hz"] == f) or (np.isnan(f) and np.isnan(r["freq_hz"]))]
        rate = np.array([r["rate_jump_deg_s"] for r in g], float)
        base = np.array([r["theta_from"] for r in g], float)
        summary.append({
            "flight_dir": flight_dir, "freq_hz": f, "n_repeats": len(g),
            "amp_deg": _nanmean([r["amp_deg"] for r in g]),
            "rate_jump_deg_s": _nanmean(rate),
            "rate_sd": (float(np.nanstd(rate, ddof=1))
                        if np.isfinite(rate).sum() > 1 else float("nan")),
            "tau_s": _nanmean([r["tau_s"] for r in g]),
            "tau_sigma_s": _nanmean([r["tau_sigma_s"] for r in g]),
            "n_settled": int(sum(bool(r["settled"]) for r in g)),
            # Repeats are only independent if restoring A and C puts the robot back where it
            # started. This is the number that says whether it did.
            "baseline_drift_deg": float(np.std(base, ddof=1)) if len(base) > 1 else np.nan,
            "agree_med_deg": _nanmean([r["agree_med_deg"] for r in g]),
            "datum": kind.split(" ")[0],
        })
    return summary


def _print_table(summary):
    print(f"\n{'freq':>6} {'n':>3} {'amp':>7} {'rate':>9} {'sd':>7} {'tau':>7} "
          f"{'+-':>6} {'settled':>8} {'drift':>7} {'agree':>6}")
    for s in summary:
        print(f"{s['freq_hz']:6.0f} {s['n_repeats']:3d} {s['amp_deg']:7.2f} "
              f"{s['rate_jump_deg_s']:9.2f} {s['rate_sd']:7.2f} {s['tau_s']:7.3f} "
              f"{s['tau_sigma_s']:6.3f} {s['n_settled']:4d}/{s['n_repeats']:<3d} "
              f"{s['baseline_drift_deg']:7.2f} {s['agree_med_deg']:6.2f}")


def sweep_report(sweep_root, out_dir=None, solve=True, stride=3):
    """Every take of a `tilt_run --sweep` campaign -> one rate-vs-frequency table.

    Reads `sweep_index.csv`, skips anything whose outcome was not `ok` (a GPIO14 abort has a
    truncated ramp and no settled baseline, so it is not data), solves any take that has no
    `axis.csv` yet, and pools the repeats per frequency.
    """

    from controller.pose import disc_axis

    root = Path(sweep_root)
    # `--sweep` writes one index at the root; a single `run_chunk` writes one per frequency
    # subdirectory. Accept either, so a campaign that was interrupted is still analysable.
    if (root / "sweep_index.csv").exists():
        idx = list(csv.DictReader(open(root / "sweep_index.csv")))
    else:
        idx = [r for f in sorted(root.glob("*/index.csv"))
               for r in csv.DictReader(open(f))]
    if not idx:
        raise SystemExit(f"{root}: no sweep_index.csv and no */index.csv")
    rows = []
    for r in idx:
        if r["outcome"] != "ok":
            print(f"  skip {r['freq_hz']} Hz repeat {r['repeat']}: {r['outcome']}")
            continue
        take = Path(r["flight"])
        if solve and not (take / "axis.csv").exists():
            print(f"  solving {take.name} ({r['freq_hz']} Hz repeat {r['repeat']})")
            disc_axis.solve(take, progress=False, stride=stride)
        if not (take / "axis.csv").exists():
            continue
        t, axis, agree = load(take)
        pts = timeline(r["log"]) if Path(r["log"]).exists() else []
        try:
            up, n_used, spread, kind = datum(t, axis, pts)
        except SystemExit as e:
            print(f"  {take.name}: no datum ({e})")
            continue
        tilt = tilt_from(axis, up)
        kills = [k for p in pts for k in p["kills"]] or find_kills(t, tilt)
        for i, tk in enumerate(kills, 1):
            rot = rotation_from_baseline(t, axis, tk)
            got = transient(t, tilt if rot is None else rot, agree, tk,
                            spin_hz=float(r["freq_hz"]))
            if got:
                rows.append({"freq_hz": float(r["freq_hz"]), "repeat": int(r["repeat"]),
                             "take": take.name, "datum_spread_deg": round(spread, 3),
                             **got})
    if not rows:
        raise SystemExit(f"{root}: no analysable takes")

    out = Path(out_dir or (root / "report"))
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "transients.csv", rows,
           ["freq_hz", "repeat", "take", "t_kill", "theta_from", "theta_to", "amp_deg",
            "t10", "t90", "rate_jump_deg_s", "tau_s", "tau_sigma_s", "settled",
            "baseline_sd_deg", "agree_med_deg", "datum_spread_deg", "note"])
    summary = _summarise(rows, str(root))
    _write(out / "summary.csv", summary, list(summary[0]))
    _print_table(summary)
    print(f"\n-> {out}")
    return summary


#: dataviz palette, validated for CVD separation in light mode.
C_TILT, C_FIT, C_MARK = "#2f6fd0", "#e0721a", "#1a8f6a"
INK, MUTED, GRID = "#1c1e21", "#6b7280", "#dfe3e8"


def plot_take(take_dir, log_path=None, out_path=None, use_log=True):
    """Tilt against time for one take, with the kill, the fit and the period marked."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    take = Path(take_dir)
    t, axis, agree = load(take)
    t = t - t[0]
    pts = []
    if use_log and log_path and Path(log_path).exists():
        e0 = sync.read_log(log_path)[0]
        base = e0[0][0] if e0 else 0.0
        pts = [{**p, "t_start": p["t_start"] - base, "t_end": p["t_end"] - base,
                "kills": [k - base for k in p["kills"]]} for p in timeline(log_path)]
    up, n_used, spread, kind = datum(t, axis, pts)
    tilt = tilt_from(axis, up)
    kills = [k for p in pts for k in p["kills"]] or find_kills(t, tilt)
    freq = pts[0]["freq"] if pts else float("nan")

    fig, ax = plt.subplots(figsize=(11, 4.6), facecolor="white")
    ax.plot(t, tilt, color=C_TILT, lw=0.5, alpha=0.35)
    ax.plot(t, _smooth(t, tilt), color=C_TILT, lw=1.8, label="tilt from the coils-off rest attitude")
    rows = []
    for tk in kills:
        r = transient(t, tilt, agree, tk)
        if not r:
            continue
        rows.append(r)
        ax.axvline(tk, color=C_MARK, lw=1.4, ls="--")
        ax.annotate("coils A,C cut", (tk, ax.get_ylim()[1]), xytext=(4, -10),
                    textcoords="offset points", fontsize=9, color=C_MARK, va="top")
        if np.isfinite(r["tau_s"]):
            w = (t >= tk) & (t <= tk + POST_TO_S)
            amp = r["amp_deg"]
            fit = r["theta_from"] + amp * (1 - np.exp(-(t[w] - tk) / r["tau_s"]))
            ax.plot(t[w], fit, color=C_FIT, lw=2.2, ls="--",
                    label=f"exponential fit, tau {r['tau_s']:.2f} s")
        for lab, key in (("10%", "t10"), ("90%", "t90")):
            if np.isfinite(r[key]):
                ax.plot([r[key]], [r["theta_from"] + (0.1 if lab == "10%" else 0.9)
                                   * r["amp_deg"]], "o", color=C_FIT, ms=7)
    ax.set_title(f"{take.name}  --  {freq:g} Hz" if np.isfinite(freq) else take.name,
                 fontsize=11.5, color=INK, loc="left")
    ax.set_xlabel("time (s)", fontsize=9.5, color=MUTED)
    ax.set_ylabel("tilt from rest (deg)", fontsize=9.5, color=MUTED)
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=MUTED, loc="upper left")
    if rows:
        r = rows[0]
        ax.text(0.995, 0.03,
                f"step {r['amp_deg']:+.2f} deg   "
                + (f"tau {r['tau_s']:.2f} s   period 2*pi*tau {2 * np.pi * r['tau_s']:.2f} s   "
                   f"rate {r['rate_jump_deg_s']:.2f} deg/s" if np.isfinite(r["tau_s"])
                   else "no response above the 2 deg floor"),
                transform=ax.transAxes, ha="right", fontsize=9, color=INK)
    fig.text(0.01, 0.005, f"datum: {kind}, {n_used} rows, spread {spread:.2f} deg",
             fontsize=8, color=MUTED)
    fig.tight_layout()
    out = Path(out_path or (take / "angle.png"))
    fig.savefig(out, dpi=140, facecolor="white")
    plt.close(fig)
    print(f"-> {out}")
    return out, rows


#: How many robust sigma of pre-cut tilt before a repeat is called invalid.
#:
#: The failures are visible BEFORE the cut. Across this campaign the modal repeats sit at
#: 9.6-19.8 deg of pre-cut tilt (median 16.8, IQR 13.9-17.9) while the outliers read 39.1,
#: 40.5, 43.0 and 45.2 -- the drone had already tilted up during the ramp or hold, so the
#: measurement was taken from a wrong initial condition rather than being a wrong response.
#: A band of median +- 3 robust sigma flags them and no good run.
#:
#: This is a PRECONDITION, not an outlier test: it asks whether the run started where it was
#: supposed to, which is causal and per-take, where the modal-basin test needs the population
#: and only notices after the fact.
PRE_TILT_SIGMA = 3.0

#: Half-width of the modal basin, in degrees. Wide enough to hold the genuine repeat spread
#: (the clean frequencies scatter under 3 deg) and far narrower than the excursions the
#: outlier runs make (-68, +129, +144 deg were all seen).
MODAL_BAND_DEG = 15.0

#: Time grid for a pooled transient, relative to the kill.
POOL_FROM_S, POOL_TO_S, POOL_DT = -PRE_S, POST_TO_S, 0.01


def fit_second_order(t, y):
    """Fit a damped second-order step response. Returns the ALIGNMENT PERIOD, among others.

    The measured transient is not first order. At 20 Hz the pooled curve jumps to +1.5 deg
    within 0.2 s, undershoots to +0.5 deg at 0.9 s, and settles at +1.8 deg by 1.4 s -- a
    damped oscillation. An exponential has nothing to grip on that, which is why `tau` came
    back at 4.7 s against a visible settling time of 1.4 s and flagged itself unsettled.

    The model is the unit step response of a second-order system,

        y(t) = A [ 1 - e^{-z w t} ( cos(wd t) + (z w / wd) sin(wd t) ) ],   wd = w sqrt(1-z^2)

    and the reported period is ``2 pi / wd`` -- the period of the alignment oscillation, which
    is what "how long does it take to align" actually means for an underdamped response.

    Fitted by a coarse grid over (w, z) refined once, rather than by an optimiser: two
    parameters, a cheap residual, and no dependency on scipy being present.
    """

    t = np.asarray(t, float)
    y = np.asarray(y, float)
    m = t >= 0
    t, y = t[m], y[m]
    if len(t) < 20:
        return None

    def model(w, z):
        wd = w * np.sqrt(max(1.0 - z * z, 1e-9))
        e = np.exp(-z * w * t)
        return 1.0 - e * (np.cos(wd * t) + (z * w / wd) * np.sin(wd * t))

    def best_amp(shape):
        d = float(shape @ shape)
        return (float(shape @ y) / d) if d > 1e-12 else 0.0

    def resid(w, z):
        sh = model(w, z)
        a = best_amp(sh)
        return float(np.mean((y - a * sh) ** 2)), a

    grid_w = np.geomspace(0.5, 60.0, 60)
    grid_z = np.linspace(0.05, 1.5, 40)
    best = None
    for w in grid_w:
        for z in grid_z:
            r, a = resid(w, z)
            if best is None or r < best[0]:
                best = (r, w, z, a)
    r0, w0, z0, _ = best
    for w in np.linspace(max(w0 * 0.6, 0.2), w0 * 1.6, 40):
        for z in np.linspace(max(z0 - 0.25, 0.02), z0 + 0.25, 40):
            r, a = resid(w, z)
            if r < best[0]:
                best = (r, w, z, a)
    r, w, z, a = best
    wd = w * np.sqrt(max(1.0 - z * z, 1e-9))
    return {"amp_deg": a, "omega_n": w, "zeta": z,
            "period_s": float(2 * np.pi / wd) if z < 1.0 else float("nan"),
            "settle_s": float(4.0 / (z * w)) if z * w > 0 else float("nan"),
            "resid_deg": float(np.sqrt(r)),
            "underdamped": bool(z < 1.0)}


def pool_transients(curves, spin_hz=None):
    """Average N repeats of the same step, aligned on their kill instants, then fit.

    A single repeat here is a 1-2 deg step buried in ~0.5 deg of segmentation noise, so a
    per-repeat `tau` is mostly unfittable and the few that do fit are not to be believed.
    Averaging first is what the repeats are FOR: the step is common to all of them and the
    noise is not, so pooling n of them improves the ratio by sqrt(n) before anything is
    fitted. This is the difference between "no rate at 20 Hz" and a measured one.

    ``curves`` is ``[(t_relative_to_kill, tilt_deg), ...]``. Returns the pooled grid, the
    mean curve, and the same metrics `transient` reports.
    """

    grid = np.arange(POOL_FROM_S, POOL_TO_S, POOL_DT)
    stack = []
    for t_rel, tilt in curves:
        t_rel = np.asarray(t_rel, float)
        tilt = np.asarray(tilt, float)
        if len(t_rel) < 10 or t_rel[0] > POOL_FROM_S or t_rel[-1] < POOL_TO_S:
            continue
        y = np.interp(grid, t_rel, tilt)
        # Each repeat is levelled on its OWN pre-kill baseline. The robot does not return to
        # exactly the same attitude between repeats, and averaging absolute angles would put
        # that drift into the step.
        # Null the once-per-rev wobble before pooling, not after: it is the dominant term
        # in a single repeat and pooling does not remove it (see `rev_window`).
        y = _smooth(grid, y, rev_window(spin_hz))
        stack.append(y - np.median(y[grid < 0]))
    if len(stack) < 2:
        return None
    mean = np.mean(stack, axis=0)
    sem = np.std(stack, axis=0, ddof=1) / np.sqrt(len(stack))
    # Require the pooled step to clear three standard errors of the pooled curve, with an
    # absolute floor so a pathologically quiet record cannot certify an arbitrarily small
    # step. This is what lets 20 Hz report a rate (1.7 deg at 0.24 sem, ~7 sigma) while
    # 10 Hz is still refused (1.3 deg at 0.94 sem, ~1.4 sigma).
    sem_med = float(np.median(sem))
    got = transient(grid, mean, np.zeros_like(grid), 0.0,
                    min_amp=max(3.0 * sem_med, 0.3), spin_hz=spin_hz)
    if got is not None:
        got["n_pooled"] = len(stack)
        got["sem_deg"] = sem_med
        got["sigma"] = abs(got["amp_deg"]) / sem_med if sem_med > 0 else float("inf")
        so = fit_second_order(grid, mean)
        if so:
            got.update({f"so_{k}": v for k, v in so.items()})
    return grid, mean, sem, got


def campaign(root, out_dir=None, which=None, metric="azimuth"):
    """Pool every chunk of a campaign into rate-vs-frequency. The deliverable.

    Reads each `<root>/<f>hz/index.csv`, keeps the good takes, measures each kill as a
    `rotation_from_baseline`, pools the repeats per frequency (`pool_transients`, which nulls
    the once-per-rev wobble first), and writes a summary plus two figures: the transients
    themselves, and the frequency dependence of how far and how fast the robot turns.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root)
    out = Path(out_dir or (root / "report"))
    out.mkdir(parents=True, exist_ok=True)
    pooled, summary = {}, []
    all_pre, pre_by = [], {}
    for idx_path in sorted(root.glob("*/index.csv")):
        idx = [r for r in csv.DictReader(open(idx_path)) if r["outcome"] == "ok"]
        if not idx:
            continue
        f = float(idx[0]["freq_hz"])
        curves, pre_tilts = [], []
        for r in idx:
            take = Path(r["flight"])
            if not (take / "axis.csv").exists() and not (take / "axis_minor.csv").exists():
                continue
            t, axis, _q = load(take, which=which)
            pts = timeline(r["log"]) if Path(r["log"]).exists() else []
            # `azimuth` needs the rest datum; fall back to total rotation without one.
            rest = None
            if metric == "azimuth" and pts:
                try:
                    rest = datum(t, axis, pts)[0]
                except SystemExit:
                    rest = None
            for tk in [k for p in pts for k in p["kills"]]:
                rot = (azimuth_from_rest(t, axis, rest, tk) if rest is not None
                       else rotation_from_baseline(t, axis, tk))
                if rot is None:
                    continue
                m = (t >= tk + POOL_FROM_S - 0.2) & (t <= tk + POOL_TO_S + 0.2)
                pre_m = (t >= tk - PRE_S) & (t < tk)
                ref_up = rest if rest is not None else None
                pre_tilt = float("nan")
                if ref_up is not None and pre_m.sum() > 20:
                    pre_tilt = float(np.median(
                        tilt_from(orient_continuous(axis, ref_up), ref_up)[pre_m]))
                curves.append((t[m] - tk, rot[m]))
                pre_tilts.append(pre_tilt)
        # Per-repeat measurements, so the spread across repeats is a SAMPLE VARIANCE of
        # independent trials rather than the standard error of one pooled curve. The two
        # answer different questions: the SEM says how well the mean is known, the sample
        # variance says how much the rig varies from run to run -- which is what tells you
        # whether five repeats was enough and whether a single flight can be trusted.
        per = []
        for t_rel, rot in curves:
            grid_i = np.arange(POOL_FROM_S, POOL_TO_S, POOL_DT)
            if len(t_rel) < 10 or t_rel[0] > POOL_FROM_S or t_rel[-1] < POOL_TO_S:
                continue
            # NOT pre-smoothed. `transient` does its own smoothing, one revolution wide for
            # the jump gradient and ~0.25 s for the 10-90% band, and it cannot widen a
            # window that has already been applied. Smoothing here with rev_window(f)
            # (200-250 ms at every frequency) put a filter 5-6x WIDER than the event on
            # the trace before the jump was ever measured -- a 10 deg jump at the measured
            # 250 deg/s lasts 40 ms -- which suppressed every gradient the fit reported.
            y = np.interp(grid_i, t_rel, rot)
            y = y - np.median(y[grid_i < 0])
            gi = transient(grid_i, y, np.zeros_like(grid_i), 0.0, min_amp=0.0, spin_hz=f)
            if gi:
                per.append(gi)

        got = pool_transients(curves, spin_hz=f)
        if got is None:
            print(f"  {f:g} Hz: {len(curves)} usable repeats, skipped")
            continue
        grid, mean, sem, g = got
        pooled[f] = (grid, mean, sem, g, per)
        all_pre.extend(x for x in pre_tilts if x == x)
        pre_by[f] = list(pre_tilts)
        amps = np.array([x["amp_deg"] for x in per], float)
        # Rate over the MODAL repeats only, and as a MAGNITUDE.
        #
        # A rate is a speed: the alignment cannot proceed at -200 deg/s. The sign was leaking
        # in from failed runs whose azimuth went the other way (-68 deg at 50 Hz), because
        # rate = 0.8 * d_azimuth / (t90 - t10) inherits the sign of the step. Those runs are
        # not slow alignments, they are different events, and including them also blew the
        # 10 Hz variance up by four orders of magnitude once the metric became azimuth.
        keep = ((np.abs(amps - np.median(amps)) <= MODAL_BAND_DEG) if len(amps)
                else np.array([], bool))
        rates = np.abs(np.array([x["rate_jump_deg_s"] for x in per], float)[keep])
        rates = rates[np.isfinite(rates)]
        # The least-squares gradient over the same 10-90% band. Lower repeat-to-repeat
        # variance than the removed chord at 7 of 8 frequencies (96.9 -> 21.7 at 20 Hz, 264 -> 14.5
        # at 40), because it fits every sample in the band rather than resting on two
        # crossing times. Both are anchored on `amp` and share its failure modes, which is
        # why the headline rate is now the JUMP_DEG gradient instead.
        fits = np.abs(np.array([x.get("rate_fit_deg_s", np.nan) for x in per], float)[keep])
        relus = np.abs(np.array([x.get("rate_relu_deg_s", np.nan) for x in per], float)[keep])
        lags = np.array([x.get("relu_lag_s", np.nan) for x in per], float)[keep]
        fits = fits[np.isfinite(fits)]
        # Above ~50 Hz the repeats stop being one population: most land on the same modal
        # response while the odd one settles somewhere completely different (+52 deg and
        # -8 deg were both seen at 60 Hz, against a mode of +8). Those outlier traces are
        # smooth and well formed -- the robot found another equilibrium, it is not the
        # estimator breaking. A mean and a sample SD over a mixture describe neither
        # population, so the median and the MAD are reported beside them, and `n_outlier`
        # counts the repeats further than 3 MAD from the median. `control/theory.md` 24.7.
        med = float(np.median(amps)) if len(amps) else float("nan")
        # Fraction of repeats in the MODAL BASIN. Above ~50 Hz the spread is not measurement
        # error, it is a minority of runs settling somewhere else entirely: at 60 Hz the five
        # repeats read 47.2, 14.2, 46.3, 54.5, 129.4 deg. The survivors match every clean
        # frequency to within a degree, so MAD describes "how often it goes elsewhere" rather
        # than how well the angle is known -- which is worth its own column instead of being
        # read off the spread. `control/theory.md` 24.5.
        in_basin = np.abs(amps - med) <= MODAL_BAND_DEG if len(amps) else np.array([])
        n_modal = int(in_basin.sum())
        modal_amps = amps[in_basin] if len(amps) else amps
        mad = float(np.median(np.abs(amps - med))) * 1.4826 if len(amps) else float("nan")
        n_out = int(np.sum(np.abs(amps - med) > max(3.0 * mad, 1.0))) if len(amps) else 0
        var = float(np.var(amps, ddof=1)) if len(amps) > 1 else float("nan")
        rvar = float(np.var(rates, ddof=1)) if len(rates) > 1 else float("nan")
        summary.append({"freq_hz": f, "n_repeats": g["n_pooled"],
                        "rot_median_deg": round(med, 3),
                        "rot_mad_deg": round(mad, 3),
                        "n_modal": n_modal,
                        "modal_frac": round(n_modal / max(len(amps), 1), 3),
                        "modal_mean_deg": round(float(modal_amps.mean()), 3)
                        if len(modal_amps) else float("nan"),
                        "modal_sd_deg": round(float(np.std(modal_amps, ddof=1)), 3)
                        if len(modal_amps) > 1 else float("nan"),
                        "n_outlier": n_out,
                        "rot_mean_deg": round(float(amps.mean()), 3) if len(amps) else float("nan"),
                        "rot_var_deg2": round(var, 4),
                        "rot_sd_deg": round(float(np.sqrt(var)), 3) if var == var else float("nan"),
                        "rot_sem_deg": round(float(np.sqrt(var / len(amps))), 3)
                        if var == var else float("nan"),
                        "rate_relu_med": round(float(np.nanmedian(relus)), 3)
                        if np.isfinite(relus).any() else float("nan"),
                        "rate_relu_mad": round(float(np.nanmedian(np.abs(relus - np.nanmedian(relus))))
                                               * 1.4826, 3) if np.isfinite(relus).sum() > 1 else float("nan"),
                        "relu_lag_ms": round(float(np.nanmedian(lags)) * 1e3, 1)
                        if np.isfinite(lags).any() else float("nan"),
                        "rate_fit_med": round(float(np.median(fits)), 3) if len(fits) else float("nan"),
                        "rate_fit_mad": round(float(np.median(np.abs(fits - np.median(fits))))
                                              * 1.4826, 3) if len(fits) > 1 else float("nan"),
                        "rate_mean_deg_s": round(float(rates.mean()), 3) if len(rates) else float("nan"),
                        "rate_var": round(rvar, 3),
                        "rate_sd_deg_s": round(float(np.sqrt(rvar)), 3) if rvar == rvar else float("nan"),
                        "n_rate": int(len(rates)),
                        "rotation_deg": round(g["amp_deg"], 3),
                        "sem_deg": round(g["sem_deg"], 4),
                        "sigma": round(g["sigma"], 1),
                        "t10_s": round(g["t10"], 4), "t90_s": round(g["t90"], 4),
                        "rate_jump_deg_s": round(g["rate_jump_deg_s"], 3),
                        "period_s": round(g.get("so_period_s", float("nan")), 4),
                        "zeta": round(g.get("so_zeta", float("nan")), 3)})
    if not summary:
        raise SystemExit(f"{root}: nothing poolable")

    # Self-calibrating precondition band from the whole campaign's pre-cut tilts.
    ap = np.array(all_pre, float)
    if len(ap) > 8:
        med = float(np.median(ap))
        sig = float(np.median(np.abs(ap - med))) * 1.4826
        lo, hi = med - PRE_TILT_SIGMA * sig, med + PRE_TILT_SIGMA * sig
        print(f"pre-cut tilt band: {lo:.1f}-{hi:.1f} deg "
              f"(median {med:.1f}, robust sigma {sig:.1f})")
        for srow in summary:
            pv = np.array([x for x in pre_by.get(srow["freq_hz"], []) if x == x], float)
            n_bad = int(((pv < lo) | (pv > hi)).sum()) if len(pv) else 0
            srow["n_precondition_fail"] = n_bad
    _write(out / "campaign.csv", summary, list(summary[0]))

    cmap = plt.get_cmap("viridis")
    fs = sorted(pooled)
    col = {f: cmap(0.08 + 0.84 * i / max(len(fs) - 1, 1)) for i, f in enumerate(fs)}

    # Zoomed on the step, not the whole window.
    #
    # The first version drew -1 to +6 s with a shaded SEM band per frequency. Nine bands over
    # a 6 s window is mostly overlap, and five of those seconds are plateau where every curve
    # sits on top of every other -- the frequency dependence lives entirely in the first few
    # hundred milliseconds. So: the cut is centred, the tail is cropped to where the curves
    # have separated, and the rise is given most of the width.
    fig, (top, bot) = plt.subplots(2, 1, figsize=(11, 8), facecolor="white",
                                   gridspec_kw={"height_ratios": [1, 1], "hspace": 0.32})
    for f in fs:
        grid, mean, sem, g, _per = pooled[f]
        top.plot(grid, mean, color=col[f], lw=1.7, label=f"{f:g} Hz")
    top.set_xlim(-0.4, 2.0)
    top.axvline(0, color="#6b7280", lw=1.2, ls="--")
    top.axhline(0, color="#c9ced4", lw=0.8)
    _style(top, "Swing of the lean bearing after the cut",
           "time from the cut (s)", "swing (deg)")
    top.legend(frameon=False, fontsize=8, labelcolor=MUTED, ncol=5, loc="lower right")

    # The rise alone, stretched. This is where the frequencies actually differ.
    for f in fs:
        grid, mean, sem, g, _per = pooled[f]
        bot.plot(grid, mean, color=col[f], lw=2.0, label=f"{f:g} Hz")
    bot.set_xlim(-0.05, 0.45)
    bot.axvline(0, color="#6b7280", lw=1.2, ls="--")
    bot.axhline(0, color="#c9ced4", lw=0.8)
    _style(bot, "The rise, stretched  (same curves, first 450 ms)",
           "time from the cut (s)", "swing (deg)")
    fig.tight_layout()
    fig.savefig(out / "transients.png", dpi=140, facecolor="white")
    plt.close(fig)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4), facecolor="white")
    x = [s["freq_hz"] for s in summary]
    # Error bars are the SAMPLE standard deviation across repeats, not the standard error:
    # the question is how much a single run varies, not how well the mean is pinned.
    # Median and MAD, not mean and SD: above 50 Hz the repeats are a mixture (24.5) and a
    # mean over two populations describes neither. The individual repeats are drawn as dots
    # so a bimodal point is visible as such rather than hidden inside an error bar.
    a1.errorbar(x, [s["rot_median_deg"] for s in summary],
                yerr=[s["rot_mad_deg"] for s in summary], color=C_TILT, lw=2, marker="o",
                ms=6, capsize=3, elinewidth=1.4, zorder=3)
    for s_ in summary:
        for pt in pooled[s_["freq_hz"]][4]:
            a1.plot([s_["freq_hz"]], [pt["amp_deg"]], "o", color=C_TILT, ms=3, alpha=0.35)
    for s_ in summary:
        if s_["n_outlier"]:
            a1.annotate(f"{s_['n_outlier']} outlier" + ("s" if s_["n_outlier"] > 1 else ""),
                        (s_["freq_hz"], s_["rot_median_deg"]), xytext=(0, -16),
                        textcoords="offset points", ha="center", fontsize=7.5, color="#b45309")
    _style(a1, "Lean-direction change  (median +- MAD; dots = repeats)",
           "drive frequency (Hz)", "azimuth change (deg)")
    rmed, rmad = [], []
    for s_ in summary:
        rr = np.array([x_["rate_jump_deg_s"] for x_ in pooled[s_["freq_hz"]][4]], float)
        rr = rr[np.isfinite(rr)]
        m_ = float(np.median(rr)) if len(rr) else float("nan")
        rmed.append(m_)
        rmad.append(float(np.median(np.abs(rr - m_))) * 1.4826 if len(rr) else float("nan"))
        for v in rr:
            a2.plot([s_["freq_hz"]], [v], "o", color=C_FIT, ms=3, alpha=0.35)
    a2.errorbar(x, rmed, yerr=rmad, color=C_FIT, lw=2, marker="o", ms=6, capsize=3,
                elinewidth=1.4, zorder=3)
    _style(a2, "Tilt rate, 10-90%  (median +- MAD)", "drive frequency (Hz)", "rate (deg/s)")
    fig.tight_layout()
    fig.savefig(out / "rate_vs_frequency.png", dpi=140, facecolor="white")
    plt.close(fig)

    print(f"\n{'freq':>5} {'n':>3} | {'median':>8} {'MAD':>7} "
          f"| {'modal':>6} {'mean':>8} {'sd':>6} | {'rate fit':>9} {'MAD':>7} "
          f"{'jump':>8} {'HINGE':>9} {'mad':>7} {'lag ms':>7}")
    for s in summary:
        rates_here = [x["rate_jump_deg_s"] for x in pooled[s["freq_hz"]][4]]
        rates_here = [x for x in rates_here if x == x]
        rmed = float(np.median(rates_here)) if rates_here else float("nan")
        print(f"{s['freq_hz']:5.0f} {s['n_repeats']:3d} | {s['rot_median_deg']:8.2f} "
              f"{s['rot_mad_deg']:7.2f} | {s['n_modal']:2d}/{s['n_repeats']:<3d} "
              f"{s['modal_mean_deg']:8.2f} {s['modal_sd_deg']:6.2f} | "
              f"{s['rate_fit_med']:9.2f} {s['rate_fit_mad']:7.2f} {rmed:8.2f}"
              f" {s.get('rate_relu_med', float('nan')):9.2f}"
              f" {s.get('rate_relu_mad', float('nan')):7.2f}"
              f" {s.get('relu_lag_ms', float('nan')):7.1f}")
    print(f"\n-> {out}")
    return summary


def trial_panels(root, out_path=None, which=None, metric="azimuth"):
    """One CHOSEN trial per frequency with the fit drawn on it, not a pooled average.

    A pooled curve is the wrong picture for judging this fit. Repeats do not share a dead
    time to the millisecond, so averaging rounds the corner the fit is built on and shows a
    ramp gentler than any trial actually flew -- the reader ends up checking the fit against
    a curve no run produced. The trial shown is the one whose rate is nearest the median of
    its frequency, so it is representative rather than flattering.

    Drawn per panel: the pre-cut level, the level the swing reaches, a dashed vertical at
    the instant A and C drop, the fitted ramp extended back to where it meets the pre-cut
    level, and the dead time between the two.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root)
    picks = []
    for idx_path in sorted(root.glob("*/index.csv")):
        idx = [r for r in csv.DictReader(open(idx_path)) if r["outcome"] == "ok"]
        if not idx:
            continue
        f = float(idx[0]["freq_hz"])
        cand = []
        for r in idx:
            take = Path(r["flight"])
            if not (take / "axis.csv").exists() and not (take / "axis_minor.csv").exists():
                continue
            if not Path(r["log"]).exists():
                continue
            t, axis, _q = load(take, which=which)
            pts = timeline(r["log"])
            try:
                rest = datum(t, axis, pts)[0]
            except SystemExit:
                continue
            for tk in [k for p_ in pts for k in p_["kills"]]:
                rot = azimuth_from_rest(t, axis, rest, tk)
                if rot is None:
                    continue
                res = transient(t, rot, np.zeros_like(t), tk, min_amp=0.0, spin_hz=f)
                if res and np.isfinite(res.get("rate_relu_deg_s", np.nan)):
                    cand.append((f, take.name, t, rot, tk, res))
        if not cand:
            continue
        med = float(np.median([c[5]["rate_relu_deg_s"] for c in cand]))
        picks.append(min(cand, key=lambda c: abs(c[5]["rate_relu_deg_s"] - med)))

    if not picks:
        print("no trials to draw")
        return None

    n = len(picks)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.1 * nrow), squeeze=False)
    C_RAW, C_SM, C_RAMP, C_PRE, C_POST, C_CUT = ("#c9ccd1", "#1f2933", "#c1432c",
                                                 "#2f6f9f", "#3f8f5a", "#8a5cf6")
    for k, (f, name, t, rot, tk, res) in enumerate(picks):
        ax = axes[k // ncol][k % ncol]
        rel = t - tk
        m = (rel >= -0.30) & (rel <= 0.70)
        ys = _smooth(t[m], rot[m], 1.0 / f)
        pre_lvl = float(np.median(rot[(rel >= -PRE_S) & (rel < 0)]))
        ax.plot(rel[m] * 1e3, rot[m] - pre_lvl, lw=0.8, color=C_RAW, label="raw", zorder=1)
        ax.plot(rel[m] * 1e3, ys - pre_lvl, lw=1.8, color=C_SM,
                label="smoothed (1 rev)", zorder=3)

        ax.axhline(0.0, color=C_PRE, lw=1.4, ls="--", zorder=2,
                   label="pre-cut tilt (mean)")
        lag, span = res["relu_lag_s"], res["relu_win_s"]
        rate = res["rate_relu_deg_s"]
        sgn = 1.0
        w_pk = (rel >= 0) & (rel <= SWING_MAX_S)
        if w_pk.sum():
            dev = (ys - pre_lvl)[(rel[m] >= 0) & (rel[m] <= SWING_MAX_S)]
            if dev.size:
                sgn = 1.0 if dev[np.argmax(np.abs(dev))] > 0 else -1.0
                ax.axhline(float(dev[np.argmax(np.abs(dev))]), color=C_POST, lw=1.4,
                           ls="--", zorder=2, label="post-cut tilt (swing peak)")

        ax.axvline(0.0, color=C_CUT, lw=1.6, ls="--", zorder=4, label="A,C off")
        if np.isfinite(lag) and np.isfinite(span) and np.isfinite(rate):
            t0 = lag
            xs = np.linspace(t0, t0 + span, 40)
            ax.plot(xs * 1e3, sgn * rate * (xs - t0), lw=2.4, color=C_RAMP, zorder=5,
                    label="fitted ramp")
            ax.axvspan(0.0, max(t0, 0.0) * 1e3, color=C_CUT, alpha=0.10, zorder=0,
                       label="dead time")
            ax.annotate(f"{abs(rate):.0f} deg/s\ndead {lag*1e3:.0f} ms",
                        xy=(0.97, 0.05), xycoords="axes fraction", ha="right", va="bottom",
                        fontsize=8.5, color=C_RAMP)
        ax.set_title(f"{f:g} Hz   {name}", fontsize=9.5)
        ax.set_xlim(-300, 700)
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=8)
        if k % ncol == 0:
            ax.set_ylabel("azimuth from pre-cut (deg)", fontsize=9)
        if k // ncol == nrow - 1:
            ax.set_xlabel("time from A,C off (ms)", fontsize=9)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    h, l = axes[0][0].get_legend_handles_labels()
    seen, hh, ll = set(), [], []
    for a_, b_ in zip(h, l):
        if b_ not in seen:
            seen.add(b_); hh.append(a_); ll.append(b_)
    fig.legend(hh, ll, loc="lower center", ncol=len(ll), fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Alignment fit on one representative trial per drive frequency "
                 "(median-rate trial, not an average)", fontsize=12)
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    out = Path(out_path or (root / "report" / "trial_fits.png"))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")
    return out


def rate_scatter(root, out_path=None, which=None, metric="azimuth"):
    """Alignment rate against drive frequency: every repeat, coloured by frequency.

    Two panels because there are two defensible definitions of "rate" and they differ by a
    near-constant factor, so they must never be mixed:

    * **jump** -- the gradient of the first JUMP_DEG (10 deg) rise after the kill. This is
      the headline rate. It is anchored in absolute degrees, so it neither dilutes when the
      swing drifts on after the jump (30 and 50 Hz) nor vanishes when a 10-90% crossing
      fails to resolve (60 and 70 Hz).
    * **least squares** -- the gradient over the 10% to 90% band of the total swing, kept for
      continuity. Formerly also a **chord** across that band, removed 2026-09-10: two noisy
      samples reading the same flawed window the least-squares line already reads in full.
      90% crossings. This is what the 2026-09-03 report used (verified against its own stored
      values) and it is the conventional rise-time number.
    * **least-squares** -- the gradient of a line fitted through every sample in that same
      band. Same window, so equally objective, but it uses all ~40 samples instead of two
      crossings that are each one noisy sample.

    Measured across this campaign, the least-squares slope has the lower repeat-to-repeat MAD
    at seven of eight frequencies (5.70 -> 1.24 deg/s at 20 Hz, 10.61 -> 4.18 at 40), and it
    The jump gradient and the least-squares gradient are different measurements, not two
    estimates of one number, and must never be mixed in a single series.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root)
    per = {}
    for idx_path in sorted(root.glob("*/index.csv")):
        idx = [r for r in csv.DictReader(open(idx_path)) if r["outcome"] == "ok"]
        if not idx:
            continue
        f = float(idx[0]["freq_hz"])
        rows = []
        for r in idx:
            take = Path(r["flight"])
            if not (take / "axis.csv").exists() and not (take / "axis_minor.csv").exists():
                continue
            t, axis, _q = load(take, which=which)
            pts = timeline(r["log"]) if Path(r["log"]).exists() else []
            # Same metric as `campaign`. These disagreed for a while -- the scatter stayed on
            # total rotation after the campaign moved to azimuth -- and reported 17 deg/s at
            # 20 Hz against the campaign's 121 for the same takes.
            rest = None
            if metric == "azimuth" and pts:
                try:
                    rest = datum(t, axis, pts)[0]
                except SystemExit:
                    rest = None
            for tk in [k for p in pts for k in p["kills"]]:
                rot = (azimuth_from_rest(t, axis, rest, tk) if rest is not None
                       else rotation_from_baseline(t, axis, tk))
                if rot is None:
                    continue
                g = np.arange(POOL_FROM_S, POOL_TO_S, POOL_DT)
                m = (t >= tk + POOL_FROM_S - 0.2) & (t <= tk + POOL_TO_S + 0.2)
                y = _smooth(g, np.interp(g, t[m] - tk, rot[m]), rev_window(f))
                y = y - np.median(y[g < 0])
                gi = transient(g, y, np.zeros_like(g), 0.0, min_amp=0.0, spin_hz=f)
                if gi:
                    rows.append(gi)
        if rows:
            per[f] = rows

    fs = sorted(per)
    cmap = plt.get_cmap("turbo")
    col = {f: cmap(0.06 + 0.88 * i / max(len(fs) - 1, 1)) for i, f in enumerate(fs)}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), facecolor="white")
    # ANGLE on the left, RATE on the right, one metric each.
    #
    # The rate is the JUMP_DEG gradient. It replaced the 10-90% metrics on 2026-09-10 at the
    # operator's reading of the traces: the fitted
    # line is the best straight-line slope through it, and on a curved rise the fit reads
    # line is anchored on `amp`, so it is only as good as that swing is clean --
    # lower relative scatter at seven of nine frequencies (50 Hz CV 0.605 -> 0.341, 80 Hz
    # 0.105 -> 0.039) -- and comparing across frequencies needs precision more than it needs
    # the constant. The least-squares column stays in `campaign.csv` for continuity with the
    # report, which used it.
    for ax, key, name, unit in (
            (axes[0], "amp_deg", "Swing of the lean bearing", "swing (deg)"),
            (axes[1], "rate_fit_deg_s", "Alignment rate (least-squares)", "rate (deg/s)")):
        band = []
        for f in fs:
            amps = np.array([x["amp_deg"] for x in per[f]], float)
            m = np.median(amps)
            keep = np.abs(amps - m) <= MODAL_BAND_DEG
            v = np.array([x[key] for x in per[f]], float)[keep]
            if key != "amp_deg":
                v = np.abs(v)
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            jitter = (np.random.default_rng(int(f)).random(len(v)) - 0.5) * 2.0
            ax.scatter(np.full(len(v), f) + jitter, v, s=34, color=col[f], alpha=0.75,
                       edgecolor="white", linewidth=0.6, zorder=3,
                       label=f"{f:g} Hz" if ax is axes[0] else None)
            sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
            ax.errorbar([f], [float(np.mean(v))], yerr=[sd], color=col[f], lw=0,
                        elinewidth=2.0, capsize=5, capthick=2.0, zorder=4)
            ax.plot([f], [float(np.mean(v))], "_", color=col[f], ms=18, mew=2.4, zorder=5)
            band.append((f, float(np.mean(v)), sd))
        # Join the error bars into a continuous +-1 SD ribbon. Points alone invite reading a
        # trend between them that the spread may not support; the ribbon shows where the
        # frequencies are actually separated and where their scatter overlaps.
        if len(band) > 1:
            bx = np.array([b[0] for b in band], float)
            bm = np.array([b[1] for b in band], float)
            bs = np.array([b[2] for b in band], float)
            ax.fill_between(bx, bm - bs, bm + bs, color="#6b7280", alpha=0.13, lw=0,
                            zorder=1)
            ax.plot(bx, bm, color="#4b5563", lw=1.6, alpha=0.75, zorder=2)
        _style(ax, name + "  (band = +-1 SD, modal repeats)", "drive frequency (Hz)", unit)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="upper left", ncol=2)
    fig.suptitle("How far the lean bearing swings when coils A and C are cut, and how fast",
                 fontsize=12.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(out_path or (root / "report" / "rate_scatter.png"))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, facecolor="white")
    plt.close(fig)

    print(f"{'freq':>5} {'n':>3} | {'jump mean':>11} {'var':>9} {'sd':>7} "
          f"| {'fit mean':>9} {'var':>9} {'sd':>7}")
    for f in fs:
        amps = np.array([x["amp_deg"] for x in per[f]], float)
        keep = np.abs(amps - np.median(amps)) <= MODAL_BAND_DEG
        c = np.abs(np.array([x["rate_jump_deg_s"] for x in per[f]], float)[keep])
        q = np.abs(np.array([x["rate_fit_deg_s"] for x in per[f]], float)[keep])
        c, q = c[np.isfinite(c)], q[np.isfinite(q)]
        cv = float(np.var(c, ddof=1)) if len(c) > 1 else float("nan")
        qv = float(np.var(q, ddof=1)) if len(q) > 1 else float("nan")
        print(f"{f:5.0f} {len(c):3d} | {c.mean():11.2f} {cv:9.2f} {np.sqrt(cv):7.2f} "
              f"| {q.mean():9.2f} {qv:9.2f} {np.sqrt(qv):7.2f}")
    print(f"\n-> {out}")
    return out


def fit_figure(root, out_path=None, which=None, freqs=None):
    """Pooled angle-vs-time per frequency with BOTH rate fits drawn on it.

    The two estimators differ only in how they use the same 10-90% band:

    * **jump** -- the gradient of the first 10 deg rise after the kill, over its own span
      decide it. `0.8 * amp / (t90 - t10)`.
    * **least squares** -- the best-fit line through every sample between those crossings,
      about 40 of them at this sample rate.

    On a concave rise a 10-90% line is the shallower (it follows the flattening middle, while
    the fitted line follows the flattening middle), which is why the least-squares number
    the jump gradient reads only the sharp leading edge), and the two must never be mixed.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root)
    picks = []
    for idx_path in sorted(root.glob("*/index.csv")):
        idx = [r for r in csv.DictReader(open(idx_path)) if r["outcome"] == "ok"]
        if not idx:
            continue
        f = float(idx[0]["freq_hz"])
        if freqs and f not in freqs:
            continue
        curves = []
        for r in idx:
            take = Path(r["flight"])
            if not (take / "axis.csv").exists():
                continue
            t, axis, _q = load(take, which=which)
            pts = timeline(r["log"]) if Path(r["log"]).exists() else []
            rest = None
            try:
                rest = datum(t, axis, pts)[0]
            except SystemExit:
                pass
            for tk in [k for p in pts for k in p["kills"]]:
                rot = (azimuth_from_rest(t, axis, rest, tk) if rest is not None
                       else rotation_from_baseline(t, axis, tk))
                if rot is None:
                    continue
                m = (t >= tk + POOL_FROM_S - 0.2) & (t <= tk + POOL_TO_S + 0.2)
                curves.append((t[m] - tk, rot[m]))
        got = pool_transients(curves, spin_hz=f)
        if got:
            picks.append((f, got))
    if not picks:
        raise SystemExit(f"{root}: nothing to draw")

    n = len(picks)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.5 * rows),
                            facecolor="white", squeeze=False)
    for ax, (f, (grid, mean, sem, g)) in zip(axs.ravel(), picks):
        ax.fill_between(grid, mean - sem, mean + sem, color=C_TILT, alpha=0.18, lw=0)
        ax.plot(grid, mean, color=C_TILT, lw=1.8, label="pooled angle")
        t10, t90, amp = g["t10"], g["t90"], g["amp_deg"]
        if np.isfinite(t10) and np.isfinite(t90) and t90 > t10:
            y10, y90 = g["theta_from"] + 0.1 * amp, g["theta_from"] + 0.9 * amp
            ax.plot([t10, t90], [y10, y90], color=C_FIT, lw=2.4, ls="--",
                    label=f"jump {abs(g['rate_jump_deg_s']):.0f} deg/s")
            ax.plot([t10, t90], [y10, y90], "o", color=C_FIT, ms=6)
            if np.isfinite(g.get("rate_fit_deg_s", np.nan)):
                band = (grid >= t10) & (grid <= t90)
                sl, ic = np.polyfit(grid[band], mean[band], 1)
                xs = np.array([t10 - 0.15, t90 + 0.15])
                ax.plot(xs, sl * xs + ic, color=C_MARK, lw=2.4,
                        label=f"least squares {abs(sl):.0f} deg/s")
            ax.axvspan(t10, t90, color="#f2f4f7", zorder=0)
        ax.axvline(0, color="#9aa0a6", lw=1.1, ls=":")
        ax.set_xlim(-0.5, min(3.0, grid[-1]))
        _style(ax, f"{f:g} Hz  (n={g['n_pooled']})", "time from cut (s)", "angle (deg)")
        ax.legend(frameon=False, fontsize=7.5, labelcolor=MUTED, loc="lower right")
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Rate fits on the pooled angle: first 10 deg jump vs least squares over the "
                 "10-90% band", fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(out_path or (root / "report" / "rate_fits.png"))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, facecolor="white")
    plt.close(fig)
    print(f"-> {out}")
    return out


def _style(ax, title, xl, yl):
    ax.set_title(title, fontsize=11.5, color=INK, loc="left")
    ax.set_xlabel(xl, fontsize=9.5, color=MUTED)
    ax.set_ylabel(yl, fontsize=9.5, color=MUTED)
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def _nanmean(vals):
    """Mean of the finite entries, or nan. `np.nanmean` warns on an all-nan slice, and a
    frequency with no response is all-nan by design -- that is the answer, not a problem."""

    v = np.asarray([x for x in vals if x is not None], dtype=float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


def _write(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------- self-check


def _self_check():
    rng = np.random.default_rng(0)
    fs, tau_true, amp_true = 200.0, 0.8, 40.0   # 40 deg matches the measured swing
    t = np.arange(0, 60, 1 / fs)
    kills = [10.0, 30.0]
    tilt = np.full_like(t, 2.0)
    for tk in kills:
        m = t >= tk
        tilt[m] += amp_true * (1 - np.exp(-(t[m] - tk) / tau_true))
        back = t >= tk + 8.0                      # the coils come back on
        tilt[back] -= amp_true * (1 - np.exp(-(t[back] - tk - 8.0) / tau_true))
    noisy = tilt + rng.normal(0, 0.6, t.shape)
    agree = np.full_like(t, 5.0)

    r = transient(t, noisy, agree, kills[0])
    assert abs(r["amp_deg"] - amp_true) < 0.5, r["amp_deg"]
    assert abs(r["tau_s"] - tau_true) < 0.15, r["tau_s"]
    assert r["settled"], "an 8 s window on a 0.8 s tau must settle"
    # The jump gradient, on a known exponential. The first JUMP_DEG of an
    # amp*(1-exp(-t/tau)) rise is reached at t1 = -tau*ln(1 - JUMP/amp), so the mean
    # gradient across the jump is JUMP/t1. `spin_hz` is passed because the metric smooths
    # over one revolution: without it the window is SMOOTH_S, wider than the jump itself,
    # which is exactly the blunting the one-revolution rule exists to avoid.
    rj = transient(t, noisy, agree, kills[0], spin_hz=50.0)
    t1 = -tau_true * np.log(1.0 - JUMP_DEG / amp_true)
    assert abs(rj["rate_jump_deg_s"] - JUMP_DEG / t1) < 0.25 * JUMP_DEG / t1, \
        (rj["rate_jump_deg_s"], JUMP_DEG / t1)
    assert rj["jump_deg"] >= JUMP_DEG, rj["jump_deg"]
    assert rj["rate_fit_deg_s"] < 0.5 * amp_true / tau_true, \
        ("the 10-90% gradient is expected to under-read badly", rj["rate_fit_deg_s"])

    # The hinge, on a response shaped like the real one: DEAD TIME then an underdamped
    # swing that overshoots and comes back. A settling exponential is the wrong test case
    # for it -- there is no turning point, so departure-to-peak has nothing to find and the
    # measured 110 Hz trace (flat 50 ms, peak 65 deg at 200 ms, back to 18 deg by 500 ms)
    # is not that shape at all. Two things are asserted, and neither is circular: the lag is
    # recovered from an injected dead time, and the noisy fit is held to what the SAME
    # estimator reads off the noiseless curve, which tests its noise robustness rather than
    # restating the physics it was handed.
    dead, wn, zeta = 0.05, 18.0, 0.25
    wd = wn * np.sqrt(1 - zeta ** 2)
    osc = np.full_like(t, 2.0)
    md = t >= kills[0] + dead
    td = t[md] - kills[0] - dead
    osc[md] += amp_true * (1 - np.exp(-zeta * wn * td)
                           * (np.cos(wd * td) + zeta * wn / wd * np.sin(wd * td)))
    clean = transient(t, osc, agree, kills[0], min_amp=0.0, spin_hz=50.0)
    noisy_o = transient(t, osc + rng.normal(0, 0.6, t.shape), agree, kills[0],
                        min_amp=0.0, spin_hz=50.0)
    assert abs(clean["relu_lag_s"] - dead) < 0.03, (clean["relu_lag_s"], dead)
    assert abs(noisy_o["rate_relu_deg_s"] - clean["rate_relu_deg_s"]) \
        < 0.30 * clean["rate_relu_deg_s"], (noisy_o["rate_relu_deg_s"], clean["rate_relu_deg_s"])
    # and the span must stop at the overshoot, not run into the return
    assert clean["relu_win_s"] < SWING_MAX_S, clean["relu_win_s"]

    # A response smaller than the jump threshold must report no rate, not a small one.
    small = np.full_like(t, 2.0)
    m_ = t >= kills[0]
    small[m_] += 6.0 * (1 - np.exp(-(t[m_] - kills[0]) / tau_true))
    rsm = transient(t, small + rng.normal(0, 0.2, t.shape), agree, kills[0], spin_hz=50.0)
    assert not np.isfinite(rsm["rate_jump_deg_s"]), rsm
    assert "jump" in rsm.get("note", ""), rsm

    # no response must report no rate rather than a number made of noise
    flat = np.full_like(t, 2.0) + rng.normal(0, 0.6, t.shape)
    r0 = transient(t, flat, agree, kills[0])
    assert not np.isfinite(r0["rate_jump_deg_s"]), r0
    assert "no response" in r0.get("note", ""), r0

    # the log-free step detector must find both kills and not invent a third
    got = find_kills(t, noisy)
    assert len(got) == 2, f"restores must not read as kills: {got}"
    for k in kills:
        assert min(abs(g - k) for g in got) < 0.4, (k, got)

    # a step that never settles must widen its own sigma, not hide it
    slow = np.full_like(t, 2.0)
    m = t >= kills[0]
    # tau = 4 s against a 6 s window: risen enough to clear MIN_AMP_DEG, nowhere near
    # settled. (tau = 20 s was the wrong test case -- it only rises 1.55 deg inside the
    # window, so "no response" is the right answer there and the sigma path never runs.)
    slow[m] += amp_true * (1 - np.exp(-(t[m] - kills[0]) / 4.0))
    rs = transient(t, slow, agree, kills[0])
    assert not rs["settled"], rs
    assert "extrapolation" in rs.get("note", ""), rs

    print("alignment_rate: self-check passed (tau, rate, no-response, kill finder, sigma)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take", nargs="?")
    ap.add_argument("--log", default=None, help="sweep.log (default: found beside the take)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sweep", default=None,
                    help="a results/alignment_rate/<stamp> root: solve and pool every take")
    ap.add_argument("--stride", type=int, default=3, help="solve every Nth frame (--sweep)")
    ap.add_argument("--campaign", default=None,
                    help="a results/alignment_rate/<stamp> root: pool every chunk")
    ap.add_argument("--which", default=None, choices=("minor", "conic"))
    ap.add_argument("--metric", default="azimuth", choices=("azimuth", "rotation"),
                    help="azimuth = lean-direction change (recommended); rotation = total")
    ap.add_argument("--fits", default=None,
                    help="a campaign root: pooled angle with both rate fits drawn")
    ap.add_argument("--trials", default=None,
                    help="a campaign root: one representative trial per frequency, fit drawn")
    ap.add_argument("--scatter", default=None,
                    help="a campaign root: rate-vs-frequency scatter, coloured per frequency")
    ap.add_argument("--plot", action="store_true", help="tilt-vs-time png for one take")
    ap.add_argument("--no-log", action="store_true",
                    help="ignore any sweep.log and find the kills in the data")
    a = ap.parse_args()
    if a.fits:
        fit_figure(a.fits, out_path=a.out, which=a.which)
        sys.exit()
    if a.trials:
        trial_panels(a.trials, out_path=a.out, which=a.which, metric=a.metric)
        sys.exit()
    if a.scatter:
        rate_scatter(a.scatter, out_path=a.out, which=a.which, metric=a.metric)
        sys.exit()
    if a.campaign:
        campaign(a.campaign, out_dir=a.out, which=a.which, metric=a.metric)
        sys.exit()
    if a.sweep:
        sweep_report(a.sweep, out_dir=a.out, stride=a.stride)
        sys.exit()
    if a.plot:
        plot_take(a.take, log_path=a.log, out_path=a.out, use_log=not a.no_log)
        sys.exit()
    if a.take is None:
        _self_check()
        sys.exit()
    report(a.take, log_path=a.log, out_dir=a.out, use_log=not a.no_log)
