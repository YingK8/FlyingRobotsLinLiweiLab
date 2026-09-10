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

#: Radial lean (deg) below which the azimuth is DROPPED rather than believed. Azimuth is the
#: direction of lean, so its noise is sigma/sin(radial) and it has a coordinate singularity at
#: the datum -- with the measured 3.18 deg of per-frame axis scatter that is 6 deg/frame of
#: azimuth noise at 30 deg of lean but 61 at 3 deg, which at 19.6 ms sampling is 3100 deg/s.
#: That is the whole explanation for the near-vertical spikes: 82.4% of physically impossible
#: jumps sit below 5 deg of lean against 5.5% of samples, a 15x enrichment.
#:
#: It is NOT a hemisphere or sign error, which was the first hypothesis and was tested and
#: rejected: only 0.75% of frames sit below the rest datum, they sit at LARGE lean (median
#: 39.2 deg), take 2026-09-09_212646 has zero of them and still shows both its jumps, and
#: enforcing the constraint a priori removes 7% of jumps and leaves 93%.
#:
#: 5 deg is the knee of the cost curve: 1 deg drops 25% of jumps for 0.7% of samples, 5 deg
#: drops 82% for 5.5%, 8 deg drops 96% for 16.8%. `disc_video` uses 1.0 for the same idea,
#: which is "meaningless" rather than "noisier than the signal".
AZIMUTH_MIN_RADIAL_DEG = 5.0

#: Sample-to-sample azimuth DISPLACEMENT (deg) above which the step is not physical and the
#: ramp containing it is refused. A displacement and not a rate: at stride 1 the record is
#: solved every 4.3 ms, so any rate threshold that made sense at stride 4 now condemns the
#: real event (see `relu_window`). The corrupt samples this exists for jump 100-280 deg
#: between neighbours, while the fastest genuine step measured across the campaign is 16 deg
#: -- so 45 sits in a wide gap rather than on a judgement call, and it stays put whatever
#: stride the take was solved at. The AZIMUTH_MIN_RADIAL_DEG mask removes 82% of these;
#: this refuses to fit the rest instead of reporting a gradient made of one corrupt sample.
IMPOSSIBLE_STEP_DEG = 45.0

#: How long after the cut to look for the response's PEAK. It has to be wide enough to hold
#: a delayed rise -- 60 Hz takes 2026-09-09_213341 and 2026-09-10_000608 do nothing for
#: 220-270 ms and then climb to 175 deg -- and short enough that the oscillation's SECOND
#: swing cannot be mistaken for the first. Both hold at 1 s: the first overshoot of an
#: underdamped response is always the largest, and measured across this campaign the return
#: swings come back to 60-80% of it, peaking around 500 ms.
PEAK_MAX_S = 1.0

#: How high a swing must reach, as a fraction of the response's full height in the window,
#: to count as THE peak rather than a preliminary hump.
#:
#: It only has to clear the humps, and they are small: the ones that used to capture the fit
#: are 10 deg of a 178 deg record. Set high it does damage instead, because the FIRST swing
#: is not always the tallest -- 30 Hz take 2026-09-10_011809 peaks at ~50 deg and a later
#: swing reaches 77, so at 0.7 the crossing skipped the response entirely and fitted the
#: second rising edge with a 367 ms "dead time".
#:
#: Scanned over all 101 kill windows in the campaign (2026-09-10). Fits landing later than
#: 200 ms after the cut go 5, 4, 3, 2, 1, 1, 1, 1 as the fraction falls 0.70 -> 0.20, while
#: the number of windows fitted stays at 88-90 and the median rate does not move off
#: 740 deg/s. So it is a plateau, not a trade, and 0.30 sits in the middle of it -- 3x the
#: hump level and well below the smallest genuine first swing.
PEAK_FRAC = 0.30

#: How long after that crossing to look for the swing's actual top. Long enough to contain
#: the last of the rise, short enough that the following swing cannot get in.
PEAK_SETTLE_S = 0.25

#: Where the foot of the rising edge sits, as a fraction of the rise above its minimum.
#: Small on purpose: the fit is meant to cover the edge, not just its steepest middle, and
#: `BUFFER_FRAC` already trims the ends afterwards. 10% skips the part of the record that is
#: still flat without eating into the edge itself.
FOOT_FRAC = 0.10

#: How far before the cut a fitted ramp's extrapolated onset may land before the trial is
#: refused. Physically the answer is zero -- nothing responds before the step -- so this is
#: purely the fit's own noise on the intercept: two samples at the stride-1 rate of 231 Hz.
LAG_TOL_S = 0.010


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
    """Lean DIRECTION about the rest datum, as an angle from its pre-cut direction.

    This, not the total rotation, is what the coil cut actually changes. Measured across the
    campaign, the azimuth swings 41-50 deg at EVERY frequency from 10 to 90 Hz -- MAD 0.83 deg
    at 20 Hz, 1.06 at 40 -- while the radial tilt barely moves and if anything goes slightly
    negative (the robot ends a little more upright).

    ~45 deg is what the coil geometry predicts: with A and C at 0 and 180 deg removed, the
    remaining asymmetry from B and D at 90/270 sits 45 deg away. So the number is a geometric
    constant being recovered, not a drive-dependent response, which is why it does not scale
    with frequency the way `rotation_from_baseline` appears to. That apparent growth is mostly
    geometry: a fixed azimuth swing sweeps a longer great-circle arc at a larger radial tilt.

    RETURNED VALUES ARE IN [0, 180] AND NEVER NEGATIVE
    --------------------------------------------------
    The number is the angle BETWEEN two directions in the plane about the datum -- where the
    robot leans now, and where it leaned in the second before the cut. An angle between two
    directions is by definition in [0, 180]: there is no sense to it, because the robot lives
    in the upper hemisphere and the field left by cutting A and C is symmetric about the A-C
    line, so a lean that swings one way and one that swings the other are the same event seen
    from opposite sides.

    It is computed as that angle DIRECTLY -- an arccos of two unit vectors -- and not by
    unwrapping an `arctan2` and referencing it afterwards, which is how it was done until
    2026-09-10. `unwrap` puts each take on its own arbitrary +-360 branch, so pooling repeats
    mixed 54 deg with -306; wrapping the referenced angle back into +-180 to fix that then
    put a discontinuity at exactly 180 deg, and a swing that genuinely reached it jumped the
    full 360 -- which is the 50 Hz take 2026-09-09_212525 diving from +140 to -120 deg
    between two samples, and every other reported sign flip. An arccos has no branch to
    choose, so there is nothing left to flip: the angle simply turns around at 180 and comes
    back, which is what the geometry does too.

    The cost, stated plainly: near zero the measurement is rectified, so a zero-mean wobble
    in the pre-cut window reads as a small positive floor of about 0.8 sigma rather than as
    zero. That is a property of the quantity, not of this estimator -- an unsigned angle
    cannot be negative -- and it is why the pre-cut level is taken as a median over the whole
    window rather than as an assumed zero.
    """

    up = np.asarray(up, float)
    up = up / np.linalg.norm(up)
    a = orient_continuous(axis, up)

    # The lean direction: the axis with its component along the datum removed, normalised.
    # This is a unit vector in the plane perpendicular to `up`, and its direction IS the
    # azimuth -- carried as a vector so no angle is ever unwrapped.
    u = a - np.outer(a @ up, up)
    n = np.linalg.norm(u, axis=1)

    # Samples where the lean is too small for its direction to mean anything are dropped
    # BEFORE the reference is built and interpolated across afterwards. Near the datum the
    # direction's noise goes as sigma / sin(lean), so a nearly-upright sample points
    # essentially at random; below 5 deg that is 82% of the wild excursions in the record.
    lean = tilt_from(a, up)
    ok = (lean >= AZIMUTH_MIN_RADIAL_DEG) & (n > 1e-9)
    if ok.sum() < 20:
        return None
    u = u / np.maximum(n, 1e-12)[:, None]

    m = (t >= t_kill - pre_s) & (t < t_kill) & ok
    if m.sum() < 20:
        return None
    # The reference direction is the MEAN pre-cut lean direction, not one sample of it.
    ref = u[m].mean(0)
    nr = float(np.linalg.norm(ref))
    if nr < 1e-6:            # the pre-cut lean direction never settled -- no reference exists
        return None
    ref /= nr

    dev = np.full(len(t), np.nan)
    dev[ok] = np.degrees(np.arccos(np.clip(u[ok] @ ref, -1.0, 1.0)))
    if (~ok).any():
        dev = np.interp(t, t[ok], dev[ok])
    return dev


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


def relu_window(ts, ys, t_kill, amp_sign=None, sd_base=None, allow_early=False):
    """The rising edge of the response: trough-before-peak to peak. ``(slope, lag, span)``.

    THE RULE
    --------
    Find the PEAK of the response -- the largest value the smoothed trace reaches within
    ``PEAK_MAX_S`` of the cut. Walk back to the last turning point before it, which is the
    trough the rise starts from. That trough-to-peak segment IS the alignment event. Trim
    ``BUFFER_FRAC`` off each end so the turning points themselves do not flatten the line,
    fit the middle, and extrapolate back to the pre-cut level for the dead time.

    There is no search, no candidate list and no ranking: the peak is unique and the trough
    before it is determined by the peak, so the segment is a function of the data alone.

    WHY IT IS ANCHORED ON THE PEAK AND NOT ON THE FIRST QUALIFYING RISE
    ------------------------------------------------------------------
    The previous rule (2026-09-10, morning) took the FIRST consecutive pair of turning points
    differing by at least ``JUMP_DEG``. Two failure modes, both found by the operator reading
    the per-repeat panels, and both fixed by anchoring on the peak instead:

    * **Initial humps win by being first.** 60 Hz take 2026-09-09_213341 rises 18 deg in a
      small early bump, falls back, and only then makes its real 175 deg climb; the old rule
      fitted the bump at 246 deg/s. 2026-09-10_000608 is the same shape and read 182. The
      peak of the record is not in either bump, so neither can be selected now.
    * **A rough edge gets chopped into slivers.** At 120 Hz the rising edge carries enough
      scatter to put turning points inside itself, and the first pair between two of them is
      a fraction of the edge -- 2026-09-09_213218 was fitted over 20 ms of a 190 ms rise.
      Intermediate turning points are simply passed over now, because the segment runs from
      the trough to the peak whatever happens between them.

    It also retires two thresholds that existed only to prop the old rule up: ``STEEP_FRAC``
    (a steepness bar to reject shallow first pairs) and ``MAX_LAG_S`` (a bar on how late a
    ramp could start, which was rejecting the real rise in exactly the two takes above).

    WHAT IS STILL REFUSED
    ---------------------
    A rise smaller than ``JUMP_DEG``; a segment containing a physically impossible
    sample-to-sample step; and a fit whose extrapolated onset lands before the cut, which
    says the rise was already underway when the coils dropped.
    """

    ts, ys = np.asarray(ts, float), np.asarray(ys, float)
    post = np.flatnonzero(ts >= t_kill)
    if post.size < 5:
        return float("nan"), float("nan"), float("nan")
    i0 = int(post[0])
    y, x = ys[i0:], ts[i0:]

    # The peak of the response: the FIRST swing that reaches the height of the record, not
    # the tallest sample in it. A plain argmax over the window assumes the first overshoot is
    # strictly the largest, and when the damping is light enough the later swings come within
    # a percent of it -- 60 Hz take 2026-09-09_213341 peaks at 175 deg at 480 ms and again at
    # 179 at 873, so argmax chose the third swing and the segment straddled a whole down-up
    # cycle, fitting a rising 162 deg edge at MINUS 144 deg/s.
    #
    # So: find where the trace first reaches `PEAK_FRAC` of the window's full height, then
    # take the largest sample in the `PEAK_SETTLE_S` that follow. The fraction is what makes
    # an early hump ineligible -- 213341's hump is 18 deg of a 178 deg record, nowhere near
    # 70% -- and the short settle window is what makes "the peak of that swing" robust to
    # scatter without needing a local-maximum test that noise would trip.
    win = np.flatnonzero(x <= t_kill + PEAK_MAX_S)
    if win.size < 5:
        return float("nan"), float("nan"), float("nan")
    yw = y[win]
    lo, hi = float(yw.min()), float(yw.max())
    if hi - lo < JUMP_DEG:
        return float("nan"), float("nan"), float("nan")
    cross = int(win[int(np.argmax(yw >= lo + PEAK_FRAC * (hi - lo)))])
    settle = np.flatnonzero((x >= x[cross]) & (x <= x[cross] + PEAK_SETTLE_S))
    q_ = int(settle[int(np.argmax(y[settle]))]) if settle.size else cross
    if q_ < 2:
        return float("nan"), float("nan"), float("nan")

    # The trough the rise starts from: the LAST time before the peak that the trace was at
    # its lowest, within a tolerance. Not "walk back while the trace is non-increasing" --
    # that was tried first and it terminates after two samples on anything noisy, because
    # noise alone breaks monotonicity; it returned no rate at all on the self-check's noisy
    # underdamped case. Taking the last visit to the minimum instead is what makes it the
    # start of the FINAL approach to the peak: on 60 Hz take 2026-09-09_213341 the trace sits
    # near its minimum at the cut AND again at 215 ms after the early hump falls back, and
    # only the later one begins the 175 deg climb.
    #
    # "At its lowest" is the FOOT of the rise, not the exact minimum. A trace that dips
    # without returning all the way down never satisfies an exact-minimum test, and 60 Hz
    # take 2026-09-10_000608 is that shape: it sits at 0 at the cut, humps to 18 deg, falls
    # back only to 5, then climbs to 178. Anchored on the exact minimum the foot stayed at
    # the cut and the fit was dragged across 490 ms of mostly-flat record at 210 deg/s.
    #
    # The band is the larger of two scales, so neither a quiet record nor a noisy one breaks
    # it: twice the pre-cut scatter of this same smoothed trace, and FOOT_FRAC of the rise.
    mn = float(np.min(y[:q_ + 1]))
    tol = FOOT_FRAC * (y[q_] - mn)
    if sd_base is not None and np.isfinite(sd_base) and sd_base > 0:
        tol = max(tol, 2.0 * sd_base)
    low = np.flatnonzero(y[:q_ + 1] <= mn + tol)
    p_ = int(low[-1]) if low.size else 0

    # And the top of the edge is the FIRST entry into the top band, mirroring the foot. The
    # response often reaches its level and then sits there: 30 Hz take 2026-09-10_011809 is
    # flat until 76 ms, climbs 10 -> 46 deg by 132 ms, and then PLATEAUS at 45-49 deg until
    # 216 ms. Taking the highest sample put the end of the edge at 213 ms, so the fit spent
    # most of its span on the plateau and read 167 deg/s against a 625 deg/s rise -- and the
    # shallow line, run backwards, crossed the pre-cut level 85 ms BEFORE the cut, which got
    # the trial refused for a dead time it never had.
    top = np.flatnonzero(y[p_:q_ + 1] >= y[q_] - tol)
    if top.size:
        q_ = p_ + int(top[0])
    if q_ - p_ < 2 or y[q_] - y[p_] < JUMP_DEG:
        return float("nan"), float("nan"), float("nan")

    # A rise is only a rise if every step inside it is physical -- one corrupt sample would
    # otherwise BE the peak and drag the segment to itself.
    steps = np.abs(np.diff(y[p_:q_ + 1]))
    if steps.size and steps.max() > IMPOSSIBLE_STEP_DEG:
        return float("nan"), float("nan"), float("nan")

    # Trim the buffer, then fit the middle. Backed off if the rise is too short to spare it
    # -- a biased slope beats no slope, and the bias is toward under-reading.
    n_ = q_ - p_ + 1
    cut = int(round(BUFFER_FRAC * n_))
    while cut > 0 and n_ - 2 * cut < 4:
        cut -= 1
    a_, b_ = p_ + cut, q_ - cut
    slope, icept = (float(v) for v in np.polyfit(x[a_:b_ + 1], y[a_:b_ + 1], 1))
    if slope <= 0:
        return float("nan"), float("nan"), float("nan")

    # Dead time by extrapolation, not by "the first sample that moved". Run the fitted ramp
    # back to where it crosses the level the trace held before it, and the lag is that
    # crossing minus the cut. Reading it off the first moving sample instead makes it a
    # function of where the samples happen to fall and of whichever noise excursion crossed
    # first; a two-line intersection uses every sample in both. The level spans the pre-cut
    # baseline and the flat dead time, since it runs from the cut to the rise's own start.
    flat = (ts >= t_kill - PRE_S) & (ts <= x[p_])
    lag = float("nan")
    if flat.sum() >= 3:
        lag = float((float(np.median(ys[flat])) - icept) / slope - t_kill)

    # A dead time cannot be negative: the response does not precede the cut. When the
    # extrapolation lands before t_kill the rise was ALREADY UNDERWAY when the coils dropped,
    # so it is not the response to them and the trial is refused rather than credited with a
    # rate -- 50 Hz take 2026-09-09_234425 was a -94 ms "dead time" with its fitted ramp drawn
    # entirely left of zero. It doubles as the quiescence precondition, which is why there is
    # no separate flatness test. The tolerance is two samples of fit noise, not an allowance.
    #
    # `allow_early` scopes it to SINGLE repeats, which is the only place the question makes
    # sense. On a pooled curve a negative lag is not evidence that any run responded early --
    # it is the averaging: repeats land 33-102 ms apart, so the mean edge is smeared
    # symmetrically and starts a median of 40 ms before the cut even though no repeat did.
    # Applying a per-repeat precondition to an average refused the pooled fit at 10 of 11
    # frequencies; the pooled rate is reported with its lag as measured instead.
    if np.isfinite(lag) and lag < -LAG_TOL_S and not allow_early:
        return float("nan"), float("nan"), float("nan")
    return slope, lag, float(x[q_] - x[p_])


def transient(t, tilt, agree, t_kill, min_amp=MIN_AMP_DEG, spin_hz=None, pooled=False):
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
    # THE RESPONSE GOES UP. Both metrics fed to this function are non-negative angular
    # separations from an attitude the robot held before the cut -- `azimuth_from_rest` from
    # the pre-cut lean DIRECTION, `rotation_from_baseline` from the pre-cut axis -- so the
    # only thing the coil cut can do to either is increase it. The sign was inferred from the
    # data until 2026-09-10 and it was a liability rather than a degree of freedom: on 10 Hz
    # take 2026-09-09_210122 the largest early excursion was downward, which pointed the ramp
    # search at a falling limb and returned -522 deg/s.
    #
    # A dominant DOWNWARD excursion is therefore not a swing to be followed -- it is the
    # trial telling us the pre-cut window was not a settled reference. If the lean direction
    # was still moving through that second, the mean of it points somewhere the robot never
    # sat, the trace starts high and falls toward it, and there is no alignment event to
    # measure. That trial is refused rather than fitted, the same way a ramp that starts
    # before the cut is (see `relu_window`). 210122 is exactly this: it drops 40 deg in the
    # 76 ms after the cut.
    early = (ts >= t_kill) & (ts <= t_kill + SWING_MAX_S)
    sgn, ref_settled = 1.0, True
    if early.sum() >= 5:
        dev = ys_j[early] - theta_from
        far = float(dev[np.argmax(np.abs(dev))])
        if np.isfinite(far) and far < 0:
            ref_settled = False
            out["note"] = (f"pre-cut lean direction not settled -- the trace falls "
                           f"{abs(far):.1f} deg below it within {SWING_MAX_S:.2f} s of the "
                           f"cut, so there is no reference to align away from")

    # The headline rate: a hinge pinned at the known cut instant. It spans the kill, so it
    # needs the flat pre-kill samples too -- `w` above starts AT the kill and would leave the
    # intercept resting on the ramp alone.
    w2 = (t >= t_kill - PRE_S - 0.2) & (t <= t_kill + POST_TO_S)
    if ref_settled and w2.sum() >= 12:
        ys2 = _smooth(t[w2], tilt[w2], span_j)
        # The threshold must be the scatter of the trace the search actually reads.
        # `baseline_sd_deg` is measured on the RAW tilt, which carries the full once-per-rev
        # wobble -- several degrees at low frequency -- so using it here set a departure
        # threshold (and a peak floor of 3x it) far above anything the smoothed trace does,
        # and 10, 40 and 50 Hz reported no rate on every repeat.
        pre2 = (t[w2] >= t_kill - PRE_S) & (t[w2] < t_kill)
        sd2 = float(ys2[pre2].std()) if pre2.sum() >= 5 else out["baseline_sd_deg"]
        m_relu, lag, win = relu_window(t[w2], ys2, t_kill, sgn, sd_base=sd2,
                                       allow_early=pooled)
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
                    min_amp=max(3.0 * sem_med, 0.3), spin_hz=spin_hz, pooled=True)
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
    skipped_nyquist = []
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
            # NYQUIST GATE. The once-per-rev wobble sits AT the drive frequency and is
            # large -- 18.2 deg in azimuth at 40 Hz against a ~45 deg swing -- and
            # `rev_window` nulls it with a boxcar of a whole number of revolutions, a null
            # that exists only if the revolution is RESOLVED. A take solved below 2*f has
            # that wobble folded onto |f - n*fs| where the boxcar cannot touch it: at 40 Hz
            # a stride-4 solve (51 Hz) folds it to 11.1 Hz, whose 92 ms period is comparable
            # to the ~150 ms ramp being measured, and the fitted rate came out 2.1x high.
            #
            # The take is kept on disk and in index.csv -- this refuses to average a
            # corrupted rate into the frequency, nothing more. The only takes this can catch
            # are ones whose video was deleted on 2026-09-10 and so cannot be re-solved.
            fs_take = 1.0 / float(np.median(np.diff(t))) if len(t) > 2 else 0.0
            if fs_take < 2.0 * f:
                skipped_nyquist.append((f, take.name, fs_take))
                continue
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
        if skipped_nyquist:
            print(f"  {len(skipped_nyquist)} take(s) REFUSED -- solved below 2x the drive "
                  f"frequency, so the once-per-rev wobble is aliased (kept on disk):")
            for f_, n_, fs_ in skipped_nyquist:
                print(f"    {f_:5.0f} Hz  {n_}  solved {fs_:5.1f} Hz, needs {2*f_:.0f}")
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
    _style(top, "Azimuthal swing $\\Delta\\theta_{az}$ after the cut",
           "time from the cut (s)", "$\\Delta\\theta_{az}$ (deg)")
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
        rr = np.array([x_["rate_relu_deg_s"] for x_ in pooled[s_["freq_hz"]][4]], float)
        rr = rr[np.isfinite(rr)]
        m_ = float(np.median(rr)) if len(rr) else float("nan")
        rmed.append(m_)
        rmad.append(float(np.median(np.abs(rr - m_))) * 1.4826 if len(rr) else float("nan"))
        for v in rr:
            a2.plot([s_["freq_hz"]], [v], "o", color=C_FIT, ms=3, alpha=0.35)
    a2.errorbar(x, rmed, yerr=rmad, color=C_FIT, lw=2, marker="o", ms=6, capsize=3,
                elinewidth=1.4, zorder=3)
    # The CURRENT metric. This panel plotted `rate_jump_deg_s` -- the gradient of the first
    # 10 deg of rise -- until 2026-09-10, and before that the 10-90% band its title still
    # named; both are retired and neither agreed with `campaign.csv`, which is the table a
    # reader compares this figure against.
    _style(a2, "Alignment rate, rising edge  (median +- MAD)",
           "drive frequency (Hz)", "rate (deg/s)")
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


def trial_panels(root, out_path=None, which=None, metric="azimuth", only_hz=None):
    """One CHOSEN trial per frequency with the fit drawn on it, not a pooled average.

    With ``only_hz`` set, EVERY repeat at that one frequency is drawn instead, one panel per
    take, in the order they were flown. That is the figure to reach for when a frequency's
    rate scatters (50 Hz sits at CV 0.63 against 0.12 at 80): the per-frequency summary can
    only say the spread is large, while the panels say which repeats caused it and whether
    the cause is the fit or the flight.

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
        if only_hz is not None and abs(f - only_hz) > 0.5:
            continue
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
        if only_hz is not None:
            picks.extend(sorted(cand, key=lambda c: c[1]))
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
        # The AZIMUTH ITSELF, not the azimuth minus its pre-cut level. Subtracting the level
        # to put the baseline on zero draws every dip below it as a negative angle, which the
        # quantity cannot be -- it is a separation between two directions, in [0, 180]. The
        # pre-cut level is drawn where it actually sits instead.
        pre_lvl = float(np.median(rot[(rel >= -PRE_S) & (rel < 0)]))
        ax.plot(rel[m] * 1e3, rot[m], lw=0.8, color=C_RAW, label="raw", zorder=1)
        ax.plot(rel[m] * 1e3, ys, lw=1.8, color=C_SM,
                label="smoothed (1 rev)", zorder=3)

        ax.axhline(pre_lvl, color=C_PRE, lw=1.4, ls="--", zorder=2,
                   label="pre-cut tilt (mean)")
        ax.set_ylim(bottom=0.0)
        lag, span = res["relu_lag_s"], res["relu_win_s"]
        rate = res["rate_relu_deg_s"]
        sgn, peak_dev = 1.0, None
        w_pk = (rel >= 0) & (rel <= SWING_MAX_S)
        if w_pk.sum():
            dev = (ys - pre_lvl)[(rel[m] >= 0) & (rel[m] <= SWING_MAX_S)]
            if dev.size:
                peak_dev = float(dev[np.argmax(np.abs(dev))])
                sgn = 1.0 if peak_dev > 0 else -1.0
                ax.axhline(pre_lvl + peak_dev, color=C_POST, lw=1.4,
                           ls="--", zorder=2, label="post-cut tilt (swing peak)")

        ax.axvline(0.0, color=C_CUT, lw=1.6, ls="--", zorder=4, label="A,C off")
        if np.isfinite(lag) and np.isfinite(span) and np.isfinite(rate):
            t0 = lag
            # Stop the drawn line at the swing peak. The fitted slope is steeper than the
            # chord across the ramp by design -- BUFFER_FRAC trims the shoulders and fits the
            # middle -- so a line drawn over the ramp's full span climbs well above anything
            # the robot did, which reads as a bad fit rather than as a deliberately trimmed
            # one.
            span_d = span
            if peak_dev and abs(rate) > 0:
                span_d = min(span, abs(peak_dev) / abs(rate))
            xs = np.linspace(t0, t0 + span_d, 40)
            ax.plot(xs * 1e3, pre_lvl + sgn * rate * (xs - t0), lw=2.4, color=C_RAMP,
                    zorder=5, label="fitted ramp")
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
            ax.set_ylabel("azimuth from pre-cut direction (deg)", fontsize=9)
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
    if only_hz is not None:
        rr = np.array([p_[5]["rate_relu_deg_s"] for p_ in picks], float)
        fig.suptitle(f"{only_hz:g} Hz -- every repeat, alignment fit drawn "
                     f"(n={len(picks)}, rate {np.median(rr):.0f} +- "
                     f"{np.median(np.abs(rr - np.median(rr))):.0f} deg/s MAD)", fontsize=12)
    else:
        fig.suptitle("Alignment fit on one representative trial per drive frequency "
                     "(median-rate trial, not an average)", fontsize=12)
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    default = ("trial_fits.png" if only_hz is None else f"trials_{only_hz:03.0f}hz.png")
    out = Path(out_path or (root / "report" / default))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")
    return out


def rate_scatter(root, out_path=None, which=None, metric="azimuth"):
    """Swing and alignment rate against drive frequency: every repeat, coloured by frequency.

    Left panel is the azimuthal swing $\\Delta\\theta_{az}$, right panel is the rate: the gradient of the
    first steep ramp of at least JUMP_DEG after the coils drop, which is the operator's
    definition and the only rate this module now reports. The two 10-90% metrics that used to
    share this figure are gone -- the crossing chord on 2026-09-10, the least-squares slope
    over the same band with it. Both were anchored on `amp`, the median over 4-6 s after the
    cut, and the response does not settle there: it overshoots and swings back, so `amp`
    could sit near the baseline or on the wrong side of it, and on a synthetic step with a
    known answer the least-squares band read 17.1 deg/s against a true 50.0.

    Points are jittered in x only, so a column's vertical spread is the real repeat-to-repeat
    scatter at that frequency.
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
                # On the take's OWN timebase, unsmoothed. Resampling to the 10 ms pooling
                # grid and pre-smoothing with `rev_window` (0.25 s at every frequency) is
                # what `campaign` used to do and it was removed there for the same reason:
                # the boxcar is 3-6x wider than the 40-80 ms event, so it read gradients
                # several times too low -- 85-200 deg/s here against the campaign's
                # 355-1437 on the very same takes. `transient` does its own smoothing, one
                # revolution wide, which is the narrowest window that still nulls the wobble.
                gi = transient(t, rot, np.zeros_like(t), tk, min_amp=0.0, spin_hz=f)
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
            (axes[0], "amp_deg", "Azimuthal swing $\\Delta\\theta_{az}$", "$\\Delta\\theta_{az}$ (deg)"),
            (axes[1], "rate_relu_deg_s", "Alignment rate (first steep ramp)",
             "rate (deg/s)")):
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
    fig.suptitle("Azimuthal swing $\\Delta\\theta_{az}$ when coils A and C are cut, and how fast",
                 fontsize=12.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(out_path or (root / "report" / "rate_scatter.png"))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, facecolor="white")
    plt.close(fig)

    print(f"{'freq':>5} {'n':>3} | {'swing':>7} {'sd':>6} "
          f"| {'rate':>8} {'sd':>7} {'CV':>5} | {'dead ms':>7} {'sd':>5}")
    for f in fs:
        amps = np.array([x["amp_deg"] for x in per[f]], float)
        keep = np.abs(amps - np.median(amps)) <= MODAL_BAND_DEG
        a = amps[keep]
        r = np.abs(np.array([x["rate_relu_deg_s"] for x in per[f]], float)[keep])
        d = np.array([x["relu_lag_s"] for x in per[f]], float)[keep] * 1e3
        r, d = r[np.isfinite(r)], d[np.isfinite(d)]
        rs = float(np.std(r, ddof=1)) if len(r) > 1 else float("nan")
        print(f"{f:5.0f} {len(r):3d} | {a.mean():7.2f} {a.std():6.2f} "
              f"| {r.mean():8.1f} {rs:7.1f} {rs / max(r.mean(), 1e-9):5.2f} "
              f"| {d.mean():7.1f} {d.std():5.1f}")
    print(f"\n-> {out}")
    return out


def fit_figure(root, out_path=None, which=None, freqs=None):
    """Pooled angle per frequency with the rate fit on it, AND what pooling costs.

    This is not a second way of looking at `--trials`. It answers one question that per-trial
    panels cannot: is the pooled curve safe to fit? Every panel carries both numbers -- the
    rate the current rule reads off the POOLED curve, and the median of the rates it reads
    off the INDIVIDUAL repeats -- so the penalty is printed rather than argued about.

    It is expected to be negative, and the reason is worth keeping in view. Repeats do not
    share a dead time to the millisecond (measured here: 59 ms median, but 33-102 ms across
    frequencies), so averaging them smears the corner the fit is built on and the pooled edge
    is gentler than any single run actually flew. That is precisely why the headline numbers
    in `campaign.csv` are per-repeat and this figure is a diagnostic.

    Both numbers come from the same estimator, so the difference between them is the smearing
    and nothing else. Until 2026-09-10 this figure drew two RETIRED metrics instead -- the
    first-10-deg jump gradient and a least-squares line over the 10-90% band -- and its
    caption compared them to each other, which is a comparison of two things neither of which
    is reported any more.
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
        curves, singles = [], []
        for r in idx:
            take = Path(r["flight"])
            if not (take / "axis.csv").exists() and not (take / "axis_minor.csv").exists():
                continue
            t, axis, _q = load(take, which=which)
            if len(t) > 2 and 1.0 / float(np.median(np.diff(t))) < 2.0 * f:
                continue                      # aliased -- see the Nyquist gate in `campaign`
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
                # the same estimator on the single repeat, for the comparison
                one = transient(t, rot, np.zeros_like(t), tk, min_amp=0.0, spin_hz=f)
                if one and np.isfinite(one.get("rate_relu_deg_s", np.nan)):
                    singles.append(float(one["rate_relu_deg_s"]))
        got = pool_transients(curves, spin_hz=f)
        if got:
            # When does the POOLED curve start moving? Measured against its own pre-cut
            # level and scatter, so it is the pooling artifact and not a threshold choice.
            grid_, mean_ = got[0], got[1]
            ys_ = _smooth(grid_, mean_, 1.0 / f)
            pre_ = (grid_ >= -PRE_S) & (grid_ < 0)
            lvl_ = float(np.median(ys_[pre_]))
            sd_ = float(ys_[pre_].std())
            up_ = np.flatnonzero((grid_ > -PRE_S) & (ys_ > lvl_ + 2.0 * sd_))
            onset = float(grid_[up_[0]]) if up_.size else float("nan")
            picks.append((f, got, float(np.median(singles)) if singles else float("nan"),
                          len(singles), onset))
    if not picks:
        raise SystemExit(f"{root}: nothing to draw")

    n = len(picks)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.5 * rows),
                            facecolor="white", squeeze=False)
    costs, onsets, fitted = [], [], 0
    for ax, (f, (grid, mean, sem, g), med_single, n_single, onset) in zip(axs.ravel(), picks):
        if np.isfinite(onset):
            onsets.append(onset)
        ax.fill_between(grid, mean - sem, mean + sem, color=C_TILT, alpha=0.18, lw=0)
        ax.plot(grid, mean, color=C_TILT, lw=1.8, label="pooled angle")
        rate, lag, span = (g.get("rate_relu_deg_s", np.nan), g.get("relu_lag_s", np.nan),
                           g.get("relu_win_s", np.nan))
        if np.isfinite(rate) and np.isfinite(lag) and np.isfinite(span):
            lvl = g["theta_from"]
            xs = np.linspace(lag, lag + span, 40)
            ax.plot(xs, lvl + rate * (xs - lag), color=C_FIT, lw=2.4,
                    label=f"pooled fit {rate:.0f} deg/s")
            ax.axvspan(0.0, max(lag, 0.0), color="#f2f4f7", zorder=0)
            fitted += 1
        else:
            ax.plot([], [], " ", label="pooled fit REFUSED")
        if np.isfinite(onset):
            ax.axvline(onset, color=C_MARK, lw=1.2, ls="--")
            ax.plot([], [], " ", label=f"pooled edge starts {onset * 1e3:+.0f} ms")
        if np.isfinite(med_single):
            ax.plot([], [], " ", label=f"per-repeat median {med_single:.0f} deg/s")
            if np.isfinite(rate) and med_single > 0:
                costs.append(rate / med_single - 1.0)
        ax.axvline(0, color="#9aa0a6", lw=1.1, ls=":")
        ax.set_xlim(-0.3, 0.8)
        _style(ax, f"{f:g} Hz  (pooled n={g['n_pooled']}, repeats n={n_single})",
               "time from cut (s)", "angle from pre-cut direction (deg)")
        ax.legend(frameon=False, fontsize=7.5, labelcolor=MUTED, loc="lower right")
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    om = np.median(onsets) * 1e3 if onsets else float("nan")
    cm = np.median(costs) * 100 if costs else float("nan")
    fig.suptitle(
        f"Rate fit on the POOLED angle ({fitted} of {n} frequencies), against the median of "
        f"the individual repeats\n"
        f"Pooling reads {cm:+.0f}% of the per-repeat rate; pooled edge starts {om:+.0f} ms "
        "from the cut (median).\nRepeats land 33-102 ms apart, so averaging smears an edge "
        "that is itself only ~50 ms wide -- which also flattens the rate's variation with "
        "drive frequency.",
        fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = Path(out_path or (root / "report" / "rate_fits.png"))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, facecolor="white")
    plt.close(fig)
    print(f"pooled fit accepted at {fitted}/{n} frequencies; pooled edge onset median "
          f"{om:+.0f} ms")
    if costs:
        print(f"pooling reads {np.median(costs) * 100:+.0f}% of the per-repeat rate "
              f"(range {min(costs) * 100:+.0f}% to {max(costs) * 100:+.0f}%)")
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
    ap.add_argument("--only-hz", type=float, default=None,
                    help="with --trials: draw EVERY repeat at this one frequency instead")
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
        trial_panels(a.trials, out_path=a.out, which=a.which, metric=a.metric,
                     only_hz=a.only_hz)
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
