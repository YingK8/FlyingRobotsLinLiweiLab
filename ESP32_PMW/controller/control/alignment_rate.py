#!/usr/bin/env python3
"""How fast the robot re-aligns after two coils are cut, per drive frequency.

    uv run python controller/control/alignment_rate.py                  # self-check
    uv run python controller/control/alignment_rate.py <take_dir>       # one take
    uv run python controller/control/alignment_rate.py <take> --log <sweep.log>
    uv run python controller/control/alignment_rate.py --settle <campaign_root>

`--settle` answers a DIFFERENT question from the rest of this module: not how fast the lean
starts, but where the axis ends up and how long until it stays there. Resting axes, a +-10%
settling time, the 10-90% rise time, and the precession cone about the running average axis.
See theory.md 25, and read 25.1 before quoting any of the three timing numbers by name.

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

    A POINT CAN OPEN ON ITS KILL, NOT ONLY ON ITS `FREQ_`
    ----------------------------------------------------
    24.2 records a label placed last labelling nothing. This is the same failure at the other
    end: the board is reset to start a schedule, the first serial bytes are boot garbage, and
    the opening `FREQ_<f>HZ` is lost inside it. `2026-09-09_213502` (70 Hz) is one -- 18 null
    bytes in its first 4 KB, no `FREQ_070HZ`, but `SETTLE_070HZ`, `HOLD_070HZ`, `KILL_070HZ`
    and `DOWN_070HZ` all present and correct.

    Requiring `FREQ_` to open the point threw that take away, and it was 1 of only 5 flown at
    70 Hz -- the thinnest frequency in the campaign. So any of the labels opens a point, and
    the frequency comes from the label's own tag, which every one of them carries.

    This does NOT weaken 23.2's rule that the label is the only event and no offset is fitted.
    The kill instant still comes from `KILL_<f>HZ` and from nothing else; what changed is only
    which label is allowed to open the record it belongs to. `t_start` becomes the first label
    seen rather than the ramp's start, and nothing reads it except the debug plot's axis.
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
        elif (k := KILL_RE.match(name)):
            if cur is None:                  # opening FREQ_ lost to boot garbage
                cur = {"freq": float(k.group(1)), "t_start": t, "kills": [], "t_end": None}
                pts.append(cur)
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


def _basis(ref):
    """Orthonormal ``(e1, e2)`` spanning the plane perpendicular to ``ref``.

    The tangent plane is the right coordinate for anything the axis does off ``ref``
    (`pose/theory.md` 20.7): it has no ``1/sin(theta)`` singularity, unlike azimuth, so a
    small motion near the reference stays small instead of being amplified into noise.
    """

    ref = np.asarray(ref, float)
    ref = ref / np.linalg.norm(ref)
    e1 = np.cross(ref, [0.0, 0.0, 1.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(ref, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(ref, e1)


def angles(axis, ref):
    """``(radial_deg, azimuth_deg)`` of the axis about a reference direction.

    Radial is the polar angle from ``ref`` -- how far the rotor axis has tipped. Azimuth is
    where it tipped TO, measured in the plane perpendicular to ``ref`` and unwrapped, so a
    steadily precessing axis gives a straight ramp rather than a sawtooth.

    The pair separates two motions the single tilt scalar conflates: a cone of constant
    half-angle is CONSTANT in radial and LINEAR in azimuth, while a lean that grows and stays
    is a step in radial with azimuth fixed.
    """

    e1, e2 = _basis(ref)
    ref = np.asarray(ref, float)
    ref = ref / np.linalg.norm(ref)
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


def _smooth(t, y, span_s=SMOOTH_S, odd=False):
    """Box smoother with EDGE padding. ``odd=True`` centres it exactly; see the warning.

    `np.convolve(..., "same")` zero-pads, which drags the first half-window toward 0 --
    and the first half-window is exactly where the step being measured begins. That put
    `frac` far negative right after the kill and fitted tau at 8.7 s against a true 0.8.

    THERE IS A HALF-SAMPLE BIAS AT EVEN `k`, AND IT IS LEFT IN BY DEFAULT
    --------------------------------------------------------------------
    A boxcar is symmetric, so applied centred it has exactly linear phase and zero group
    delay (theory.md 25.3). That argument needs a true centre to sit on. At EVEN `k` the
    centre falls between two samples, and `pad = k // 2` then `[:len(y)]` leaves the output
    half a sample early -- ~2.4 ms at the 204 Hz stride-1 rate.

    Forcing `k` odd fixes it, and `odd=True` does that. It is NOT the default, because it is
    not a free change: re-running `--campaign` with it moves `rate_mean_deg_s` at 60 Hz from
    620.6 to 263.3 deg/s and `rate_relu_med` at 40 Hz from 677.3 to 902.8. The published
    `campaign.csv` is reproduced to 0.00% by the default path and not by the fixed one.

    2.4 ms does not do that on its own. `relu_window` picks the edge by DISCRETE index --
    the last sample in the bottom band, the first in the top -- so a half-sample shift can
    move a threshold crossing by a whole sample, and on a ~50 ms edge carrying a handful of
    samples that is a large lever on the fitted slope. The frequencies it moves most are the
    ones 24.5 and 24.7 already call unreliable (60 Hz is 5 of 9 repeats).

    So the settling path (which times a step and must not have a timing bias) passes
    `odd=True`, and the 24 rate path keeps the behaviour its published table was made with
    until someone re-runs that table deliberately. Whether to flip it is an operator's call,
    not a side effect of adding a new metric.
    """

    k = max(1, int(round(span_s / max(np.median(np.diff(t)), 1e-9))))
    if odd:
        k += 1 - k % 2
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


# ---------------------------------------------------------------- settling and precession
#
# A DIFFERENT QUESTION FROM THE ONE ABOVE. Everything before this point measures how fast the
# lean STARTS -- `rate_relu_deg_s` is the gradient of the rising edge. This section measures
# where the axis ENDS UP and how long until it stays there, which needs the axis itself rather
# than a scalar departure from where it used to be. See theory.md 25.
#
# The two names are the standard ones and are not interchangeable (theory.md 25.1):
#
#   settling time -- t_kill to the last instant the response LEAVES the +-band about its
#                    final value. The headline here.
#   rise time     -- 10% to 90% of the final value. Reported alongside, and the same
#                    quantity `campaign.csv` already carries as `t10_s` / `t90_s`.

#: Fraction of the swing that defines the settled band. The operator asked for 10%; the
#: control-systems default is 2% or 5%. It is an argument and not a constant because the
#: choice is a convention rather than a measurement.
SETTLE_BAND = 0.10

#: Where the FINAL resting axis is read, relative to the kill. `tilt_schedule.DROP_MS` leaves
#: the robot leaning 5.0 s and then the DOWN_ ramp starts, so 1.5 s is what there is. The
#: window is ALSO clamped to the DOWN_ label -- see `resting_axis` and theory.md 25.5.
FINAL_FROM_S = 3.5

#: The 2-4 Hz mechanical mode, `control/theory.md` 24.10: `f = 3.86 - 0.0155 * f_drive`, most
#: likely the 8 mm takeoff rod. This is the rise-peak-fall structure that makes a rise time
#: mis-specified on the RAW trace; nulling it is what leaves an average axis that can settle.
#: Used as a SEED only -- the null lands on the peak this take actually has, within
#: MECH_LO_HZ..MECH_HI_HZ, and 24.10's relation is reported against it as a cross-check.
MECH_A, MECH_B = 3.86, -0.0155
#: The band the mode is actually looked for in. Wide enough that the median over it is an
#: off-peak floor, and wide enough to hold the line wherever the take puts it.
MECH_LO_HZ, MECH_HI_HZ = 1.5, 6.0

#: A `dft` peak has to clear this multiple of the median amplitude over the search grid before
#: it is believed. Below it the line is not there and the caller falls back to its assumption.
PEAK_OVER_FLOOR = 3.0

#: Minimum swing for a settling time to mean anything. Below it there is no step to settle
#: from, and the band would be narrower than the segmentation noise.
MIN_SWING_DEG = 5.0

#: The settled tail: how much of the end of the window is treated as "arrived", and how far
#: inside the band its wander has to sit before a settling time off that band means anything.
#:
#: TAIL_CLEAR IS 1.0 BECAUSE THAT IS THE DEFINITION, NOT A MARGIN. "Settled" means inside the
#: band; a tail that wanders wider than the band has not settled, whatever the last crossing
#: says. Any value below 1.0 is an extra safety margin, and it decides the answer -- over this
#: campaign 0.3 keeps 6 of 98 repeats, 0.5 keeps 36, 0.7 keeps 61, 1.0 keeps 77, 2.0 keeps 94.
#: A number that swings the result by 13x is not something to pick quietly, so the margin is
#: not taken here at all: every reported settling time carries its own `tail_over_band` in the
#: CSV, and a reader who wants a stricter cut can make it from that column. See theory.md 25.6.
TAIL_S = 1.5
TAIL_CLEAR = 1.0


def _polar(n, ref):
    """``(radial_deg, azimuth_deg)`` of ONE axis about ``ref``. The vector form of `angles`.

    Azimuth is in [0, 360) here rather than unwrapped: a single attitude has no history to
    unwrap against.
    """

    e1, e2 = _basis(ref)
    ref = np.asarray(ref, float)
    ref = ref / np.linalg.norm(ref)
    n = np.asarray(n, float)
    n = n / np.linalg.norm(n)
    if float(n @ ref) < 0.0:            # a LINE, like everything else in this module
        n = -n
    return (float(np.degrees(np.arccos(np.clip(n @ ref, -1.0, 1.0)))),
            float(np.degrees(np.arctan2(n @ e2, n @ e1))) % 360.0)


def cone_line(t, axis, t0, t1, f_drive):
    """``(f_hz, amp_deg, measured)``: the once-per-rev cone frequency, MEASURED not assumed.

    `campaign` sets `spin_hz = freq_hz` and never checks it, and every `rev_window` null in
    this module rests on that. It is not a safe assumption: the rotor is friction-capped on
    the takeoff rod and saturates well below the field above ~50 Hz drive, so the null can sit
    somewhere the wobble is not.

    At stride 1 the solve is ~204 Hz and the line is simply RESOLVED, so it can be read off
    instead of assumed -- which is a thing stride 4 could not do, and the reason this function
    did not exist before. `dft` is used rather than `np.fft` because the pose stream is
    non-uniform (see its docstring). Above ~92 Hz drive the line passes Nyquist and cannot be
    measured; the caller is told so via ``measured`` rather than being handed a fold.
    """

    m = (t >= t0) & (t < t1)
    if m.sum() < 64:
        return float(f_drive), float("nan"), False
    fs = 1.0 / float(np.median(np.diff(t[m])))
    lo, hi = max(0.5 * f_drive, 1.0), min(1.6 * f_drive, 0.45 * fs)
    if hi <= lo + 1.0:
        return float(f_drive), float("nan"), False

    # The cone lives in the plane perpendicular to the window's own mean axis, where it has
    # no 1/sin(theta) amplification (20.7). One component is enough to locate the line.
    ref = _hemisphere(axis[m], axis[m][0]).mean(0)
    if np.linalg.norm(ref) < 1e-9:
        return float(f_drive), float("nan"), False
    ref /= np.linalg.norm(ref)
    e1, _e2 = _basis(ref)
    c1 = np.degrees(_hemisphere(axis[m], ref) @ e1)

    grid = np.arange(lo, hi, 0.05)
    amp = dft(c1, t[m], grid)
    i = int(np.argmax(amp))
    if amp[i] < PEAK_OVER_FLOOR * float(np.median(amp)):
        return float(f_drive), float("nan"), False
    return float(grid[i]), float(amp[i]), True


def mech_line(t, y, f_drive):
    """``(f_hz, measured)``: the 2-4 Hz mechanical mode, seeded from 24.10 and refined here.

    24.10 fitted `f = 3.86 - 0.0155 * f_drive` across the campaign and concluded the mode is
    mechanical -- it does not track the drive the way a coning or nutation term would. That
    line is the seed; the null is placed on whatever peak this take actually has near it,
    because a null in the wrong place attenuates rather than removes.
    """

    seed = MECH_A + MECH_B * float(f_drive)
    if len(t) < 64:
        return max(seed, 0.5), False
    # Searched over a real BAND, not +-1 Hz of the seed. Two reasons. The floor test needs
    # somewhere off-peak to measure a floor -- over a narrow grid the median sits on the
    # shoulder of the peak itself, which is why an earlier +-1 Hz version reported
    # `mech_measured = 0` on takes with an obvious 0.17 deg line in them. And the seed is a
    # fit across the campaign, not a per-take truth: on `2026-09-10_012419` (40 Hz, seed
    # 3.24) the line is at 2.40 Hz. 24.10's relation is a cross-check here, not a constraint.
    grid = np.arange(MECH_LO_HZ, MECH_HI_HZ, 0.02)
    amp = dft(y, t, grid)
    i = int(np.argmax(amp))
    if amp[i] < PEAK_OVER_FLOOR * float(np.median(amp)):
        return float(max(seed, MECH_LO_HZ)), False
    return float(grid[i]), True


def average_axis(t, axis, cone_hz, mech_hz):
    """``(avg_unit_axis, valid_mask, span_s)``. The trajectory with both wobbles removed.

    This is the "average axis of rotation vs time" the settling time is read from, and the
    thing precession is measured ABOUT. Two wobbles come out, by two DIFFERENT methods, and
    the difference is the point:

    * the once-per-rev cone, by a centred boxcar of a whole number of rotor revolutions
      (`rev_window`) -- an exact null at the cone and every harmonic of it. ~0.25 s.
    * the 2-4 Hz mechanical mode of 24.10, by least-squares removal of that one line
      (`_fit_line_at`) -- NOT by a second boxcar.

    The first window does not already remove the second mode: it is ~0.25 s, and
    `sinc(3.1 Hz * 0.25 s) = 0.47`, so half the mode passes straight through. That residue is
    the rise-peak-fall the zip MANIFEST names when it says a rise time is mis-specified here,
    and it is not cosmetic -- measured on three 40 Hz repeats, leaving it in reads settling
    times of 0.87, 2.16 and 0.54 s against 0.23, 0.92 and 0.19 s with it gone. Without this
    removal the metric times the rod, not the alignment.

    WHY THE SECOND ONE IS NOT A BOXCAR
    ----------------------------------
    Because it would cost 0.33 s of width, and the settling times being measured are 0.2-0.9 s.
    A centred boxcar biases a settling time LATE whenever the event is shorter than the
    window -- measured against known exponentials, 1.02x at 0.46 s but 1.83x at 0.12 s
    (theory.md 25.4). Removing the line by least squares costs no time-domain width at all,
    so the only smoother left in the path is the 0.25 s cone window and the gate in `settling`
    has something most repeats can clear.

    ZERO PHASE DELAY, AND NOTHING TO COMPENSATE
    -------------------------------------------
    Each boxcar is symmetric and applied centred at odd length, so its phase is exactly linear
    with group delay zero (`_smooth`); two cascaded are still zero. This is a property of the
    filter, not an approximation to be corrected afterwards -- a causal moving average would
    need a `(k-1)/2` sample correction and a one-pole IIR a frequency-dependent one, and both
    would put the correction's own error into the timing. `_self_check` asserts it against a
    ramp rather than restating it.

    The cost is at the ENDS. `_smooth` pads `mode="edge"`, so the outer half-window of each
    pass is biased toward the boundary value. Those samples are returned as invalid rather
    than silently used: the guard is half the total span, and the caller has a 1 s baseline
    and a 1.5 s settled window to spend it out of.
    """

    a = np.asarray(axis, float)
    out, span = a, 0.0
    if cone_hz and np.isfinite(cone_hz) and cone_hz > 0:
        span = rev_window(cone_hz)
        out = np.column_stack([_smooth(t, a[:, i], span, odd=True) for i in range(3)])
    if mech_hz and np.isfinite(mech_hz) and mech_hz > 0:
        wide = 1.0 / mech_hz
        for i in range(3):
            trend = _smooth(t, out[:, i], wide, odd=True)
            out[:, i] -= _fit_line_at(t, out[:, i] - trend, mech_hz)
    n = np.linalg.norm(out, axis=1)
    out = out / np.maximum(n, 1e-12)[:, None]
    guard = 0.5 * span
    valid = (t >= t[0] + guard) & (t <= t[-1] - guard) & (n > 1e-9)
    return out, valid, span


def _fit_line_at(t, y, f_hz):
    """The component of ``y`` at exactly ``f_hz``, by least squares. Returns it to subtract.

    `deproject`'s method, applied per axis component: one sinusoid, fitted and removed, with
    no passband distortion either side of it -- and, the reason it is used here, NO WIDTH.
    A boxcar long enough to null 3 Hz is 0.33 s, which is longer than most of the settling
    times being measured and biases them up to 1.8x (theory.md 25.4). A least-squares line
    removal costs nothing in the time domain, so the only smoother left in the path is the
    0.25 s cone window.

    ``y`` is passed already high-passed by the caller -- the trace minus a boxcar-smoothed
    copy of itself -- so the settling curve is mostly gone before the fit sees it and cannot
    leak into the amplitude. Whatever does leak, the fit rejects anyway: it takes only the
    component at this one frequency.
    """

    t = np.asarray(t, float)
    A = np.c_[np.cos(2 * np.pi * f_hz * t), np.sin(2 * np.pi * f_hz * t)]
    coef, *_ = np.linalg.lstsq(A, np.asarray(y, float), rcond=None)
    return A @ coef


def precession(t, axis, avg, ref, cone_hz, window):
    """Motion of the axis ABOUT its running average: how big, and is it a cone or a shake.

    ``half_deg`` is the instantaneous angle between the axis and the average axis -- the cone
    half-angle, and the "precession vs time" trace.

    ``(c1, c2)`` are the residual's components in the tangent plane of ``ref``, which is where
    20.7 established this motion should be read. A lock-in on ``c1 + i c2`` at ``cone_hz``
    splits it into CO-rotating and COUNTER-rotating parts, and their ratio says which motion
    it is: a circular cone puts everything into one sense, a linear wobble splits evenly
    between the two. 20.7 measured 9.6:1 in favour of circular on one take by this test; the
    ratio is returned per repeat so it stops being one take's number.

    Reported direction-agnostically as larger/smaller, so it does not depend on which way the
    rig happens to spin (`PHASES_CW` is labelled inverted on this bench).
    """

    dot = np.clip(np.sum(np.asarray(axis, float) * np.asarray(avg, float), axis=1), -1.0, 1.0)
    half = np.degrees(np.arccos(dot))
    e1, e2 = _basis(ref)
    r = np.asarray(axis, float) - np.asarray(avg, float)
    c1, c2 = np.degrees(r @ e1), np.degrees(r @ e2)

    m = np.asarray(window, bool)
    ratio = float("nan")
    if m.sum() >= 32 and cone_hz and np.isfinite(cone_hz) and cone_hz > 0:
        z = c1[m] + 1j * c2[m]
        ph = 2j * np.pi * cone_hz * t[m]
        co = 2.0 * abs(np.mean(z * np.exp(-ph)))
        ct = 2.0 * abs(np.mean(z * np.exp(ph)))
        hi_, lo_ = max(co, ct), min(co, ct)
        ratio = float(hi_ / lo_) if lo_ > 1e-12 else float("inf")
    return half, c1, c2, ratio


def prec_envelope(t, c1, c2, cone_hz):
    """RMS cone half-angle over one rotor revolution: the precession ENVELOPE vs time.

    The instantaneous half-angle is not a plottable trace -- it swings through the full cone
    once per revolution, so at 40 Hz over 14 repeats it is a solid block of ink and shows
    nothing about stability. What the eye needs is the envelope, and the RMS over a whole
    number of revolutions is exactly that: the same window that nulls the cone in
    `average_axis` here MEASURES it, because the mean of a squared sinusoid over whole
    periods is its mean square regardless of phase.
    """

    w = rev_window(cone_hz) if cone_hz and np.isfinite(cone_hz) and cone_hz > 0 else SMOOTH_S
    return np.sqrt(np.maximum(
        _smooth(t, np.asarray(c1) ** 2 + np.asarray(c2) ** 2, w, odd=True), 0.0))


def _band_settle(t, prog, t_kill, t_stop, valid, band=SETTLE_BAND):
    """Settling time from a two-sided +-band about the final level of ``prog``. IN-PLANE ONLY.

    This is the textbook reading of a step response, and it is what the annotated figure draws
    because it is the one a band can be drawn on. It is reported ALONGSIDE `settle_10pct_s`
    rather than instead of it, because the two are not the same measurement and the difference
    is physical, not cosmetic.

    `settle_10pct_s` tests `Delta`, the angle to the final axis in three dimensions, so it
    counts motion that left the initial->final arc sideways. This one does not: it only asks
    how far along the arc the axis has got. Measured over ten 40 Hz repeats they agree to
    0.01-0.06 s on eight of them and disagree by 0.51 and 0.80 s on the other two, which are
    exactly the repeats with the largest out-of-plane excursion. `perp_max_deg` is carried in
    the CSV so that gap can be attributed rather than guessed at.

    Where they differ, `settle_10pct_s` is the stricter and the more physical: an axis 3 deg
    off to the side has not arrived, whatever its progress along the arc says.
    """

    m = valid & (t >= t_kill) & (t <= t_stop)
    pre = valid & (t >= t_kill - PRE_S) & (t < t_kill)
    fin = valid & (t >= t_kill + FINAL_FROM_S) & (t <= t_stop)
    if m.sum() < 20 or pre.sum() < 20 or fin.sum() < 10:
        return float("nan")
    ini_l, fin_l = float(np.median(prog[pre])), float(np.median(prog[fin]))
    tol = band * abs(fin_l - ini_l)
    if tol <= 0:
        return float("nan")
    out = np.flatnonzero(np.abs(prog[m] - fin_l) > tol)
    if not out.size:
        return 0.0
    if out[-1] >= m.sum() - 2:
        return float("nan")
    return float(t[m][out[-1]] - t_kill)


def great_circle(avg, n_0, n_f):
    """``(progress_deg, out_of_plane_deg)`` of a trajectory along the initial->final arc.

    `Delta`, the angle to the final axis, is what the settling band is applied to, and it is
    the right scalar for that: it is unsigned, so a departure from `n_f` in ANY direction
    increases it and a one-sided test on it is already a two-sided band in every direction at
    once. What it cannot do is SHOW anything -- it folds an overshoot past `n_f` onto the same
    side as a shortfall, so a plot of it cannot be read the way a step response is read.

    This is the signed companion, for the figures. Build an orthonormal frame on the great
    circle through `n_0` and `n_f`: ``b1 = n_0``, ``b2`` the part of `n_f` perpendicular to it.
    Then ``progress`` is the angle around that circle from `n_0`, which starts near 0, ends
    near the swing, and goes PAST it on an overshoot; ``out_of_plane`` is the component that
    left the arc, which is the part of the motion the swing does not describe.
    """

    b1 = np.asarray(n_0, float) / np.linalg.norm(n_0)
    r = np.asarray(n_f, float) - float(np.asarray(n_f, float) @ b1) * b1
    nr = float(np.linalg.norm(r))
    if nr < 1e-9:                       # n_0 and n_f coincide: no arc to measure along
        return np.zeros(len(avg)), np.zeros(len(avg))
    b2 = r / nr
    b3 = np.cross(b1, b2)
    a = np.asarray(avg, float)
    return (np.degrees(np.arctan2(a @ b2, a @ b1)),
            np.degrees(np.arcsin(np.clip(a @ b3, -1.0, 1.0))))


#: Search window for the peak of the precession envelope after the cut. Measured, the peak
#: lands at 0.13-0.38 s on the 40 Hz repeats, so 1.5 s is generous and still well clear of
#: the settled window the final level is read from.
PREC_PEAK_MAX_S = 1.5

#: Floor on the excursion (peak minus settled level) for a precession settling time to mean
#: anything. Below it the cut did not measurably excite the cone and there is no decay to
#: time. Measured excursions at 40 Hz run 3.2-12.1 deg, so this refuses only the flat ones.
MIN_PREC_EXC_DEG = 1.0


def precession_settle(t, env, t_kill, t_stop, valid, span_s, band=SETTLE_BAND):
    """How long the CONE takes to stop ringing, as opposed to how long the axis takes to move.

    The axis metric times a step: one level, a cut, another level. This times a PULSE. The
    envelope sits at its driven level, jumps when the coils are cut, and decays -- so the
    quantity that plays the role of the swing is the EXCURSION, ``peak - final``, and the band
    is 10% of that about the final level. Same convention as `settling`, applied to the thing
    that actually changes.

    Returns ``peak_deg``, ``t_peak_s``, ``final_deg``, ``settle_s``, ``tau_s``, ``asym_deg``.

    WHY A DECAY CONSTANT IS REPORTED NEXT TO THE SETTLING TIME
    ---------------------------------------------------------
    Because on this campaign the settling time is often not available and the decay constant
    is. The window is 5 s (`DROP_MS`) and the envelope is still falling at the end of it --
    measured on five 40 Hz repeats, the median drops another 9-22% between 2-3 s and 4-5 s.
    A band about a "final" level that is not final refuses, correctly, and says nothing.

    An exponential fitted from the peak does not need the asymptote to be reached to measure
    how fast it is being approached, so it survives where the band does not. Three parameters
    -- ``C + A exp(-(t - t_peak) / tau)`` -- with ``A`` and ``C`` solved linearly at each
    ``tau`` on a grid, in the manner of `fit_second_order` and for the same reason: two
    nested loops and no dependency on scipy being present.

    That constant is the one 11.3 wants. Gyroscopic action alone gives steady coning at fixed
    half-angle; only dissipation spirals it in. So `tau_s` is the aerodynamic damping `c_t`
    made visible, and it is the number this campaign was carrying all along.
    """

    out = {"peak_deg": float("nan"), "t_peak_s": float("nan"), "final_deg": float("nan"),
           "settle_s": float("nan"), "tau_s": float("nan"), "asym_deg": float("nan"),
           "note": ""}
    m = valid & (t >= t_kill) & (t <= t_stop)
    pk = valid & (t >= t_kill) & (t <= t_kill + PREC_PEAK_MAX_S)
    fin_m = valid & (t >= t_kill + FINAL_FROM_S) & (t <= t_stop)
    if m.sum() < 20 or pk.sum() < 10 or fin_m.sum() < 10:
        out["note"] = "post-kill window too short"
        return out

    i = int(np.argmax(env[pk]))
    out["peak_deg"] = float(env[pk][i])
    out["t_peak_s"] = float(t[pk][i] - t_kill)
    out["final_deg"] = float(np.median(env[fin_m]))
    exc = out["peak_deg"] - out["final_deg"]
    if exc < MIN_PREC_EXC_DEG:
        out["note"] = (f"the cut moved the cone by {exc:.2f} deg, under the "
                       f"{MIN_PREC_EXC_DEG:.1f} deg floor -- no decay to time")
        return out

    # Decay constant first: it is the number that survives an unfinished window.
    out.update(_decay_fit(t[m] - t_kill - out["t_peak_s"], env[m],
                          t[m] - t_kill >= out["t_peak_s"]))

    x, y = t[m] - t_kill, np.abs(env[m] - out["final_deg"])
    tol = band * exc
    tail = x >= max(x[-1] - TAIL_S, 0.5 * x[-1])
    if tail.sum() > 5 and float(np.max(y[tail])) > TAIL_CLEAR * tol:
        out["note"] = (f"the envelope is still moving {float(np.max(y[tail])):.2f} deg at the "
                       f"end of the window against a {tol:.2f} deg band -- still decaying")
        return out
    outside = np.flatnonzero(y > tol)
    if not outside.size:
        out["settle_s"] = 0.0
    elif outside[-1] >= len(x) - 2:
        out["note"] = f"still outside the {100 * band:.0f}% band when the down-ramp starts"
    elif float(x[outside[-1]]) < span_s:
        out["note"] = (f"settle {float(x[outside[-1]]):.3f} s is inside the {span_s:.3f} s "
                       f"smoother -- not measurable")
    else:
        out["settle_s"] = float(x[outside[-1]])
    return out


def _decay_fit(t_rel, y, mask):
    """``C + A exp(-t / tau)`` from the peak on. ``A`` and ``C`` linear at each grid ``tau``."""

    x, yy = np.asarray(t_rel, float)[mask], np.asarray(y, float)[mask]
    if len(x) < 30:
        return {}
    best = None
    for tau in np.geomspace(0.05, 20.0, 140):
        A = np.c_[np.exp(-x / tau), np.ones_like(x)]
        coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
        r = float(np.mean((yy - A @ coef) ** 2))
        if best is None or r < best[0]:
            best = (r, tau, coef)
    r, tau, coef = best
    # A decay that fits with a NEGATIVE amplitude is a rise, not a decay, and a tau at the
    # end of the grid is the fit running away rather than converging. Neither is a number.
    if coef[0] <= 0 or tau >= 19.0 or tau <= 0.06:
        return {}
    return {"tau_s": float(tau), "asym_deg": float(coef[1])}


def resting_axis(t, axis, t0, t1, min_n=20):
    """``(unit_axis, n, spread_deg)`` over a window, or ``None``. A settled attitude.

    Averaged from the RAW axis, not the smoothed one: a window of 1 s or more is already
    many cone periods long, so it nulls the wobble by itself, and it does so without the
    edge bias `average_axis` carries at its ends.
    """

    m = (t >= t0) & (t < t1)
    if m.sum() < min_n:
        return None
    V = _hemisphere(axis[m], axis[m][0])       # sign-align BEFORE averaging, or the mean
    n = V.mean(0)                              #   shortens and reads as tilt
    norm = float(np.linalg.norm(n))
    if norm < 1e-9:
        return None
    n = n / norm
    spread = float(np.degrees(np.arccos(np.clip(V @ n, -1.0, 1.0))).std())
    return n, int(m.sum()), spread


def settling(t, delta, t_kill, t_stop, swing, span_s=0.0, band=SETTLE_BAND):
    """``dict`` with the settling time and the rise time, or a ``note`` saying why not.

    ``delta`` is the angle from the running average axis to the FINAL resting axis, so it
    starts at ``swing`` and ends at zero. Two standard quantities are read off it:

    * **settling time**: the last instant ``delta`` exceeds ``band * swing``. Last, not first
      -- entering the band and coming back out again is not settled, and a response with a
      2-4 Hz mode in it does exactly that, which is the whole reason the mode is nulled first.
    * **rise time**: 10% to 90% of the final value, on the response ``swing - delta``.

    REFUSES RATHER THAN CLAMPS, like everything else here. A repeat whose trace is still
    outside the band when the down-ramp starts has not settled inside the record, and the
    honest answer is that the record cannot say -- not `t_stop`, which would be a censored
    value entering the statistics as a measurement.

    ``span_s`` IS THE SMOOTHER'S SUPPORT, AND THE RISE TIME NEEDS IT
    ---------------------------------------------------------------
    A zero-phase filter is non-causal. That is the same fact as its zero group delay, seen
    from the other side: `average_axis` is centred, so it spreads the step SYMMETRICALLY
    about the cut and the smoothed trace is already part-way up at ``t_kill`` itself. Measured
    on the synthetic case in `_self_check`, a 0.66 s rise read `t10 = 0.000` -- not a fast
    response, the filter's own half-window.

    So the two quantities are read over different windows, and for a reason:

    * the SETTLING time is read from ``t_kill`` on. It is a property of the tail, far from
      the step, where the smear has nothing left to do.
    * the RISE time is read from ``t_kill - span_s``, because the rise legitimately begins
      before the cut in a non-causally filtered trace. It is a DIFFERENCE of two crossings,
      so the symmetric smear largely cancels in it -- but only largely, which is why it is
      then refused outright when it does not outlast ``span_s``.

    That refusal is `control/theory.md` 24.7's rule -- "the rate is only measurable where the
    rise outlasts the smoother" -- restated for two cascaded windows instead of one. There it
    cost 10 Hz its rate; here the cascade is wider still, so it will cost more.
    """

    out = {"settle_s": float("nan"), "rise_s": float("nan"), "t10_s": float("nan"),
           "t90_s": float("nan"), "tail_max_deg": float("nan"), "note": ""}
    if not np.isfinite(swing) or swing < MIN_SWING_DEG:
        out["note"] = f"swing {swing:.1f} deg below the {MIN_SWING_DEG:.0f} deg floor"
        return out
    m = (t >= t_kill) & (t <= t_stop)
    if m.sum() < 20:
        out["note"] = "post-kill window too short"
        return out
    x, y = t[m] - t_kill, delta[m]

    # How far the average axis still wanders once it is supposed to have arrived. The band is
    # measured against THIS, so it has to clear it: a band the same size as the residual
    # wander is decided by the wander, and on `2026-09-10_012419` (40 Hz) that read 0.87 s or
    # 2.09 s depending on which line was removed, for a tail max of 1.190 deg against a
    # 1.190 deg band. Neither number was a measurement.
    tail = x >= max(x[-1] - TAIL_S, 0.5 * x[-1])
    out["tail_max_deg"] = float(np.max(y[tail])) if tail.sum() > 5 else float("nan")
    if np.isfinite(out["tail_max_deg"]) and out["tail_max_deg"] > TAIL_CLEAR * band * swing:
        out["note"] = (f"tail still wanders {out['tail_max_deg']:.2f} deg against a "
                       f"{band * swing:.2f} deg band -- no settled state to time")
        return out

    outside = np.flatnonzero(y > band * swing)
    if not outside.size:
        out["settle_s"] = 0.0
        out["note"] = "already inside the band at the cut"
    elif outside[-1] >= len(x) - 2:
        out["note"] = (f"still outside the {100 * band:.0f}% band at "
                       f"{x[-1]:.2f} s, when the down-ramp starts")
        return out
    elif float(x[outside[-1]]) < span_s:
        # The smoother is wider than the event. A centred boxcar biases a settling time LATE
        # when the response is shorter than the window -- measured against known exponentials
        # (theory.md 25.4): 1.02x at settle = 1.8 W, 1.11x at 0.9 W, 1.36x at 0.5 W and 1.83x
        # at 0.2 W. Refusing below one window caps the residual bias at about 10% on
        # everything that IS reported, and 24.7 set the precedent for the rate.
        out["note"] = (f"settle {float(x[outside[-1]]):.3f} s is inside the {span_s:.3f} s "
                       f"smoother -- not measurable")
        return out
    else:
        out["settle_s"] = float(x[outside[-1]])

    # FIRST crossings: unlike the settling time this one is about how quickly the motion got
    # going, and a later re-crossing is the overshoot, not the rise.
    mr = (t >= t_kill - span_s) & (t <= t_stop)
    xr, resp = t[mr] - t_kill, swing - delta[mr]
    for key, frac in (("t10_s", 0.10), ("t90_s", 0.90)):
        hit = np.flatnonzero(resp >= frac * swing)
        if hit.size:
            out[key] = float(xr[hit[0]])
    if np.isfinite(out["t10_s"]) and np.isfinite(out["t90_s"]):
        rise = out["t90_s"] - out["t10_s"]
        if rise < span_s:
            out["note"] = (out["note"] + "; " if out["note"] else "") + \
                f"rise {rise:.3f} s is inside the {span_s:.3f} s smoother -- not measurable"
            out["t10_s"] = out["t90_s"] = float("nan")
        else:
            out["rise_s"] = rise
    return out


def settle_take(take, log_path, freq_hz, which=None):
    """Every kill in one take, as settling rows. ``[]`` when the take cannot support one."""

    t, axis, _q = load(take, which=which)
    if len(t) < 200:
        return []
    fs = 1.0 / float(np.median(np.diff(t)))
    pts = timeline(log_path) if Path(log_path).exists() else []
    if not pts:
        return []
    try:
        up = datum(t, axis, pts)[0]
    except SystemExit:
        return []
    a = orient_continuous(axis, up)

    rows = []
    for p in pts:
        for tk in p["kills"]:
            # The settled window is clamped to the DOWN_ label. `POST_TO_S = 6.0` against
            # `DROP_MS = 5000` puts a second of SPIN-DOWN inside what the rest of this module
            # calls settled; a resting axis measured through a frequency ramp is not one.
            t_stop = min(tk + POST_TO_S, p["t_end"])
            fin = resting_axis(t, a, tk + FINAL_FROM_S, t_stop)
            ini = resting_axis(t, a, tk - PRE_S, tk)
            if fin is None or ini is None:
                continue
            n_f, n_fin, spread_f = fin
            n_0, n_ini, spread_0 = ini
            if float(n_f @ n_0) < 0.0:
                n_f = -n_f                      # both are lines; report them on one branch

            cone_hz, cone_amp, cone_meas = cone_line(t, a, tk - PRE_S, tk, freq_hz)
            post = (t >= tk) & (t <= t_stop)
            mech_hz, mech_meas = mech_line(
                t[post], np.degrees(np.arccos(np.clip(a[post] @ n_f, -1.0, 1.0))), freq_hz)
            avg, valid, span = average_axis(t, a, cone_hz, mech_hz)
            guard = 0.5 * span

            delta = np.degrees(np.arccos(np.clip(avg @ n_f, -1.0, 1.0)))
            pre = (t >= tk - PRE_S) & (t < tk) & valid
            if pre.sum() < 20:
                continue
            swing = float(np.median(delta[pre]))
            got = settling(t, delta, tk, t_stop - guard, swing, span_s=span)

            half, c1, c2, ratio = precession(t, a, avg, n_f, cone_hz, post & valid)
            prog, perp = great_circle(avg, n_0, n_f)
            band_s = _band_settle(t, prog, tk, t_stop - guard, valid)
            ps = precession_settle(t, prec_envelope(t, c1, c2, cone_hz), tk,
                                   t_stop - guard, valid, span)
            r0, az0 = _polar(n_0, up)
            rf, azf = _polar(n_f, up)
            rows.append({
                "freq_hz": freq_hz, "take": Path(take).name, "t_kill": round(tk, 3),
                "fs_hz": round(fs, 1), "swing_deg": round(swing, 3),
                "n0_x": round(n_0[0], 6), "n0_y": round(n_0[1], 6), "n0_z": round(n_0[2], 6),
                "nf_x": round(n_f[0], 6), "nf_y": round(n_f[1], 6), "nf_z": round(n_f[2], 6),
                "radial0_deg": round(r0, 3), "azim0_deg": round(az0, 3),
                "radialf_deg": round(rf, 3), "azimf_deg": round(azf, 3),
                "d_radial_deg": round(rf - r0, 3),
                "d_azim_deg": round((azf - az0 + 180.0) % 360.0 - 180.0, 3),
                "spread0_deg": round(spread_0, 3), "spreadf_deg": round(spread_f, 3),
                "settle_10pct_s": (round(got["settle_s"], 4)
                                   if np.isfinite(got["settle_s"]) else ""),
                "settle_band_s": round(band_s, 4) if np.isfinite(band_s) else "",
                "perp_max_deg": round(float(np.max(np.abs(perp[post & valid]))), 3)
                if (post & valid).sum() else "",
                "rise_1090_s": round(got["rise_s"], 4) if np.isfinite(got["rise_s"]) else "",
                "t10_s": round(got["t10_s"], 4) if np.isfinite(got["t10_s"]) else "",
                "t90_s": round(got["t90_s"], 4) if np.isfinite(got["t90_s"]) else "",
                "tail_max_deg": round(got["tail_max_deg"], 3)
                if np.isfinite(got["tail_max_deg"]) else "",
                "tail_over_band": round(got["tail_max_deg"] / (SETTLE_BAND * swing), 3)
                if np.isfinite(got["tail_max_deg"]) and swing > 0 else "",
                "prec_pre_deg": round(float(np.median(half[pre])), 3),
                "prec_peak_deg": _r(ps["peak_deg"]), "prec_t_peak_s": _r(ps["t_peak_s"], 4),
                "prec_final_deg": _r(ps["final_deg"]), "prec_settle_s": _r(ps["settle_s"], 4),
                "prec_tau_s": _r(ps["tau_s"], 4), "prec_asym_deg": _r(ps["asym_deg"]),
                "prec_note": ps["note"],
                "prec_post_deg": round(float(np.median(half[post & valid])), 3)
                if (post & valid).sum() else "",
                "cone_hz": round(cone_hz, 2), "cone_amp_deg": round(cone_amp, 3)
                if np.isfinite(cone_amp) else "", "cone_measured": int(cone_meas),
                "circ_ratio": round(ratio, 2) if np.isfinite(ratio) else "",
                "mech_hz": round(mech_hz, 3), "mech_measured": int(mech_meas),
                "n_pre": n_ini, "n_post": n_fin, "note": got["note"],
                # kept out of the CSV by `extrasaction="ignore"`, used by the figures
                "_t": t - tk, "_delta": delta, "_env": prec_envelope(t, c1, c2, cone_hz),
                "_c1": c1, "_c2": c2, "_gc": great_circle(avg, n_0, n_f),
                "_axis": a, "_avg": avg, "_n0": n_0, "_nf": n_f, "_up": up,
                "_valid": valid, "_post": post, "_stop": t_stop - tk,
            })
    return rows


SETTLE_COLS = ["freq_hz", "take", "t_kill", "fs_hz", "swing_deg",
               "n0_x", "n0_y", "n0_z", "nf_x", "nf_y", "nf_z",
               "radial0_deg", "azim0_deg", "radialf_deg", "azimf_deg",
               "d_radial_deg", "d_azim_deg", "spread0_deg", "spreadf_deg",
               "settle_10pct_s", "settle_band_s", "perp_max_deg", "rise_1090_s", "t10_s", "t90_s", "tail_max_deg", "tail_over_band",
               "prec_pre_deg", "prec_post_deg", "prec_peak_deg", "prec_t_peak_s",
               "prec_final_deg", "prec_settle_s", "prec_tau_s", "prec_asym_deg",
               "prec_note", "cone_hz", "cone_amp_deg",
               "cone_measured", "circ_ratio", "mech_hz", "mech_measured",
               "n_pre", "n_post", "note"]


def settle_rows(root, which=None, only_hz=None):
    """``(rows, skipped)`` for a campaign. The measurement, with no files written.

    Split out of `settle_campaign` so `settle_report.py` can draw from the same rows the CSV
    is written from, rather than re-deriving the traces from it -- the per-frame arrays a
    figure needs (`_delta`, `_env`, `_c1`, `_c2`) are deliberately not columns.
    """

    root = Path(root)
    rows, skipped = [], []
    for idx_path in sorted(root.glob("*hz/index.csv")):
        idx = [r for r in csv.DictReader(open(idx_path)) if r["outcome"] == "ok"]
        if not idx:
            continue
        f = float(idx[0]["freq_hz"])
        if only_hz is not None and abs(f - only_hz) > 0.01:
            continue
        for r in idx:
            take = Path(r["flight"])
            if not (take / "axis.csv").exists() and not (take / "axis_minor.csv").exists():
                skipped.append((f, take.name, "no axis.csv"))
                continue
            t, _axis, _q = load(take, which=which)
            # The same Nyquist gate `campaign` applies, and for the same reason: the cone sits
            # AT the drive frequency and `rev_window`'s null only exists if it is resolved.
            fs = 1.0 / float(np.median(np.diff(t))) if len(t) > 2 else 0.0
            if fs < 2.0 * f:
                skipped.append((f, take.name, f"aliased: solved at {fs:.1f} Hz"))
                continue
            got = settle_take(take, r["log"], f, which=which)
            if not got:
                skipped.append((f, take.name, "no usable kill"))
            rows.extend(got)
    return rows, skipped


def settle_campaign(root, out_dir=None, which=None, only_hz=None):
    """Settling time, resting axes and precession for every repeat in a campaign.

    Per repeat, not pooled. The campaign README records that pooling reads 19% of the
    per-repeat rate and flattens the frequency dependence, because repeats do not share a
    dead time; the same objection applies here and more so, since a settling time is a
    property of one trajectory and an average of trajectories does not have one.
    """

    import matplotlib
    matplotlib.use("Agg")

    root = Path(root)
    out = Path(out_dir) if out_dir else root / "report"
    out.mkdir(parents=True, exist_ok=True)
    rows, skipped = settle_rows(root, which=which, only_hz=only_hz)

    if not rows:
        print("no usable takes")
        return []

    _write(out / "settling.csv", rows, SETTLE_COLS)
    per_freq = _settle_by_freq(rows)
    _write(out / "settling_by_freq.csv", per_freq, list(per_freq[0]))
    for f in sorted({r["freq_hz"] for r in rows}):
        _settle_figure([r for r in rows if r["freq_hz"] == f], f,
                       out / f"axis_vs_time_{int(round(f)):03d}hz.png")
    _settle_vs_frequency(per_freq, out / "settle_vs_frequency.png")

    _print_settle(per_freq)
    if skipped:
        print(f"\n{len(skipped)} take(s) not measured:")
        for f, name, why in skipped:
            print(f"  {f:5.0f} Hz  {name}  {why}")
    print(f"\nwrote {out}/settling.csv, settling_by_freq.csv, "
          f"axis_vs_time_*.png, settle_vs_frequency.png")
    return rows


def _med_mad(vals):
    v = np.asarray([x for x in vals if x != "" and x is not None], dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return float("nan"), float("nan"), 0
    med = float(np.median(v))
    return med, float(np.median(np.abs(v - med))), int(v.size)


def _settle_by_freq(rows):
    out = []
    for f in sorted({r["freq_hz"] for r in rows}):
        g = [r for r in rows if r["freq_hz"] == f]
        s_med, s_mad, n_s = _med_mad([r["settle_10pct_s"] for r in g])
        r_med, r_mad, n_r = _med_mad([r["rise_1090_s"] for r in g])
        sw_med, sw_mad, _ = _med_mad([r["swing_deg"] for r in g])
        pr_med, _, _ = _med_mad([r["prec_post_deg"] for r in g])
        pp_med, _, _ = _med_mad([r["prec_pre_deg"] for r in g])
        pk_med, _, _ = _med_mad([r["prec_peak_deg"] for r in g])
        pf_med, _, _ = _med_mad([r["prec_final_deg"] for r in g])
        pset_med, pset_mad, n_ps = _med_mad([r["prec_settle_s"] for r in g])
        ptau_med, ptau_mad, n_pt = _med_mad([r["prec_tau_s"] for r in g])
        cz_med, _, _ = _med_mad([r["cone_hz"] for r in g])
        ci_med, _, _ = _med_mad([r["circ_ratio"] for r in g])
        mh_med, _, _ = _med_mad([r["mech_hz"] for r in g])
        dr_med, _, _ = _med_mad([r["d_radial_deg"] for r in g])
        da_med, _, _ = _med_mad([r["d_azim_deg"] for r in g])
        out.append({
            "freq_hz": f, "n_repeats": len(g),
            "settle_10pct_med_s": round(s_med, 4), "settle_10pct_mad_s": round(s_mad, 4),
            "n_settled": n_s, "n_refused": len(g) - n_s,
            "rise_1090_med_s": round(r_med, 4), "rise_1090_mad_s": round(r_mad, 4),
            "n_rise": n_r,
            "swing_med_deg": round(sw_med, 3), "swing_mad_deg": round(sw_mad, 3),
            "d_radial_med_deg": round(dr_med, 3), "d_azim_med_deg": round(da_med, 3),
            "prec_pre_med_deg": round(pp_med, 3), "prec_post_med_deg": round(pr_med, 3),
            "prec_peak_med_deg": round(pk_med, 3), "prec_final_med_deg": round(pf_med, 3),
            "prec_settle_med_s": round(pset_med, 4), "prec_settle_mad_s": round(pset_mad, 4),
            "n_prec_settled": n_ps,
            "prec_tau_med_s": round(ptau_med, 4), "prec_tau_mad_s": round(ptau_mad, 4),
            "n_prec_tau": n_pt,
            "cone_med_hz": round(cz_med, 2), "cone_over_drive": round(cz_med / f, 3),
            "circ_ratio_med": round(ci_med, 2), "mech_med_hz": round(mh_med, 3),
        })
    return out


def _print_settle(per_freq):
    print(f"{'f':>5} {'n':>3} {'settled':>7} {'t_settle':>9} {'rise':>8} {'swing':>8} "
          f"{'d_azim':>7} | {'pre':>5} {'peak':>5} {'final':>6} {'p_set':>7} {'p_tau':>6} "
          f"{'n_tau':>5}")
    for r in per_freq:
        print(f"{r['freq_hz']:5.0f} {r['n_repeats']:3d} "
              f"{r['n_settled']:3d}/{r['n_repeats']:<3d} "
              f"{r['settle_10pct_med_s']:9.3f} {r['rise_1090_med_s']:8.3f} "
              f"{r['swing_med_deg']:8.2f} {r['d_azim_med_deg']:7.1f} | "
              f"{r['prec_pre_med_deg']:5.2f} {r['prec_peak_med_deg']:5.2f} "
              f"{r['prec_final_med_deg']:6.2f} {r['prec_settle_med_s']:7.3f} "
              f"{r['prec_tau_med_s']:6.3f} {r['n_prec_tau']:3d}/{r['n_repeats']:<3d}")


def _settle_figure(rows, freq_hz, path):
    """Three panels: the average axis settling, the precession about it, and its shape."""

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 1, figsize=(9.5, 11.5), facecolor="white")
    band = 100 * SETTLE_BAND

    ax = axs[0]
    for r in rows:
        m = r["_valid"] & (r["_t"] >= -PRE_S) & (r["_t"] <= r["_stop"])
        ax.plot(r["_t"][m], r["_delta"][m], color=C_TILT, lw=0.8, alpha=0.45)
        if r["settle_10pct_s"] != "":
            ax.plot([r["settle_10pct_s"]], [SETTLE_BAND * r["swing_deg"]],
                    "o", color=C_MARK, ms=4, zorder=5)
    sw = float(np.median([r["swing_deg"] for r in rows]))
    ax.axhline(SETTLE_BAND * sw, color=C_FIT, lw=1.4, ls="--",
               label=f"{band:.0f}% of the median swing ({sw:.1f} deg)")
    ax.axvline(0.0, color=MUTED, lw=1.0)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=MUTED)
    _style(ax, f"{freq_hz:.0f} Hz  -- angle from the AVERAGE axis to the final resting axis"
                f"  (dots: settling time, n={len(rows)})",
           "time from the cut (s)", "angle to final axis (deg)")

    ax = axs[1]
    for r in rows:
        m = r["_valid"] & (r["_t"] >= -PRE_S) & (r["_t"] <= r["_stop"])
        ax.plot(r["_t"][m], r["_env"][m], color=C_TILT, lw=0.9, alpha=0.5)
    ax.axvline(0.0, color=MUTED, lw=1.0)
    _style(ax, "precession about that average axis -- RMS cone half-angle over one rev",
           "time from the cut (s)", "half-angle (deg)")

    ax = axs[2]
    rep = max(rows, key=lambda r: (r["_post"] & r["_valid"]).sum())
    lc = spiral(ax, rep)
    if lc is not None:
        cb = fig.colorbar(lc, ax=ax, pad=0.02)
        cb.set_label("time from the cut (s)", color=MUTED, fontsize=9)
        cb.ax.tick_params(colors=MUTED, labelsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def spiral(ax, rep, t_from=0.0, t_to=None, cmap="viridis"):
    """The residual in the tangent plane, coloured by TIME. The cone decaying, as a spiral.

    Plotted as a plain line this is a disc of ink: several hundred revolutions overdrawn, and
    nothing in it says which pass came first. The motion underneath is a cone whose half-angle
    decays (the envelope falls 8-15 deg to ~4 deg over the window), so what it should look
    like is a spiral winding inward -- and it only looks like one if time is visible.

    A `LineCollection` with one colour per segment is the way to do that without resampling:
    every solved frame is drawn, and the colour carries the axis the plane cannot.
    """

    from matplotlib.collections import LineCollection

    t = rep["_t"]
    t_to = rep["_stop"] if t_to is None else t_to
    m = rep["_valid"] & (t >= t_from) & (t <= t_to)
    x, y, tt = rep["_c1"][m], rep["_c2"][m], t[m]
    if len(x) < 4:
        return None
    seg = np.stack([np.column_stack([x[:-1], y[:-1]]),
                    np.column_stack([x[1:], y[1:]])], axis=1)
    lc = LineCollection(seg, cmap=cmap, linewidths=0.85, alpha=0.9)
    lc.set_array(tt[:-1])
    ax.add_collection(lc)
    lim = 1.08 * float(np.max(np.hypot(x, y)))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    _style(ax, f"residual in the tangent plane, coloured by time — the cone spiralling in\n"
               f"circ:linear {rep['circ_ratio']}, cone {rep['cone_hz']} Hz",
           "e1 (deg)", "e2 (deg)")
    return lc


def _settle_vs_frequency(per_freq, path):
    import matplotlib.pyplot as plt

    f = [r["freq_hz"] for r in per_freq]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), facecolor="white")
    axs[0].errorbar(f, [r["settle_10pct_med_s"] for r in per_freq],
                    yerr=[r["settle_10pct_mad_s"] for r in per_freq],
                    fmt="o-", color=C_TILT, ms=4, lw=1.2, capsize=3)
    # n settled / n repeats on every point. 10 Hz is 3 of 22 and 70 Hz is 4 of 4; without
    # this they are drawn identically and read as equally solid.
    for r in per_freq:
        axs[0].annotate(f"{r['n_settled']}/{r['n_repeats']}",
                        (r["freq_hz"], r["settle_10pct_med_s"]), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=7, color=MUTED)
    _style(axs[0], f"settling time ({100 * SETTLE_BAND:.0f}% band)  -- label is n settled / n",
           "drive frequency (Hz)", "median +- MAD (s)")
    axs[1].errorbar(f, [r["swing_med_deg"] for r in per_freq],
                    yerr=[r["swing_mad_deg"] for r in per_freq],
                    fmt="o-", color=C_FIT, ms=4, lw=1.2, capsize=3)
    _style(axs[1], "swing: initial to final resting axis",
           "drive frequency (Hz)", "median +- MAD (deg)")
    axs[2].plot(f, [r["prec_post_med_deg"] for r in per_freq], "o-",
                color=C_MARK, ms=4, lw=1.2, label="after the cut")
    axs[2].plot(f, [r["prec_pre_med_deg"] for r in per_freq], "o--",
                color=MUTED, ms=4, lw=1.0, label="before")
    axs[2].legend(frameon=False, fontsize=8.5, labelcolor=MUTED)
    _style(axs[2], "precession half-angle about the average axis",
           "drive frequency (Hz)", "median (deg)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


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


def _r(v, nd=3):
    """Round for the CSV, or an empty cell if it is not a number. Never writes `nan`."""

    return round(float(v), nd) if v is not None and np.isfinite(v) else ""


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

    _settle_self_check()
    print("alignment_rate: self-check passed (tau, rate, no-response, kill finder, sigma, "
          "settling, zero-delay, cone)")


def _settle_self_check():
    """The settling metric, on a trajectory whose answer is known analytically."""

    rng = np.random.default_rng(1)
    fs, cone_f, mech_f = 204.0, 40.0, 3.2
    t = np.arange(0, 20, 1 / fs)
    tk, tau = 8.0, 0.30

    # A known axis: n0 -> nf as a settling exponential, plus a circular cone at cone_f and a
    # smaller circular one at the mechanical rate. Both wobbles are put in the tangent plane
    # of nf, which is where `precession` reads them.
    n0 = np.array([0.20, -0.10, -0.974]); n0 /= np.linalg.norm(n0)
    nf = np.array([0.05, 0.18, -0.982]); nf /= np.linalg.norm(nf)
    swing_true = float(np.degrees(np.arccos(np.clip(n0 @ nf, -1, 1))))
    e1, e2 = _basis(nf)
    frac = np.where(t >= tk, 1.0 - np.exp(-(t - tk) / tau), 0.0)
    base = n0[None, :] + frac[:, None] * (nf - n0)[None, :]
    # The rotor cone is CIRCULAR -- 20.7 measured equal amplitudes in quadrature. The rod
    # mode is put in as LINEAR, along one tangent direction, which is both what a swinging
    # mast does and the only shape that shows up in `delta` at all: a circular wobble of
    # constant half-angle leaves `arccos(avg . nf)` constant and would make the test below
    # pass against a filter that does nothing.
    cone_deg, mech_deg = 2.0, 0.8
    wob = (np.radians(cone_deg) * (np.cos(2 * np.pi * cone_f * t)[:, None] * e1
                                   + np.sin(2 * np.pi * cone_f * t)[:, None] * e2)
           + np.radians(mech_deg) * np.cos(2 * np.pi * mech_f * t)[:, None] * e2)
    axis = base + wob + rng.normal(0, 2e-4, base.shape)
    axis /= np.linalg.norm(axis, axis=1)[:, None]

    # ZERO GROUP DELAY. A monotone ramp must cross every level at the same time smoothed as
    # raw. This is the assertion that stands in for "compensating" a delay a centred boxcar
    # does not have -- a causal moving average of the same length would fail it by (k-1)/2
    # samples, which at these spans is 25 to 32 of them.
    ramp = np.column_stack([np.linspace(0, 1, len(t)), np.zeros(len(t)), np.ones(len(t))])
    sm, val, _sp = average_axis(t, ramp / np.linalg.norm(ramp, axis=1)[:, None],
                                cone_f, mech_f)
    raw_u = ramp / np.linalg.norm(ramp, axis=1)[:, None]
    for lev in (0.2, 0.4, 0.6):
        i_raw = int(np.argmax(raw_u[:, 0] >= lev))
        i_sm = int(np.argmax((sm[:, 0] >= lev) & val))
        assert abs(i_raw - i_sm) <= 1, ("group delay is not zero", lev, i_raw, i_sm)

    f_cone, _amp, meas = cone_line(t, axis, tk - PRE_S, tk, cone_f)
    assert meas and abs(f_cone - cone_f) < 0.2, (f_cone, cone_f)

    avg, valid, span = average_axis(t, axis, f_cone, mech_f)
    delta = np.degrees(np.arccos(np.clip(avg @ nf, -1.0, 1.0)))
    pre = (t >= tk - PRE_S) & (t < tk) & valid
    swing = float(np.median(delta[pre]))
    assert abs(swing - swing_true) < 0.3, (swing, swing_true)

    got = settling(t, delta, tk, tk + 5.0, swing, span_s=span)
    # A settling exponential enters the 10% band at -tau*ln(0.1) and never leaves.
    assert abs(got["settle_s"] + tau * np.log(SETTLE_BAND)) < 0.05, got
    # The 10-90% rise of the same exponential is tau*ln(9) = 0.659 s, against a 0.56 s
    # cascade. It clears the gate, and the measurement comes back ~11% wide -- which IS the
    # smoother, symmetric smear and all, and is why the gate is set at the full span and not
    # at some fraction of it.
    assert abs(got["rise_s"] - tau * np.log(9.0)) < 0.15, got
    assert got["rise_s"] > tau * np.log(9.0), \
        ("a centred boxcar can only widen a rise, never sharpen it", got["rise_s"])

    # A rise SHORTER than the cascade is refused rather than reported as the filter's width.
    # tau = 0.05 s is a 0.11 s rise inside a 0.56 s smoother: 24.7's rule, which cost 10 Hz
    # its rate on one window and will cost more on two.
    fast = n0[None, :] + np.where(t >= tk, 1.0 - np.exp(-(t - tk) / 0.05),
                                  0.0)[:, None] * (nf - n0)[None, :] + wob
    fast /= np.linalg.norm(fast, axis=1)[:, None]
    avg_q, val_q, sp_q = average_axis(t, fast, f_cone, mech_f)
    d_q = np.degrees(np.arccos(np.clip(avg_q @ nf, -1.0, 1.0)))
    sw_q = float(np.median(d_q[(t >= tk - PRE_S) & (t < tk) & val_q]))
    got_q = settling(t, d_q, tk, tk + 5.0, sw_q, span_s=sp_q)
    assert not np.isfinite(got_q["rise_s"]) and "not measurable" in got_q["note"], got_q

    # Removing BOTH wobbles is what makes that possible. The once-per-rev boxcar does not
    # touch the rod mode -- sinc(3.2 * 0.25) = 0.23, so a quarter of it survives, and on the
    # real 40 Hz takes leaving it in reads settling times of 0.87/2.16/0.54 s against
    # 0.23/0.92/0.19 s with it gone. Asserted on the settled TAIL, where the only thing left
    # to wobble is the mode itself.
    one, val1, _ = average_axis(t, axis, f_cone, None)
    d1 = np.degrees(np.arccos(np.clip(one @ nf, -1.0, 1.0)))
    tail, tail1 = (t > tk + 3) & valid, (t > tk + 3) & val1
    assert np.ptp(d1[tail1]) > 4.0 * np.ptp(delta[tail]), \
        ("removing the rod mode must flatten the tail",
         np.ptp(d1[tail1]), np.ptp(delta[tail]))

    # Circular vs linear, told apart by the lock-in and not by eye.
    post = (t >= tk) & (t <= tk + 5.0) & valid
    _h, _c1, _c2, ratio = precession(t, axis, avg, nf, f_cone, post)
    assert ratio > 5.0, ("a circular cone must lock in on one sense", ratio)
    lin = base + np.radians(cone_deg) * np.cos(2 * np.pi * cone_f * t)[:, None] * e1
    lin /= np.linalg.norm(lin, axis=1)[:, None]
    avg_l, val_l, _ = average_axis(t, lin, f_cone, mech_f)
    _h, _c1, _c2, ratio_l = precession(t, lin, avg_l, nf, f_cone,
                                       (t >= tk) & (t <= tk + 5.0) & val_l)
    assert ratio_l < 2.0, ("a linear wobble must split evenly", ratio_l)

    # A trace still outside the band when the window ends is REFUSED, not censored to the
    # window length. tau = 3 s against 5 s of record: risen, nowhere near settled.
    slow = n0[None, :] + np.where(t >= tk, 1.0 - np.exp(-(t - tk) / 3.0),
                                  0.0)[:, None] * (nf - n0)[None, :]
    slow /= np.linalg.norm(slow, axis=1)[:, None]
    avg_s, val_s, _ = average_axis(t, slow, f_cone, mech_f)
    d_s = np.degrees(np.arccos(np.clip(avg_s @ nf, -1.0, 1.0)))
    got_s = settling(t, d_s, tk, tk + 5.0, float(np.median(d_s[(t < tk) & val_s])))
    assert not np.isfinite(got_s["settle_s"]), got_s
    # Either refusal is the right one and the tail gate is the stricter, so it fires first:
    # at tau = 3 s the trace is still 5.67 deg from its final axis where it is supposed to
    # have arrived, which is not a slow settle, it is no settled state at all.
    assert ("still outside" in got_s["note"] or "no settled state" in got_s["note"]), got_s

    # No swing, no settling time -- a floor, not a small number.
    flat = np.tile(n0, (len(t), 1)) + rng.normal(0, 1e-3, (len(t), 3))
    flat /= np.linalg.norm(flat, axis=1)[:, None]
    avg_f, val_f, _ = average_axis(t, flat, f_cone, mech_f)
    d_f = np.degrees(np.arccos(np.clip(avg_f @ n0, -1.0, 1.0)))
    got_f = settling(t, d_f, tk, tk + 5.0, float(np.median(d_f[(t < tk) & val_f])))
    assert not np.isfinite(got_f["settle_s"]) and "floor" in got_f["note"], got_f

    # THE PRECESSION SETTLING TIME, on an envelope whose answer is known.
    # A pulse: flat at 2 deg, jumps to 12, decays to 3 with tau = 0.8 s.
    tau_p, pre_p, pk_p, fin_p = 0.8, 2.0, 12.0, 3.0
    env = np.full_like(t, pre_p)
    md = t >= tk
    env[md] = fin_p + (pk_p - fin_p) * np.exp(-(t[md] - tk) / tau_p)
    val = np.ones_like(t, dtype=bool)
    got = precession_settle(t, env, tk, tk + 11.0, val, 0.25)
    assert abs(got["peak_deg"] - pk_p) < 0.05, got
    assert abs(got["final_deg"] - fin_p) < 0.05, got
    assert abs(got["tau_s"] - tau_p) < 0.08, got
    assert abs(got["asym_deg"] - fin_p) < 0.1, got
    # |env - final| falls below 10% of the excursion at tau*ln(10)
    assert abs(got["settle_s"] - tau_p * np.log(10.0)) < 0.05, got

    # A window that ends before the decay finishes must REFUSE the settling time and still
    # return tau -- which is the whole reason tau is reported. 5 s against tau = 4 s.
    env2 = np.full_like(t, pre_p)
    env2[md] = fin_p + (pk_p - fin_p) * np.exp(-(t[md] - tk) / 4.0)
    g2 = precession_settle(t, env2, tk, tk + 5.0, val, 0.25)
    assert not np.isfinite(g2["settle_s"]), g2
    assert "still decaying" in g2["note"], g2
    assert np.isfinite(g2["tau_s"]) and abs(g2["tau_s"] - 4.0) < 1.0, g2

    # A cut that does not move the cone has no decay to time.
    g3 = precession_settle(t, np.full_like(t, 4.0) + rng.normal(0, 0.01, t.shape),
                           tk, tk + 5.0, val, 0.25)
    assert not np.isfinite(g3["settle_s"]) and "floor" in g3["note"], g3

    # The resting axis must recover nf from the settled window, and _polar must round-trip.
    fin = resting_axis(t, axis, tk + FINAL_FROM_S, tk + 5.0)
    assert fin is not None and np.degrees(np.arccos(abs(fin[0] @ nf))) < 0.1, fin
    r, az = _polar(nf, n0)
    assert abs(r - swing_true) < 1e-6, (r, swing_true)
    assert 0.0 <= az < 360.0, az


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take", nargs="?")
    ap.add_argument("--log", default=None, help="sweep.log (default: found beside the take)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sweep", default=None,
                    help="a results/alignment_rate/<stamp> root: solve and pool every take")
    # KEEP THIS AT 1. The cone sits AT the drive frequency and `rev_window`'s null only
    # exists if it is resolved; `disc_axis.solve` was fixed to 1 and this was missed.
    ap.add_argument("--stride", type=int, default=1, help="solve every Nth frame (--sweep)")
    ap.add_argument("--settle", default=None,
                    help="a campaign root: resting axes, settling time and precession")
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
    if a.settle:
        settle_campaign(a.settle, out_dir=a.out, which=a.which, only_hz=a.only_hz)
        sys.exit()
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
