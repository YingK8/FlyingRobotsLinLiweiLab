#!/usr/bin/env python3
"""Why the azimuth traces jump 100-200 deg in one sample, and what it costs the rate.

    uv run python controller/control/azimuth_jumps.py            # self-check
    uv run python controller/control/azimuth_jumps.py <campaign> # scan a sweep root

Analysis only -- nothing here drives a coil and nothing here edits `alignment_rate` or
`disc_axis`. The prototype fix is `keep_radial()`; wiring it in is a one-line change left
for whoever owns that file.

THE SYMPTOM
-----------
`report/trial_fits.png` from `results/alignment_rate/20260909_205843` shows near-vertical
steps in the azimuth trace. Two named by the operator, both reproduced here exactly:

    take               freq   t - t_kill   d(azimuth)   dt        implied rate   radial
    2026-09-09_213757  70 Hz   +188 ms      +69.4 deg   20.6 ms     3376 deg/s   8.97 -> 3.50
    2026-09-09_213757  70 Hz   +211 ms      +96.3 deg   23.3 ms     4136 deg/s   3.50 -> 4.04
    2026-09-09_213757  70 Hz   +564 ms     -125.4 deg   15.6 ms    -8033 deg/s   2.65 -> 4.60
    2026-09-09_212646  50 Hz   +185 ms     +157.3 deg   21.2 ms     7435 deg/s   4.81 ->15.06
    2026-09-09_212646  50 Hz   +407 ms     -141.3 deg   21.7 ms    -6515 deg/s   3.47 -> 2.76

The last column is the whole story: the median radial lean over those same windows is
29.3 deg (70 Hz) and 17.2 deg (50 Hz). Every jump happens where the lean has collapsed to
a few degrees.

ROOT CAUSE: A COORDINATE SINGULARITY, NOT A HEMISPHERE FLIP
-----------------------------------------------------------
`azimuth_from_rest` measures the DIRECTION of lean about the rest datum. The direction of a
zero-length vector is undefined, and near-zero is worse than undefined -- it is defined and
wrong, because the perpendicular component that `arctan2` reads is then smaller than the
estimator's own noise. Measured on this campaign: the frame-to-frame scatter of the solved
axis during the settled pre-cut hold is **3.18 deg** (median over all 81 kills, p90 6.73).
The azimuth error that scatter produces is `sigma / sin(radial)`, so

    radial = 30 deg  ->   6 deg of azimuth noise per frame  ->   320 deg/s   (invisible)
    radial =  5 deg  ->  37 deg                             ->  1850 deg/s   (visible)
    radial =  3 deg  ->  61 deg                             ->  3100 deg/s   (the figure)

which is the observed onset, to within the width of the bins. Across the campaign 82.4% of
the impossible jumps sit below 5 deg of lean, against 5.5% of window samples -- a 15x
enrichment, and the strongest single number in this file.

THE HEMISPHERE HYPOTHESIS, TESTED AND EXCLUDED
----------------------------------------------
The operator's reading is that the spikes are a pose-level sign flip: the robot never turns
over, so any reconstruction below the datum is wrong and should be reflected. The constraint
is right, it is available at every frame, and it is worth enforcing -- but it is not what
makes these spikes, and four independent measurements say so:

* **The reflect count and the jumps do not overlap.** 218 of 29128 kill-window frames
  (0.75%) have the raw `axis.csv` direction below the rest datum. They sit at a median
  radial of **39.2 deg** (p90 41.2) against 14.3 deg for all frames -- i.e. at LARGE lean,
  where the fixed `disc_axis.ORIENT_REF` runs near perpendicular, and NOT at the small-lean
  frames where the jumps are. On the two named takes, 2026-09-09_212646 has **zero**
  below-datum frames in the entire recording and still shows both jumps.
* **Enforcing the constraint changes almost nothing.** Counting impossible jumps in the kill
  windows under three sign rules: raw `axis.csv` 688, `_hemisphere` against the rest datum
  639, `orient_continuous` 639. The a-priori upper-hemisphere rule removes 7% of them and
  leaves 93%. It also agrees with `orient_continuous` sample for sample on every take here.
  **The constraint is already enforced downstream**, so moving it into `disc_axis` fixes
  nothing in the rate -- it would only tidy the sign in `axis.csv` for whoever plots that
  file raw (`pose/disc_video`), which is worth doing on its own merits and is not this bug.
* **The pose estimate is not worse at the jumps.** `vs_conic_deg` (the minor-axis vs conic
  disagreement, the file's own quality column) reads median 8.03 deg at the jump samples
  against 6.88 deg everywhere else. A wrong `fuse` branch is a 30-70 deg line error by that
  function's own docstring; nothing of the kind is present.
* **The axis is not moving.** Its great-circle speed -- sign-invariant, so blind to every
  hemisphere question -- has median 125 deg/s over the kill windows and p99.9 **1002 deg/s**.
  At the jump samples it is a median of 323 deg/s (5.9 deg per frame). The robot is not
  turning 150 deg in 20 ms; only the coordinate is.

Two smaller findings from the same scan, neither of which is the cause:

* `disc_axis.ORIENT_REF = (0, 0, -1)` is a fixed world direction, and the rest attitude sits
  a median of **72.6 deg** from it (world +z is camera A's axis, not up). 424 kill-window
  frames come within 5 deg of the 90 deg singular line, which is exactly the mechanism that
  produces the 218 below-datum frames. Six takes carry them, three in long contiguous runs
  (721-1002 frames on 2026-09-09_213341, 2026-09-10_000608 and 2026-09-09_214051) -- a run
  that long is the robot genuinely leaning past the reference, not an isolated bad frame.
  Referencing the sign to the take's own rest datum instead of a fixed vector would remove
  it at source.
* The `+-180` wrap of the change (the last line of `azimuth_from_rest`) puts a 360 deg step
  wherever the trace crosses half a turn from its pre-cut median: 39 such steps across the
  campaign, **0** of them inside a fitted ramp. It draws a vertical line in the figure and
  changes no rate.
* `np.unwrap` is cleared by construction: it wraps each step into +-180, so it can neither
  create a 150 deg step nor repair one.

HOW MUCH IT COSTS
-----------------
`relu_window` fits the first turning-point-to-turning-point rise of `JUMP_DEG`, and a
singularity spike is the steepest rise in any window it lands in. Across the campaign:

    45 of 84 kills carry at least one jump inside their window
    30 of 84 have one INSIDE the fitted ramp -- those rates are the corrupted ones

(the second number is what `scan()` prints, and it moves with whatever `relu_window` is
doing on the day: `alignment_rate.py` was being edited while this was measured, so re-run
the scan before quoting it. The first number depends on nothing but the axis and the datum.)

and the bias is upward at every affected frequency, median `rate_relu_deg_s` per frequency,
current code vs the same code with samples below 5 deg of lean dropped:

    freq    now   masked   ratio        freq    now   masked   ratio
     10     487      292    1.67         70     951      865    1.10
     20     828      689    1.20         80    1879     1117    1.68
     30    1013      801    1.26         90    1176     1176    1.00
     40    1542     1293    1.19        100     883      883    1.00
     50     870      710    1.22        110     599      599    1.00
     60     485      457    1.06

So yes, it biases the rate, and worse than a uniform bias would be: 10-80 Hz are inflated
by 6-67% and 90-110 Hz are untouched, because at high drive the lean is large enough that
the axis never comes near the datum. That is a frequency-DEPENDENT distortion of exactly
the rate-vs-frequency trend the campaign exists to measure.

THE FIX AND ITS PRICE
---------------------
`keep_radial()`: drop the samples whose lean is below `MIN_RADIAL_DEG` before the azimuth
is unwrapped. Same idea as `pose/disc_video.AZIMUTH_MIN_RADIAL_DEG`, which blanks the
overlay trace for the same reason, but at 5 deg rather than that file's 1.0 -- 1.0 deg is
the threshold for "meaningless", 5 deg is the threshold for "noisier than the signal",
and this module needs the second one. Measured cost of the choice:

    threshold   jumps removed   window samples dropped
      1 deg         24.9 %             0.7 %
      3 deg         58.6 %             3.0 %
      5 deg         82.4 %             5.5 %      <- the knee
      8 deg         95.8 %            16.8 %

It belongs in `azimuth_from_rest` and not in each caller: `trial_panels`, `rate_scatter`
and both pooling paths all route through that one function, so one guard there covers every
one of them.

**The trade-off is real and is not a rounding error.** Masking does not recover the
azimuth during the pole passage -- nothing can, the measurement is not in the data -- it
only stops the undefined stretch being read as motion. Three consequences to accept:

1. A kill whose swing genuinely carries the axis THROUGH the datum loses the samples where
   it crosses, and its azimuth either side is separated by a real, unmeasurable 180 deg
   ambiguity that `unwrap` will guess at. Those kills should be reported, not fitted.
2. It costs 5.5% of samples at 5 deg, but not evenly: they are concentrated at the low
   frequencies where the lean is small, so 10 Hz pays most of it.
3. Every published rate for 10-80 Hz moves down. The numbers in the table above are the
   correction, and they are large enough that any figure already circulated is wrong.

The honest alternative, if the pole passages turn out to be common enough to matter, is to
stop using azimuth for those takes and report `rotation_from_baseline` instead -- the
great-circle angle from the pre-cut attitude has no singularity anywhere. It measures a
different thing (total rotation, not direction of lean), which is why it is not simply
swapped in here.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from controller.control import alignment_rate as ar

#: Sample-to-sample azimuth rate above which the sample is not motion. The axis's own
#: great-circle speed over all 81 kill windows of `20260909_205843` has p99.9 = 1002 deg/s
#: and the measured alignment rates are 500-2000 deg/s, so 3000 is ~3x anything physical
#: while still being under the 3100 deg/s the noise produces at 3 deg of lean. The result is
#: insensitive to the exact value: the jumps cluster at 3000-15000 deg/s with nothing
#: between 1000 and 3000.
IMPOSSIBLE_DEG_S = 3000.0

#: Below this much lean the azimuth is noisier than the signal. From the measured 3.18 deg
#: of per-frame axis scatter: `3.18 / sin(5 deg)` is 37 deg of azimuth noise per frame,
#: which at the 19.6 ms sampling is 1850 deg/s -- already at the edge. Anything lower is
#: unusable. Drops 5.5% of the campaign's kill-window samples and removes 82% of the jumps.
MIN_RADIAL_DEG = 5.0


def jumps(t, azimuth, radial, thr_deg_s=IMPOSSIBLE_DEG_S):
    """Indices ``i`` where the step from ``i`` to ``i+1`` is physically impossible.

    Returns ``(idx, rate_deg_s, radial_at_jump)``, where the radial reported is the SMALLER
    of the two samples -- the singularity is a property of whichever end is nearer the pole.
    """

    t = np.asarray(t, float)
    azi = np.asarray(azimuth, float)
    rad = np.asarray(radial, float)
    dt = np.diff(t)
    rate = np.where(dt > 0, np.diff(azi) / np.where(dt > 0, dt, 1.0), 0.0)
    idx = np.flatnonzero(np.abs(rate) > thr_deg_s)
    return idx, rate[idx], np.minimum(rad[idx], rad[idx + 1])


def axis_speed(t, axis):
    """Great-circle speed of the axis LINE, deg/s, one shorter than ``t``.

    The control measurement. It is sign-invariant (`abs` of the dot, as `tilt_from` does)
    and frame-independent, so it is blind to every hemisphere, unwrap and datum question --
    if the azimuth says 8000 deg/s and this says 300, the motion is not there.
    """

    a = np.asarray(axis, float)
    cos = np.abs((a[1:] * a[:-1]).sum(1)).clip(0.0, 1.0)
    dt = np.diff(np.asarray(t, float))
    return np.degrees(np.arccos(cos)) / np.where(dt > 0, dt, np.inf)


def reflected(axis, up):
    """Frames the a-priori upper-hemisphere constraint would flip: ``axis . up < 0``.

    The robot never turns over, so this is decidable per frame with no appeal to continuity.
    `alignment_rate._hemisphere` already applies exactly this rule; the mask is here to
    COUNT the frames, because the question was whether they coincide with the azimuth jumps.
    Measured on `20260909_205843`: they do not -- median lean 39.2 deg at these frames
    against 2.5 deg at the jumps.
    """

    up = np.asarray(up, float)
    return np.asarray(axis, float) @ (up / np.linalg.norm(up)) < 0.0


def keep_radial(t, axis, up, min_radial_deg=MIN_RADIAL_DEG):
    """PROTOTYPE FIX. Boolean mask of the samples whose lean has a defined direction.

    Applied as ``t[m], axis[m]`` before `alignment_rate.azimuth_from_rest`, which is why it
    returns a mask and not an angle: `transient` and `relu_window` both read the actual
    sample times, so dropping rows is safe where inserting NaN is not (`_smooth` convolves
    and would spread one NaN over a whole window).

    The radial angle is taken sign-invariantly through `alignment_rate.angles`, so this is
    the same [0, 90] number the rest of the module measures leans with -- and so the mask is
    unaffected by whichever hemisphere the frame arrived in.
    """

    radial, _azi = ar.angles(np.asarray(axis, float), np.asarray(up, float))
    return radial >= float(min_radial_deg)


def scan(root, thr_deg_s=IMPOSSIBLE_DEG_S):
    """Every ``outcome=ok`` take of a campaign -> one row per kill. Prints, returns rows.

    Reads the per-frequency `*/index.csv` layout `tilt_sweep` writes. A take with no solved
    axis or no `sweep.log` has no datum and is skipped, the same rule `sweep_report` uses.
    """

    root = Path(root)
    rows = []
    for idx_path in sorted(root.glob("*/index.csv")):
        for r in csv.DictReader(open(idx_path)):
            if r["outcome"] != "ok" or not Path(r["log"]).exists():
                continue
            take = Path(r["flight"])
            if not (take / "axis.csv").exists() and not (take / "axis_minor.csv").exists():
                continue
            try:
                t, axis, _q = ar.load(take)
            except (IndexError, KeyError, OSError):     # an empty or half-written axis file
                continue
            pts = ar.timeline(r["log"])
            try:
                rest = ar.datum(t, axis, pts)[0]
            except SystemExit:
                continue
            radial, _ = ar.angles(axis, rest)
            speed = axis_speed(t, axis)
            flip = reflected(axis, rest)
            for i_k, tk in enumerate([k for p in pts for k in p["kills"]], 1):
                azi = ar.azimuth_from_rest(t, axis, rest, tk)
                if azi is None:
                    continue
                rel = t - tk
                w = (rel >= -ar.PRE_S) & (rel <= ar.POST_TO_S)
                idx, rate, rad = jumps(t[w], azi[w], radial[w], thr_deg_s)
                res = ar.transient(t, azi, np.zeros_like(t), tk,
                                   min_amp=0.0, spin_hz=float(r["freq_hz"]))
                lag = res["relu_lag_s"] if res else np.nan
                win = res["relu_win_s"] if res else np.nan
                rel_w = rel[w]
                in_ramp = int(sum(1 for i in idx
                                  if np.isfinite(lag) and lag <= rel_w[i + 1] <= lag + win))
                rows.append({
                    "freq_hz": float(r["freq_hz"]), "repeat": int(r["repeat"]),
                    "take": take.name, "kill": i_k, "n_jumps": int(idx.size),
                    "n_in_ramp": in_ramp,
                    "n_reflected": int(flip[w].sum()),
                    "min_radial_at_jump": float(rad.min()) if idx.size else float("nan"),
                    "radial_at_reflected": (float(np.median(radial[w & flip]))
                                            if (w & flip).any() else float("nan")),
                    "median_radial_deg": float(np.median(radial[w])),
                    "max_axis_speed_deg_s": float(np.nanmax(speed[w[:-1]])),
                    "rate_relu_deg_s": float(res["rate_relu_deg_s"]) if res else float("nan"),
                })
    print(f"{len(rows)} kills, {sum(1 for x in rows if x['n_jumps'])} with a jump over "
          f"{thr_deg_s:.0f} deg/s, {sum(1 for x in rows if x['n_in_ramp'])} with one inside "
          f"the fitted ramp, {sum(1 for x in rows if x['n_reflected'])} with a below-datum frame")
    print(f"\n{'freq':>5} {'kills':>6} {'hit':>4} {'in ramp':>8} {'med radial':>11} "
          f"{'radial @ jump':>14} {'reflected':>10} {'radial @ refl':>14}")
    for f in sorted({x["freq_hz"] for x in rows}):
        g = [x for x in rows if x["freq_hz"] == f]
        mn = [x["min_radial_at_jump"] for x in g if x["n_jumps"]]
        rf = [x["radial_at_reflected"] for x in g if x["n_reflected"]]
        print(f"{f:5.0f} {len(g):6d} {sum(1 for x in g if x['n_jumps']):4d} "
              f"{sum(1 for x in g if x['n_in_ramp']):8d} "
              f"{np.median([x['median_radial_deg'] for x in g]):11.2f} "
              f"{(np.median(mn) if mn else float('nan')):14.2f} "
              f"{sum(x['n_reflected'] for x in g):10d} "
              f"{(np.median(rf) if rf else float('nan')):14.2f}")
    return rows


def _self_check():
    """Synthetic pole passage. Hermetic -- no take on disk, no hardware, no serial."""

    rng = np.random.default_rng(7)
    dt = 0.0196                       # measured stride-4 sampling of the solved axis
    # Starts well before the cut: `azimuth_from_rest` needs 20 pre-kill samples for its
    # reference median, which at this stride is 0.4 s.
    t = np.arange(-0.6, 1.2, dt)
    up = np.array([0.0, 0.0, 1.0])
    t_kill = 0.0                      # the pole passage is at 0.6 s, well inside the window

    def lean(radial_deg, turn_deg_s=100.0):
        """A lean of the given profile turning at a sane rate, plus the measured scatter."""

        rad = np.radians(np.asarray(radial_deg, float) * np.ones_like(t))
        phi = np.radians(turn_deg_s * t)
        a = np.c_[np.sin(rad) * np.cos(phi), np.sin(rad) * np.sin(phi), np.cos(rad)]
        # 3.18 deg is the per-frame axis scatter measured on the pre-cut holds of
        # 20260909_205843 -- the number that sets where the singularity starts to bite.
        # Split over the two components that actually tilt a unit vector, so the RESULTING
        # per-frame tilt is 3.18 deg rather than sqrt(2) times it.
        a = a + np.radians(3.18 / np.sqrt(2.0)) * rng.standard_normal(a.shape)
        return a / np.linalg.norm(a, axis=1, keepdims=True)

    # Dips to 1 deg at t = 0.6 s and comes back: a pole passage, nothing more. The real
    # takes go lower -- 0.03 deg during the rest hold, 2.65 deg inside the 70 Hz kill window.
    near = lean(np.minimum(1.0 + 18.0 * np.abs(t - 0.6) / 0.6, 25.0))

    # The motion itself stays physical. This is the control that proves the jumps below are
    # a coordinate artefact and not a fast synthetic robot.
    assert np.nanmax(axis_speed(t, near)) < IMPOSSIBLE_DEG_S, "synthetic axis moves too fast"

    azi = ar.azimuth_from_rest(t, near, up, t_kill, pre_s=0.5)
    assert azi is not None
    radial, _ = ar.angles(near, up)
    idx, _rate, rad = jumps(t, azi, radial)
    assert idx.size, "the pole passage should produce impossible azimuth rates"
    # ...and they must sit at the pole, which is the hypothesis this whole module tests.
    # Stated on the median, as the campaign scan states it: the tail reaches 18.3 deg there
    # (p90 6.2), so a hard ceiling on the maximum would be a claim the real data does not make.
    assert np.median(rad) < MIN_RADIAL_DEG, f"jumps at median radial {np.median(rad):.1f} deg"

    m = keep_radial(t, near, up)
    assert 0.5 < m.mean() < 1.0, f"mask kept {m.mean():.0%} -- expected a dip, not a wipe"
    azi2 = ar.azimuth_from_rest(t[m], near[m], up, t_kill, pre_s=0.5)
    idx2, _, _ = jumps(t[m], azi2, radial[m])
    assert idx2.size < 0.25 * idx.size, f"{idx2.size}/{idx.size} jumps survive the mask"

    # A 25 deg lean turning at the same rate is the 90-110 Hz case: well conditioned, so the
    # mask must not fire and no jump may appear. Guards against a fix that eats good data.
    far = lean(25.0)
    assert keep_radial(t, far, up).all(), "mask dropped samples from a well-conditioned lean"
    idx3, _, _ = jumps(t, ar.azimuth_from_rest(t, far, up, t_kill, pre_s=0.5),
                       ar.angles(far, up)[0])
    assert idx3.size == 0, "a 25 deg lean should never produce an impossible azimuth rate"

    # The hemisphere hypothesis, as a check and not as an assertion about the campaign: a
    # deliberately reflected frame at LARGE lean is caught by `reflected` and by
    # `_hemisphere`, and -- the point -- costs the azimuth nothing, because the sign is
    # re-resolved before the angle is taken. That is why enforcing the constraint upstream
    # in `disc_axis` does not remove these jumps.
    flipped = far.copy()
    flipped[40] = -flipped[40]
    assert reflected(flipped, up).sum() == 1, "reflected() should see exactly the flipped frame"
    a_ok = ar._hemisphere(flipped, up)
    assert np.allclose(a_ok, far), "_hemisphere should undo an a-priori-wrong sign exactly"
    idx4, _, _ = jumps(t, ar.azimuth_from_rest(t, flipped, up, t_kill, pre_s=0.5),
                       ar.angles(flipped, up)[0])
    assert idx4.size == 0, "a lone hemisphere flip at large lean must not make an azimuth jump"

    print("azimuth_jumps: self-check ok")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        scan(sys.argv[1])
    else:
        _self_check()
