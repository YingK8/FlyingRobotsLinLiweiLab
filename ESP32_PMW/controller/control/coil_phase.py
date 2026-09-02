#!/usr/bin/env python3
"""Per-channel coil CURRENT phase: measure it, and derive the trim that flattens it. NO CAMERAS.

The ESP32 commands a phase. The robot responds to the phase of the coil *current*, and the
series RLC puts those two apart by an amount that differs channel to channel. That
difference is an uncontrolled contribution to rotor tilt, mixed in with the amplitude
imbalance `coil_balance.py` already trims. `theory.md` 22 is the write-up; this is the tool.

WHY ONLY TWO NUMBERS PER CHANNEL
--------------------------------
For a series RLC the current phase collapses to

    theta_k(f) = atan( Q_k * (f/f0_k - f0_k/f) )

which contains no V, no absolute R, L or C -- only the resonant frequency and the Q. That
matters more here than it looks. The CS path's absolute gain is the least trustworthy
number on this board: the VNH5019 sense ratio spreads 4670-10110 part to part (+-30 %,
`hw_references/VNH5019_CS.png`), `R4` in the CS->ADC divider has no value in the BOM
(`docs/PCB_Design_Documentation.md` 157), and the empirical `SENS ~ 15.3 A/V` is ~3.2x what
a 1 kohm load implies, unexplained. **All of it cancels in a phase measurement**, because a
real scalar gain drops out of an argument. The same cancellation is why the magnitude fit
below solves for (A, f0, Q) and throws A away.

TWO INDEPENDENT ROUTES TO THE SAME NUMBER
-----------------------------------------
  * `fit_run()`   -- the SHAPE of |I|(f) over a ramp gives (f0, Q) per channel. Costs
                     nothing: every run CSV already logs `i_a..i_d`.
  * `measure()`   -- a `probe=` lock-in burst reports the phase DIRECTLY at a held
                     frequency, via the firmware, which is the only place that knows the
                     commanded phase exactly.

They share no arithmetic, so agreement between them is the check that the lock-in is real
and not an artefact of its own reference. Disagreement means one of them is wrong, and
`compare()` prints both rather than averaging them.

WHAT THIS IS NOT
----------------
Not field-oriented control. `theory.md` 17.2 rules FOC out because commutation needs signed
per-phase current sampled synchronously, every cycle, in real time; the CS pin is unsigned
and paced at ~1 kHz. This is a *calibration*: one frequency, held for a second, coherently
averaged over hundreds of cycles, off the flight path. 17.2 still stands.

    uv run python controller/control/coil_phase.py                 # self-check, no hardware
    uv run python controller/control/coil_phase.py --fit <csv>     # (f0,Q) from a run
    uv run python controller/control/coil_phase.py --measure       # drive the coils
"""

from __future__ import annotations

import math
import sys
import time

import numpy as np

from controller.control import constants as C
from controller.control.link import SerialComm, parse_probe

TAGS = "ABCD"

# Frequencies the probe sweep visits. Chosen to bracket the resonance the 800 uF bank
# predicts (~150 Hz) from both sides -- a sweep that only approaches f0 from below cannot
# separate f0 from Q, which is exactly how `COIL_SERIES_C_F` came to be fitted wrong in the
# first place (`constants.py`, and 18.6).
PROBE_F_HZ = (90.0, 110.0, 130.0, 150.0, 170.0, 190.0, 210.0)
PROBE_MS = 1200          # per point. 7 points is ~8.4 s of drive, against a 55 s ramp cap.
MIN_COHERENCE = 0.90     # below this the burst is noise and the point is dropped, not used
FIT_MIN_POINTS = 12
EDGE_BINS = 2            # the |I| peak must sit this far inside the swept band


def theta(f, f0, q):
    """Current phase re. the commanded phase, in degrees. Negative below resonance.

    The whole model. `f` may be an array. See the module docstring for why this needs no
    R, L, C or drive amplitude.
    """

    f = np.asarray(f, float)
    return np.degrees(np.arctan(q * (f / f0 - f0 / f)))


def _mag(f, a, f0, q):
    """|I|(f) for a series RLC, normalised so `a` is the current at resonance."""

    return a / np.sqrt(1.0 + (q * (f / f0 - f0 / f)) ** 2)


def fit_channel(f, i):
    """Fit (f0_hz, q) from the SHAPE of one channel's |I|(f). Returns None if it cannot.

    Refuses rather than returns a number it cannot support:

      * fewer than `FIT_MIN_POINTS` usable samples, or
      * the |I| peak sits at the edge of the swept band.

    The second guard is the one that matters. A ramp that stops below resonance still fits
    happily -- `curve_fit` will hand back an f0 above the band with a plausible-looking
    covariance -- and that is precisely the failure that put 400 uF into `constants.py`
    (18.6: "it was fitted in the one band where it is the only thing visible").
    """

    from scipy.optimize import curve_fit

    f = np.asarray(f, float)
    i = np.asarray(i, float)
    ok = np.isfinite(f) & np.isfinite(i) & (i > 0)
    f, i = f[ok], i[ok]
    if len(f) < FIT_MIN_POINTS:
        return None

    order = np.argsort(f)
    f, i = f[order], i[order]
    peak = int(np.argmax(i))
    if peak < EDGE_BINS or peak > len(f) - 1 - EDGE_BINS:
        return None     # the band does not bracket resonance; f0 and Q are not separable

    try:
        p, cov = curve_fit(_mag, f, i, p0=[float(i[peak]), float(f[peak]), 1.0],
                           bounds=([0.0, 20.0, 0.02], [1e4, 500.0, 50.0]), maxfev=20000)
    except Exception:
        return None
    sig = np.sqrt(np.diag(cov)) if np.all(np.isfinite(cov)) else np.full(3, np.nan)
    return {"f0_hz": float(p[1]), "q": float(p[2]),
            "f0_sigma": float(sig[1]), "q_sigma": float(sig[2]), "n": int(len(f))}


def fit_run(path):
    """Per-channel (f0, Q) from a run CSV. -> {tag: fit-or-None}. Reuses takeoff_report.load."""

    from controller.control.takeoff_report import load     # local: takeoff_report imports us

    d = load(path)
    return {t: fit_channel(d["f_hz"], d["i_" + t.lower()]) for t in TAGS}


def spread_deg(f, fits):
    """Per-channel theta at `f`, referenced to the four-channel mean.

    Absolute phase is deliberately not reported. The CS path's own delay (VNH5019
    t_DSENSE is 20-50 us, 1.4-3.6 deg at 200 Hz) is common to four identical channels and
    cancels here; it does not cancel in an absolute number. What tilts the rotation axis is
    the channel-to-channel difference anyway, so nothing is lost by referencing the mean.
    """

    th = np.array([theta(f, fits[t]["f0_hz"], fits[t]["q"]) if fits.get(t) else np.nan
                   for t in TAGS])
    return th - np.nanmean(th)


def trim_deg(f, fits):
    """The per-channel phase to SUBTRACT from the command so the currents come out level."""

    return -spread_deg(f, fits)


def measure(port=None, freqs=PROBE_F_HZ, ms=PROBE_MS, verbose=True):
    """Drive a `probe=` lock-in burst at each frequency. -> {f_hz: ProbePoint}.

    Each point is bracketed by its own `stop`, so the thermal clock in `link._note_drive`
    closes per burst rather than counting the gaps between them as drive.
    """

    total_s = len(freqs) * ms / 1000.0
    if total_s > C.MAX_RAMP_S:
        raise SystemExit(f"{len(freqs)} points x {ms} ms = {total_s:.0f}s of drive, "
                         f"over the {C.MAX_RAMP_S:.0f}s cap -- shorten the sweep")

    # Same gate `fly()` uses. `ai/` is gitignored, so a fresh clone has no thermal model and
    # must refuse to energise rather than guess the coils are cold.
    from ai.thermal import coil_thermal
    coil_thermal.wait_until_safe()

    link = SerialComm(port=port)
    out = {}
    try:
        link.reset_device()
        time.sleep(1.5)
        for f in freqs:
            link.handle_serial_comm(f"probe={f:.1f}:{ms}")
            deadline = time.monotonic() + ms / 1000.0 + 3.0
            pt = None
            while time.monotonic() < deadline and pt is None:
                line = link.handle_serial_comm()
                if not line:
                    time.sleep(0.002)
                    continue
                pt = parse_probe(line)
            link.handle_serial_comm("stop")
            time.sleep(0.2)
            if pt is None:
                print(f"  {f:6.1f} Hz  no answer -- skipped")
                continue
            if pt.coherence is not None and pt.coherence < MIN_COHERENCE:
                print(f"  {f:6.1f} Hz  coherence {pt.coherence:.2f} < {MIN_COHERENCE} "
                      f"-- dropped, not used")
                continue
            out[f] = pt
            if verbose:
                rel = np.array(pt.phase_deg) - float(np.mean(pt.phase_deg))
                print(f"  {f:6.1f} Hz  " + "  ".join(
                    f"{t}{d:+6.2f}" for t, d in zip(TAGS, rel)) + "  (re. mean)")
    finally:
        link.handle_serial_comm("stop")
        link.close()
    if not out:
        raise SystemExit("no usable probe points -- is the firmware built with probe= ?")
    return out


def fit_probe(points):
    """(f0, Q) per channel from measured phases. -> {tag: fit-or-None}.

    The second route. `tan(theta) = Q (f/f0 - f0/f)` is linear in the two unknowns once
    you have theta at three or more frequencies, so this needs no magnitude at all.
    """

    from scipy.optimize import curve_fit

    f = np.array(sorted(points))
    fits = {}
    for k, t in enumerate(TAGS):
        th = np.array([points[x].phase_deg[k] for x in f])
        ok = np.isfinite(th)
        if ok.sum() < 3:
            fits[t] = None
            continue
        try:
            p, cov = curve_fit(lambda x, f0, q: theta(x, f0, q), f[ok], th[ok],
                               p0=[150.0, 1.0], bounds=([20.0, 0.02], [500.0, 50.0]),
                               maxfev=20000)
        except Exception:
            fits[t] = None
            continue
        sig = np.sqrt(np.diag(cov)) if np.all(np.isfinite(cov)) else np.full(2, np.nan)
        fits[t] = {"f0_hz": float(p[0]), "q": float(p[1]),
                   "f0_sigma": float(sig[0]), "q_sigma": float(sig[1]), "n": int(ok.sum())}
    return fits


def report(fits, at_hz=None):
    """Print (f0, Q) per channel and the phase spread they imply. Returns the trim table."""

    at = at_hz or C.F_HOVER_TRACK_HZ
    print(f"per-channel series RLC, from the |I|(f) shape:")
    for t in TAGS:
        fit = fits.get(t)
        if not fit:
            print(f"   {t}  no fit -- band does not bracket resonance, or too few points")
            continue
        print(f"   {t}  f0 = {fit['f0_hz']:6.1f} +-{fit['f0_sigma']:4.1f} Hz   "
              f"Q = {fit['q']:5.2f} +-{fit['q_sigma']:4.2f}   ({fit['n']} pts)")

    if not all(fits.get(t) for t in TAGS):
        print("\nincomplete -- no trim table until all four channels fit")
        return None

    sp = spread_deg(at, fits)
    print(f"\ncurrent phase at {at:.0f} Hz, referenced to the four-channel mean:")
    for t, d in zip(TAGS, sp):
        print(f"   {t}  {d:+6.2f} deg")
    pk = float(np.nanmax(sp) - np.nanmin(sp))
    tick = 360.0 * 25e-6 * at
    print(f"   spread {pk:.2f} deg peak-to-peak")
    print(f"\nagainst {tick:.2f} deg, the commanded-phase quantum at {at:.0f} Hz "
          f"(25 us ISR tick, PwmController.cpp:146)")
    print("   -- the analog error is the larger of the two"
          if pk > tick else
          "   -- the digital quantum dominates; a trim finer than it buys nothing")

    trim = trim_deg(at, fits)
    print("\nPHASE_TRIM_DEG for src/drive_common.h (subtracted from the command):")
    print("static const float PHASE_TRIM_DEG[NUM_CHANNELS] = {"
          + ", ".join(f"{v:+.2f}f" for v in trim) + "};")
    return tuple(float(v) for v in trim)


def compare(ramp_fits, probe_fits, at_hz=None):
    """Print the two independent routes side by side. Never averages them."""

    at = at_hz or C.F_HOVER_TRACK_HZ
    print(f"\n{'':4}{'ramp |I|(f)':>22}{'probe lock-in':>22}{'theta @':>10}")
    print(f"{'':4}{'f0 / Q':>22}{'f0 / Q':>22}{f'{at:.0f} Hz':>10}")
    for k, t in enumerate(TAGS):
        a, b = ramp_fits.get(t), probe_fits.get(t)
        sa = f"{a['f0_hz']:6.1f} / {a['q']:4.2f}" if a else "-- / --"
        sb = f"{b['f0_hz']:6.1f} / {b['q']:4.2f}" if b else "-- / --"
        d = (f"{float(theta(at, b['f0_hz'], b['q']) - theta(at, a['f0_hz'], a['q'])):+6.2f}"
             if a and b else "    --")
        print(f"  {t} {sa:>21} {sb:>21} {d:>9}  (probe - ramp)")
    print("\n  the two share no arithmetic; a disagreement means one of them is wrong")


def demo():
    # 1. The model itself.
    assert abs(float(theta(150.0, 150.0, 1.0))) < 1e-9, "no phase shift at resonance"
    assert float(theta(210.0, 150.0, 1.0)) > 0, "current lags above resonance"
    assert float(theta(100.0, 150.0, 1.0)) < 0, "current leads below resonance"
    # Higher Q means a sharper phase swing for the same detuning -- the reviewer's
    # d(theta) ~ 2 Q (df/f0), which is the small-detuning limit of the same expression.
    assert float(theta(160.0, 150.0, 2.0)) > float(theta(160.0, 150.0, 1.0))
    lin = float(theta(153.0, 150.0, 1.0))
    approx = math.degrees(2 * 1.0 * (3.0 / 150.0))
    assert abs(lin - approx) < 0.05, (lin, approx)

    # 2. The magnitude fit recovers (f0, Q) from shape alone...
    f = np.linspace(90.0, 220.0, 140)
    truth = (152.0, 0.85)
    i = _mag(f, 4.2, *truth)
    got = fit_channel(f, i)
    assert abs(got["f0_hz"] - truth[0]) < 0.5, got
    assert abs(got["q"] - truth[1]) < 0.02, got

    # 3. ...and is INDEPENDENT of the drive amplitude and the CS gain, which is the whole
    # reason a phase calibration survives a +-30 % sense ratio that an amplitude one does not.
    scaled = fit_channel(f, i * 3.17)
    assert abs(scaled["f0_hz"] - got["f0_hz"]) < 1e-3, (scaled, got)
    assert abs(scaled["q"] - got["q"]) < 1e-6, (scaled, got)

    # 4. A band that stops short of resonance must REFUSE, not invent an f0 above it.
    # This is the 400 uF failure from 18.6, reproduced.
    below = f < 140.0
    assert fit_channel(f[below], i[below]) is None, "a ramp short of f0 must not fit"
    assert fit_channel(f[:6], i[:6]) is None, "too few points must not fit"

    # 5. Four detuned channels -> a spread, and a trim that cancels it.
    fits = {t: {"f0_hz": f0, "q": 0.85, "f0_sigma": 0.4, "q_sigma": 0.02, "n": 140}
            for t, f0 in zip(TAGS, (145.0, 150.0, 155.0, 160.0))}
    sp = spread_deg(190.0, fits)
    assert abs(float(np.mean(sp))) < 1e-9, "a spread about the mean must have zero mean"
    assert sp[0] > sp[-1], "the lowest-f0 channel lags most at 190 Hz"
    tr = trim_deg(190.0, fits)
    assert np.allclose(sp + tr, 0.0), "the trim must cancel the spread it was built from"

    # 6. The phase-only fit recovers the same (f0, Q) as the magnitude fit, from angles
    # alone. This is the cross-check `compare()` prints, run against synthetic truth.
    from controller.control.link import ProbePoint
    pts = {x: ProbePoint(f_hz=x, amps=(1.0,) * 4,
                         phase_deg=tuple(float(theta(x, fits[t]["f0_hz"], fits[t]["q"]))
                                         for t in TAGS))
           for x in PROBE_F_HZ}
    pf = fit_probe(pts)
    for t in TAGS:
        assert abs(pf[t]["f0_hz"] - fits[t]["f0_hz"]) < 0.5, (t, pf[t])
        assert abs(pf[t]["q"] - fits[t]["q"]) < 0.02, (t, pf[t])

    pk = float(np.max(sp) - np.min(sp))
    print(f"coil_phase: theta() matches the 2Q(df/f0) limit; (f0,Q) recovered from |I|(f) "
          f"shape\n  gain-independent to 1e-6; short bands refused; magnitude and "
          f"phase fits agree\n  a 145-160 Hz f0 spread gives {pk:.1f} deg pk-pk at 190 Hz "
          f"vs a {360.0 * 25e-6 * 190.0:.2f} deg command quantum\n  ok")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--measure":
        pts = measure()
        report(fit_probe(pts))
    elif args and args[0] == "--fit":
        report(fit_run(args[1]))
    else:
        demo()
