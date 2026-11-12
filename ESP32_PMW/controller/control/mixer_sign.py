#!/usr/bin/env python3
"""Which way does `az=` tilt the disk? Measure it, do not assume it.

`main_flight.cpp`'s `applyMixer` drops the carrier on the az-facing coils so the disk
tilts toward `az`, and the comment there says plainly "Verify sign on rig". Nothing ever
did. A feedforward tilt trim built on the wrong sign DOUBLES the tilt instead of
cancelling it, so this is the calibration that has to come first.

Measured 2026-09-01: the rotor normal sits ~1.5 deg off the datum axis at rest and grows
to ~3.1 deg by 105 Hz, in a fixed azimuth near 150 deg (concentration 0.97). See
`theory.md` 18.14.

Method: capture the rotor, hold a fixed sub-liftoff frequency, then command each of four
azimuths in turn at a fixed `mag` and record where the rotor normal actually goes. The
response azimuth minus the commanded azimuth is the mixer's sign and offset.

    uv run python controller/control/mixer_sign.py

HOLD_HZ is deliberately below the 98-126 Hz liftoff band measured on 2026-09-01: the robot
must stay on the pad, because this measures the disk's response and not a flight.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from controller.control import ramp
from controller.control.link import SerialComm

# 115 Hz, NOT 70. At 70 the robot is seated hard on the rod and physically cannot tilt:
# the first attempt at this measured its resting tilt four times over and learned nothing
# about the mixer (theory.md 18.15). Liftoff was measured at 98-126 Hz on 2026-09-01, so
# 115 leaves it nearly unloaded -- free to respond -- without commanding a flight.
HOLD_HZ = 115.0
MAG = 0.30              # double the lateral loop's cap: identification wants a big signal
SETTLE_S = 0.8          # let the disk reach the new tilt before believing it
HOLD_S = 1.2            # then average over this
# PER-COIL, not a polar sweep. `applyMixer` uses max(0, cos(az - COIL_AZ)), so pointing az
# exactly at a coil drops THAT COIL ALONE: its neighbours sit at cos(+-90) = 0 and the
# opposite coil's cos(180) = -1 is clamped away. So az on a coil axis is individual
# amplitude control, and no firmware command is needed for it.
COIL_AZ = {0.0: "A", 90.0: "B", 180.0: "C", 270.0: "D"}
MAG_STEPS = (0.20, 0.40)       # depth of the drop: 12 % and 24 % of collective
RESET_S = 0.5                  # mag=0 between steps -- every coil back to full

def _steps(mags=None):
    """(az, mag, label) in order, with a full-amplitude reset between each measurement."""

    for az, tag in COIL_AZ.items():
        for m in (MAG_STEPS if mags is None else mags):
            yield az, m, f"{tag} @ mag {m:.2f}"
            yield az, 0.0, None          # reset: all four coils back to max


def _circmean(deg):
    a = np.radians(np.asarray(deg, float))
    return np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360.0


def _concentration(deg):
    """0 = uniform, 1 = one direction. Below ~0.5 the mean azimuth means nothing."""

    a = np.radians(np.asarray(deg, float))
    return float(np.hypot(np.cos(a).mean(), np.sin(a).mean()))


def run(port=None, width=640, height=400, fps=210, hold_hz=None, mags=None):
    """Ramp, hold, then drop each coil in turn and watch where the rotor normal goes."""

    from controller.calib.rig import StereoRig
    from controller.viz import live_viz

    hold_hz = HOLD_HZ if hold_hz is None else hold_hz
    segs = ((2.0, 8.0, 6.0, ramp.EASE, 2.0), (8.0, hold_hz, 10.0, ramp.EASE, 2.0))
    ramp.check(segs)
    steps = list(_steps(mags))
    print(f"per-coil id: {ramp.label(segs)}, hold {hold_hz:.0f} Hz, "
          f"{sum(1 for x in steps if x[2])} measurements")

    link = ticks = None
    out = []
    try:
        link = SerialComm(port=port)
        link.reset_device()
        time.sleep(1.5)
        pair = StereoRig.load().sources()
        ticks = live_viz.stereo_frames(specs=pair, width=width, height=height, fps=fps)

        for c in ramp.seq_lines(segs):
            link.handle_serial_comm(c)
        t0 = time.monotonic()
        ramping, k, t_step, samples, lost = True, 0, t0, [], 0

        for tick in ticks:
            now = time.monotonic()
            while link.handle_serial_comm() is not None:
                pass
            lost = lost + 1 if tick.xyz_mm is None else 0
            if lost > 120:
                print("  departed -- stopping the sweep", flush=True)
                break

            if ramping:
                if now - t0 > ramp.duration_s(segs) + 1.0:
                    ramping, t_step = False, now
                    az, mag, lab = steps[0]
                    link.handle_serial_comm(f"az={az:.0f}")
                    link.handle_serial_comm(f"mag={mag:.3f}")
                    if lab:
                        print(f"  {lab} ...", flush=True)
                continue

            az, mag, lab = steps[k]
            dur = (SETTLE_S + HOLD_S) if lab else RESET_S
            if lab and tick.pose is not None and now - t_step > SETTLE_S:
                samples.append((tick.pose.theta_deg, tick.pose.phi_deg,
                                tick.xyz_mm[2] if tick.xyz_mm is not None else np.nan))
            if now - t_step < dur:
                continue

            if lab and samples:
                th = np.median([x[0] for x in samples])
                ph = _circmean([x[1] for x in samples])
                cn = _concentration([x[1] for x in samples])
                z = np.nanmedian([x[2] for x in samples])
                out.append((lab, az, mag, th, ph, cn, z, len(samples)))
            k += 1
            if k >= len(steps):
                break
            samples, t_step = [], now
            az, mag, lab = steps[k]
            link.handle_serial_comm(f"az={az:.0f}")
            link.handle_serial_comm(f"mag={mag:.3f}")
            if lab:
                print(f"  {lab} ...", flush=True)
    finally:
        if link is not None:
            for c in ("mag=0", "stop"):
                link.handle_serial_comm(c)
            link.close()
        if ticks is not None:
            ticks.close()

    if not out:
        print("no measurements -- it departed before the sweep began")
        return out
    print(f"\n{'step':>14} {'tilt':>7} {'tilt az':>9} {'conc':>6} {'z':>8} {'n':>5}")
    for lab, az, mag, th, ph, cn, z, n in out:
        print(f"{lab:>14} {th:6.2f}d {ph:8.1f}d {cn:6.2f} {z:7.2f}mm {n:5d}")
    base = np.median([o[3] for o in out])
    print(f"\nmedian tilt across all steps {base:.2f} deg")
    if len(out) < 2:
        # A spread over one sample is 0.00 by construction, and printing "NO RESPONSE"
        # off it reports a false negative as a measurement. Observed 2026-09-01: the
        # sweep departed during coil B and the one-sample verdict read as proof the
        # mixer does nothing.
        print(f"INCONCLUSIVE: only {len(out)} coil(s) measured before it departed. "
              f"A spread needs at least two, so this says NOTHING about the mixer.")
        return out
    spread = max(o[3] for o in out) - min(o[3] for o in out)
    print(f"tilt SPREAD across steps      {spread:.2f} deg")
    if spread < 0.3:
        print("  NO RESPONSE: dropping a coil did not move the rotor normal. Either the "
              "robot is still constrained by the rod, or the mixer has no authority here.")
    else:
        best = max(out, key=lambda o: o[3]); worst = min(out, key=lambda o: o[3])
        print(f"  most tilt  {best[0]} -> {best[3]:.2f} deg at {best[4]:.0f} deg")
        print(f"  least tilt {worst[0]} -> {worst[3]:.2f} deg at {worst[4]:.0f} deg")
    return out


if __name__ == "__main__":
    run(port=sys.argv[1] if len(sys.argv) > 1 else None)
