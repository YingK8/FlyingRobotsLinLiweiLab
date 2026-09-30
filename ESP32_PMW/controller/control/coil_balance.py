#!/usr/bin/env python3
"""Measure the coil current imbalance and derive the trim that cancels it. NO CAMERAS.

The four channels draw unequal current at equal commanded duty -- 3.39 / 4.70 / 4.73 /
3.77 A measured 2026-08-29, a 39 % spread (`theory.md` 18.8). Resolved against the coil
azimuths that is a **1.63 A dipole at 145 deg**, and the rotor's measured lean is at
**146 deg** (18.15). One degree apart, from two unrelated instruments -- so the tilt this
project has been chasing is, at least in part, a field asymmetry, which 20.3 argues is the
one candidate cause that IS addressable from the host.

WHAT CAN AND CANNOT BE CANCELLED
--------------------------------
`applyMixer` is 2-DOF: one azimuth, one magnitude. Decomposing the four per-coil deviations:

  * **dipole** -- the part that points somewhere, and the part that tilts the rotor. A
    2-DOF mixer cancels it exactly. This is 1.63 A of the imbalance.
  * **quadrupole** (A+C against B+D) -- a squeeze with no net direction. NOT reachable by
    az/mag at all, and it does not produce a net lateral force either. This is 0.35 A.

So the dipole is 4.7x the part we cannot touch, which is why a 2-DOF trim is worth doing.

THE DIRECTION IS NOT THE ONE THE TILT SUGGESTS
----------------------------------------------
`applyMixer` uses `max(0, cos(az - COIL_AZ))`, so it **weakens** the coils facing `az`.
The strong coils are B (90 deg) and C (180 deg), facing ~135 deg. So the correction points
AT the strong side, `az ~ 145 deg` -- not at `tilt + 180`. That is worth stating because
both 0 deg and 315 deg have been flown and 145 deg never has.

    uv run python controller/control/coil_balance.py            # measure and recommend
"""

from __future__ import annotations

import math
import sys
import time

import numpy as np

from controller.control import constants as C
from controller.control import ramp
from controller.control.link import SerialComm, parse_telemetry

COIL_AZ_DEG = (0.0, 90.0, 180.0, 270.0)     # src/main_flight.cpp COIL_AZ, MEASURED by sweep
TAGS = "ABCD"
MIX_GAIN = 0.6                              # src/main_flight.cpp, must match
HOLD_HZ = 90.0                              # below the 98-126 Hz liftoff band; robot stays put
SETTLE_S = 1.5
SAMPLE_S = 3.0


def decompose(amps):
    """(dipole_amps, dipole_az_deg, quadrupole_amps, per-coil deviation)."""

    I = np.asarray(amps, float)
    dev = I - I.mean()
    a = np.radians(COIL_AZ_DEG)
    x, y = float((dev * np.cos(a)).sum()), float((dev * np.sin(a)).sum())
    quad = float(dev[0] - dev[1] + dev[2] - dev[3])
    return math.hypot(x, y), math.degrees(math.atan2(y, x)) % 360.0, quad, dev


def trim_for(amps):
    """(az_deg, mag) that cancels the dipole, or (None, 0) when there is nothing to cancel.

    `mag` follows from the mixer's own law. The drop on the coil facing `az` is
    `MIX_GAIN * mag`, and it has to remove that coil's excess as a FRACTION of the mean:

        MIX_GAIN * mag = dipole_per_coil / mean

    `dipole/2` is the per-coil amplitude of a dipole of that magnitude, since the projection
    onto four azimuths spreads it over two active coils.
    """

    dip, az, _, _ = decompose(amps)
    mean = float(np.mean(amps))
    if mean <= 0 or dip < 0.05:
        return None, 0.0
    mag = (dip / 2.0) / mean / MIX_GAIN
    return az, min(mag, 1.0)


def measure(port=None, hold_hz=HOLD_HZ, verbose=True):
    """Ramp to `hold_hz`, hold, and average the per-coil currents. Returns amps (A,B,C,D)."""

    segs = ((2.0, 8.0, 6.0, ramp.EASE, 2.0), (8.0, hold_hz, 8.0, ramp.EASE, 2.0))
    ramp.check(segs)
    link = SerialComm(port=port)
    rows = []
    try:
        link.reset_device()
        time.sleep(1.5)
        for c in ramp.seq_lines(segs):
            # Spaced, not back-to-back. Sent with no gap the firmware reached FLIGHT late or
            # not at all and every `freq=` was then refused; 0.12 s apart it is reliable.
            # `fly()` gets this for free because its sends are one per control tick.
            link.handle_serial_comm(c)
            time.sleep(0.12)
        t0 = time.monotonic()
        end = t0 + ramp.duration_s(segs) + SETTLE_S + SAMPLE_S
        while time.monotonic() < end:
            line = link.handle_serial_comm()
            if not line:
                time.sleep(0.002)
                continue
            tel = parse_telemetry(line)
            if (len(tel.amps) == 4 and tel.freq and tel.freq > hold_hz * 0.95
                    and time.monotonic() - t0 > ramp.duration_s(segs) + SETTLE_S):
                rows.append(tel.amps)
    finally:
        link.handle_serial_comm("stop")
        link.close()
    if len(rows) < 3:
        raise SystemExit(f"only {len(rows)} telemetry samples at {hold_hz:.0f} Hz -- "
                         f"nothing to average")
    amps = tuple(float(v) for v in np.median(np.array(rows), axis=0))
    if verbose:
        report(amps, n=len(rows), hold_hz=hold_hz)
    return amps


def report(amps, n=None, hold_hz=None):
    dip, az, quad, dev = decompose(amps)
    mean = float(np.mean(amps))
    print(f"per-coil current at {hold_hz or HOLD_HZ:.0f} Hz"
          + (f", median of {n} samples" if n else "") + ":")
    for t, i, d in zip(TAGS, amps, dev):
        print(f"   {t}  {i:5.2f} A   {d:+5.2f} from the mean")
    print(f"   mean {mean:.2f} A, spread {100 * (max(amps) - min(amps)) / mean:.0f} %")
    print(f"\ndipole      {dip:.2f} A at {az:.0f} deg   -- steerable, and what tilts the rotor")
    print(f"quadrupole  {quad:+.2f} A            -- NOT steerable by a 2-DOF mixer")
    t_az, t_mag = trim_for(amps)
    if t_az is None:
        print("\nalready balanced within 0.05 A -- no trim worth commanding")
    else:
        print(f"\n  TRIM:  resultant weak direction {t_az:.0f} deg, strength {t_mag:.2f}")
        print(f"         weakens the coils facing {t_az:.0f} deg, which are the strong ones")
    return t_az, t_mag


def demo():
    # The 2026-08-29 measurement, and the claim this module rests on.
    amps = (3.39, 4.70, 4.73, 3.77)
    dip, az, quad, dev = decompose(amps)
    assert abs(dip - 1.63) < 0.02, dip
    assert abs(az - 145.0) < 1.5, az
    assert abs(quad + 0.35) < 0.02, quad
    assert abs(az - 146.0) < 5.0, "the imbalance no longer points where the rotor leans"

    # A perfectly balanced set has no dipole and asks for no trim.
    assert decompose((4.0, 4.0, 4.0, 4.0))[0] < 1e-9
    assert trim_for((4.0, 4.0, 4.0, 4.0)) == (None, 0.0)

    # A pure dipole on A alone points at A, and the trim points back at the STRONG coil.
    d, a, q, _ = decompose((5.0, 4.0, 3.0, 4.0))
    assert abs(a - 0.0) < 1e-6 or abs(a - 360.0) < 1e-6, a
    assert abs(q) < 1e-9, "a pure dipole must carry no quadrupole"
    t_az, t_mag = trim_for((5.0, 4.0, 3.0, 4.0))
    assert abs(t_az) < 1e-6 or abs(t_az - 360) < 1e-6, t_az

    t_az, t_mag = trim_for(amps)
    print(f"coil_balance: dipole {dip:.2f} A at {az:.0f} deg vs a rotor lean at 146 deg; "
          f"quadrupole {quad:+.2f} A is unreachable\n"
          f"  trim -> weak direction {t_az:.0f} deg, strength {t_mag:.2f}\n  ok")


if __name__ == "__main__":
    if "--measure" in sys.argv:
        measure()
    else:
        demo()


SWEEP_HZ = (95.0, 80.0, 60.0, 40.0)      # all below the 98-126 Hz liftoff band
SWEEP_HOLD_S = 3.0


def sweep(port=None, freqs=SWEEP_HZ, verbose=True):
    """Per-coil current at several frequencies, to test whether the imbalance rotates.

    Two measurements of this rig disagree completely -- 1.63 A at 145 deg above 140 Hz on
    2026-08-29, 0.49 A at 18 deg at 90 Hz today, with coil A going from weakest to
    strongest. The candidate explanation is frequency: the channels are series-resonant
    near 174 Hz, so above ~140 Hz the per-coil L and C spread dominates the impedance while
    at 90 Hz it is mostly R. This sweep is the test.

    It needs NO cameras and does NOT need the rotor to capture -- the quantity is electrical.
    Descending order, because `freq=` is accepted only in FLIGHT and the ramp has to reach
    the top first.
    """

    segs = ((2.0, 8.0, 6.0, ramp.EASE, 2.0), (8.0, max(freqs), 10.0, ramp.EASE, 2.0))
    ramp.check(segs)
    link = SerialComm(port=port)
    out = {}
    try:
        link.reset_device()
        time.sleep(1.5)
        for c in ramp.seq_lines(segs):
            # Spaced, not back-to-back. Sent with no gap the firmware reached FLIGHT late or
            # not at all and every `freq=` was then refused; 0.12 s apart it is reliable.
            # `fly()` gets this for free because its sends are one per control tick.
            link.handle_serial_comm(c)
            time.sleep(0.12)
        t0 = time.monotonic()
        # WAIT FOR THE STATE, do not assume it. `cmdFreq` refuses unless state == FLIGHT and
        # answers `!freq state=<n>`; an earlier version slept for the ramp's nominal duration
        # and stepped anyway, so every `freq=` was rejected, the reported frequency never
        # moved off the ramp's last value, and the sample filter then discarded everything.
        # Three of four sweep points came back empty and it looked like a telemetry problem.
        flight, deadline = False, t0 + ramp.duration_s(segs) + 8.0
        while time.monotonic() < deadline:
            line = link.handle_serial_comm()
            if line and parse_telemetry(line).state == 2:
                flight = True
                break
            if not line:
                time.sleep(0.002)
        if not flight:
            raise SystemExit("firmware never reported FLIGHT -- the ramp did not finish, so "
                             "`freq=` would be refused and every hold would be empty")
        for f in sorted(freqs, reverse=True):
            link.handle_serial_comm(f"freq={f:.2f}")
            t1, rows = time.monotonic(), []
            while time.monotonic() - t1 < SETTLE_S + SWEEP_HOLD_S:
                line = link.handle_serial_comm()
                if not line:
                    time.sleep(0.002)
                    continue
                tel = parse_telemetry(line)
                if (len(tel.amps) == 4 and tel.freq and abs(tel.freq - f) < 0.1 * f
                        and time.monotonic() - t1 > SETTLE_S):
                    rows.append(tel.amps)
            if len(rows) >= 2:
                out[f] = tuple(float(v) for v in np.median(np.array(rows), axis=0))
                if verbose:
                    print(f"  {f:5.0f} Hz  " + "  ".join(
                        f"{t}={i:5.2f}" for t, i in zip(TAGS, out[f]))
                        + f"   ({len(rows)} samples)", flush=True)
            elif verbose:
                print(f"  {f:5.0f} Hz  -- only {len(rows)} samples, skipped", flush=True)
    finally:
        link.handle_serial_comm("stop")
        link.close()
    if verbose and out:
        print(f"\n{'f':>7} {'dipole':>9} {'at':>7} {'quad':>8}   trim")
        for f, amps in sorted(out.items(), reverse=True):
            dip, az, quad, _ = decompose(amps)
            t_az, t_mag = trim_for(amps)
            print(f"{f:6.0f}Hz {dip:8.2f}A {az:6.0f}d {quad:+7.2f}A   "
                  + (f"weak dir {t_az:.0f} deg, strength {t_mag:.2f}" if t_az else "none"))
        azs = [decompose(a)[1] for a in out.values()]
        spread = max(azs) - min(azs)
        print(f"\ndipole direction spans {spread:.0f} deg across {min(out):.0f}-{max(out):.0f} Hz"
              + ("  -- FREQUENCY DEPENDENT, trim at the frequency you fly"
                 if spread > 30 else "  -- stable with frequency"))
    return out
