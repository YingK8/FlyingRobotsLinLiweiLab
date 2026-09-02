#!/usr/bin/env python3
"""
Voltage to current to field: what limits the drive, and what waveform to use. theory.md sec. 15.6.

Measured per channel, from `calculations/coil_capacitors.xlsx`: R = 12.6 to 18 ohm,
L = 6.64 to 6.85 mH. Those two numbers decide almost everything here.

    At 92 Hz, X_L = 3.9 ohm against R = 15 ohm, so Q = 0.26.

Three consequences, and they run against the usual advice:

    ohmic          The chain is essentially purely resistive. Series resonance cancels X_L and
                   so recovers about 3 %. It only starts paying above ~350 Hz, where X_L ~ R.
    rail-limited   12 V across 15 ohm is 0.55 A rms, against a 2 A continuous rating. The
                   binding limit is the supply, not heating, so "improve efficiency" here means
                   "get more fundamental current per rail volt", not "waste less as heat".
    first harmonic The cycle averages that produce force and torque project onto n = 1 only.
                   Waveform shape inside a cycle is therefore irrelevant to both, and only the
                   fundamental phasor matters. `fundamental` computes it for any waveform.

Together those say the present bipolar square drive is already the right choice: it delivers
4/pi = 1.27x the fundamental per rail volt that a sine does. A sine would save 19 % of the
heating, but heating is not what binds.

    uv run python ai/design/drive_model.py --self-check
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from controller.control import constants as C

R_CHANNEL = C.COIL_CHANNEL_R_OHM
L_CHANNEL = C.COIL_CHANNEL_L_H
V_RAIL = C.V_RAIL
V_RAIL_MAX = C.V_RAIL_MAX
I_CONTINUOUS = C.I_CONTINUOUS
I_TRIP = C.I_TRIP


def impedance(f, r=R_CHANNEL, l=L_CHANNEL, c_series=None):
    """Channel impedance magnitude at ``f``. ``c_series`` adds a series resonating capacitor."""

    x = 2.0 * math.pi * f * l
    if c_series:
        x -= 1.0 / (2.0 * math.pi * f * c_series)
    return math.hypot(r, x)


def resonant_c(f, l=L_CHANNEL):
    """Series capacitance that cancels the reactance at ``f``."""

    return 1.0 / (l * (2.0 * math.pi * f) ** 2)


def resonance_gain(f, r=R_CHANNEL, l=L_CHANNEL):
    """Current gain from adding the resonating capacitor. 1.0 means it buys nothing."""

    return impedance(f, r, l) / r


def max_current(f, v_rail=V_RAIL, r=R_CHANNEL, l=L_CHANNEL, resonant=False, square=True):
    """Peak fundamental current the rail can drive, in amps.

    A bipolar square of amplitude ``v_rail`` has a fundamental of 4/pi times that, which is
    why the square wins whenever the limit is voltage rather than heat.
    """

    z = r if resonant else impedance(f, r, l)
    v1 = v_rail * (4.0 / math.pi if square else 1.0)
    return v1 / z


def power(i_peak, r=R_CHANNEL, n_channels=4, harmonic_factor=1.0):
    """Ohmic dissipation in watts. ``harmonic_factor`` is I_rms^2 / I_1,rms^2."""

    return n_channels * (i_peak / math.sqrt(2.0)) ** 2 * r * harmonic_factor


def fundamental(wave, n=4096):
    """Complex first-harmonic phasor of a periodic waveform, and its RMS.

    ``wave`` is called on theta in [0, 2pi). Returns ``(amplitude_1, rms_total)``. The ratio
    of their squares is how much of the heating does no useful work: everything outside n = 1
    dissipates and contributes exactly nothing to the cycle-averaged torque or force.
    """

    th = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    y = np.asarray([wave(t) for t in th], dtype=float)
    a1 = 2.0 * np.mean(y * np.cos(th))
    b1 = 2.0 * np.mean(y * np.sin(th))
    return math.hypot(a1, b1), float(np.sqrt(np.mean(y * y)))


SQUARE = lambda t: 1.0 if math.sin(t) >= 0 else -1.0
SINE = math.sin


def trapezoid(rise_frac):
    """Trapezoidal current: a square with edges ramped over ``rise_frac`` of the period."""

    def w(t):
        x = (t % (2.0 * math.pi)) / (2.0 * math.pi)
        r = max(rise_frac, 1e-9)
        if x < r:
            return x / r
        if x < 0.5 - r:
            return 1.0
        if x < 0.5 + r:
            return (0.5 + r - x) / (2 * r) * 2 - 1
        if x < 1.0 - r:
            return -1.0
        return (x - 1.0 + r) / r - 1.0
    return w


def coast_time(f, i_spin, k_drag=3.0e-10):
    """Rotor spin-down time constant, I*w / (k_d f^2), in seconds.

    Long compared with a field period, which is what makes burst duty-cycling thinkable: the
    rotor carries through the off time on its own inertia.
    """

    return i_spin * 2.0 * math.pi * f / (k_drag * f * f)


def _self_check():
    from ai.design import spatial_model as sm

    p = sm.robot_params()

    # 1. The chain is ohmic at the operating frequency and only stops being so far above it.
    g92, g350 = resonance_gain(92.0), resonance_gain(350.0)
    assert 1.02 < g92 < 1.05, g92
    assert g350 > 1.35, g350
    print(f"impedance   : |Z|/R = {g92:.3f} at 92 Hz, {g350:.3f} at 350 Hz"
          f"  -> resonance buys {100*(g92-1):.1f} % where you run")
    print(f"              resonating C at 92 Hz would be {resonant_c(92.0)*1e6:.1f} uF")

    # 2. Rail-limited, not thermally limited. This is the number that inverts the advice.
    i_sq = max_current(92.0, square=True)
    i_si = max_current(92.0, square=False)
    assert i_sq < I_CONTINUOUS, (i_sq, I_CONTINUOUS)
    assert abs(i_sq / i_si - 4.0 / math.pi) < 1e-9
    print(f"rail        : {V_RAIL:.0f} V gives {i_sq:.3f} A peak fundamental (square), "
          f"{i_si:.3f} A (sine), against a {I_CONTINUOUS:.0f} A continuous rating")
    print(f"              -> voltage-limited by {I_CONTINUOUS/i_sq:.1f}x. "
          f"Rail ceiling {V_RAIL_MAX:.0f} V would give {max_current(92.0, V_RAIL_MAX):.3f} A.")

    # 3. The first-harmonic theorem, as arithmetic. A square puts 4/pi of its amplitude in the
    # fundamental and wastes 19 % of its heating on harmonics that average to nothing.
    a_sq, rms_sq = fundamental(SQUARE)
    a_si, rms_si = fundamental(SINE)
    assert abs(a_sq - 4.0 / math.pi) < 1e-3, a_sq
    assert abs(a_si - 1.0) < 1e-6 and abs(rms_si - 1 / math.sqrt(2)) < 1e-6
    waste = 1.0 - (a_sq / math.sqrt(2)) ** 2 / rms_sq**2
    assert 0.18 < waste < 0.20, waste
    print(f"waveform    : square fundamental {a_sq:.4f} (= 4/pi), "
          f"{100*waste:.1f} % of its heating makes no average torque")

    # For the SAME fundamental, a sine costs less heat; for the same RAIL, a square delivers
    # more fundamental. Which one wins is decided by which limit binds, and here it is the rail.
    heat_ratio = (rms_sq / a_sq) ** 2 / (rms_si / a_si) ** 2
    print(f"              same fundamental: sine costs {1/heat_ratio:.3f}x the heat; "
          f"same rail: square gives {4/math.pi:.3f}x the fundamental")
    print("              rail binds, so the square drive already in the firmware is correct")

    # 4. A trapezoid interpolates between the two, and buys nothing either way.
    for rf in (0.05, 0.15, 0.25):
        a, rms = fundamental(trapezoid(rf))
        print(f"              trapezoid rise {rf:.2f}: fundamental {a:.4f}, "
              f"heat per unit fundamental {(rms/a)**2/((rms_si/a_si)**2):.3f}x sine")

    # 5. Burst duty-cycling is thinkable because the rotor coasts far longer than a period.
    tc = coast_time(92.0, p.I_spin)
    assert tc > 0.5, tc
    print(f"coast       : rotor time constant {tc:.3f} s at 92 Hz, against a "
          f"{1/92.0*1e3:.1f} ms field period ({tc*92.0:.0f} cycles)")

    # 6. Power, at the currents the model actually uses.
    print(f"power       : 4 channels at 1.25 A peak = {power(1.25):.1f} W ohmic; "
          f"at the rail limit {i_sq:.2f} A = {power(i_sq):.1f} W")

    print("self-check PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-check", action="store_true")
    ap.parse_args()
    _self_check()
