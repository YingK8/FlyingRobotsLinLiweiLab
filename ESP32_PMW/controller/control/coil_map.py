#!/usr/bin/env python3
"""Coil placement and strength, from duty Jacobians measured at several frequencies.

    uv run python controller/control/coil_map.py

WHAT A JACOBIAN COLUMN ACTUALLY CONTAINS
----------------------------------------
`tilt_servo.identify` drops one coil's duty and reads the lean, giving column `k` of a
2x4 Jacobian at whatever frequency it was held at. That column is not a property of the
coil alone. Writing it out (`theory.md` 25.1):

    J[:, k]  proportional to  g_k * G_k(f) * [cos, sin](alpha_k + psi - theta_k(f))

with three separable pieces:

    alpha_k     where coil k sits, as an azimuth. FIXED.
    g_k         how strong its field is at the robot. FIXED.
    theta_k(f)  the phase its own series RLC adds to the CURRENT, atan(Q(f/f0 - f0/f)).
                Known from `coil_phase`, and it SWINGS TENS OF DEGREES across a ramp.

A Jacobian at one frequency cannot separate `alpha_k` from `theta_k(f)`: they enter as a
sum, and one measurement of a sum is not two numbers. **That is the entire reason this
module wants a frequency sweep** and `tilt_servo --identify --freqs ...` exists.

Sweeping `f` and removing the known `theta_k(f)` leaves a constant per coil. Whether it
IS constant is then a falsification test of the `coil_phase` fit that produced `theta`,
using data `coil_phase` never saw -- so the two measurements check each other instead of
merely coexisting. `fit_geometry` reports that scatter and refuses on it.

WHAT THIS CANNOT DO
-------------------
It does not give `psi`. `theory.md` 24.4 is explicit that nothing measured on the mast
transfers to free flight -- the seated Jacobian is a property of the rod. `psi` folds into
every column identically, so it is absorbed into a single global offset here and reported
as unidentifiable rather than silently attributed to the coils. `lockin_tilt` is what
measures it, airborne.
"""

from __future__ import annotations

import math

import numpy as np

#: Per-coil cross-frequency scatter of `alpha_k` allowed before the fit is refused, in
#: degrees. Above it either `theta_k(f)` is wrong -- a bad `coil_phase` fit, or a stale
#: capacitor bank -- or the seated response is not the linear map this model assumes.
#:
#: **Set from a measurement, and weaker than it looks.** Sweeping a deliberately wrong f0
#: over the 70-190 Hz band at Q ~ 0.9, and reading the scatter it produces:
#:
#:     f0 error   x0.90   x0.85   x0.80   x0.75   x0.707   x0.60
#:     scatter     2.6     3.7     4.9     5.9     6.8      8.8   deg
#:
#: theta(f) is a smooth arctan and a scaled f0 shifts it in a way the circular mean
#: largely absorbs, so this is a poor discriminator of a MODEST f0 error and a good one
#: only for a gross error or a structural breakdown. 5 deg catches roughly a 20 % error
#: and up -- including x0.707, which is exactly the 400 -> 800 uF bank change that voided
#: every fit before 2026-09-01 -- while leaving ~4x margin over the ~1.1 deg of angle
#: scatter that realistic column noise contributes.
#:
#: An earlier value of 15 deg was picked by eye and would have passed the bank change
#: silently, which is the one case this gate exists for. Do not raise it without
#: re-measuring the table.
MAX_ALPHA_SCATTER_DEG = 5.0


def theta_deg(f_hz, f0_hz, q):
    """The series-RLC current phase, `coil_phase.theta`'s law, in degrees.

    Repeated here as one line rather than imported so this module can be self-checked
    without a probe run; `coil_phase` remains the authority for FITTING (f0, Q).
    """

    f = np.asarray(f_hz, dtype=float)
    return np.degrees(np.arctan(q * (f / f0_hz - f0_hz / f)))


class CoilMap:
    """Per-coil azimuth and relative strength, plus the global offset they are known to."""

    def __init__(self, alpha_deg, gain, scatter_deg, offset_deg, freqs):
        self.alpha_deg = np.asarray(alpha_deg, float)
        self.gain = np.asarray(gain, float)
        self.scatter_deg = np.asarray(scatter_deg, float)
        self.offset_deg = float(offset_deg)
        self.freqs = list(freqs)

    def __repr__(self):
        a = ", ".join(f"{x:.0f}" for x in self.alpha_deg)
        g = ", ".join(f"{x:.2f}" for x in self.gain)
        return (f"CoilMap(alpha=[{a}] deg (+ an unidentifiable {self.offset_deg:.0f} deg), "
                f"gain=[{g}], max scatter {self.scatter_deg.max():.1f} deg)")

    def versus_firmware(self, coil_az=(0.0, 90.0, 180.0, 270.0)):
        """Measured azimuths minus what `main_flight.cpp::COIL_AZ` assumes.

        `COIL_AZ` is a seed. If the differences are not all near equal, `applyMixer`
        steers by a map the coils do not have and every `az=` ever sent was pointed
        somewhere else -- which is a deliverable, not a check.
        """

        d = (self.alpha_deg - np.asarray(coil_az, float)) % 360.0
        # Only the SPREAD is meaningful: a common rotation is the unidentifiable offset.
        spread = ((d - d[0] + 180.0) % 360.0) - 180.0
        return spread


def fit_geometry(jac_by_freq, f0_hz, q, ref_coil=0):
    """`{f: (2,4) Jacobian}` + per-coil (f0, Q) -> a `CoilMap`. Raises on refusal.

    Each column is turned into an angle and a length, the known `theta_k(f)` is removed
    from the angle, and what remains should not depend on `f`.
    """

    freqs = sorted(jac_by_freq)
    if len(freqs) < 2:
        raise ValueError(
            f"{len(freqs)} frequency(ies): alpha_k and theta_k(f) enter as a sum and one "
            f"measurement of a sum is not two numbers. Sweep at least 2, ideally 5.")
    f0 = np.asarray(f0_hz, float)
    qq = np.asarray(q, float)
    n = f0.size

    ang = np.zeros((len(freqs), n))
    mag = np.zeros((len(freqs), n))
    for i, f in enumerate(freqs):
        J = np.asarray(jac_by_freq[f], float)
        if J.shape != (2, n):
            raise ValueError(f"Jacobian at {f} Hz is {J.shape}, expected (2, {n})")
        for k in range(n):
            ang[i, k] = math.degrees(math.atan2(J[1, k], J[0, k]))
            mag[i, k] = float(np.hypot(J[0, k], J[1, k]))
        # ADD it back. The measured angle is `alpha + psi - theta(f)` -- the RLC lag
        # subtracts -- so recovering the geometric part means adding theta, not
        # subtracting it. Getting this sign backwards doubles the frequency dependence
        # instead of cancelling it, and the scatter gate catches it (61.9 deg on coil B),
        # which is what the gate is for.
        ang[i] += theta_deg(f, f0, qq)

    # Circular mean per coil, and the scatter about it. Circular, because these are angles:
    # an arithmetic mean of 359 and 1 is 180, which is the wrong answer by half a turn.
    alpha = np.zeros(n)
    scatter = np.zeros(n)
    for k in range(n):
        r = np.radians(ang[:, k])
        c, s = float(np.mean(np.cos(r))), float(np.mean(np.sin(r)))
        alpha[k] = math.degrees(math.atan2(s, c)) % 360.0
        d = ((ang[:, k] - alpha[k] + 180.0) % 360.0) - 180.0
        scatter[k] = float(np.sqrt(np.mean(d ** 2)))

    if scatter.max() > MAX_ALPHA_SCATTER_DEG:
        worst = int(np.argmax(scatter))
        raise ValueError(
            f"coil {'ABCD'[worst]} azimuth scatters {scatter[worst]:.1f} deg across "
            f"{len(freqs)} frequencies, over the {MAX_ALPHA_SCATTER_DEG} deg limit. "
            f"alpha_k is fixed by geometry, so a frequency-dependent residual means the "
            f"theta_k(f) being removed is wrong -- re-run coil_phase on the bank actually "
            f"fitted (f0 goes as C^-1/2, so the 400->800 uF change voids any older fit).")

    # `psi` and any constant rotation of the whole array are the same number here, so the
    # map is reported relative to one coil and the offset is named as unidentifiable.
    offset = alpha[ref_coil]
    alpha = (alpha - offset) % 360.0
    gain = mag.mean(axis=0)
    gain = gain / gain.max()
    return CoilMap(alpha, gain, scatter, offset, freqs)


def demo():
    rng = np.random.default_rng(7)
    n = 4
    alpha_true = np.array([0.0, 88.0, 181.0, 274.0])      # not the assumed 0/90/180/270
    gain_true = np.array([1.0, 0.72, 0.95, 0.81])         # 18.18's 29% asymmetry, roughly
    f0 = np.array([150.0, 148.0, 152.0, 149.0])
    q = np.array([0.9, 1.0, 0.85, 0.95])
    psi = 37.0                                            # a global rotation, unidentifiable
    freqs = [70.0, 100.0, 130.0, 160.0, 190.0]

    def jac(f, noise=0.0, theta_scale=1.0):
        th = theta_deg(f, f0, q) * theta_scale
        a = np.radians(alpha_true + psi - th)
        J = np.vstack([gain_true * np.cos(a), gain_true * np.sin(a)])
        return J + rng.normal(scale=noise, size=J.shape)

    # 1. Recovery. The angles come back relative to coil A, and the offset is reported
    #    rather than folded into them.
    m = fit_geometry({f: jac(f) for f in freqs}, f0, q)
    rel_true = (alpha_true - alpha_true[0]) % 360.0
    assert np.allclose(m.alpha_deg, rel_true, atol=1.0), (m.alpha_deg, rel_true)
    assert np.allclose(m.gain, gain_true / gain_true.max(), atol=0.02), m.gain
    assert abs(((m.offset_deg - (alpha_true[0] + psi) + 180) % 360) - 180) < 1.0, m.offset_deg

    # 2. ONE FREQUENCY IS NOT ENOUGH, and the refusal says why. This is the argument for
    #    sweeping, tested rather than asserted in a comment.
    try:
        fit_geometry({160.0: jac(160.0)}, f0, q)
    except ValueError as e:
        assert "sum" in str(e), e
    else:
        raise AssertionError("a single-frequency fit was accepted; it is degenerate")

    # 3. A WRONG theta -- a stale capacitor bank, say -- must be caught by the scatter
    #    and not averaged into a plausible-looking azimuth. This is the cross-check that
    #    makes coil_phase and coil_map validate each other.
    # 0.707 is not an arbitrary wrongness: f0 goes as C^-1/2, so it is exactly what a fit
    # from the old 400 uF bank looks like against the 800 uF one actually fitted. That is
    # the specific mistake this gate exists to catch.
    try:
        fit_geometry({f: jac(f) for f in freqs}, f0 * 0.707, q)
    except ValueError as e:
        assert "scatter" in str(e) and "coil_phase" in str(e), e
    else:
        raise AssertionError("a stale (f0, Q) was accepted; the scatter should refuse it")

    # 4. Noise degrades it gracefully rather than silently.
    mn = fit_geometry({f: jac(f, noise=0.02) for f in freqs}, f0, q)
    assert np.allclose(mn.alpha_deg, rel_true, atol=4.0), mn.alpha_deg
    assert mn.scatter_deg.max() > m.scatter_deg.max()
    # ...and realistic column noise must not trip the gate, or it would refuse good runs.
    assert mn.scatter_deg.max() < MAX_ALPHA_SCATTER_DEG, mn.scatter_deg

    # 5. The comparison against the firmware's assumed map. Coil B really is 2 deg off
    #    90 and D 4 deg off 270; `applyMixer` steers by the assumption.
    spread = m.versus_firmware()
    assert abs(spread[1] - (-2.0)) < 1.0 and abs(spread[3] - (4.0)) < 1.0, spread

    print(f"coil_map: recovers alpha and gain from {len(freqs)} frequencies; refuses a "
          f"single frequency\n  (alpha and theta enter as a sum) and refuses a stale "
          f"(f0, Q) via the scatter;\n  reports the global rotation as unidentifiable\n  ok")


if __name__ == "__main__":
    demo()
