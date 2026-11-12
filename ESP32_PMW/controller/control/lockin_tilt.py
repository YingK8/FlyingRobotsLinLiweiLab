#!/usr/bin/env python3
"""The lateral actuation gain as ONE complex number, fitted from a rotating command.

    uv run python controller/control/lockin_tilt.py

WHAT IS BEING MEASURED
----------------------
`hover_model.K_LAT_DEFAULT = 0.05` is marked "SEED GUESS: identify on the rig" and never
was. `attitude.TiltController.rot_deg` has no default because `fit_rotation` has always
refused. `theory.md` 20.5 shows the two are not independent quantities: both lateral axes
share one row of `K`, so a rotation between commanded and realised direction enters the
loop as a COMPLEX gain. `k_lat` is its modulus and `psi` its argument, and identifying
them together is the only way to get either honestly.

So the model is one complex regression. With the command written as a phasor
`u = mag * exp(j*az)` and the response as `w = tilt_x + j*tilt_y`,

    w = k * u + c,      k = k_lat_eff * exp(j*psi)

`c` is the rest tilt -- the robot sits ~1.6 deg off-axis toward ~146 deg (18.15) and
leans 3-5 deg under drive toward the field's strong side (18.18). Fitting it rather than
subtracting a datum is what stops that bias being read as gain: a constant offset in `w`
correlates with any command that does not average to zero over the record, and the record
IS short.

WHY A REGRESSION AND NOT A QUADRATURE LOCK-IN
---------------------------------------------
A textbook lock-in multiplies by `exp(-j*2*pi*f*t)` and averages, which needs a whole
number of cycles to reject the bias and the harmonics. `theory.md` 25.5 measured the
airborne window at **0.546-0.550 s**, which at the 0.5 Hz dither is 0.27 of a cycle. Over
a partial arc the quadrature integral does not separate `k` from `c` at all, and a lock-in
run on one anyway returns a number with no error bars.

The regression is the same estimator when the arc is complete and degrades gracefully when
it is not -- and, crucially, it *reports* the degradation: the conditioning of `[u, 1]` is
a direct function of how much of the circle the command visited, so `sigma_psi` grows
without bound as the arc shrinks. That is the difference between a short record and a
short record you can tell is short.

**It still refuses.** `MIN_ARC_DEG` is not a tuning knob; below it the fit is a straight
line through an arc and the split between `k` and `c` is arbitrary.
"""

from __future__ import annotations

import math

import numpy as np

#: Command arc the record must cover before a fit is attempted, in degrees. Below half a
#: turn the command phasor and the constant are barely distinguishable and the fitted
#: `psi` follows the arc's midpoint rather than the plant.
MIN_ARC_DEG = 180.0

#: The tilt-vector noise floor at the 0.25 s estimator window, measured in `attitude.py`
#: (0.49 deg at 0.25 s, 0.11 at 0.50, 0.02 at 1.00). `fit_rotation` refuses below it and
#: so does this. The response must clear it by `MIN_SNR`.
FLOOR_DEG = 0.49
MIN_SNR = 5.0

#: Above this the loop is outside the region `theory.md` 20.5 certifies -- both lateral
#: axes share one gain row, so a rotation is a complex loop gain and the low-gain margin
#: collapses from 21.9 dB to 0.9 dB by 67.5 deg.
MAX_PSI_ERR_DEG = 69.4


class Fit:
    """A complex-gain fit and everything needed to disbelieve it."""

    def __init__(self, k, c, sigma_k, arc_deg, n, resid_deg):
        self.k, self.c = complex(k), complex(c)
        self.sigma_k = float(sigma_k)
        self.arc_deg, self.n, self.resid_deg = float(arc_deg), int(n), float(resid_deg)

    @property
    def k_lat_eff(self):
        """|k|, in degrees of tilt per unit `mag`."""
        return abs(self.k)

    @property
    def psi_deg(self):
        return math.degrees(math.atan2(self.k.imag, self.k.real)) % 360.0

    @property
    def sigma_psi_deg(self):
        """1-sigma on the argument. |k| / sigma is the SNR; the angle error is its inverse."""
        return math.degrees(self.sigma_k / abs(self.k)) if abs(self.k) > 0 else float("inf")

    @property
    def snr(self):
        return abs(self.k) / self.sigma_k if self.sigma_k > 0 else float("inf")

    def __repr__(self):
        return (f"Fit(k_lat_eff={self.k_lat_eff:.4f} deg/mag, psi={self.psi_deg:.1f}"
                f"+-{self.sigma_psi_deg:.1f} deg, arc={self.arc_deg:.0f} deg, n={self.n})")


def fit(az_deg, mag, tilt_x, tilt_y):
    """Complex least squares for (k, c). Returns a `Fit`, or raises with the reason.

    `mag` may be a scalar or per-sample. Raises rather than returning a poor fit: every
    caller in this tree treats a refusal as a result, and a number with no arc behind it
    is the failure mode `fit_rotation` exists to avoid.
    """

    az = np.asarray(az_deg, dtype=float)
    tx = np.asarray(tilt_x, dtype=float)
    ty = np.asarray(tilt_y, dtype=float)
    m = np.full_like(az, float(mag)) if np.isscalar(mag) else np.asarray(mag, dtype=float)
    ok = np.isfinite(az) & np.isfinite(tx) & np.isfinite(ty) & np.isfinite(m) & (m > 0)
    az, tx, ty, m = az[ok], tx[ok], ty[ok], m[ok]
    if az.size < 8:
        raise ValueError(f"{az.size} usable samples -- too few to fit two complex numbers")

    # Arc actually covered. Not the range of `az`, which a command that oscillates over a
    # small sector would report as large: this is the total swept angle, unwrapped.
    arc = float(np.abs(np.diff(np.unwrap(np.radians(az)))).sum())
    arc_deg = math.degrees(arc)
    if arc_deg < MIN_ARC_DEG:
        raise ValueError(
            f"command swept {arc_deg:.0f} deg, under the {MIN_ARC_DEG:.0f} deg minimum. "
            f"Over a shorter arc the command phasor and the constant offset are not "
            f"separable and the fitted angle follows the arc, not the plant. "
            f"theory.md 25.5: the airborne window is ~0.55 s, so a 0.5 Hz dither covers "
            f"0.27 of a turn -- lengthen the window (fly the balanced duty vector) or "
            f"raise the dither rate.")

    u = m * np.exp(1j * np.radians(az))
    w = tx + 1j * ty
    a = np.column_stack([u, np.ones_like(u)])
    sol, *_ = np.linalg.lstsq(a, w, rcond=None)
    k, c = sol
    resid = w - a @ sol
    # Complex residual variance per real DOF: 2n real observations, 4 real parameters.
    dof = max(2 * az.size - 4, 1)
    s2 = float(np.vdot(resid, resid).real) / dof
    cov = np.linalg.inv(a.conj().T @ a).real
    sigma_k = math.sqrt(max(s2 * cov[0, 0], 0.0))
    return Fit(k, c, sigma_k, arc_deg, az.size, math.sqrt(s2))


def accept(f: Fit) -> tuple[bool, str]:
    """Gate a fit the way the sysid stages do. Returns (ok, reason)."""

    if f.snr < MIN_SNR:
        return False, (f"response {f.k_lat_eff:.3f} deg/mag against a {f.sigma_k:.3f} "
                       f"scatter -- SNR {f.snr:.1f} under {MIN_SNR}. Not fitting a "
                       f"rotation to noise.")
    if f.k_lat_eff < MIN_SNR * FLOOR_DEG:
        return False, (f"|k| = {f.k_lat_eff:.3f} deg/mag is under {MIN_SNR}x the "
                       f"{FLOOR_DEG} deg estimator floor")
    if f.sigma_psi_deg > 15.0:
        return False, f"psi = {f.psi_deg:.0f} +- {f.sigma_psi_deg:.0f} deg is too loose to use"
    return True, ""


def demo():
    rng = np.random.default_rng(0)

    def record(turns, k_true, psi_deg, bias=(1.6, 0.4), noise=FLOOR_DEG, n=None,
               mag=0.30, rate_hz=0.5, fs=190.0):
        n = n or max(8, int(round(turns / rate_hz * fs)))
        t = np.arange(n) / fs
        az = (360.0 * rate_hz * t) % 360.0
        k = k_true * np.exp(1j * math.radians(psi_deg))
        w = k * mag * np.exp(1j * np.radians(az)) + complex(*bias)
        w = w + rng.normal(scale=noise, size=n) + 1j * rng.normal(scale=noise, size=n)
        return az, mag, w.real, w.imag

    # 1. A clean record recovers both numbers, bias and all. The bias is 1.6 deg against a
    #    3 deg response -- big enough that ignoring it would bend the answer.
    az, mag, tx, ty = record(3.0, k_true=10.0, psi_deg=215.0)
    f = fit(az, mag, tx, ty)
    assert abs(f.k_lat_eff - 10.0) < 0.3, f
    assert abs((f.psi_deg - 215.0 + 180) % 360 - 180) < 3.0, f
    assert abs(f.c - complex(1.6, 0.4)) < 0.3, f.c
    ok, why = accept(f)
    assert ok, why

    # 2. THE FAILURE THIS FILE IS ABOUT. A dither that commands a direction with no
    #    magnitude produces no response, and must not yield an angle. This is the exact
    #    fault of theory.md 25.1, which stood in the runner for the project's whole life.
    az0, _, tx0, ty0 = record(3.0, k_true=0.0, psi_deg=215.0)
    f0 = fit(az0, 0.30, tx0, ty0)
    ok, why = accept(f0)
    assert not ok and "noise" in why.lower(), (f0, why)

    # 3. THE CONSTRAINT 25.5 MEASURED. 0.55 s of window at 0.5 Hz is 0.27 of a turn, and
    #    over that arc `k` and `c` are not separable. It must refuse rather than fit.
    try:
        az1, mag1, tx1, ty1 = record(0.27, k_true=10.0, psi_deg=215.0)
        fit(az1, mag1, tx1, ty1)
    except ValueError as e:
        assert "swept" in str(e) and "25.5" in str(e), e
    else:
        raise AssertionError("a 0.27-turn arc was fitted; it is not separable")

    # 4. ...and the same short WINDOW is fine once the dither is fast enough to cover the
    #    circle in it. That is the third of 25.5's ways out, tested rather than asserted.
    az2, mag2, tx2, ty2 = record(1.05, k_true=10.0, psi_deg=215.0, rate_hz=1.9)
    f2 = fit(az2, mag2, tx2, ty2)
    ok, why = accept(f2)
    assert ok, why
    assert abs((f2.psi_deg - 215.0 + 180) % 360 - 180) < 8.0, f2

    # 5. The uncertainty must GROW as the arc shrinks -- that is what makes a short record
    #    detectable rather than merely wrong.
    wide = fit(*record(3.0, 10.0, 215.0)[:1] + (0.30,) + record(3.0, 10.0, 215.0)[2:])
    narrow = fit(*record(0.6, 10.0, 215.0)[:1] + (0.30,) + record(0.6, 10.0, 215.0)[2:])
    assert narrow.sigma_psi_deg > wide.sigma_psi_deg, (narrow, wide)

    # 6. Outside the certified region the caller must be told, not quietly handed a gain.
    az3, mag3, tx3, ty3 = record(3.0, k_true=10.0, psi_deg=100.0)
    f3 = fit(az3, mag3, tx3, ty3)
    assert abs((f3.psi_deg - 100.0 + 180) % 360 - 180) < 3.0, f3
    assert min(f3.psi_deg, 360 - f3.psi_deg) > MAX_PSI_ERR_DEG, "should be outside 20.5's region"

    print(f"lockin_tilt: recovers |k| and psi with a fitted bias; refuses zero excitation, "
          f"refuses a\n  0.27-turn arc (25.5), accepts the same window at 1.9 Hz; "
          f"sigma_psi grows as the arc shrinks\n  ok")


if __name__ == "__main__":
    demo()
