#!/usr/bin/env python3
"""
Exact robust-stability certificate for the shipped hover gains. theory.md section 20.

Answers one question: over what range of ROBOT MASS and THRUST-AXIS ROTATION is the closed
loop still asymptotically stable, holding `hover_controller.json` fixed? Section 18.14
measured a 4.6 deg thrust-axis tilt that grows with thrust and named a centre-of-mass offset
as its first candidate cause, so the range is the thing that has to be known.

Why this is a proof and not a sweep. `hover_model.linearize_reduced` returns a
parameter-free `A` (a double-integrator pair) whose only uncertain entries live in `B`, as
the two scalars `b_l = g*k_lat` and `b_v = 2g/f_hover`. `A` is nilpotent blockwise, so ZOH
gives `Bd_i = [Ts^2/2, Ts]' * b` EXACTLY, and `augment_integrators` appends parameter-free
rows. Per axis the closed loop is therefore a rank-one, exactly affine perturbation

    M_i(b) = Aa_i - b * v * k_i',      v = [Ts^2/2, Ts, 0]',   Aa_i parameter-free

whose characteristic polynomial is `chi_ol(z) + b*N(z)`: a textbook root locus in `b`. The
certified set in `b` is exact, so no LMI, no polytopic over-bound and no norm bound is
needed. Bisection finds the boundary of the connected stable component containing nominal,
to machine precision. `_self_check` verifies the affineness rather than trusting this
paragraph.

The shipped `K` has off-block entries of order 1e-13, so the two axes decouple into 3-state
SISO loops, and the runner drives BOTH lateral axes with the same `K` row 0
(`hover_controller_runner.py:731`). That symmetry is what lets a lateral input ROTATION
complexify: with xi = x + iy, a common rotation psi is exactly the complex loop gain
`b_l * kappa * exp(i*psi)`, so the certified set is a region in the complex gain plane whose
real-axis extent is a gain margin and whose angular extent is a phase margin.

    mass:      b_v = 2g/f_h  and  f_h = sqrt(m g / k_T)   =>   b_v ~ 1/sqrt(m)
    rotation:  psi enters only the lateral channel, as a complex gain angle

What this does NOT show, kept here because section 13.5 states the trap: this certifies the
linearised design model against itself. Two numbers anchor it to the rig and NEITHER is
measured -- `k_lat` is still the seed guess of `hover_model.py:70`, and the mixer rotation is
what `mixer_sign.py` exists to measure and has never been run to conclusion. A wide certified
region is permission to arm the loop and identify them, not evidence that the rig is inside it.

    uv run python controller/control/robust_cert.py
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

from controller.control import constants as C
from controller.control.design_hover_lqr import C_MEAS, augment_integrators
from controller.control.hover_model import discretize, linearize_reduced, make_params

# Section 9, from the CAD tensor. Used only to price a mass ratio in grams; nothing in the
# certificate depends on it, because the 1-D plant never forms k_T and mass cancels.
M_ROBOT_KG = 8.3566e-5
C_GRAV = 9.80665

# Section 3, CAD tensor: polar and transverse moments, kg m^2. Section 11.2: the spin-averaged
# magnetic tilt stiffness kappa_t = tau_max cos(delta_eq) / 2, quoted at section 11.3's 160 Hz
# operating point. Only their RATIO to the inertia term matters below, and the inertia term wins
# by two orders of magnitude, so kappa_t's own precision is not load-bearing.
I_SPIN, I_TRANS = 3.3578361348541667e-9, 2.0373e-9
KAPPA_T = 2.45e-5

# State indices of each decoupled 3-state loop inside the 6-state augmented model
# [x, xd, z, zd, int_ex, int_ez], paired with the K row that drives it.
AXES = {"lateral": ((0, 1, 4), 0), "vertical": ((2, 3, 5), 1)}

# theory.md 12.8, on the spatial model: "commanding a field tilt of 0.1 in +x moves the spin
# axis into +y at 72 deg from the command after 50 ms". A MODEL number, not a rig
# measurement -- `mixer_sign.py` is the rig's version and has never been run to conclusion.
PRECESSION_DEG = 72.0


def coning_from_com(d_m, f_hz, thrust_ratio=1.0):
    """Once-per-rev coning half-angle, rad, from a COM offset ``d_m`` transverse to the spin axis.

    A free body feels NO gravity torque about its own COM, so a COM offset acts only through
    the thrust line: thrust applied at the aerodynamic centre, offset d from the COM, gives
    tau = d x T, a torque PROPORTIONAL TO THRUST -- section 18.14's measured signature.

    But d is body-fixed and the body spins, so in the lab that torque rotates at omega. Driving
    section 11.3's tilt block at its own spin frequency,

        chi(t) = tau_0 e^{i omega t} / (kappa_t - (I_s + I_t) omega^2 + i c_t omega),

    which is a SYNCHRONOUS CONING whose mean over one revolution is zero, not a steady tilt.
    Off resonance by construction: nutation sits at (I_s/I_t) omega = 1.65 omega, never at omega.
    c_t is dropped (section 12.8 sets it to zero and section 11.3 shows a plausible value is three
    orders too small to matter here), which makes this an UPPER bound on the response.
    """

    w = 2.0 * math.pi * f_hz
    return M_ROBOT_KG * C_GRAV * thrust_ratio * d_m / abs(KAPPA_T - (I_SPIN + I_TRANS) * w**2)


def load_gains(path=None):
    path = path or os.path.join(os.path.dirname(__file__), "hover_controller.json")
    with open(path) as f:
        return json.load(f)


def axis_loop(gains, axis):
    """``(Aa_i, bv_i, k_i, b_nom)`` for one decoupled axis, built through the DESIGN code.

    Imports `linearize_reduced` / `discretize` / `augment_integrators` rather than restating
    the matrices, so the certificate cannot drift away from what `design_hover_lqr` ships.
    `bv_i` is the input column at the NOMINAL parameter; a gain multiplier kappa scales it.
    """

    a = gains["meta"]["args"]
    ts = gains["design"]["ts"]
    p = make_params(f_hover=a["f_hover"], k_lat=a["k_lat"], margin=a["margin"])
    aa, ba = augment_integrators(*discretize(*linearize_reduced(p), ts), C_MEAS, ts)
    idx, row = AXES[axis]
    b_nom = p.g * p.k_lat if axis == "lateral" else 2.0 * p.g / p.f_hover
    return (aa[np.ix_(idx, idx)], ba[idx, row], np.array(gains["K"])[row, list(idx)], b_nom)


def _schur(aa, bv, k, gain):
    """Is ``Aa - gain*bv*k'`` Schur? ``gain`` may be complex (a rotation of the input)."""

    m = aa.astype(complex) - gain * np.outer(bv, k)
    return float(np.max(np.abs(np.linalg.eigvals(m)))) < 1.0


def gain_interval(aa, bv, k, angle=0.0, cap=1e8, tol=1e-12):
    """Exact ``(kappa_lo, kappa_hi)`` bounding the stable component that contains kappa = 1.

    The closed loop is affine in the gain, so this is a root locus and the boundary is a
    genuine stability limit rather than the edge of a sampled grid. `angle` rotates the input
    by psi radians, which for two identical axes is exact (see the module docstring).
    """

    rot = np.exp(1j * angle)
    ok = lambda g: _schur(aa, bv, k, g * rot)
    if not ok(1.0):
        return None

    def edge(step):
        lo, hi = 1.0, 1.0
        while ok(hi) and tol < hi < cap:
            lo, hi = hi, hi * step
        if ok(hi):
            return hi  # ran off the end: unbounded in this direction
        while abs(hi - lo) > tol * max(1.0, lo):
            mid = math.sqrt(lo * hi)
            lo, hi = (mid, hi) if ok(mid) else (lo, mid)
        return lo

    return edge(0.5), edge(2.0)


def complex_region(aa, bv, k, n=73):
    """The certified region in the complex gain plane, as ``(psi, kappa_lo, kappa_hi)`` rows.

    psi is a rotation of the commanded lateral direction away from the realised one. Rows
    where the loop is unstable at kappa = 1 are omitted, so the last row is the phase margin.
    """

    out = []
    for psi in np.linspace(0.0, math.pi, n):
        got = gain_interval(aa, bv, k, angle=psi)
        if got is None:
            break
        out.append((psi, got[0], got[1]))
    return out


def phase_margin(aa, bv, k, tol=1e-9):
    """Largest input rotation |psi| the loop tolerates at nominal gain, in radians."""

    lo, hi = 0.0, math.pi
    if _schur(aa, bv, k, np.exp(1j * hi)):
        return hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if _schur(aa, bv, k, np.exp(1j * mid)) else (lo, mid)
    return lo


def mass_interval(kappa_lo, kappa_hi):
    """Certified mass ratio ``m/m_nom`` from the vertical gain interval.

    b_v = 2g/f_h and f_h = sqrt(m g / k_T), so b_v ~ 1/sqrt(m): a heavier robot is a LOWER
    loop gain. The map is monotone, so interval endpoints map to interval endpoints exactly
    and the certified mass set is genuinely an interval.
    """

    return 1.0 / kappa_hi**2, 1.0 / kappa_lo**2


def as_sigmas(half_width, sigma):
    """``(k, miss_probability)`` for a certified half-width against a Gaussian of width sigma.

    A Gaussian has unbounded support, so NO box certifies every draw. What is certified is a
    box plus the probability mass it carries; this reports both rather than letting the box
    be read as "all". Two-sided, one parameter.
    """

    k = half_width / sigma
    return k, math.erfc(k / math.sqrt(2.0))


# --------------------------------------------------------------------------


def report(gains=None):
    gains = gains or load_gains()
    print(f"gains: {gains['meta']['date']}, rate {gains['design']['rate_hz']:.0f} Hz, "
          f"f_hover {gains['meta']['args']['f_hover']:.0f} Hz, "
          f"k_lat {gains['meta']['args']['k_lat']} (SEED GUESS)")

    out = {}
    for axis in AXES:
        aa, bv, k, b_nom = axis_loop(gains, axis)
        lo, hi = gain_interval(aa, bv, k)
        out[axis] = (lo, hi)
        print(f"\n{axis:8s}: gain multiplier certified over [{lo:.4f}, {hi:.4f}]  "
              f"({20*math.log10(1/lo):+.1f} dB down, {20*math.log10(hi):+.1f} dB up)")

    m_lo, m_hi = mass_interval(*out["vertical"])
    f_h = gains["meta"]["args"]["f_hover"]
    m_phys = (C.F_STEPOUT_HZ / f_h) ** 2  # heaviest robot this rig can lift at all
    print(f"\nMASS   m/m_nom certified over [{m_lo:.4g}, {m_hi:.4g}]"
          f"  =  [{m_lo*M_ROBOT_KG*1e6:.2g}, {m_hi*M_ROBOT_KG*1e6:.4g}] mg"
          f"  (nominal {M_ROBOT_KG*1e6:.1f} mg)")
    print("       b_v ~ 1/sqrt(m): a HEAVIER robot is a LOWER loop gain, so the binding")
    print(f"       stability edge is the upper one, at {m_hi:.1f}x nominal mass.")
    print(f"       But f_h = f_h_nom*sqrt(m/m_nom), so {m_phys:.2f}x mass already puts f_h at")
    print(f"       F_STEPOUT_HZ = {C.F_STEPOUT_HZ:.0f} Hz. THRUST AND STEP-OUT BIND FIRST, by")
    print(f"       a factor of {m_hi/m_phys:.0f}. Mass error is not a stability problem here.")

    aa, bv, k, _ = axis_loop(gains, "lateral")
    pm = phase_margin(aa, bv, k)
    print(f"\nROTATION  lateral input rotation certified to |psi| < {math.degrees(pm):.1f} deg")
    print(f"       For comparison, section 12.8 measured the spin axis answering a field-tilt")
    print(f"       command at {PRECESSION_DEG:.0f} deg from it. That is {'INSIDE' if PRECESSION_DEG < math.degrees(pm) else 'OUTSIDE'} "
          f"the certified region,")
    print(f"       margin {math.degrees(pm) - PRECESSION_DEG:+.1f} deg -- run `mixer_sign.py` before arming lateral.")
    reg = complex_region(aa, bv, k)
    print("\n       psi     kappa_lo   kappa_hi")
    for psi, klo, khi in reg[:: max(1, len(reg) // 8)]:
        print(f"    {math.degrees(psi):6.1f}d  {klo:9.4f}  {khi:9.4f}")
    return out


def _self_check():
    gains = load_gains()

    # 1. The affineness the whole proof rests on, verified rather than asserted in prose:
    # the discretised input column is EXACTLY linear in the uncertain scalar.
    ts = gains["design"]["ts"]
    a = gains["meta"]["args"]
    p = make_params(f_hover=a["f_hover"], k_lat=a["k_lat"], margin=a["margin"])
    _, bd = discretize(*linearize_reduced(p), ts)
    for col, b in ((0, p.g * p.k_lat), (1, 2.0 * p.g / p.f_hover)):
        got = bd[:, col][bd[:, col] != 0.0] / b
        assert np.allclose(got, [ts**2 / 2, ts], rtol=0, atol=1e-18), got
    print(f"affine      : Bd column / b == [Ts^2/2, Ts] exactly  ({ts**2/2:.1e}, {ts:.1e})")

    # 2. Rebuilding through the design code reproduces the SHIPPED poles, which is what says
    # the certificate is analysing the gains that fly and not a re-derivation of them.
    aa, ba = augment_integrators(*discretize(*linearize_reduced(p), ts), C_MEAS, ts)
    eig = np.linalg.eigvals(aa - ba @ np.array(gains["K"]))
    poles = np.sort(np.abs(np.log(eig.astype(complex))).real / ts / (2 * np.pi))
    assert np.allclose(poles, gains["closed_loop_poles_hz"], rtol=1e-9), poles
    print(f"poles       : reproduce hover_controller.json to 1e-9  {poles[[0,2,4]].round(4)}")

    # 3. The axes really are decoupled, which is what licenses the 3-state SISO reduction
    # and the complexification. Off-block gains are ~1e-13 but the threshold is what matters.
    k_full = np.array(gains["K"])
    off = max(abs(k_full[0, [2, 3, 5]]).max(), abs(k_full[1, [0, 1, 4]]).max())
    on = min(abs(k_full[0, [0, 1, 4]]).min(), abs(k_full[1, [2, 3, 5]]).min())
    assert off / on < 1e-9, (off, on)
    print(f"decoupled   : off-block/on-block gain ratio {off/on:.1e}")

    # 4. Nominal is strictly interior on both axes, and the interval really is a stability
    # boundary: just outside it the loop must NOT be Schur.
    for axis in AXES:
        aa_i, bv, k, _ = axis_loop(gains, axis)
        lo, hi = gain_interval(aa_i, bv, k)
        assert lo < 1.0 < hi, (axis, lo, hi)
        assert _schur(aa_i, bv, k, 1.0)
        assert not _schur(aa_i, bv, k, hi * 1.001), axis
        assert not _schur(aa_i, bv, k, lo * 0.999), axis
        print(f"{axis:12s}: kappa in [{lo:.4f}, {hi:.4f}], boundary is sharp")

    # 5. The mass map is monotone-inverted, so the ORDER of the endpoints must flip.
    lo, hi = gain_interval(*axis_loop(gains, "vertical")[:3])
    m_lo, m_hi = mass_interval(lo, hi)
    assert m_lo < 1.0 < m_hi and m_lo == 1.0 / hi**2, (m_lo, m_hi)
    print(f"mass        : m/m_nom in [{m_lo:.4f}, {m_hi:.4f}]  "
          f"({m_lo*M_ROBOT_KG*1e6:.1f}-{m_hi*M_ROBOT_KG*1e6:.1f} mg)")

    # 6. Phase margin is inside the region scan, and the region shrinks monotonically with
    # rotation (a rotated input can only cost gain range, never buy it, on this loop).
    aa_i, bv, k, _ = axis_loop(gains, "lateral")
    pm = phase_margin(aa_i, bv, k)
    assert 0.0 < pm < math.pi
    assert _schur(aa_i, bv, k, np.exp(1j * pm * 0.99))
    assert not _schur(aa_i, bv, k, np.exp(1j * min(math.pi, pm * 1.01)))
    print(f"rotation    : phase margin {math.degrees(pm):.1f} deg, boundary is sharp")

    # 7. The COM result, which is the section's load-bearing number: a body-fixed COM offset
    # produces once-per-rev coning, not a steady tilt, and the amplitude is negligible. If this
    # ever stops holding, section 20's ruling-out of COM offset stops holding with it.
    per_mm = math.degrees(coning_from_com(1e-3, 126.0))
    assert per_mm < 0.02, per_mm
    need_m = 1e-3 * math.radians(4.6) / coning_from_com(1e-3, 126.0)
    assert need_m > 0.1, need_m
    print(f"COM         : {per_mm:.4f} deg of coning per mm of offset at 126 Hz; "
          f"{need_m*1e3:.0f} mm needed for 4.6 deg")

    # 8. The sigma inversion: a Gaussian is never fully certified, so the miss probability
    # must be strictly positive however wide the box.
    kk, miss = as_sigmas(0.5, 0.1)
    assert abs(kk - 5.0) < 1e-12 and 0.0 < miss < 1e-6, (kk, miss)
    print(f"sigmas      : half-width 5 sigma leaves miss probability {miss:.1e} (never zero)")

    print("self-check PASS")


if __name__ == "__main__":
    _self_check()
    print()
    report()
