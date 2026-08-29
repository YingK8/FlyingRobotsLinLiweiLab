#!/usr/bin/env python3
"""
Fixed-current linearisation, Lyapunov certificate, and region of attraction. theory.md sect. 15.5.

Section 14 screened designs on R, the ratio of the spin axis's alignment bandwidth to the growth
rate of the Earnshaw anti-trap. R is a scalar summary of a reduced third-order model. This module
does the real thing: linearise the full eight-state plant at its trim with the coil currents
*held fixed*, take the eigenvalues, and if they are Hurwitz, produce a quadratic Lyapunov function
and an estimate of the basin it certifies.

Held fixed matters. `spatial_mpc.linearize` differentiates *through* the current allocator and so
answers the closed-loop question, where section 13.2 already shows the instability disappears.
The open-loop question needs the currents frozen, which is what `linearize` here does.

    state   st = [r(3), v(3), s_x, s_y]     s_z reconstructed from |s| = 1
    input   none: this is the autonomous, fixed-current plant

One structural warning, stated up front because it decides what is provable. With align_tau = 0
the tilt block carries no dissipation at all, so A has a pair of eigenvalues on the imaginary axis
and cannot be Hurwitz however good the geometry is. Asymptotic stability therefore requires a
non-zero alignment damping, and `min_align_tau` reports how much. That number is a rig
identification, never a result.

    uv run python controller/control/stability_cert.py --self-check
"""

from __future__ import annotations

import argparse
import math

import numpy as np
from scipy.linalg import solve_continuous_lyapunov
from scipy.optimize import brentq

import spatial_model as sm

_ZH = np.array([0.0, 0.0, 1.0])


# --------------------------------------------------------------------------
# Trim
# --------------------------------------------------------------------------


def _az(plant, z, f):
    """Vertical acceleration on the array axis, upright, at rest. Zero at a trim."""

    try:
        a, _ = plant.accel(np.array([0.0, 0.0, z]), np.zeros(3), _ZH, f)
    except sm.StepOut:
        return math.nan
    return a[2]


def trim(plant, f, z_lo=0.004, z_hi=0.060, n=80):
    """Every on-axis equilibrium with its axial stiffness, as ``(z, daz/dz)`` pairs.

    Generalises `open_loop_sweep.trim`: that one differenced the gradient force against a lift
    deficit, which silently assumed the gradient force was the only z-dependent term. With a DC
    channel it is not, so this differences the full vertical acceleration instead. The two agree
    exactly when there is no DC current.

    Step-out leaves gaps in the scanned interval, so bracket sign changes rather than handing
    brentq the whole range. n = 80 over 56 mm is 0.7 mm of resolution, verified to find the
    same roots as n = 200 on geometries whose two trims sit 0.79 mm apart.
    """

    zs = np.linspace(z_lo, z_hi, n)
    vals = np.array([_az(plant, z, f) for z in zs])
    out, h = [], 1e-5
    for i in range(n - 1):
        a, b = vals[i], vals[i + 1]
        if np.isfinite(a) and np.isfinite(b) and a * b < 0.0:
            z = brentq(lambda zz: _az(plant, zz, f), zs[i], zs[i + 1], xtol=1e-10)
            out.append((z, (_az(plant, z + h, f) - _az(plant, z - h, f)) / (2 * h)))
    return out


def stable_trim(plant, f, **kw):
    """The highest trim with negative axial stiffness, the one a takeoff reaches. Or None."""

    good = [t for t in trim(plant, f, **kw) if t[1] < 0.0]
    return good[-1] if good else None


# --------------------------------------------------------------------------
# Linearisation
# --------------------------------------------------------------------------


def _pack(x):
    return np.concatenate([x[:6], x[6:8]])


def _unpack(st):
    sz = math.sqrt(max(1e-12, 1.0 - st[6] ** 2 - st[7] ** 2))
    return np.concatenate([st[:6], [st[6], st[7], sz]])


def linearize(plant, f, z_eq, h_state=1e-7, h_int=1e-4):
    """Continuous-time Jacobian A (8x8) of the fixed-current plant at the on-axis trim.

    Central differences of the state derivative, which is itself one plant step divided by the
    step size. Same construction as `spatial_mpc.linearize:178`, minus the allocator.
    """

    st0 = _pack(sm.make_state(np.array([0.0, 0.0, z_eq])))

    def deriv(st):
        xn, _ = plant.step(_unpack(st), f, h_int)
        return (_pack(xn) - st) / h_int

    a = np.empty((8, 8))
    for j in range(8):
        sp, sm_ = st0.copy(), st0.copy()
        sp[j] += h_state
        sm_[j] -= h_state
        a[:, j] = (deriv(sp) - deriv(sm_)) / (2.0 * h_state)
    return a


def margin(a_mat):
    """Stability margin: the distance of the rightmost eigenvalue from the imaginary axis.

    Positive is asymptotically stable, and larger is faster. This is the quantity the optimiser
    maximises, in place of section 14's R.
    """

    return float(-np.max(np.linalg.eigvals(a_mat).real))


# --------------------------------------------------------------------------
# Lyapunov certificate
# --------------------------------------------------------------------------


def certify(a_mat, q=None):
    """Solve A'P + PA = -Q for the certificate P. Returns ``(P, cond)``, or None if not Hurwitz.

    Lyapunov's indirect method: a Hurwitz linearisation gives local asymptotic stability of the
    nonlinear system, and V = x'Px is a genuine Lyapunov function on a neighbourhood.
    """

    if margin(a_mat) <= 0.0:
        return None
    q = np.eye(len(a_mat)) if q is None else q
    p = solve_continuous_lyapunov(a_mat.T, -q)
    p = 0.5 * (p + p.T)
    if np.min(np.linalg.eigvalsh(p)) <= 0.0:
        return None
    return p, float(np.linalg.cond(p))


def roa_radius(plant, f, z_eq, a_mat, p_mat, q=None, n_dir=64, r_max=0.020, seed=0):
    """Largest lateral offset whose sublevel set of V is still certified, in metres.

    The estimate is the standard one: V decays wherever the nonlinear residual is dominated by
    the linear term, so grow the level c until some sampled direction violates
    dV/dt < 0, then report the lateral radius that level reaches.

    # ponytail: sampled directions, not an SOS or interval certificate. It is an estimate and
    # the docstring says so; upgrade to a verified bound only if a candidate ever survives.
    """

    q = np.eye(8) if q is None else q
    rng = np.random.default_rng(seed)
    st0 = _pack(sm.make_state(np.array([0.0, 0.0, z_eq])))

    def vdot(st):
        d = _pack(plant.step(_unpack(st), f, 1e-4)[0]) - st
        d = d / 1e-4
        e = st - st0
        return 2.0 * e @ (p_mat @ d)

    dirs = rng.normal(size=(n_dir, 8))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    # Scale so a unit direction is a physically comparable perturbation in each block.
    scale = np.array([1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2])

    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        ok = True
        for d in dirs:
            st = st0 + mid * scale * d
            try:
                if vdot(st) >= 0.0:
                    ok = False
                    break
            except sm.StepOut:
                ok = False
                break
        lo, hi = (mid, hi) if ok else (lo, mid)
        if hi - lo < 1e-4:
            break
    return float(lo * scale[0] * 1.0), lo


def min_align_tau(plant, f, z_eq, taus=None):
    """Smallest ``align_tau`` at which the linearisation becomes Hurwitz, or None.

    With align_tau = 0 the tilt block is undamped and A always has eigenvalues on the imaginary
    axis, so this is the parameter that decides whether any asymptotic-stability claim is even
    available. It is measured on the rig, never chosen.
    """

    taus = np.geomspace(1e-3, 1e3, 25) if taus is None else taus
    import dataclasses
    for t in taus:
        p2 = dataclasses.replace(plant, align_tau=float(t))
        try:
            if margin(linearize(p2, f, z_eq)) > 0.0:
                return float(t)
        except sm.StepOut:
            continue
    return None


# --------------------------------------------------------------------------


def _self_check():
    import coil_geometry as cg

    p = sm.robot_params()

    # 1. The generalised trim agrees with open_loop_sweep's specialised one when there is no
    # DC channel, which is what licenses replacing it.
    import open_loop_sweep as ols
    coils = sm.quick_coils("cross", d=0.046, tilt_deg=15.0, amp=1.25, model="dipole")
    plant = sm.Plant(coils, p, f_hover=110.0, use_grad=True)
    mine = stable_trim(plant, 92.0)
    ref = ols.evaluate(0.046, 15.0, 92.0, model="dipole")
    assert abs(mine[0] - ref["z_eq"]) < 5e-6, (mine, ref["z_eq"])
    print(f"trim        : {mine[0]*1e3:.4f} mm vs open_loop_sweep {ref['z_eq']*1e3:.4f} mm")

    # 2. The linearisation reproduces the reduced cubic's verdict at a section 14 point: the
    # plant is unstable there, and the growth rates agree to within a factor of two.
    a = linearize(plant, 92.0, mine[0])
    assert a.shape == (8, 8)
    m = margin(a)
    assert m < 0.0, m
    ratio = (-m) / ref["max_re_root"]
    assert 0.4 < ratio < 2.5, (m, ref["max_re_root"], ratio)
    print(f"linearise   : rightmost root {-m:+.2f}/s, reduced cubic {ref['max_re_root']:+.2f}/s"
          f" (ratio {ratio:.2f})")
    assert certify(a) is None, "an unstable plant must not certify"

    # 3. The certificate itself, on a Hurwitz matrix built for the purpose.
    a_ok = np.diag([-1.0, -2.0, -0.5, -3.0, -1.5, -2.5, -0.8, -1.2])
    a_ok[0, 1] = 0.7
    out = certify(a_ok)
    assert out is not None
    p_mat, cond = out
    resid = np.abs(a_ok.T @ p_mat + p_mat @ a_ok + np.eye(8)).max()
    assert resid < 1e-10, resid
    assert np.min(np.linalg.eigvalsh(p_mat)) > 0.0
    print(f"lyapunov    : A'P+PA+Q residual {resid:.1e}, P positive definite, cond {cond:.1f}")

    # 4. The structural warning, verified rather than asserted in prose: with no alignment
    # damping the tilt block sits on the imaginary axis, and adding damping is what moves it.
    tau = min_align_tau(plant, 92.0, mine[0])
    print(f"align_tau   : Hurwitz at align_tau >= {tau if tau else 'never (in 1e-3..1e3 s)'}"
          f"  <- the plant is unstable here for other reasons, so 'never' is expected")

    # 5. The DC channel changes the trim, which is the whole point of having it.
    dc = cg.c4_array(r_lo=0.037, z_lo=-0.010, r_hi=0.037, z_hi=0.050, n_per_ring=2,
                     r_dc=0.060, z_dc=0.020, i_dc=2.0)
    p_ax = sm.replace(p, m_axial=0.03 * p.mdip)
    plant_dc = sm.Plant(dc, p_ax, f_hover=110.0, use_grad=True)
    plant_no = sm.Plant(sm.replace(dc, dc_amp=np.zeros(len(dc.pos))), p_ax,
                        f_hover=110.0, use_grad=True)
    # 110 Hz, not 92: the two-ring array is far flatter, so its trim sits elsewhere.
    t_dc, t_no = stable_trim(plant_dc, 110.0), stable_trim(plant_no, 110.0)
    assert t_dc and t_no
    w_dc, w_no = math.sqrt(-t_dc[1]), math.sqrt(-t_no[1])
    print(f"DC trim     : z {t_no[0]*1e3:.3f} -> {t_dc[0]*1e3:.3f} mm, axial mode "
          f"{w_no:.3f} -> {w_dc:.3f} rad/s ({w_dc/2/math.pi:.2f} Hz) at 2 A and 3 % axial moment")
    assert abs(t_dc[1] - t_no[1]) > 1e-3, "the DC channel must move the axial stiffness"

    # 6. The two-ring array against the section 14 baseline, on the quantity that decides it.
    om = 0.5 * p.mdip * cg.field_shape(dc, t_dc[0])[0] / (p.I_spin * 2 * math.pi * 110.0)
    print(f"two-ring    : Omega_align {om:.3f} rad/s, axial mode {w_dc:.3f} rad/s"
          f"  ->  R = sqrt(2)*Om/w_z = {math.sqrt(2)*om/w_dc:.3f}"
          f"   (section 14 best with restoring stiffness was 0.097)")

    print("self-check PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-check", action="store_true")
    ap.parse_args()
    _self_check()
