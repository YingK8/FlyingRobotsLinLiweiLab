#!/usr/bin/env python3
"""
Open-loop passive stability of the coil matrix. See theory.md section 14.

An earlier sweep (write-up removed; theory.md section 14 supersedes it) found coil
geometries where the quasi-static lateral stiffness
C_net = C_tilt + C_grad goes negative, and read that as passive lateral trapping. It is not:
C_net < 0 is the s -> 0 limit of a third-order loop whose middle block is the spin axis, and
the spin axis is bandwidth-limited by its own angular momentum. The screening quantity is

    R = Omega_align / lambda_grad,   Omega_align = kappa_t / (I_s w),
                                     lambda_grad = sqrt(C_grad / m)

: how fast the axis can answer a field tilt, against how fast the Earnshaw anti-trap grows.
R < 1 means the tilt term is rolled off before it can act and C_net is irrelevant.

This module sweeps every knob against R and reports where the feasible region is:

    A  coil spacing d, outward tilt, drive frequency f
    B  drive amplitude and 2x2 pitch
    C  quadrature phase error, i.e. field ellipticity
    D  the rotor itself: dipole moment, spin inertia, mass, and uniform scale
    E  nonlinear trajectories at the best candidates, under both field models

Every quantity comes from spatial_model.py unchanged. Both field backends run on every
point, so section 12.9's point-dipole error is a measured column rather than a caveat.

    uv run python controller/control/open_loop_sweep.py --self-check
    uv run python controller/control/open_loop_sweep.py
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

import spatial_model as sm

OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "open_loop_sweep"

COIL_R = 0.0105       # loop radius, m, as quick_coils builds it
COIL_THICK = 0.022    # physical axial extent, the filename's 22 mm
LOCK_MAX = 0.8        # section 12.3's safety limit on sin(phi)
CLEARANCE_MIN = 0.002  # that sweep's own criterion: 2 mm above the coil envelope

_ZH = np.array([0.0, 0.0, 1.0])


# --------------------------------------------------------------------------
# Parameterization
# --------------------------------------------------------------------------


def scale_robot(robot, k=1.0, mdip=1.0, i_spin=1.0, mass=1.0):
    """A rotor scaled by linear factor ``k``, or by per-quantity factors.

    Under a uniform scale the magnets and the body scale together: dipole moment and mass go
    as k^3, the inertias as k^5, and the hover frequency as k^-1/2 because blade thrust goes
    as f^2 D^4 against a weight going as k^3. `hover_scale` returns that last factor, since
    the caller has to move f with it to hold the lift fraction and so z_eq fixed.
    """

    return replace(
        robot,
        mdip=robot.mdip * mdip * k**3,
        mass=robot.mass * mass * k**3,
        I_spin=robot.I_spin * i_spin * k**5,
        I_trans=robot.I_trans * i_spin * k**5,
    )


def hover_scale(k):
    """Hover frequency factor for a rotor at linear scale ``k``."""

    return k**-0.5


def array(d, tilt_deg, amp, pitch, model, phase_err_deg=0.0):
    """`quick_coils` plus a quadrature error on the sin-pair of channels."""

    coils = sm.quick_coils(
        "cross", d=d, tilt_deg=tilt_deg, amp=amp, pitch=pitch, model=model
    )
    if phase_err_deg:
        # Channels 1 and 3 carry v; shifting the pair bodily breaks quadrature with the
        # u pair, which is exactly what turns the circular field elliptical.
        phase = coils.phase.copy()
        phase[np.isin(coils.channel, (1, 3))] += math.radians(phase_err_deg)
        coils = replace(coils, phase=phase)
    return coils


def envelope_top(coils):
    """Highest z reached by any coil body, as a cylinder of radius R and length 22 mm."""

    nz = np.abs(coils.nhat[:, 2])
    rim = COIL_R * np.sqrt(np.maximum(1.0 - nz**2, 0.0))
    return float(np.max(coils.pos[:, 2] + rim + 0.5 * COIL_THICK * nz))


# --------------------------------------------------------------------------
# Trim and stiffness
# --------------------------------------------------------------------------


def _fz(plant, f, need):
    """Axial force residual on the symmetry axis: the gradient force minus the lift deficit."""

    def g(z):
        try:
            _, info = plant.accel(np.array([0.0, 0.0, z]), np.zeros(3), _ZH, f)
        except sm.StepOut:
            return math.nan
        return info["f_grad"][2] - need

    return g


def trim(plant, f, z_lo=0.006, z_hi=0.045, n=150):
    """Every on-axis equilibrium in [z_lo, z_hi], with its axial stiffness.

    `Plant.accel` raises StepOut inside the bracket wherever the field is too weak to hold
    lock, so brentq cannot be handed the interval directly. Scan first, treating step-out as
    a gap, then bracket each sign change that survives.
    """

    need = plant.robot.mass * sm.GRAVITY * (1.0 - (f / plant.f_hover) ** 2)
    g = _fz(plant, f, need)
    zs = np.linspace(z_lo, z_hi, n)
    vals = np.array([g(z) for z in zs])

    out = []
    h = 1e-5
    for i in range(n - 1):
        a, b = vals[i], vals[i + 1]
        if np.isfinite(a) and np.isfinite(b) and a * b < 0.0:
            z = brentq(g, zs[i], zs[i + 1], xtol=1e-9)
            out.append((z, (g(z + h) - g(z - h)) / (2.0 * h)))
    return out


def evaluate(d=0.046, tilt_deg=15.0, f=92.0, amp=1.25, pitch=0.021, model="dipole",
             phase_err_deg=0.0, robot=None, f_hover=110.0, k_drag=3.0e-10, beta=0.20):
    """Every stability metric at one design point, or None if it has no stable trim."""

    robot = robot if robot is not None else sm.robot_params()
    coils = array(d, tilt_deg, amp, pitch, model, phase_err_deg)
    plant = sm.Plant(coils, robot, f_hover=f_hover, k_drag=k_drag, beta=beta, use_grad=True)

    stable = [(z, kz) for z, kz in trim(plant, f) if kz < 0.0]
    if not stable:
        return None
    z_eq, k_z = stable[-1]   # the highest stable trim, the one takeoff reaches

    h = 1e-5
    at = lambda x: np.array([x, 0.0, z_eq])

    def n_x(x):
        u, v, _, _ = sm.field_at(at(x), coils, None, False)
        return sm.rotation_axis(u, v)[0]

    g_n = (n_x(h) - n_x(-h)) / (2.0 * h)

    def fx(x):
        _, info = plant.accel(at(x), np.zeros(3), _ZH, f)
        return info["f_grad"][0]

    c_grad = (fx(h) - fx(-h)) / (2.0 * h)

    # The same derivative with the moment frozen at its on-axis value. The difference from
    # c_grad is the part of the lateral force that comes from the moment re-orienting, which
    # is also what puts a residual in the Laplace trace below.
    u0, v0, _, _ = sm.field_at(at(0.0), coils, None, True)
    n0 = sm.rotation_axis(u0, v0)
    phi0, a_ax, b_ax, lock = sm.phase_lag(u0, v0, robot.mdip, k_drag * f * f)
    _, m_cos, m_sin = sm.cycle_average(_ZH, u0, v0, n0, robot.mdip, phi0)

    def f_fixed(dr, i):
        _, _, du, dv = sm.field_at(at(0.0) + dr, coils, None, True)
        return sm.gradient_force(m_cos, m_sin, du, dv)[i]

    ex, ez = np.array([h, 0.0, 0.0]), np.array([0.0, 0.0, h])
    c_grad_fixed = (f_fixed(ex, 0) - f_fixed(-ex, 0)) / (2.0 * h)
    k_z_fixed = (f_fixed(ez, 2) - f_fixed(-ez, 2)) / (2.0 * h)

    lift = robot.mass * sm.GRAVITY * (f / f_hover) ** 2
    c_tilt = lift * g_n
    c_net = c_tilt + c_grad

    b_mean = 0.5 * (a_ax + b_ax)
    kappa_t = 0.5 * robot.mdip * b_mean * math.cos(phi0)
    omega = 2.0 * math.pi * f
    om_align = kappa_t / (robot.I_spin * omega)
    lam_grad = math.sqrt(c_grad / robot.mass) if c_grad > 0 else 0.0

    roots = lateral_roots(robot.mass, beta, c_grad, c_net, kappa_t, robot.I_spin, omega)
    return dict(
        d=d, tilt_deg=tilt_deg, f=f, amp=amp, pitch=pitch, model=model,
        phase_err_deg=phase_err_deg, z_eq=z_eq, k_z=k_z, G_n=g_n,
        C_grad=c_grad, C_grad_fixed_m=c_grad_fixed, C_tilt=c_tilt, C_net=c_net,
        k_z_fixed_m=k_z_fixed, trace_resid=2.0 * c_grad + k_z,
        trace_resid_fixed_m=2.0 * c_grad_fixed + k_z_fixed, B_mean=b_mean, lock=lock, ellipticity=b_ax / a_ax,
        kappa_t=kappa_t, Omega_align=om_align, lambda_grad=lam_grad,
        R=om_align / lam_grad if lam_grad > 0 else math.inf,
        max_re_root=float(np.max(roots.real)),
        clearance=z_eq - envelope_top(coils),
    )


def lateral_roots(mass, beta, c_grad, c_net, kappa_t, i_spin, omega, c_t=0.0):
    """Roots of section 14's lateral characteristic polynomial, complex coefficients.

        m s^3 + m(beta + p) s^2 + (m beta p - C_grad) s - C_net p = 0,
        p = kappa_t / (c_t + i I_s w)

    Complex, not a modelling shortcut: the gyroscopic term genuinely splits forward whirl
    from backward whirl, so the two lateral axes do not decouple into a real pair.
    """

    p = kappa_t / complex(c_t, i_spin * omega)
    return np.roots([mass, mass * (beta + p), mass * beta * p - c_grad, -c_net * p])


def trajectory(row, robot=None, offset=1e-5, dt=2e-4, t_max=3.0, box=0.010):
    """Open loop from ``offset`` off axis at the trim point. Returns (verdict, t, path)."""

    robot = robot if robot is not None else sm.robot_params()
    coils = array(row["d"], row["tilt_deg"], row["amp"], row["pitch"], row["model"],
                  row["phase_err_deg"])
    plant = sm.Plant(coils, robot, f_hover=row.get("f_hover", 110.0), use_grad=True)

    u, v, _, _ = sm.field_at(np.array([0.0, 0.0, row["z_eq"]]), coils, None, False)
    x = sm.make_state(np.array([offset, 0.0, row["z_eq"]]), s=sm.rotation_axis(u, v))

    path = [x[0:3].copy()]
    for k in range(int(t_max / dt)):
        try:
            x, _ = plant.step(x, row["f"], dt)
        except sm.StepOut:
            return "step-out", k * dt, np.array(path)
        path.append(x[0:3].copy())
        if math.hypot(x[0], x[1]) > box:
            return "escaped", k * dt, np.array(path)
    return "held", t_max, np.array(path)


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------


def _rows(points, **fixed):
    out = []
    for kw in points:
        r = evaluate(**kw, **fixed)
        if r is not None:
            out.append(r)
    return out


def stage_a(models=("dipole", "loop")):
    """Coil spacing, outward tilt, drive frequency. The main surface."""

    ds = np.arange(0.034, 0.0601, 0.001)
    tilts = np.arange(0.0, 30.1, 2.0)
    fs = np.arange(60.0, 130.1, 5.0)
    pts = [dict(d=float(d), tilt_deg=float(t), f=float(f))
           for d in ds for t in tilts for f in fs]
    out = []
    for m in models:
        out += _rows(pts, model=m)
    return pd.DataFrame(out).assign(stage="A")


def stage_b(best, models=("dipole", "loop")):
    """Drive amplitude and 2x2 pitch, at the stage-A front."""

    amps = np.geomspace(0.4, 8.0, 17)
    pitches = np.arange(0.015, 0.0271, 0.002)
    pts = [dict(d=best["d"], tilt_deg=best["tilt_deg"], f=best["f"], amp=float(a))
           for a in amps]
    pts += [dict(d=best["d"], tilt_deg=best["tilt_deg"], f=best["f"], pitch=float(p))
            for p in pitches]
    out = []
    for m in models:
        out += _rows(pts, model=m)
    return pd.DataFrame(out).assign(stage="B")


def stage_c(best, models=("dipole", "loop")):
    """Quadrature phase error: field ellipticity, and what it costs."""

    pts = [dict(d=best["d"], tilt_deg=best["tilt_deg"], f=best["f"],
                phase_err_deg=float(e)) for e in np.arange(0.0, 30.1, 2.0)]
    out = []
    for m in models:
        out += _rows(pts, model=m)
    return pd.DataFrame(out).assign(stage="C")


def stage_d(best, model="loop"):
    """The rotor. Section 14 predicts the exponents; this measures them."""

    base = sm.robot_params()
    rows = []
    axes = {
        "mdip": ("mdip", np.geomspace(0.25, 8.0, 13)),
        "i_spin": ("i_spin", np.geomspace(0.125, 4.0, 13)),
        "mass": ("mass", np.geomspace(0.5, 2.0, 9)),
    }
    for name, (kw, factors) in axes.items():
        for x in factors:
            r = evaluate(d=best["d"], tilt_deg=best["tilt_deg"], f=best["f"], model=model,
                         robot=scale_robot(base, **{kw: float(x)}))
            if r is not None:
                rows.append({**r, "axis": name, "factor": float(x)})

    # Uniform scale moves the hover frequency with it, so f moves too and the lift fraction,
    # and with it z_eq, is held fixed. That is what makes this a one-parameter family.
    for k in np.geomspace(0.125, 2.0, 17):
        k = float(k)
        r = evaluate(d=best["d"], tilt_deg=best["tilt_deg"], f=best["f"] * hover_scale(k),
                     model=model, robot=scale_robot(base, k=k),
                     f_hover=110.0 * hover_scale(k), k_drag=3.0e-10 * k**5)
        if r is not None:
            rows.append({**r, "axis": "scale", "factor": k})
    return pd.DataFrame(rows).assign(stage="D")


def fit_exponent(df, axis, col="R"):
    """Power-law exponent of ``col`` against the scale factor, by least squares in logs."""

    sub = df[df["axis"] == axis]
    if len(sub) < 3:
        return None
    return float(np.polyfit(np.log(sub["factor"]), np.log(sub[col]), 1)[0])


def solve_scale(df, target=1.0):
    """The uniform rotor scale at which R reaches ``target``, from the fitted exponent."""

    sub = df[df["axis"] == "scale"].sort_values("factor")
    e = fit_exponent(df, "scale")
    if e is None or not len(sub):
        return None
    return float(sub["factor"].iloc[0] * (target / sub["R"].iloc[0]) ** (1.0 / e))


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------


def plot_stage_a(df, path, f_slice=None):
    """R and C_net over (d, tilt) at one frequency, with the three feasibility boundaries."""

    import matplotlib.pyplot as plt

    df = df[df["model"] == "loop"].copy()
    ok = (df["lock"] < LOCK_MAX) & (df["clearance"] > CLEARANCE_MIN)
    if f_slice is None:
        # The slice worth showing is the one where the restoring region actually lives, not
        # the one holding the global R maximum: that maximum sits at C_net > 0.
        counts = df[ok & (df["C_net"] < 0)].groupby("f").size()
        f_slice = float(counts.idxmax()) if len(counts) else float(df["f"].iloc[0])
    sl = df[df["f"] == f_slice]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)
    for ax, col, label, cmap in (
        (axes[0], "R", r"$R = \Omega_{align}/\lambda_{grad}$   (stable needs $>1$)", "viridis"),
        (axes[1], "C_net", r"$C_{net}$ [N/m]   (restoring is $<0$)", "coolwarm"),
    ):
        piv = sl.pivot_table(index="tilt_deg", columns="d", values=col)
        x, y = piv.columns.values * 1e3, piv.index.values
        lim = np.nanmax(np.abs(piv.values)) if col == "C_net" else None
        im = ax.pcolormesh(x, y, piv.values, shading="nearest", cmap=cmap,
                           vmin=-lim if lim else None, vmax=lim if lim else None)
        fig.colorbar(im, ax=ax, label=label)
        for src, lev, c, ls in (("C_net", 0.0, "k", "-"),
                                ("lock", LOCK_MAX, "#d62728", "-"),
                                ("clearance", CLEARANCE_MIN, "w", "--")):
            p2 = sl.pivot_table(index="tilt_deg", columns="d", values=src)
            if np.nanmin(p2.values) < lev < np.nanmax(p2.values):
                ax.contour(x, y, p2.values, levels=[lev], colors=c,
                           linewidths=1.6, linestyles=ls)
        ax.set_xlabel("channel radius $d$ [mm]")
        ax.set_ylabel("outward coil tilt [deg]")
        ax.set_title(f"{col}   at $f$ = {f_slice:.0f} Hz")
    fig.suptitle("black: $C_{net}=0$   red: lock $=0.8$   white dashed: 2 mm clearance   "
                 "blank: no stable trim", fontsize=9)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_thresholds(df, cands, path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), constrained_layout=True)
    cols = [("z_eq", "trim height [mm]", 1e3), ("C_net", r"$C_{net}$ [N/m]", 1.0),
            ("R", r"$R = \Omega_{align}/\lambda_{grad}$", 1.0),
            ("lock", r"$\sin\varphi$ (step-out at 1)", 1.0)]
    for ax, (col, label, scale) in zip(axes.ravel(), cols):
        for d, t in cands:
            for model, ls in (("dipole", "--"), ("loop", "-")):
                sub = df[np.isclose(df["d"], d) & np.isclose(df["tilt_deg"], t)
                         & (df["model"] == model)]
                sub = sub.sort_values("f")
                ax.plot(sub["f"], sub[col] * scale, ls,
                        label=f"{d*1e3:.0f} mm, {t:.0f} deg, {model}")
        ax.set_xlabel("drive frequency $f$ [Hz]")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        if col in ("C_net",):
            ax.axhline(0.0, color="k", lw=0.8)
        if col == "R":
            ax.axhline(1.0, color="r", lw=1.0)
            ax.set_yscale("log")
        if col == "lock":
            ax.axhline(LOCK_MAX, color="r", lw=1.0)
    axes[0, 0].legend(fontsize=7)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_scaling(df, path):
    import matplotlib.pyplot as plt

    predicted = {"mdip": 0.5, "i_spin": -1.0, "mass": 0.5, "scale": -1.5}
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.8), constrained_layout=True)
    for ax, axis in zip(axes, predicted):
        sub = df[df["axis"] == axis].sort_values("factor")
        if len(sub) < 3:
            continue
        ax.loglog(sub["factor"], sub["R"], "o-", ms=3, label="measured")
        ref = sub["R"].iloc[len(sub) // 2] * (
            sub["factor"] / sub["factor"].iloc[len(sub) // 2]) ** predicted[axis]
        ax.loglog(sub["factor"], ref, "k--", lw=1,
                  label=f"predicted $\\propto x^{{{predicted[axis]:g}}}$")
        ax.axhline(1.0, color="#d62728", lw=1.0)
        if axis == "scale":
            k = solve_scale(df)
            if k:
                ax.axvline(k, color="#d62728", lw=1.0, ls=":")
                ax.annotate(f"$R=1$ at $k$ = {k:.3f}", (k, 1.0), textcoords="offset points",
                            xytext=(6, 8), fontsize=8, color="#d62728")
        ax.set_xlabel(f"{axis} factor")
        ax.set_ylabel("$R$")
        e = fit_exponent(df, axis)
        ax.set_title(f"{axis}:  fitted {e:+.3f}, predicted {predicted[axis]:+.1f}"
                     if e is not None else axis, fontsize=10)
        ax.grid(alpha=0.25, which="both")
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.legend(fontsize=7, loc="lower left" if predicted[axis] > 0 else "upper right")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_trajectories(runs, path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), constrained_layout=True)
    for label, verdict, t_end, path_xyz in runs:
        axes[0].plot(path_xyz[:, 0] * 1e3, path_xyz[:, 1] * 1e3, lw=1,
                     label=f"{label}: {verdict} at {t_end:.2f} s")
        ts = np.arange(len(path_xyz)) * 2e-4
        axes[1].semilogy(ts, np.hypot(path_xyz[:, 0], path_xyz[:, 1]) * 1e3, lw=1)
    axes[0].set_xlabel("x [mm]")
    axes[0].set_ylabel("y [mm]")
    axes[0].set_title("open loop from 10 um off axis")
    axes[0].legend(fontsize=7)
    axes[0].set_aspect("equal")
    axes[1].set_xlabel("t [s]")
    axes[1].set_ylabel("lateral offset [mm]")
    axes[1].grid(alpha=0.3, which="both")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def run(out_dir=OUT_DIR, reuse=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = out_dir / "stage_a.csv"
    if reuse and cache.exists():
        a = pd.read_csv(cache)
        print(f"stage A: reused {len(a)} points from {cache.name}")
    else:
        print("stage A: d x tilt x f, both field models ...")
        a = stage_a()
        a.to_csv(cache, index=False)

    loop = a[a["model"] == "loop"]
    feasible = loop[(loop["lock"] < LOCK_MAX) & (loop["clearance"] > CLEARANCE_MIN)]
    # Two conditions, and they are not the same condition. C_net < 0 says the static
    # stiffness restores; R > 1 says the spin axis can deliver it in time. Selecting on R
    # alone lands on points with no restoring stiffness at all.
    restoring = feasible[feasible["C_net"] < 0.0]
    print(f"  {len(loop)} loop points, {len(feasible)} pass lock and clearance, "
          f"{len(restoring)} of those also have C_net < 0")
    print(f"  stable (max Re root < 0): {(feasible['max_re_root'] < 0).sum()}")
    hi = feasible.loc[feasible["R"].idxmax()]
    print(f"  best R anywhere      : {hi['R']:.4f} at d={hi['d']*1e3:.0f} mm, "
          f"tilt={hi['tilt_deg']:.0f} deg, f={hi['f']:.0f} Hz, z_eq={hi['z_eq']*1e3:.1f} mm, "
          f"C_net={hi['C_net']:+.5f}")
    best = restoring.loc[restoring["R"].idxmax()].to_dict()
    print(f"  best R with C_net < 0: {best['R']:.4f} at d={best['d']*1e3:.0f} mm, "
          f"tilt={best['tilt_deg']:.0f} deg, f={best['f']:.0f} Hz, "
          f"z_eq={best['z_eq']*1e3:.1f} mm, C_net={best['C_net']:+.5f}")
    print(f"  C_net < 0 region     : d {restoring['d'].min()*1e3:.0f}-"
          f"{restoring['d'].max()*1e3:.0f} mm, f {restoring['f'].min():.0f}-"
          f"{restoring['f'].max():.0f} Hz, z_eq {restoring['z_eq'].min()*1e3:.1f}-"
          f"{restoring['z_eq'].max()*1e3:.1f} mm")
    clear_ok = loop[loop["clearance"] > CLEARANCE_MIN]
    print(f"  of {len(clear_ok)} clearance-ok points, lock rejects "
          f"{(clear_ok['lock'] >= LOCK_MAX).sum()} and C_net >= 0 rejects "
          f"{(clear_ok['C_net'] >= 0).sum()}")

    print("stage B: amplitude and pitch ...")
    b = stage_b(best)
    print("stage C: quadrature phase error ...")
    c = stage_c(best)
    print("stage D: the rotor ...")
    d = stage_d(best)
    for axis in ("mdip", "i_spin", "mass", "scale"):
        e = fit_exponent(d, axis)
        print(f"  R propto {axis}^{e:+.3f}" if e is not None else f"  {axis}: no points")
    k_needed = solve_scale(d)
    if k_needed:
        print(f"  R = 1 at uniform rotor scale k = {k_needed:.3f} "
              f"({1/k_needed:.1f}x smaller in linear dimension)")

    print("stage E: nonlinear trajectories ...")
    cands = (restoring.sort_values("R", ascending=False).head(2).to_dict("records")
             + restoring.sort_values("C_net").head(1).to_dict("records")
             + feasible.sort_values("R", ascending=False).head(1).to_dict("records")
             + [r for r in (evaluate(0.046, 15, 92, model="loop"),) if r])
    runs = []
    for row in cands:
        for model in ("dipole", "loop"):
            row = {**row, "model": model}
            verdict, t_end, path_xyz = trajectory(row)
            label = (f"{row['d']*1e3:.0f}mm/{row['tilt_deg']:.0f}deg/"
                     f"{row['f']:.0f}Hz/{model}")
            print(f"  {label:34s} {verdict} at t={t_end:.3f} s")
            runs.append((label, verdict, t_end, path_xyz))

    frames = pd.concat([a, b, c, d], ignore_index=True)
    frames.to_csv(out_dir / "sweep.csv", index=False)
    plot_stage_a(a, out_dir / "stability_map.png")
    top = (restoring.sort_values("R", ascending=False)
           .drop_duplicates(subset=["d", "tilt_deg"]).head(2))
    plot_thresholds(a, list(zip(top["d"], top["tilt_deg"])),
                    out_dir / "frequency_thresholds.png")
    plot_scaling(d, out_dir / "scaling.png")
    plot_trajectories(runs, out_dir / "trajectories.png")
    print(f"wrote {out_dir}/sweep.csv and four figures")
    return frames


def _self_check():
    # 1. The three candidate points from that sweep, under the dipole model it used.
    want = {
        (0.046, 15.0, 92.0): (18.387, -58.92, -0.03379, -0.03075),
        (0.047, 10.0, 94.0): (17.172, -47.20, -0.02825, -0.03101),
        (0.049, 14.0, 98.0): (17.817, -69.83, -0.04544, -0.00315),
    }
    for (d, t, f), (z, gn, ct, kz) in want.items():
        r = evaluate(d, t, f)
        assert abs(r["z_eq"] * 1e3 - z) < 2e-3, (d, r["z_eq"])
        assert abs(r["G_n"] - gn) < 0.05, (d, r["G_n"])
        assert abs(r["C_tilt"] - ct) < 1e-5, (d, r["C_tilt"])
        assert abs(r["k_z"] - kz) < 1e-5, (d, r["k_z"])
        assert r["C_net"] < 0.0, r["C_net"]
        print(f"candidate {d*1e3:.0f}mm/{t:.0f}deg/{f:.0f}Hz : "
              f"z_eq={r['z_eq']*1e3:.3f}mm C_net={r['C_net']:+.5f} lock={r['lock']:.3f} "
              f"R={r['R']:.4f} max Re={r['max_re_root']:+.2f}/s")

    # 2. The untilted baseline is section 12.6 exactly, and there the Laplace trace closes.
    base = evaluate(0.037, 0.0, 110.0)
    assert abs(base["C_grad"] - 0.05357) < 1e-4, base["C_grad"]
    assert abs(base["k_z"] + 0.10714) < 1e-4, base["k_z"]
    assert abs(base["trace_resid"]) < 1e-5, base["trace_resid"]
    assert abs(base["G_n"]) < 1e-3, base["G_n"]  # exactly zero by symmetry, to FD noise
    # With the moment frozen the residual has to close at the tilted points too, which is
    # what identifies the open trace there as moment re-orientation and not a bug.
    tilted = evaluate(0.046, 15.0, 92.0)
    assert tilted["trace_resid"] > 1e-3, tilted["trace_resid"]
    assert abs(tilted["trace_resid_fixed_m"]) < 0.02 * tilted["trace_resid"], tilted
    print(f"baseline    : C_grad={base['C_grad']:+.5f} k_z={base['k_z']:+.5f} "
          f"trace={base['trace_resid']:+.2e} | tilted trace={tilted['trace_resid']:+.5f}, "
          f"fixed-m {tilted['trace_resid_fixed_m']:+.2e}")

    # 3. Section 14's ceiling: alignment damping can only lower the pole, never raise it.
    p0 = tilted["kappa_t"] / complex(0.0, sm.robot_params().I_spin * 2 * math.pi * 92.0)
    for c_t in (1e-9, 1e-8, 1e-7, 1e-6):
        p = tilted["kappa_t"] / complex(c_t, sm.robot_params().I_spin * 2 * math.pi * 92.0)
        assert abs(p) <= abs(p0) + 1e-15, (c_t, abs(p), abs(p0))
    print(f"ceiling     : |p| <= Omega_align = {abs(p0):.4f} rad/s for every c_t >= 0")

    # 4. The reduced-order roots against the plant they claim to describe. This is the one
    # that matters: without it section 14 is algebra with no plant attached.
    for (d, t, f) in want:
        r = evaluate(d, t, f)
        verdict, t_end, _ = trajectory(r)
        assert verdict in ("escaped", "step-out"), (d, verdict)
        rate = math.log(1000.0) / t_end   # 10 um to 10 mm
        assert 0.5 < rate / r["max_re_root"] < 2.0, (d, rate, r["max_re_root"])
        print(f"roots       : {d*1e3:.0f}mm predicts {r['max_re_root']:+.2f}/s, plant gives "
              f"{rate:+.2f}/s ({verdict} at {t_end:.3f} s)")

    # 5. The loop backend moves the numbers without moving the verdict.
    for (d, t, f) in want:
        dip, loop = evaluate(d, t, f), evaluate(d, t, f, model="loop")
        assert loop["R"] < 1.0 and dip["R"] < 1.0, (d, dip["R"], loop["R"])
        print(f"loop model  : {d*1e3:.0f}mm  R {dip['R']:.4f} -> {loop['R']:.4f}, "
              f"C_net {dip['C_net']:+.5f} -> {loop['C_net']:+.5f}, "
              f"z_eq {dip['z_eq']*1e3:.2f} -> {loop['z_eq']*1e3:.2f} mm")

    print("self-check PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-check", action="store_true", help="assert the pinned results only")
    ap.add_argument("--out", default=str(OUT_DIR), help="output directory")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse a cached stage A, the only expensive stage")
    args = ap.parse_args()
    if args.self_check:
        _self_check()
    else:
        run(args.out, reuse=args.reuse)
