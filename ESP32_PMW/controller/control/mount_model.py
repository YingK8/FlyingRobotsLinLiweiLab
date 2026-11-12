#!/usr/bin/env python3
"""
The robot while it is still on the takeoff stand. theory.md section 26.

Every other model in this tree starts the robot in free flight, or holds it with a floor in z
alone (`z_track`). This one keeps the transverse degrees of freedom while the rod is in the
stand, because that is where the stand's uncertainties live: clearance, contact stiffness,
friction, insertion depth, and a rotor that is not concentric.

    state   r   = x + i y          COM lateral position, m
            chi = beta_x + i beta_y  tilt of the PRINCIPAL spin axis, rad
            and their rates.   The spin rate omega(t) is prescribed (synchronous, 12.3).

    body    e      COM offset from the rod's geometric axis              (body-fixed, 1x)
            gamma  principal axis tilted from the rod's geometric axis   (body-fixed, 1x)
    lab     alpha  tilt of the field's rotation axis                    (0x, 18.18)
    stand   l      COM to contact point, along the axis
            c, k, d, mu   radial clearance, penalty stiffness and damping, Coulomb friction

The contact point P is where the GEOMETRIC rod meets the support. The camera sees the
geometric axis n_geo = chi - gamma e^{i theta}, not chi. Torque on the tilt from a lateral force
F applied a distance l below the COM is -l F (26.3). The gyroscopic sign is chosen so that a
forward synchronous whirl is stiffened by (I_s - I_t) w^2: oblate rotors have no forward critical.

Simplifications that cut a real corner:
- the sin(delta) cross term of 11.2 is left out (its sign has not been checked on the rig);
- the vertical motion is not a state. Thrust only sets the seat load N = W (1 - (f/f_h)^2);
- there is one contact point, not a bearing stack. A stack is two such points, and it
  multiplies the unknowns without adding any data that could pin them down.

    uv run python controller/control/mount_model.py          # self-check, then the design table
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, replace

import numpy as np

from controller.control.robust_cert import (
    C_GRAV, I_SPIN, I_TRANS, KAPPA_T, M_ROBOT_KG, axis_loop, load_gains,
)

W_ROBOT = M_ROBOT_KG * C_GRAV


@dataclass(frozen=True)
class Mount:
    # Every default below is a GUESS unless a section is named. They exist to rank designs
    # against each other, not to predict a single run.
    e: float = 20e-6            # m. A printed body with two glued magnets; unmeasured
    gamma: float = math.radians(1.0)
    gamma_phase: float = 0.0    # rad, angle of gamma relative to e in the body
    alpha: complex = cmath.rect(math.radians(1.5), math.radians(151.0))  # 18.15/18.18 direction
    l: float = 5e-3             # m. The rod runs 7.4 mm past the rim (STL); depth unsurveyed
    c: float = 10e-6            # m radial. Slip fit of Ø1.0 rod in a 1 mm bearing ID; guess
    k: float = 2e3              # N/m. Rigid relative to m w^2 = 40 N/m at 110 Hz
    zeta_c: float = 0.2         # contact damping ratio
    mu: float = 0.2
    r_slip: float = 0.0         # m. 0 = rod turns with the bearing race; 0.5e-3 = sliding bore
    k_grad: float = 0.054       # N/m lateral anti-trap, 12.6 (cross array, z = 15 mm)
    c_t: float = 1e-10          # N m s, aero tilt damping. 11.3: small; unmeasured
    kappa: float = KAPPA_T      # 11.3 at 160 Hz; really scales with coil current

    @property
    def d(self) -> float:
        return 2.0 * self.zeta_c * math.sqrt(self.k * M_ROBOT_KG)


def l_crit() -> float:
    """Support distance below which no forward synchronous conical critical exists (26.3)."""
    return math.sqrt((I_SPIN - I_TRANS) / M_ROBOT_KG)


def seat_load(f_hz: float, f_hover: float) -> float:
    return max(0.0, W_ROBOT * (1.0 - (f_hz / f_hover) ** 2))


def spring_response(p: Mount, f_hz: float, f_hover: float = 101.0) -> tuple[complex, complex]:
    """Exact 1x amplitudes (R, X) for the LINEAR mount: c = 0, mu = 0, alpha = 0 (26.3).

    P = R - e - l(X - G), with G = gamma e^{i gamma_phase}
    -m w^2 R               = -k P                       + k_grad R
    ((Is-It) w^2 + kappa) X = l k P + N (e + l X - l G)   (no damping: the undamped bound)
    """

    w = 2.0 * math.pi * f_hz
    n = seat_load(f_hz, f_hover)
    g = cmath.rect(p.gamma, p.gamma_phase)
    a = np.array([[-M_ROBOT_KG * w * w + p.k - p.k_grad, -p.k * p.l],
                  [-p.l * p.k, (I_SPIN - I_TRANS) * w * w + p.kappa + p.k * p.l ** 2 - n * p.l]],
                 dtype=complex)
    b = np.array([p.k * (p.e - p.l * g), -p.l * p.k * (p.e - p.l * g) + n * (p.e - p.l * g)],
                 dtype=complex)
    r, x = np.linalg.solve(a, b)
    return complex(r), complex(x)


def rigid_pivot_cone(p: Mount, f_hz: float, f_hover: float = 101.0) -> float:
    """Geometric-axis 1x cone, rad, for k -> infinity (26.3). The denominator is the point."""

    w = 2.0 * math.pi * f_hz
    n = seat_load(f_hz, f_hover)
    g = cmath.rect(p.gamma, p.gamma_phase)
    den = (I_SPIN - I_TRANS - M_ROBOT_KG * p.l ** 2) * w * w + p.kappa - n * p.l
    x = (p.l * M_ROBOT_KG * w * w + n) * (p.e - p.l * g) / den
    return abs(x - g)


def critical_hz(l: float, kappa: float = KAPPA_T, f_hover: float = 101.0) -> float | None:
    """Forward synchronous conical critical of the rigid pivot, Hz; None below l_crit (26.3).

    Solves (I_s - I_t - m l^2) w^2 + kappa - N(f) l = 0 with the seat load N evaluated at the
    root, by fixed-point iteration (N moves by under 1 % per Hz here).
    """

    a = M_ROBOT_KG * l * l - (I_SPIN - I_TRANS)
    if a <= 0:
        return None
    f = 30.0
    for _ in range(50):
        kk = kappa - seat_load(f, f_hover) * l
        if kk <= 0:
            return 0.0
        f = math.sqrt(kk / a) / (2 * math.pi)
    return f


def simulate(p: Mount, f_of_t, t_end: float, f_hover: float = 101.0, dt: float = 2e-5,
             stop_at_release: bool = True, keep: int = 0, init=(0j, 0j, 0j, 0j)):
    """Integrate the seated robot. Returns a dict with the trajectory tail and the release state.

    Semi-implicit Euler, except the gyroscopic term i (I_s w / I_t) chi_dot, which is a pure
    rotation of chi_dot and is applied EXACTLY as one. Explicit Euler on it multiplies |chi_dot|
    by sqrt(1 + (I_s w dt / I_t)^2) every step, which blew a 1 N/m pivot up to 1e22 deg in 1.5 s:
    a numerical instability that looks exactly like a whirl. The contact's natural frequency
    sqrt(k/m) sets dt (omega_n dt ~ 0.1). Release is the first step at which N reaches zero.
    `fmax` is the peak contact force over the kept tail only, so the first impact does not set it.
    """

    m, it, is_ = M_ROBOT_KG, I_TRANS, I_SPIN
    g = cmath.rect(p.gamma, p.gamma_phase)
    r, rd, chi, chid = init
    th = 0.0
    n_steps = int(t_end / dt)
    tail = max(keep, 0)
    out_t, out_ng, out_th, out_f = [], [], [], []
    fmax = 0.0
    released = None
    for i in range(n_steps):
        t = i * dt
        f = f_of_t(t)
        w = 2.0 * math.pi * f
        th += w * dt
        u = cmath.exp(1j * th)
        n = seat_load(f, f_hover)
        if stop_at_release and n == 0.0:
            released = dict(t=t, f=f, r=r, v=rd, chi=chi, chid=chid)
            break
        # contact at the geometric rod, a distance l below the COM
        geo = chi - g * u
        pt = r - p.e * u - p.l * geo
        ptd = rd - p.e * 1j * w * u - p.l * (chid + g * 1j * w * u)
        fc = 0j
        rad = abs(pt)
        if rad > p.c:
            nh = pt / rad
            vn = (ptd * nh.conjugate()).real
            fn = max(0.0, p.k * (rad - p.c) + p.d * vn)
            th_hat = 1j * nh
            vs = (ptd * th_hat.conjugate()).real + w * p.r_slip
            fc = -fn * nh - p.mu * fn * math.tanh(vs / 1e-4) * th_hat
            if i >= n_steps - tail:
                fmax = max(fmax, abs(fc))
        rdd = (fc + p.k_grad * r) / m
        tau = -p.l * fc - n * (pt - r) + p.kappa * (p.alpha - chi) - p.c_t * chid
        rd += rdd * dt
        chid = chid * cmath.exp(1j * is_ * w / it * dt) + tau / it * dt
        r += rd * dt
        chi += chid * dt
        if tail and i >= n_steps - tail:
            out_t.append(t); out_ng.append(chi - g * u); out_th.append(th); out_f.append(f)
    return dict(released=released, fmax=fmax, t=np.array(out_t), n_geo=np.array(out_ng),
                theta=np.array(out_th), f=np.array(out_f))


def steady_cone(p: Mount, f_hz: float, f_hover: float = 101.0, t_hold: float = 1.5,
                dt: float = 2e-5, init=(0j, 0j, 0j, 0j)) -> tuple[float, float]:
    """(1x geometric cone in deg, peak contact force / W) after holding f_hz for t_hold.

    Started from rest, the tilt modes ring for as long as c_t lets them (seconds), so a short
    hold reads forced + free. That is what a real hold sees too; the 1x demodulation over the
    last 0.4 s rejects most of the free part, not all of it.
    """

    keep = int(0.4 / dt)
    s = simulate(p, lambda t: f_hz, t_hold, f_hover, dt, stop_at_release=False, keep=keep,
                 init=init)
    z = s["n_geo"] * np.exp(-1j * s["theta"])  # demodulate at 1x
    return math.degrees(abs(z.mean())), s["fmax"] / W_ROBOT


def release_envelope(mag_max: float = 1.0, gains=None) -> dict:
    """What the shipped lateral loop can absorb at the instant of release (26.6).

    v_cert: the largest release speed inside the biggest saturation-free invariant ellipsoid of
            the discrete closed loop (a CERTIFICATE: the trajectory never saturates).
    v_sim:  the largest release speed from which the saturated loop still converges (exact
            for this linear model, not a certificate).
    beta:   the steady thrust tilt the integrator can hold: g sin(beta) = g k_lat mag_max.
    k_lat is a seed, not a measurement (19.14), so every number here scales with it.
    """

    gains = gains or load_gains()
    aa, bcol, k, _ = axis_loop(gains, "lateral")  # bcol is already at the nominal k_lat
    mcl = aa - np.outer(bcol, k)
    from scipy.linalg import solve_discrete_lyapunov
    pmat = solve_discrete_lyapunov(mcl.T, np.eye(3))
    rho = mag_max ** 2 / float(k @ np.linalg.solve(pmat, k))
    e_v = np.array([0.0, 1.0, 0.0])
    v_cert = math.sqrt(rho / float(e_v @ pmat @ e_v))

    def converges(v):
        x = np.array([0.0, v, 0.0])
        for _ in range(20000):
            u = float(np.clip(-k @ x, -mag_max, mag_max))
            x = aa @ x + bcol * u
            if not np.all(np.isfinite(x)) or abs(x[0]) > 1.0:
                return False
        return abs(x[0]) < 1e-4

    lo, hi = 0.0, 5.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if converges(mid) else (lo, mid)
    k_lat = gains["meta"]["args"]["k_lat"]
    return dict(v_cert=v_cert, v_sim=lo, beta_deg=math.degrees(math.asin(min(1.0, k_lat * mag_max))),
                k_lat=k_lat, mag_max=mag_max, _mcl=mcl, _p=pmat, _rho=rho, _k=k, _aa=aa, _b=bcol)


def dc_coil_mean_force(m_dip: float = 3.6257e-3, z: float = 3e-3, r_off: float = 0.3e-3,
                       m_coil: float = 0.05, n: int = 720) -> tuple[float, float]:
    """(|<F>| / W, peak |m x B_dc| / tau_max) for a DC dipole coil under a horizontal spinning m.

    26.5: <m> = 0 over a revolution, so <grad(m.B)> = grad(<m>.B) = 0 to first order, wherever
    the coil is. What is left is a 1x rotating torque.
    """

    mu0 = 4e-7 * math.pi
    src = np.array([0.0, 0.0, -z])
    mc = np.array([0.0, 0.0, m_coil])

    def bfield(pos):
        d = pos - src
        rr = np.linalg.norm(d)
        return mu0 / (4 * math.pi) * (3 * d * (mc @ d) / rr ** 5 - mc / rr ** 3)

    pos0 = np.array([r_off, 0.0, 0.0])
    h = 1e-7
    fsum = np.zeros(3)
    tmax = 0.0
    for th in np.linspace(0, 2 * math.pi, n, endpoint=False):
        m = m_dip * np.array([math.cos(th), math.sin(th), 0.0])
        grad = np.array([(m @ bfield(pos0 + h * e) - m @ bfield(pos0 - h * e)) / (2 * h)
                         for e in np.eye(3)])
        fsum += grad
        tmax = max(tmax, float(np.linalg.norm(np.cross(m, bfield(pos0)))))
    return float(np.linalg.norm(fsum / n)) / W_ROBOT, tmax / (2 * KAPPA_T)


DESIGNS = {
    "current stand (l=5 mm, rigid, c=10 um)": Mount(),
    "same, rod 1 mm shallower (l=4.3 mm)": Mount(l=4.3e-3),
    "support near COM (l=2 mm)": Mount(l=2e-3),
    "compliant pivot (l=2 mm, k=1 N/m, c=0)": Mount(l=2e-3, k=1.0, c=0.0, zeta_c=0.3),
}


def design_table(freqs=(20, 30, 40, 50, 60, 70, 80, 90)) -> list[tuple]:
    rows = []
    for name, p in DESIGNS.items():
        for f in freqs:
            cone, fw = steady_cone(p, f, dt=2e-5 if p.k > 10 else 1e-4)
            rows.append((name, f, cone, fw))
    return rows


def _self_check() -> None:
    lc = l_crit()
    assert abs(lc - 3.975e-3) < 2e-6, lc
    assert critical_hz(0.99 * lc) is None and critical_hz(1.3 * lc) > 0

    # 1. free body: no support => the COM never moves, whatever the imbalance.
    free = Mount(k=0.0, k_grad=0.0, alpha=0j, kappa=0.0)
    s = simulate(free, lambda t: 80.0, 0.3, f_hover=1.0, stop_at_release=False, keep=15000)
    assert s["fmax"] == 0.0

    # 1b. torque-free, undamped spinning body started tilting: |chi_dot| is conserved (the
    # gyroscopic rotation is applied exactly, so the integrator adds no energy).
    s = simulate(replace(free, gamma=0.0, c_t=0.0), lambda t: 80.0, 0.3, f_hover=1.0,
                 stop_at_release=False, keep=2, init=(0j, 0j, 0j, 1.0 + 0j))
    ng = s["n_geo"]
    assert abs(abs(ng[-1] - ng[-2]) / 2e-5 - 1.0) < 1e-6

    # 2. linear mount (c = 0, mu = 0, alpha = 0): started ON the closed-form steady state, the
    # simulation stays there. The tilt modes are undamped here, so a start from rest would ring
    # for ever and test the demodulator instead of the equations.
    lin = Mount(c=0.0, mu=0.0, alpha=0j, c_t=0.0, zeta_c=0.0, k=400.0, k_grad=0.0)
    for f in (12.0, 30.0, 80.0):
        rr, x = spring_response(lin, f)
        w = 2 * math.pi * f
        want = math.degrees(abs(x - cmath.rect(lin.gamma, lin.gamma_phase)))
        got, _ = steady_cone(lin, f, t_hold=0.6, init=(rr, 1j * w * rr, x, 1j * w * x))
        assert abs(got - want) / want < 0.02, (f, got, want)

    # 3. the 4 mm rule: below l_crit the rigid-pivot denominator never crosses zero; above it does.
    ws = np.linspace(5, 200, 400)
    for l, crosses in ((0.8 * lc, False), (1.3 * lc, True)):
        p = Mount(l=l)
        den = [(I_SPIN - I_TRANS - M_ROBOT_KG * l * l) * (2 * math.pi * f) ** 2 + p.kappa
               - seat_load(f, 101.0) * l for f in ws]
        assert (min(den) < 0 < max(den)) == crosses, (l, min(den), max(den))

    # 4. a stiff spring tends to the rigid pivot.
    stiff = Mount(c=0.0, mu=0.0, alpha=0j, k=1e8, l=3e-3, k_grad=0.0)
    _, x = spring_response(stiff, 50.0)
    want = rigid_pivot_cone(stiff, 50.0)
    assert abs(abs(x - cmath.rect(stiff.gamma, 0)) - want) / want < 1e-3

    # 5. a DC coil under a horizontal spinning magnet: no mean force, a large 1x torque.
    fw, tq = dc_coil_mean_force()
    assert fw < 1e-6 and tq > 0.1, (fw, tq)

    # 6. release envelope: the ellipsoid really is invariant and saturation-free.
    env = release_envelope(1.0)
    x = np.array([0.0, 0.999 * env["v_cert"], 0.0])
    for _ in range(5000):
        u = float(-env["_k"] @ x)
        assert abs(u) <= 1.0 + 1e-9
        x = env["_mcl"] @ x
    assert env["v_cert"] <= env["v_sim"] + 1e-9
    print(f"self-check ok: l_crit = {lc * 1e3:.2f} mm, DC coil <F>/W = {fw:.1e}, "
          f"1x torque = {tq:.2f} tau_max")


def demo() -> None:
    env = release_envelope(1.0)
    print(f"\nrelease envelope at mag_max = 1, k_lat = {env['k_lat']} (a seed, 19.14):")
    print(f"  release speed: certified {env['v_cert'] * 1e3:.1f} mm/s, saturated-loop "
          f"{env['v_sim'] * 1e3:.1f} mm/s;  steady tilt held: {env['beta_deg']:.2f} deg")
    print("\nrigid-pivot critical speed vs support distance l (kappa = 11.3's 2.45e-5):")
    print("  " + "  ".join(f"{l:.1f}mm:{'none' if critical_hz(l * 1e-3) is None else f'{critical_hz(l * 1e-3):.0f}Hz'}"
                           for l in (3.9, 4.0, 4.1, 4.2, 4.3, 4.5, 5.0, 6.0, 7.0)))
    print("\nrigid-pivot 1x geometric cone (deg) vs l, e = 20 um, gamma = 1 deg:")
    for l in (2e-3, 3.5e-3, 4.3e-3, 5e-3, 7e-3):
        p = Mount(l=l)
        print(f"  l = {l * 1e3:.1f} mm: " + "  ".join(
            f"{f}:{math.degrees(rigid_pivot_cone(p, f)):.2f}" for f in (20, 40, 50, 60, 80, 100)))
    print("\ndesign table (simulated, 1.5 s hold): f, 1x cone deg, peak contact force / W")
    for name, f, cone, fw in design_table():
        print(f"  {name:42s} {f:4.0f} Hz  {cone:6.2f}  {fw:6.2f}")


def _read_csv(path: str) -> dict[str, np.ndarray]:
    import csv
    with open(path) as fh:
        rows = list(csv.DictReader(l for l in fh if not l.startswith("#")))
    keys = rows[0].keys() if rows else ()

    def col(k):
        return np.array([float(r[k]) if r[k] not in (None, "") else np.nan for r in rows])
    return {k: col(k) for k in keys if k in ("t", "f_hz", "x_mm", "y_mm", "z_mm", "freq_hz",
                                               "cone_pre_deg", "radial0_deg")}


def release_kick(csv_dir: str, heights_mm=(1, 2, 3, 4), half_window_s: float = 0.05) -> list[dict]:
    """T1 (26.2): lateral speed as the robot climbs out of the stand.

    For the takeoff CSVs of 2026-08-29..09-01 (`git show bad8b64^:ESP32_PMW/results/takeoff/`).
    The rod stays guided until it has risen its whole insertion depth, which has never been
    surveyed, so there is no single "release" sample. Speed is reported where the height above
    the pad median (2-30 Hz samples) first holds above each of `heights_mm` for 3 samples at
    f > 40 Hz. A local linear fit over +-half_window_s gives (v_lat, v_z) at that height.
    Pose flips -- a jump of more than 2 mm in z between samples, which puts the robot below
    the pad -- are dropped, and a crossing within 10 samples of one is not used.
    """

    import glob
    import os
    out = []
    for path in sorted(glob.glob(os.path.join(csv_dir, "*.csv"))):
        d = _read_csv(path)
        if "f_hz" not in d:
            continue
        ok = np.isfinite(d["f_hz"]) & np.isfinite(d["z_mm"])
        t, f, x, y, z = (d[k][ok] for k in ("t", "f_hz", "x_mm", "y_mm", "z_mm"))
        pad = (f > 2) & (f < 30)
        if len(t) < 200 or pad.sum() < 30:
            continue
        h = z - np.median(z[pad])
        flip = np.r_[False, np.abs(np.diff(h)) > 2.0]
        rec = dict(run=os.path.basename(path)[:15], flips=int(flip.sum()), at={})
        for hh in heights_mm:
            i = next((k for k in range(len(h) - 3) if f[k] > 40 and (h[k:k + 3] > hh).all()
                      and not flip[max(0, k - 10):k + 10].any()), None)
            if i is None:
                continue
            w = (np.abs(t - t[i]) <= half_window_s) & ~flip
            if w.sum() < 5:
                continue
            a = np.c_[np.ones(w.sum()), t[w] - t[i]]
            vx, vy, vz = (np.linalg.lstsq(a, q[w], rcond=None)[0][1] for q in (x, y, h))
            rec["at"][hh] = (float(f[i]), float(np.hypot(vx, vy)), float(vz))
        if rec["at"]:
            out.append(rec)
    return out


def cone_resonance_fit(settling_csv: str, radial_band=(10.0, 25.0)) -> dict:
    """T1b (26.3): does a LINEAR forced 1x resonance describe the pre-cut cone vs drive?

    Takes in the normal tilt state only (`radial0_deg` in `radial_band`, design A's 24.6 band).
    Fits, in log amplitude, gamma r^2 / |1 - r^2 + 2 i zeta r| (+) a (20/f)^2 against the
    no-resonance alternative floor (+) a (20/f)^2. Returns both rms log residuals.
    """

    from scipy.optimize import least_squares
    d = _read_csv(settling_csv)
    m = (np.isfinite(d["cone_pre_deg"]) & (d["radial0_deg"] > radial_band[0])
         & (d["radial0_deg"] < radial_band[1]) & (d["freq_hz"] >= 20))
    f, c = d["freq_hz"][m], d["cone_pre_deg"][m]

    def model(q, f):
        g, fc, z, a = q
        r = f / fc
        return np.hypot(g * r * r / np.abs(1 - r * r + 2j * z * r), a * (20 / f) ** 2)

    best = min((least_squares(lambda q: np.log(model(q, f)) - np.log(c), [1, fc0, z0, 1],
                              bounds=([0, 20, 0.005, 0], [10, 150, 2, 10]))
                for fc0 in (40, 45, 50, 55, 60) for z0 in (0.02, 0.05, 0.1, 0.2, 0.5)),
               key=lambda s: s.cost)
    null = least_squares(lambda q: np.log(np.hypot(q[0], q[1] * (20 / f) ** 2)) - np.log(c),
                         [1, 1], bounds=([0, 0], [10, 10]))
    med = {float(F): float(np.median(c[f == F])) for F in np.unique(f)}
    return dict(n=int(m.sum()), gamma_deg=best.x[0], fc_hz=best.x[1], zeta=best.x[2],
                rms_log=float(np.sqrt(np.mean(best.fun ** 2))),
                rms_log_null=float(np.sqrt(np.mean(null.fun ** 2))),
                fit={F: float(model(best.x, np.array([F]))[0]) for F in med}, median=med)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--kick", metavar="DIR", help="T1 on a folder of takeoff CSVs")
    ap.add_argument("--cone", metavar="CSV", help="T1b on an alignment-rate settling.csv")
    args = ap.parse_args()
    _self_check()
    if args.kick:
        print("\nT1 climb-out: run, flips, then per height h: f Hz / v_lat / v_z (mm/s)")
        for r in release_kick(args.kick):
            print(f"  {r['run']} {r['flips']:3d}  " + "  ".join(
                f"{h}mm:{a[0]:.0f}/{a[1]:.0f}/{a[2]:.0f}" for h, a in r["at"].items()))
    if args.cone:
        r = cone_resonance_fit(args.cone)
        print(f"\nT1b n={r['n']}: gamma {r['gamma_deg']:.2f} deg, fc {r['fc_hz']:.1f} Hz, "
              f"zeta {r['zeta']:.3f}; rms log resid {r['rms_log']:.2f} vs {r['rms_log_null']:.2f} "
              f"with no resonance")
        print("  f: median / fit  " + "  ".join(f"{F:.0f}:{r['median'][F]:.2f}/{r['fit'][F]:.2f}"
                                               for F in r["median"]))
    if not (args.kick or args.cone):
        demo()
