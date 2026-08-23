#!/usr/bin/env python3
"""
Multi-coil spatial plant: 3-D flight over a 16-coil rotating field. See theory.md section 12.

    state  x = [r(3) m, v(3) m/s, s(3) unit]        8 DOF, |s| = 1
    input  u = [Re I(4), Im I(4), f_drive Hz]       4 channel phasors + drive

Per step: decompose the local field into first harmonics B(th) = b_cos cos th + b_sin sin th,
solve the synchronous phase lag from the drag balance, cycle-average the magnetic torque and
slew the spin axis, then cycle-average the gradient force and integrate translation.

The configuration manifold R^3 x S^2 is exactly what pose/estimator.py measures. Roll is absent
from both: cycle-averaging removes the spin phase, and the vision estimator cannot see it.

Two properties of this plant drive everything downstream:

    step-out     The phase lag has no solution when the local field is too weak, which bounds
                 the workspace. `phase_lag` raises rather than clipping.
    divergence   Open loop, the robot slides away and steps out within a second from 2 mm off
                 axis: lift points along s, s follows the local field axis, and that axis tilts
                 ~1.4 deg per mm off centre. The plant needs a controller, hence spatial_mpc.py.

The gradient force is not a perturbation despite the GUI defaulting it off: it reaches 44% of
weight at z = 20 mm, trapping vertically and expelling laterally, as Earnshaw requires.

Ported from matlab/MultiCoilBeamformingGUI_quickGeom_rigidTilt_coil22mm.m; theory.md Appendix A
lists the departures. Physical constants come from CAD, theory.md section 3.

Self-check: uv run python controller/control/spatial_model.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MU0 = 4.0e-7 * math.pi  # H/m
# The beamforming GUI's value (line 1099). The 1-D GUIs and hover_model.py use
# 9.80665; the difference is 3.4e-4 relative and reaches nothing here.
GRAVITY = 9.81

_ZHAT = np.array([0.0, 0.0, 1.0])


def _cross3(a, b):
    """Cross product of two 3-vectors."""

    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    )


def _unit(a, eps=1e-12):
    """Normalize a 3-vector, guarding the zero case."""

    return a / max(math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]), eps)


def _norm3(a):
    """Euclidean norm of a 3-vector, without the `np.linalg.norm` dispatch."""

    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


# --------------------------------------------------------------------------
# Robot
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Robot:
    """Rigid-body and magnetic properties, all SI."""

    I_spin: float        # polar moment about the spin axis (body x), kg m^2
    I_trans: float       # transverse moment about a diameter, kg m^2
    I_tensor: np.ndarray # (3,3) about the total COM, body frame, kg m^2
    mdip: float          # total dipole moment of the two magnets, A m^2
    mass: float          # kg
    com: np.ndarray      # (3,) COM in body CAD coordinates, m

    @property
    def weight(self) -> float:
        """Weight in newtons, the scale every force here is judged against."""
        return self.mass * GRAVITY


def robot_params() -> Robot:
    """CAD body plus two N52 magnets, by the parallel-axis theorem."""

    m_body_g = 0.06
    com_body_mm = np.array([2.82, 0.0, 0.0])
    I_body_gmm2 = np.diag([3.35, 1.86, 1.86])

    m_mag_g = 11.7832e-3
    r_mag_mm = h_mag_mm = 0.79375
    mag_pos_mm = np.array([[-0.345, 0.0, 0.35], [-0.345, 0.0, -0.35]])

    n_mag = len(mag_pos_mm)
    m_total_g = m_body_g + n_mag * m_mag_g
    com_mm = (m_body_g * com_body_mm + m_mag_g * mag_pos_mm.sum(0)) / m_total_g

    # Cylinder about its own COM, axis along z.
    I_radial = (1.0 / 12.0) * m_mag_g * (3.0 * r_mag_mm**2 + h_mag_mm**2)
    I_axial = 0.5 * m_mag_g * r_mag_mm**2
    I_mag_gmm2 = np.diag([I_radial, I_radial, I_axial])

    def shift(I, m, d):
        return I + m * ((d @ d) * np.eye(3) - np.outer(d, d))

    I_gmm2 = shift(I_body_gmm2, m_body_g, com_body_mm - com_mm)
    for p in mag_pos_mm:
        I_gmm2 = I_gmm2 + shift(I_mag_gmm2, m_mag_g, p - com_mm)

    I_si = I_gmm2 * 1e-9  # 1 g mm^2 = 1e-9 kg m^2

    # N52 remanence 1.45 T, magnetization M ~= Br/mu0, so mdip = (Br/mu0) V.
    v_mag = math.pi * (r_mag_mm * 1e-3) ** 2 * (h_mag_mm * 1e-3)
    mdip = n_mag * (1.45 / MU0) * v_mag

    return Robot(
        I_spin=float(I_si[0, 0]),
        I_trans=float(I_si[1, 1]),
        I_tensor=I_si,
        mdip=mdip,
        mass=m_total_g * 1e-3,
        com=com_mm * 1e-3,
    )


# --------------------------------------------------------------------------
# Coil array
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Coils:
    """A coil array as parallel arrays over M coils. Positions in m, angles in rad."""

    pos: np.ndarray      # (M,3) centre, m
    nhat: np.ndarray     # (M,3) unit axis
    moment: np.ndarray   # (M,3) dipole moment per ampere, N pi R^2 nhat, A m^2/A
    amp: np.ndarray      # (M,) drive current amplitude, A
    phase: np.ndarray    # (M,) drive phase, rad
    channel: np.ndarray  # (M,) which of the 4 drive channels this coil belongs to

    @property
    def n_channels(self) -> int:
        return int(self.channel.max()) + 1

    def phasor_weights(self, phasor=None):
        """Per-coil (w_u, w_v) for B(th) = u cos th + v sin th."""

        if phasor is None:
            z = self.amp * np.exp(1j * self.phase)
        else:
            z = np.asarray(phasor, dtype=complex)[self.channel]
        return z.real, -z.imag


def quick_coils(mode="cross", d=0.037, tilt_deg=0.0, amp=1.25, pitch=0.021):
    """One of the GUI's two 4-channel presets, 16 coils as four 2x2 groups."""

    coil_r, turns = 0.0105, 650
    a = math.radians(tilt_deg)

    if mode == "cross":
        centres = np.array([[d, 0.0], [0.0, d], [-d, 0.0], [0.0, -d]])
        local = "radial"
    elif mode == "corners":
        centres = np.array([[d, d], [-d, d], [-d, -d], [d, -d]])
        local = "global"
    else:
        raise ValueError(f"unknown geometry mode: {mode!r}")

    # 0/270/180/90 degrees around the ring is what makes the field rotate.
    phases = np.radians([0.0, 270.0, 180.0, 90.0])
    signs = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])

    pos, nhat, channel = [], [], []
    for g, cxy in enumerate(centres):
        er = np.array([cxy[0], cxy[1], 0.0])
        er /= max(np.linalg.norm(er), 1e-12)
        et = np.array([-er[1], er[0], 0.0])

        # Rodrigues about the tangential axis: n = sin(a) er + cos(a) ez.
        K = np.array(
            [[0.0, -et[2], et[1]], [et[2], 0.0, -et[0]], [-et[1], et[0], 0.0]]
        )
        rot = math.cos(a) * np.eye(3) + (1 - math.cos(a)) * np.outer(et, et)
        rot = rot + math.sin(a) * K

        n_g = rot @ np.array([0.0, 0.0, 1.0])
        e_a, e_b = (er, et) if local == "radial" else (np.eye(3)[0], np.eye(3)[1])
        e_a, e_b = rot @ e_a, rot @ e_b
        centre3 = np.array([cxy[0], cxy[1], 0.0])

        for sa, sb in signs:
            pos.append(centre3 + 0.5 * pitch * (sa * e_a + sb * e_b))
            nhat.append(n_g)
            channel.append(g)

    nhat = np.array(nhat)
    return Coils(
        pos=np.array(pos),
        nhat=nhat,
        moment=turns * math.pi * coil_r**2 * nhat,
        amp=np.full(len(pos), float(amp)),
        phase=phases[np.array(channel)],
        channel=np.array(channel),
    )


# --------------------------------------------------------------------------
# Field
# --------------------------------------------------------------------------


def coil_basis(r, coils, want_grad=True):
    """Per-coil field and field gradient at ``r``, per ampere."""

    d = np.asarray(r, dtype=float)[None, :] - coils.pos  # (M,3)
    m = coils.moment  # (M,3)
    dist = np.maximum(np.linalg.norm(d, axis=1), 1e-9)  # (M,)
    md = np.einsum("ci,ci->c", m, d)  # (M,)
    k = MU0 / (4.0 * math.pi)

    d5 = dist**5
    b = k * (3.0 * md[:, None] * d / d5[:, None] - m / dist[:, None] ** 3)

    if not want_grad:
        return b, None

    eye = np.eye(3)[None, :, :]
    grad_b = (3.0 * k / d5)[:, None, None] * (
        md[:, None, None] * eye
        + np.einsum("ci,cj->cij", d, m)
        + np.einsum("ci,cj->cij", m, d)
        - 5.0 * (md / dist**2)[:, None, None] * np.einsum("ci,cj->cij", d, d)
    )
    return b, grad_b


def field_at(r, coils, phasor=None, want_grad=True):
    """First-harmonic field and its spatial gradients at ``r``."""

    b, grad_b = coil_basis(r, coils, want_grad)
    w_u, w_v = coils.phasor_weights(phasor)
    u, v = w_u @ b, w_v @ b
    if not want_grad:
        return u, v, None, None
    return u, v, np.einsum("c,cij->ij", w_u, grad_b), np.einsum("c,cij->ij", w_v, grad_b)


def rotation_axis(u, v):
    """Directed rotation axis of the local rotating field, normalize(u x v)."""

    n = _cross3(u, v)
    norm = _norm3(n)
    if not np.isfinite(norm) or norm < 1e-15:
        return np.full(3, np.nan)
    return n / norm


def ellipse_semiaxes(u, v):
    """Exact semiaxes a >= b of the field polarization ellipse, in tesla."""

    uu, uv, vv = u @ u, u @ v, v @ v
    tr = uu + vv
    det = max(uu * vv - uv * uv, 0.0)
    disc = math.sqrt(max(tr * tr - 4.0 * det, 0.0))
    return math.sqrt(max(0.5 * (tr + disc), 0.0)), math.sqrt(max(0.5 * (tr - disc), 0.0))


# --------------------------------------------------------------------------
# Synchronous phase lag
# --------------------------------------------------------------------------


class StepOut(RuntimeError):
    """The local field cannot sustain synchronous lock at this drive frequency."""

    def __init__(self, msg, ratio=math.nan):
        super().__init__(msg)
        self.ratio = ratio


def phase_lag(u, v, mdip, tau_drag):
    """Synchronous phase lag from the steady spin-drag balance."""

    if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
        raise StepOut("non-finite local field harmonics")
    if not math.isfinite(tau_drag) or tau_drag < 0:
        raise StepOut("invalid rotational drag torque")

    a, b = ellipse_semiaxes(u, v)
    den = mdip * (a + b)  # N m
    if not math.isfinite(den) or den <= 1e-18:
        raise StepOut("local field too weak to define phase locking")

    ratio = 2.0 * tau_drag / den
    if not math.isfinite(ratio):
        raise StepOut("required sin(phi) is non-finite", ratio)
    if ratio > 1.0:
        raise StepOut("required sin(phi) > 1: synchronous locking impossible", ratio)
    if ratio < 0.0:
        raise StepOut("required sin(phi) < 0: invalid drag balance", ratio)

    return math.asin(ratio), a, b, ratio


def lock_margin(u, v, mdip, tau_drag):
    """Required sin(phi) as a plain number, without raising."""

    a, b = ellipse_semiaxes(u, v)
    den = mdip * (a + b)
    return math.inf if den <= 1e-18 else 2.0 * tau_drag / den


# --------------------------------------------------------------------------
# Cycle-averaged torque and force
# --------------------------------------------------------------------------


def _phase_basis(s, u, v, n_ref):
    """Geometric phase reference shared by the torque and force models."""

    n_raw = _cross3(u, v)
    nn = _norm3(n_raw)
    if not np.isfinite(nn) or nn < 1e-15:
        return None, None, math.nan, True
    n = n_raw / nn

    # Align only to the supplied physical axis, never to +z or to s.
    if n_ref is not None:
        n_ref = np.asarray(n_ref, dtype=float)
        if np.all(np.isfinite(n_ref)) and _norm3(n_ref) > 1e-12:
            if n @ (n_ref / _norm3(n_ref)) < 0:
                n = -n
        else:
            return None, None, math.nan, True

    e_i = _cross3(n, s)
    sin_alpha = _norm3(e_i)
    if not np.isfinite(sin_alpha) or sin_alpha < 1e-10:
        return None, None, math.nan, True
    e_i = e_i / sin_alpha

    e_r = _cross3(s, e_i)
    e_r = e_r / max(_norm3(e_r), 1e-12)
    e_f = _cross3(n, e_i)
    e_f = e_f / max(_norm3(e_f), 1e-12)

    # Solve dot(B(lam), eF) = 0, then pick the branch with B(lam) along +eI.
    u_f, v_f = u @ e_f, v @ e_f
    if math.hypot(u_f, v_f) < 1e-15:
        return None, None, math.nan, True
    lam = math.atan2(-u_f, v_f)
    if (u * math.cos(lam) + v * math.sin(lam)) @ e_i < 0:
        lam += math.pi

    return e_i, e_r, lam, False


def moment_harmonics(s, u, v, n_ref, mdip, phi):
    """Harmonic coefficients of the rotating magnetic moment at one robot state."""

    s = np.asarray(s, dtype=float)
    s = s / max(_norm3(s), 1e-12)
    e_i, e_r, lam, aligned = _phase_basis(s, u, v, n_ref)

    if aligned:
        # Major axis of the field ellipse: the first right singular vector of
        # [u v] gives [cos(lam), sin(lam)].
        _, sv, vt = np.linalg.svd(np.column_stack([u, v]))
        if sv[0] < 1e-15:
            raise StepOut("field ellipse too small to define a phase reference")
        q = vt[0]
        lam = math.atan2(q[1], q[0])
        b_ref = u * math.cos(lam) + v * math.sin(lam)
        if _norm3(b_ref) < 1e-15:
            raise StepOut("invalid major-axis phase reference")
        e_i = b_ref / _norm3(b_ref)
        e_r = _cross3(s, e_i)
        if _norm3(e_r) < 1e-12:
            raise StepOut("aligned phase basis is degenerate")
        e_r = e_r / _norm3(e_r)

    delta = lam + phi
    m_cos = mdip * (math.cos(delta) * e_i - math.sin(delta) * e_r)
    m_sin = mdip * (math.sin(delta) * e_i + math.cos(delta) * e_r)
    return m_cos, m_sin


def cycle_average(s, u, v, n_ref, mdip, phi):
    """Transverse torque and moment harmonics from one shared phase basis."""

    s = np.asarray(s, dtype=float)
    s = s / max(_norm3(s), 1e-12)
    e_i, e_r, lam, aligned = _phase_basis(s, u, v, n_ref)

    if aligned:
        # Torque genuinely vanishes here; the moment still needs a reference, so
        # fall back to the ellipse major axis. See `moment_harmonics`.
        m_cos, m_sin = moment_harmonics(s, u, v, n_ref, mdip, phi)
        return np.zeros(3), m_cos, m_sin

    delta = lam + phi
    m_cos = mdip * (math.cos(delta) * e_i - math.sin(delta) * e_r)
    m_sin = mdip * (math.sin(delta) * e_i + math.cos(delta) * e_r)
    tau = 0.5 * (_cross3(m_cos, u) + _cross3(m_sin, v))
    return tau - (tau @ s) * s, m_cos, m_sin


def torque_avg(s, u, v, n_ref, mdip, phi):
    """Cycle-averaged magnetic torque, with the axial component removed."""

    return cycle_average(s, u, v, n_ref, mdip, phi)[0]


def gradient_force(m_cos, m_sin, du, dv):
    """Cycle-averaged magnetic gradient force, F = grad(m . B), in newtons."""

    return 0.5 * (du.T @ m_cos + dv.T @ m_sin)


def update_spin_axis(s, tau_avg, i_spin, w_spin, dt, n_rot=None, align_tau=0.0):
    """Slew the spin axis under transverse torque, first order in the fast spin."""

    s = np.asarray(s, dtype=float)
    s = s / max(_norm3(s), 1e-12)
    tau_perp = tau_avg - (tau_avg @ s) * s
    lnew = i_spin * w_spin * s + tau_perp * dt
    s_new = lnew / max(_norm3(lnew), 1e-12)

    if align_tau > 0.0 and n_rot is not None and np.isfinite(n_rot[0]):
        pull = n_rot - (n_rot @ s_new) * s_new
        s_new = s_new + (dt / align_tau) * pull
        s_new = s_new / max(_norm3(s_new), 1e-12)
    return s_new


def recommended_dt(tau_perp, i_spin, w_spin, f_field, max_turn=0.01, eta=0.05):
    """Largest integration step that keeps the explicit spin-axis update accurate."""

    l0 = max(i_spin * w_spin, 1e-18)
    return min(max_turn * l0 / max(tau_perp, 1e-18), eta / max(f_field, 1e-12))


# --------------------------------------------------------------------------
# Plant
# --------------------------------------------------------------------------


@dataclass
class Plant:
    """The 8-state plant: coil array, robot, and the aerodynamic coefficients."""

    coils: Coils
    robot: Robot = None
    f_hover: float = 110.0    # spin frequency at which lift equals weight, Hz
    k_drag: float = 3.0e-10   # rotational drag, tau = k_drag f^2, N m/Hz^2
    beta: float = 0.20        # translational damping, 1/s (0 disables)
    use_grad: bool = False    # include the magnetic gradient force in translation
    align_tau: float = 0.0  # spin-axis alignment time constant, s (0 = MATLAB)

    def __post_init__(self):
        if self.robot is None:
            self.robot = robot_params()

    def accel(self, r, v, s, f_drive, phasor=None):
        """Net acceleration and the diagnostics of the step, at one state."""

        u, v_h, du, dv = field_at(r, self.coils, phasor, self.use_grad)
        n_rot = rotation_axis(u, v_h)
        tau_drag = self.k_drag * f_drive**2
        phi, a_ax, b_ax, ratio = phase_lag(u, v_h, self.robot.mdip, tau_drag)

        tau, m_cos, m_sin = cycle_average(s, u, v_h, n_rot, self.robot.mdip, phi)
        f_grad = (
            gradient_force(m_cos, m_sin, du, dv) if self.use_grad else np.zeros(3)
        )

        lift = (f_drive / self.f_hover) ** 2 * GRAVITY
        a = lift * s - GRAVITY * _ZHAT + f_grad / self.robot.mass
        if self.beta > 0:
            a = a - self.beta * v

        info = {
            "u": u,
            "v": v_h,
            "n_rot": n_rot,
            "phi": phi,
            "ratio": ratio,
            "semiaxes": (a_ax, b_ax),
            "tau": tau,
            "f_grad": f_grad,
        }
        return a, info

    def step(self, x, f_drive, dt, phasor=None):
        """Advance one step. ``x`` is (9,) = [r(3), v(3), s(3)]; returns (x, info)."""

        r, v, s = x[0:3], x[3:6], x[6:9]
        u, v_h, du, dv = field_at(r, self.coils, phasor, self.use_grad)
        n_rot = rotation_axis(u, v_h)
        tau_drag = self.k_drag * f_drive**2
        phi, a_ax, b_ax, ratio = phase_lag(u, v_h, self.robot.mdip, tau_drag)

        tau = torque_avg(s, u, v_h, n_rot, self.robot.mdip, phi)
        w_spin = 2.0 * math.pi * f_drive
        s_new = update_spin_axis(
            s, tau, self.robot.I_spin, w_spin, dt, n_rot, self.align_tau
        )

        # The moment is rebuilt on the UPDATED axis, matching the MATLAB: within
        # a step, lift and gradient force must see the same s.
        f_grad = np.zeros(3)
        if self.use_grad:
            _, m_cos, m_sin = cycle_average(
                s_new, u, v_h, n_rot, self.robot.mdip, phi
            )
            f_grad = gradient_force(m_cos, m_sin, du, dv)

        lift = (f_drive / self.f_hover) ** 2 * GRAVITY
        a = lift * s_new - GRAVITY * _ZHAT + f_grad / self.robot.mass
        if self.beta > 0:
            a = a - self.beta * v

        v_new = v + dt * a
        r_new = r + dt * v_new

        info = {
            "n_rot": n_rot,
            "phi": phi,
            "ratio": ratio,
            "semiaxes": (a_ax, b_ax),
            "tau": tau,
            "f_grad": f_grad,
            "accel": a,
        }
        return np.concatenate([r_new, v_new, s_new]), info


def make_state(r, v=(0.0, 0.0, 0.0), s=(0.0, 0.0, 1.0)):
    """Pack a state vector, normalizing the spin axis."""

    s = np.asarray(s, dtype=float)
    return np.concatenate(
        [np.asarray(r, float), np.asarray(v, float), s / np.linalg.norm(s)]
    )


# The GUI's recommended passive-hover start point.
GUI_R0 = np.array([0.002, 0.0, 0.020084])

# n_rot at GUI_R0, pinned so a refactor that changes the field cannot pass quietly.
# Regenerate it from here rather than trusting the GUI's own stored s0, which belongs to a
# different geometry: it tilts -0.00056 in x where this coil table gives +0.0488.
N_ROT_AT_R0 = np.array([0.048798659933, 0.0, 0.998808635580])


def _self_check():
    from scipy.optimize import brentq

    rng = np.random.default_rng(0)
    p = robot_params()
    coils = quick_coils()

    print(f"I_spin      = {p.I_spin:.6e} kg m^2")
    print(f"I_trans     = {p.I_trans:.6e} kg m^2   (I_t/I_s = {p.I_trans/p.I_spin:.4f})")
    print(f"mass        = {p.mass:.6e} kg          (weight {p.weight*1e6:.1f} uN)")
    print(f"mdip        = {p.mdip:.6e} A m^2")

    # 1. The rotation axis: exactly +z on the symmetry axis, and pinned off it.
    on_axis = rotation_axis(*field_at(np.array([0.0, 0.0, 0.020084]), coils)[:2])
    assert np.allclose(on_axis, [0.0, 0.0, 1.0], atol=1e-12), on_axis
    u, v, du, dv = field_at(GUI_R0, coils)
    n_rot = rotation_axis(u, v)
    print(f"n_rot(r0)   = [{n_rot[0]:+.9f}, {n_rot[1]:+.9f}, {n_rot[2]:.9f}]")
    assert np.allclose(n_rot, N_ROT_AT_R0, atol=1e-9), n_rot

    # 1b. How far the point-dipole stub is from the exact on-axis loop field, at
    # the two distances that matter. Printed every run so the model's weakest
    # assumption stays a number rather than a caveat.
    r_coil, turns = 0.0105, 650
    solo = Coils(
        pos=np.zeros((1, 3)),
        nhat=np.array([[0.0, 0.0, 1.0]]),
        moment=turns * math.pi * r_coil**2 * np.array([[0.0, 0.0, 1.0]]),
        amp=np.array([1.0]),
        phase=np.array([0.0]),
        channel=np.array([0]),
    )
    errs = []
    for dist in (0.0349, 0.0526):  # nearest and farthest coils from GUI_R0
        approx = coil_basis(np.array([0.0, 0.0, dist]), solo)[0][0, 2]
        exact = MU0 * turns * r_coil**2 / (2 * (r_coil**2 + dist**2) ** 1.5)
        errs.append(100 * (approx - exact) / exact)
    print(f"dipole stub : +{errs[0]:.1f}% at 35 mm, +{errs[1]:.1f}% at 53 mm vs exact loop")
    assert 5.0 < errs[1] < errs[0] < 20.0, errs

    # 2. Analytic gradient against the central differences the MATLAB uses.
    h = 1e-6
    for target, exact in ((0, du), (1, dv)):
        num = np.empty((3, 3))
        for j in range(3):
            rp, rm = GUI_R0.copy(), GUI_R0.copy()
            rp[j] += h
            rm[j] -= h
            num[:, j] = (field_at(rp, coils)[target] - field_at(rm, coils)[target]) / (
                2 * h
            )
        assert np.allclose(num, exact, rtol=2e-5, atol=1e-12), (num, exact)

    # 3. curl B = div B = 0 away from the sources: symmetric and traceless.
    _, grad_b = coil_basis(GUI_R0, coils)
    assert np.allclose(grad_b, np.swapaxes(grad_b, 1, 2), atol=1e-12)
    assert np.allclose(np.trace(grad_b, axis1=1, axis2=2), 0.0, atol=1e-9)

    # 4. Circular-field limit reduces to theory.md section 5.3.
    bmag = 5e-3
    uc, vc = np.array([bmag, 0, 0]), np.array([0, bmag, 0])
    tau_max = p.mdip * bmag
    tau_d = 0.2 * tau_max
    phi, a_ax, b_ax, ratio = phase_lag(uc, vc, p.mdip, tau_d)
    assert abs(a_ax - bmag) < 1e-15 and abs(b_ax - bmag) < 1e-15
    assert abs(math.sin(phi) - tau_d / tau_max) < 1e-12, phi

    # ... and step-out is reported, not clipped.
    try:
        phase_lag(uc, vc, p.mdip, 1.5 * tau_max)
    except StepOut as exc:
        assert exc.ratio > 1.0
    else:
        raise AssertionError("step-out was not raised")

    # 5. Torque is transverse to s, and vanishes as s approaches n_rot.
    s_tilt = np.array([0.2, 0.1, 1.0])
    s_tilt /= np.linalg.norm(s_tilt)
    phi_r, *_ = phase_lag(u, v, p.mdip, p.mdip * 1e-6)
    tau = torque_avg(s_tilt, u, v, n_rot, p.mdip, phi_r)
    assert abs(tau @ s_tilt) < 1e-12 * max(np.linalg.norm(tau), 1e-12) + 1e-18
    assert np.linalg.norm(torque_avg(n_rot, u, v, n_rot, p.mdip, phi_r)) == 0.0
    tilts = [1e-2, 1e-3, 1e-4]
    mags = []
    for eps in tilts:
        s_e = n_rot + eps * np.array([1.0, 0.0, 0.0])
        s_e /= np.linalg.norm(s_e)
        mags.append(np.linalg.norm(torque_avg(s_e, u, v, n_rot, p.mdip, phi_r)))
    assert mags[0] > mags[1] > mags[2], mags

    # 6. On the symmetry axis, without the gradient force (the GUI's default),
    # n_rot is exactly +z, lift is exactly vertical, and the robot hovers.
    plant = Plant(coils, p)
    axis0 = np.array([0.0, 0.0, 0.020084])
    x = make_state(axis0, s=(0.0, 0.0, 1.0))
    dt = 1e-3
    for _ in range(2000):
        x, info = plant.step(x, plant.f_hover, dt)
        assert abs(np.linalg.norm(x[6:9]) - 1.0) < 1e-12
    drift = np.linalg.norm(x[0:3] - axis0)
    print(f"2 s on-axis : drift {drift*1e9:.3f} nm, lock ratio {info['ratio']:.4f}")
    assert drift < 1e-9, drift

    # 6b. Two millimetres off it, the same open loop slides away and steps out.
    x = make_state(GUI_R0, s=N_ROT_AT_R0)
    for k in range(20000):
        try:
            x, info = plant.step(x, plant.f_hover, dt)
        except StepOut as exc:
            lost = np.linalg.norm(x[0:3] - GUI_R0)
            print(
                f"off-axis    : step-out at t={k*dt:.2f} s, "
                f"{lost*1e3:.1f} mm out, needed sin(phi)={exc.ratio:.2f}"
            )
            break
    else:
        raise AssertionError("open loop did not step out; expected it to")
    assert k * dt < 5.0, k * dt

    # 7. The gradient force: a vertical trap and a lateral anti-trap, which is
    # the only pair of signs Earnshaw's theorem allows. Both pinned, because the
    # lateral term is a destabilizing force of the same order as the tilt slide.
    grad = Plant(coils, p, use_grad=True)

    def fz(z):
        return grad.accel(
            np.array([0.0, 0.0, z]), np.zeros(3), np.array([0.0, 0.0, 1.0]), 110.0
        )[1]["f_grad"][2]

    z_trap = brentq(fz, 0.010, 0.030)
    stiffness = (fz(z_trap + 1e-4) - fz(z_trap - 1e-4)) / 2e-4
    _, info = grad.accel(
        np.array([0.002, 0.0, z_trap]), np.zeros(3), np.array([0.0, 0.0, 1.0]), 110.0
    )
    fx = info["f_grad"][0]
    print(
        f"F_grad      : trap at z={z_trap*1e3:.3f} mm, dFz/dz={stiffness:+.4f} N/m; "
        f"lateral {100*fx/p.weight:+.2f} % of weight at 2 mm"
    )
    assert 0.014 < z_trap < 0.017, z_trap
    assert stiffness < 0.0, stiffness  # traps vertically
    assert fx > 0.05 * p.weight, fx  # and expels laterally

    _, info = grad.accel(
        np.array([0.0, 0.0, 0.020084]), np.zeros(3), np.array([0.0, 0.0, 1.0]), 110.0
    )
    share = abs(info["f_grad"][2]) / p.weight
    print(f"            : {100*share:.1f} % of weight at z=20 mm, pulling down")
    assert share > 0.3, share

    # 8. Euler's step bound is numerical, far below the model's own timescales.
    dt_rec = recommended_dt(
        np.linalg.norm(info["tau"]), p.I_spin, 2 * math.pi * plant.f_hover, plant.f_hover
    )
    print(f"dt bound    = {dt_rec*1e3:.3f} ms (explicit Euler, not physics)")

    # A random state must not raise: the model is defined off the axis too.
    for _ in range(20):
        rr = GUI_R0 + rng.normal(0, 3e-3, 3)
        ss = rng.normal(0, 1, 3)
        grad.accel(rr, np.zeros(3), ss / np.linalg.norm(ss), plant.f_hover)

    print("self-check PASS")


if __name__ == "__main__":
    _self_check()
