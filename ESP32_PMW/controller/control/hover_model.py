#!/usr/bin/env python3
"""Hover plant model -- Python port of the physics embedded in
model/frequency_modulation_piecewise_vertical_gui.m, extended with a lateral
axis for the state-space hover controller.

Full nonlinear state s = [x, xd, z, zd, delta, omega]:
  x_dd     = g * k_lat * mag                      (lateral tilt authority)
  z_dd     = g * ((omega/(2*pi*f_hover))^2 - 1)   (lift ~ f_robot^2)
  delta_d  = 2*pi*f_field - omega                 (field-robot phase lag)
  omega_d  = (tau_max*sin(delta) - k_drag*f_r*|f_r|) / I_robot,  f_r = omega/2pi

Inputs u = (mag_signed, f_field_hz). k_lat (rad of disk tilt per unit mixer
mag) is the single parameter NOT derivable from the MATLAB model -- seed
guess, re-identify on the rig (see design_hover_lqr.py --k-lat).

Reduced 4-state design model (x, xd, z, zd): rotation treated as a
phase-locked inner actuator (f_robot ~ f_field); its 15.6 Hz / zeta~0.02
mode is above a 30 Hz camera's Nyquist so the outer loop must not try to
shape it -- see linearize_full() eigenvalues for the justification numbers.

Self-check: uv run python ai/hover_model.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import cont2discrete

GRAVITY = 9.80665  # m/s^2 (MATLAB line 613)

# Disk moment of inertia, kg m^2 -- verbatim expression from the .m file
# lines 25-28 (hub + 2 arms + tip masses, SI).
I_ROBOT = 3.89e-9 + 2.0 * (
    1.0 / 12.0 * 1.17832e-5 * (3.0 * 0.79375**2 * 1e-6 + 0.79375**2 * 1e-6)
    + 1.17832e-5 * (0.496875**2 * 1e-6)
)

# 23-point measured drag-torque table (N m, negative = opposing), 10:10:230 Hz
# -- verbatim from the .m file lines 30-55.
DRAG_FREQ_HZ = np.arange(10.0, 231.0, 10.0)
DRAG_TORQUE_NM = np.array([
    -7.20311e-08, -8.54759e-07, -1.18389e-06, -1.55412e-06, -2.06800e-06,
    -2.56102e-06, -2.88399e-06, -3.59653e-06, -3.83331e-06, -4.99189e-06,
    -5.72497e-06, -6.29224e-06, -7.17918e-06, -8.19104e-06, -9.07986e-06,
    -1.00964e-05, -1.08262e-05, -1.21433e-05, -1.34310e-05, -1.48856e-05,
    -1.68245e-05, -1.88440e-05, -2.10429e-05,
])

F_HOVER_HZ_DEFAULT = 140.0   # GUI default; firmware main_flight.cpp HOVER_HZ=150 -> parameterize
K_LAT_DEFAULT = 0.05         # SEED GUESS -- rad tilt per unit mag; identify on rig
TORQUE_MARGIN_DEFAULT = 5.0  # tau_max / |drag at f_hover| (GUI default)


def fit_k_drag() -> tuple[float, float]:
    """Zero-intercept least-squares fit drag = -k_drag*f^2 (MATLAB lines 57-62).
    Returns (k_drag [N m/Hz^2], R^2)."""
    f2 = DRAG_FREQ_HZ**2
    k_drag = float(np.sum(f2 * (-DRAG_TORQUE_NM)) / np.sum(f2**2))
    fitted = -k_drag * f2
    ss_res = float(np.sum((DRAG_TORQUE_NM - fitted) ** 2))
    ss_tot = float(np.sum((DRAG_TORQUE_NM - DRAG_TORQUE_NM.mean()) ** 2))
    return k_drag, 1.0 - ss_res / ss_tot


@dataclass
class HoverParams:
    """Plant parameters + hover-trim derived quantities."""
    f_hover: float = F_HOVER_HZ_DEFAULT
    k_lat: float = K_LAT_DEFAULT
    margin: float = TORQUE_MARGIN_DEFAULT
    g: float = GRAVITY
    I_robot: float = I_ROBOT
    k_drag: float = field(default_factory=lambda: fit_k_drag()[0])
    # derived (filled in __post_init__)
    tau_max: float = 0.0
    delta_trim: float = 0.0
    omega_trim: float = 0.0

    def __post_init__(self):
        # tau_max = margin * drag torque at the hover frequency (MATLAB 554-556)
        self.tau_max = self.margin * self.k_drag * self.f_hover**2
        self.delta_trim = math.asin(1.0 / self.margin)
        self.omega_trim = 2.0 * math.pi * self.f_hover


def make_params(f_hover: float = F_HOVER_HZ_DEFAULT, k_lat: float = K_LAT_DEFAULT,
                margin: float = TORQUE_MARGIN_DEFAULT) -> HoverParams:
    return HoverParams(f_hover=f_hover, k_lat=k_lat, margin=margin)


def nonlinear_dynamics(t: float, s: np.ndarray, u: tuple[float, float],
                       p: HoverParams) -> np.ndarray:
    """Full 6-state truth model. s=[x,xd,z,zd,delta,omega], u=(mag_signed, f_field_hz)."""
    _, xd, _, zd, delta, omega = s
    mag, f_field = u
    f_robot = omega / (2.0 * math.pi)
    return np.array([
        xd,
        p.g * p.k_lat * mag,
        zd,
        p.g * ((f_robot / p.f_hover) ** 2 - 1.0),
        2.0 * math.pi * f_field - omega,
        (p.tau_max * math.sin(delta) - p.k_drag * f_robot * abs(f_robot)) / p.I_robot,
    ])


def linearize_reduced(p: HoverParams) -> tuple[np.ndarray, np.ndarray]:
    """4-state design model about hover trim. States [x, xd, z, zd],
    inputs [mag_signed, delta_f_hz] (delta_f = f_field - f_hover).
    Rotation assumed phase-locked: f_robot == f_field, so the vertical
    channel sees d(z_dd)/d(delta_f) = 2 g / f_hover."""
    A = np.array([[0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0],
                  [0.0, 0.0, 0.0, 0.0]])
    B = np.array([[0.0, 0.0],
                  [p.g * p.k_lat, 0.0],      # ONLY entry depending on the unknown k_lat
                  [0.0, 0.0],
                  [0.0, 2.0 * p.g / p.f_hover]])
    return A, B


def linearize_full(p: HoverParams) -> tuple[np.ndarray, np.ndarray]:
    """6-state Jacobian about hover trim (documentation / eigen-analysis).
    States [x, xd, z, zd, delta, omega], inputs [mag_signed, f_field_hz]."""
    w0 = p.omega_trim
    dwd_ddelta = p.tau_max * math.cos(p.delta_trim) / p.I_robot
    dwd_domega = -p.k_drag * w0 / (2.0 * math.pi**2) / p.I_robot
    dzdd_domega = 2.0 * p.g / (2.0 * math.pi * p.f_hover)  # per rad/s at trim
    A = np.zeros((6, 6))
    A[0, 1] = 1.0
    A[2, 3] = 1.0
    A[3, 5] = dzdd_domega
    A[4, 5] = -1.0
    A[5, 4] = dwd_ddelta
    A[5, 5] = dwd_domega
    B = np.zeros((6, 2))
    B[1, 0] = p.g * p.k_lat
    B[4, 1] = 2.0 * math.pi
    return A, B


def discretize(A: np.ndarray, B: np.ndarray, ts: float) -> tuple[np.ndarray, np.ndarray]:
    """ZOH discretization at sample time ts [s]."""
    n, m = A.shape[0], B.shape[1]
    Ad, Bd, *_ = cont2discrete((A, B, np.eye(n), np.zeros((n, m))), ts, method="zoh")
    return Ad, Bd


if __name__ == "__main__":
    k_drag, r2 = fit_k_drag()
    p = make_params()
    A4, B4 = linearize_reduced(p)
    A6, _ = linearize_full(p)
    eig = np.linalg.eigvals(A6)
    rot = eig[np.abs(eig.imag) > 1.0]  # the phase-lock pair
    vgain = B4[3, 1]

    print(f"I_robot     = {p.I_robot:.12e} kg m^2")
    print(f"k_drag      = {k_drag:.12e} N m/Hz^2   (R^2 = {r2:.4f})")
    print(f"tau_max     = {p.tau_max:.4e} N m  (margin {p.margin} @ {p.f_hover:.0f} Hz)")
    print(f"delta_trim  = {math.degrees(p.delta_trim):.2f} deg")
    print(f"vert. gain  = {vgain:.4f} m/s^2 per Hz  (2g/f_hover)")
    print(f"full-model eigenvalues: {np.sort_complex(eig)}")
    print(f"phase-lock mode: {abs(rot[0].imag)/(2*math.pi):.1f} Hz, "
          f"zeta = {-rot[0].real/abs(rot[0]):.3f}")

    # Port-correctness asserts against the MATLAB-derived reference numbers.
    assert abs(k_drag - 3.908983509946592e-10) < 1e-16, k_drag
    assert abs(p.I_robot - 3.900767435994792e-9) < 1e-15, p.I_robot
    assert abs(math.degrees(p.delta_trim) - 11.5370) < 1e-3
    assert abs(vgain - 2 * GRAVITY / 140.0) < 1e-12
    assert abs(rot[0].real - (-2.233)) < 0.01 and abs(abs(rot[0].imag) - 98.07) < 0.2
    print("self-check PASS")
