#!/usr/bin/env python3
"""
Nonlinear MPC for the multi-coil spatial plant in `spatial_model.py`. theory.md section 13.

Why MPC and not a second LQR: three properties a gain matrix cannot express.

    moving feasibility   Synchronous lock needs 2 k_drag f^2 <= mdip (a+b)(r, I), and (a+b)
                         falls off with position. The workspace is a region, not a box, and it
                         moves with the drive frequency.
    authority via tilt   Lift points along the spin axis and the axis is slewed by a torque, so
                         lateral force is two integrations from the input, with ~1 Hz precession
                         in the same band as heave. It has to be predicted, not just fed back.
    an unstable plant    Open loop the robot leaves in under a second (spatial_model.py).

    u = [tilt_x, tilt_y, f_drive]

The honest input is four complex channel phasors plus drive frequency, nine numbers. At any
point the four channels span the field with rank 3, so they set rotation axis, amplitude and
ellipticity independently: three numbers suffice. `allocate` commands a circular field of fixed
amplitude about normalize(tilt_x, tilt_y, 1), inverting the 3x4 channel basis by pseudo-inverse,
which is minimum-norm and therefore minimum-ohmic-loss.

This also removes the one un-identified number in `hover_model.py`: `K_LAT_DEFAULT` is a seed
guess for lateral authority, which here is derived from the field.

Scheme: real-time iteration. One SQP pass per control step, warm-started from the previous
solution shifted forward, inputs move-blocked. The plant costs ~105 us per step and a gradient
over B blocks costs (3B+1) rollouts, which is what fits the budget.

Self-check: uv run python controller/control/spatial_mpc.py
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.optimize import minimize

from spatial_model import (
    GRAVITY,
    Plant,
    StepOut,
    coil_basis,
    ellipse_semiaxes,
    field_at,
    make_state,
    quick_coils,
    robot_params,
)


@dataclass
class Weights:
    """Stage-cost weights, each normalised by a tolerance so the terms compare."""

    pos: float = 1.0
    vel: float = 0.05
    acc: float = 0.02
    jerk: float = 0.02
    energy: float = 0.005
    slew: float = 0.05
    lock: float = 50.0
    terminal: float = 8.0

    # Bryson tolerances: the value at which each term counts as one unit of bad.
    pos_tol: float = 2e-3     # m
    vel_tol: float = 0.05     # m/s
    acc_tol: float = 2.5      # m/s^2
    jerk_tol: float = 25.0    # m/s^3
    i_tol: float = 3.0        # per-channel current, A: ohmic loss up to a factor
    slew_tol: np.ndarray = field(
        default_factory=lambda: np.array([0.05, 0.05, 10.0])
    )


@dataclass
class Limits:
    """Hard bounds and the soft feasibility margin."""

    i_max: float = 3.0            # per-channel current ceiling, A
    f_min: float = 60.0           # drive frequency, Hz
    f_max: float = 260.0          # ceiling; `SpatialMPC` tightens it to the lock bound
    tilt_max: float = 0.30        # field-axis tilt as tan(angle); 0.30 = 16.7 deg
    lock_max: float = 0.80        # largest allowed required sin(phi); below 1 for the Q~20 swing
    box_xy: float = 0.020         # workspace where the field map is trusted, m
    box_z: tuple = (0.006, 0.034) # m


class SpatialMPC:
    """Receding-horizon controller over `spatial_model.Plant`."""

    def __init__(
        self,
        plant: Plant,
        weights: Weights = None,
        limits: Limits = None,
        horizon: int = 10,    # control steps predicted
        ts: float = 0.05,     # control step, s
        substeps: int = 5,    # plant Euler steps per control step: predictor accuracy
        blocks: int = 3,      # move blocking; inputs free for this many steps, then trim
        iters: int = 1,       # SQP iterations per step; 1 works because of the LQR warm start
        b_amp: float = 4.5e-3,
    ):
        self.plant = plant
        self.w = weights or Weights()
        self.lim = limits or Limits()
        self.horizon = horizon
        self.ts = ts
        self.substeps = substeps
        self.blocks = blocks
        self.iters = iters
        self.b_amp = b_amp

        # The lock bound, as a frequency bound. `allocate` commands a circular
        # field of fixed amplitude, so a + b = 2 b_amp and the feasibility
        # condition 2 k_drag f^2 <= mdip (a+b) collapses to a ceiling on f alone.
        # That makes it a *bound*, which SLSQP enforces exactly and for free,
        # rather than a penalty it can wander past.
        #
        # It is not the whole story: the amplitude is only achievable while the
        # allocation stays inside i_max, and far off axis or at large tilt it does
        # not. The residual coupling between position, current and lock is what
        # the soft term in `rollout` still covers.
        f_lock = math.sqrt(
            self.lim.lock_max * plant.robot.mdip * 2.0 * b_amp / (2.0 * plant.k_drag)
        )
        self.f_max = min(self.lim.f_max, f_lock)

        self.u_trim = np.array([0.0, 0.0, plant.f_hover])
        self.U = np.tile(self.u_trim, (blocks, 1))

        # Decision variables are scaled to comparable sensitivity before they
        # reach SLSQP. Unscaled, a tilt of ~0.01 and a frequency of ~110 differ by
        # four orders of magnitude, and SLSQP's single absolute finite-difference
        # step cannot suit both: with the default 1.5e-8 the tilt gradient came
        # out as rounding noise from the rollout's ~60 Euler steps, and the
        # controller sat at trim while the robot flew away.
        self.scale = np.array([0.05, 0.05, 20.0])
        self.fd_eps = 2e-3  # in scaled units: 1e-4 of tilt, 0.04 Hz
        self.K = None  # LQR warm-start gain, built on first use
        self._pinv = None  # channel basis inverse, refreshed per solve

    # -- actuator ---------------------------------------------------------

    def channel_basis(self, r):
        """(3,4) field per ampere of each channel at ``r``, in tesla per amp."""

        b, _ = coil_basis(np.asarray(r, float), self.plant.coils, want_grad=False)
        ch = self.plant.coils.channel
        return np.column_stack(
            [b[ch == g].sum(0) for g in range(self.plant.coils.n_channels)]
        )

    def allocate(self, tilt, r=None):
        """Channel phasors realizing a circular field about ``normalize(tilt, 1)``."""

        pinv = self._pinv if r is None else self._pinv_at(r)
        n = np.array([tilt[0], tilt[1], 1.0])
        n /= math.sqrt(n @ n)
        e1 = np.array([1.0, 0.0, -n[0] / max(n[2], 1e-9)])
        e1 /= math.sqrt(e1 @ e1)
        e2 = np.cross(n, e1)
        z = pinv @ (self.b_amp * e1) - 1j * (pinv @ (self.b_amp * e2))

        # Saturate on the current ceiling, keeping the axis and shape and giving
        # up amplitude. Without this the allocation "achieves" b_amp anywhere by
        # demanding unbounded current, so a prediction that leaves the array
        # never loses lock and the cost landscape fills with 1e21 spikes. The
        # drive saturates; the model has to as well.
        peak = float(np.abs(z).max())
        if peak > self.lim.i_max:
            z = z * (self.lim.i_max / peak)
        return z

    def _pinv_at(self, r):
        """Right inverse of the 3x4 channel basis at ``r``, as a (4,3) array."""

        b = self.channel_basis(r)
        return b.T @ np.linalg.inv(b @ b.T)

    # -- linear model and the warm start ----------------------------------

    def linearize(self, r0, f0=None):
        """Numerical Jacobians of the reduced 8-state model about hover at ``r0``."""

        f0 = self.plant.f_hover if f0 is None else f0
        st0 = np.concatenate([np.asarray(r0, float), np.zeros(3), np.zeros(2)])
        u0 = np.array([0.0, 0.0, f0])
        h_int = 1e-4

        def deriv(st, u):
            sz = math.sqrt(max(1e-12, 1.0 - st[6] ** 2 - st[7] ** 2))
            x = np.concatenate([st[:6], [st[6], st[7], sz]])
            xn, _ = self.plant.step(
                x, u[2], h_int, phasor=self.allocate(u[:2], r=x[0:3])
            )
            return (np.concatenate([xn[:6], xn[6:8]]) - st) / h_int

        A = np.zeros((8, 8))
        B = np.zeros((8, 3))
        for i in range(8):
            e = np.zeros(8)
            e[i] = 1e-7
            A[:, i] = (deriv(st0 + e, u0) - deriv(st0 - e, u0)) / 2e-7
        for j, step_j in enumerate((1e-6, 1e-6, 1e-3)):
            e = np.zeros(3)
            e[j] = step_j
            B[:, j] = (deriv(st0, u0 + e) - deriv(st0, u0 - e)) / (2 * step_j)
        return A, B

    def design_lqr(self, r0, r_scale=50.0):
        """Continuous LQR about hover at ``r0``, stored as `self.K` (3x8)."""

        A, B = self.linearize(r0)
        w = self.w
        Q = np.diag(
            [1 / w.pos_tol**2] * 3 + [1 / w.vel_tol**2] * 3 + [1 / 0.02**2] * 2
        )
        R = r_scale * np.diag([1 / 0.05**2, 1 / 0.05**2, 1 / 10.0**2])
        P = solve_continuous_are(A, B, Q, R)
        self.K = np.linalg.inv(R) @ B.T @ P
        self.lqr_poles = np.linalg.eigvals(A - B @ self.K)
        self.open_poles = np.linalg.eigvals(A)
        return self.K

    def lqr_input(self, x, target):
        """Stabilising input from the linear gain, clipped to the bounds."""

        if self.K is None:
            self.design_lqr(x[0:3])
        st = np.concatenate([x[0:6], x[6:8]])
        ref = np.concatenate([np.asarray(target, float), np.zeros(5)])
        u = self.u_trim - self.K @ (st - ref)
        u[0] = np.clip(u[0], -self.lim.tilt_max, self.lim.tilt_max)
        u[1] = np.clip(u[1], -self.lim.tilt_max, self.lim.tilt_max)
        u[2] = np.clip(u[2], self.lim.f_min, self.f_max)
        return u

    # -- prediction -------------------------------------------------------

    def _expand(self, U):
        """Move-blocked inputs to one per horizon step, padded with trim."""

        seq = np.tile(self.u_trim, (self.horizon, 1))
        n = min(self.blocks, self.horizon)
        seq[:n] = U[:n]
        return seq

    def rollout(self, x0, U, target, collect=False):
        """Cost of an input sequence, and optionally the predicted trajectory."""

        seq = self._expand(U)
        x = x0.copy()
        dt = self.ts / self.substeps
        cost = 0.0
        prev_u = self._u_prev
        prev_a = None
        traj = [] if collect else None

        for k in range(self.horizon):
            tilt, f = seq[k, :2], seq[k, 2]
            phasor = self.allocate(tilt, r=x[0:3])

            try:
                for _ in range(self.substeps):
                    x, info = self.plant.step(x, f, dt, phasor=phasor)
            except StepOut:
                return (1e6 + 1e3 * (self.horizon - k), traj)

            r, v, a = x[0:3], x[3:6], info["accel"]
            e = r - target

            cost += self.w.pos * (e @ e) / self.w.pos_tol**2
            cost += self.w.vel * (v @ v) / self.w.vel_tol**2
            cost += self.w.acc * (a @ a) / self.w.acc_tol**2
            if prev_a is not None:
                j = (a - prev_a) / self.ts
                cost += self.w.jerk * (j @ j) / self.w.jerk_tol**2
            prev_a = a

            amps = np.abs(phasor)
            cost += self.w.energy * (amps @ amps) / self.w.i_tol**2

            du = (seq[k] - prev_u) / self.w.slew_tol
            cost += self.w.slew * (du @ du)
            prev_u = seq[k]

            # Exact penalty on the lock margin. Kept soft rather than as an SLSQP
            # constraint because a constrained solve finite-differences the
            # constraint too, which does not fit the real-time budget.
            # ponytail: soft lock margin; promote to a hard constraint if the
            # trajectory is ever seen riding the boundary.
            over = info["ratio"] - self.lim.lock_max
            if over > 0:
                cost += self.w.lock * (over / (1.0 - self.lim.lock_max)) ** 2

            # Same treatment for the workspace and the current ceiling.
            out = max(abs(r[0]), abs(r[1])) - self.lim.box_xy
            out = max(out, self.lim.box_z[0] - r[2], r[2] - self.lim.box_z[1])
            if out > 0:
                cost += self.w.lock * (out / 1e-3) ** 2
            over_i = amps.max() - self.lim.i_max
            if over_i > 0:
                cost += self.w.lock * (over_i / 0.1) ** 2

            if collect:
                traj.append((x.copy(), info))

        e = x[0:3] - target
        cost += self.w.terminal * self.w.pos * (e @ e) / self.w.pos_tol**2
        cost += self.w.terminal * self.w.vel * (x[3:6] @ x[3:6]) / self.w.vel_tol**2
        return (cost, traj)

    # -- solve ------------------------------------------------------------

    def solve(self, x, target, u_prev=None):
        """One real-time iteration. Returns ``(u, diagnostics)``."""

        self._pinv = np.linalg.pinv(self.channel_basis(x[0:3]))
        self._u_prev = self.u_trim if u_prev is None else np.asarray(u_prev, float)

        # Warm start: LQR now, trim afterwards. Seeding from a stabilising input
        # rather than from trim is what makes one SQP iteration enough. Started
        # at trim, SLSQP could not find a descent direction at all -- doing
        # nothing is a deep local minimum here, because a tilt costs position
        # immediately (precession sends the robot 90 degrees off the command) and
        # only pays off later.
        self.U = np.vstack(
            [self.lqr_input(x, target), np.tile(self.u_trim, (self.blocks - 1, 1))]
        )

        lo = np.array([-self.lim.tilt_max, -self.lim.tilt_max, self.lim.f_min])
        hi = np.array([self.lim.tilt_max, self.lim.tilt_max, self.f_max])
        bounds = [(lo[i % 3], hi[i % 3]) for i in range(3 * self.blocks)]

        sc = np.tile(self.scale, self.blocks)
        off = np.tile(self.u_trim, self.blocks)
        bounds = [
            ((lo[i % 3] - off[i]) / sc[i], (hi[i % 3] - off[i]) / sc[i])
            for i in range(3 * self.blocks)
        ]

        t0 = time.perf_counter()
        res = minimize(
            lambda z: self.rollout(
                x, (z * sc + off).reshape(self.blocks, 3), target
            )[0],
            (self.U.ravel() - off) / sc,
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": self.iters, "ftol": 1e-8, "eps": self.fd_eps},
        )
        self.U = np.clip((res.x * sc + off).reshape(self.blocks, 3), lo, hi)

        return self.U[0].copy(), {
            "cost": float(res.fun),
            "solve_s": time.perf_counter() - t0,
            "nfev": res.nfev,
        }

    def lqr_loop(self, x, target):
        """The linear law alone, for comparison against the constrained solve."""

        return self.lqr_input(x, target)

    def apply(self, x, u, dt):
        """Step the plant under a held input, returning ``(x, info, ok)``."""

        phasor = self.allocate(u[:2])
        info = None
        n = max(1, int(round(dt / (self.ts / self.substeps))))
        for _ in range(n):
            try:
                x, info = self.plant.step(
                    x, u[2], self.ts / self.substeps, phasor=phasor
                )
            except StepOut:
                return x, info, False
        return x, info, True


def closed_loop(mpc, x0, targets, duration, log=None):
    """Run the loop against the plant. ``targets`` is a callable t -> (3,) in m."""

    x = x0.copy()
    u = mpc.u_trim.copy()
    n = int(round(duration / mpc.ts))
    out = {k: [] for k in ("t", "r", "v", "s", "u", "ratio", "amps", "solve_s", "ref")}

    for k in range(n):
        t = k * mpc.ts
        ref = np.asarray(targets(t), float)
        u, diag = mpc.solve(x, ref, u_prev=u)
        x, info, ok = mpc.apply(x, u, mpc.ts)
        if not ok:
            out["lost_at"] = t
            break

        out["t"].append(t)
        out["r"].append(x[0:3].copy())
        out["v"].append(x[3:6].copy())
        out["s"].append(x[6:9].copy())
        out["u"].append(u.copy())
        out["ref"].append(ref)
        out["ratio"].append(info["ratio"])
        out["amps"].append(np.abs(mpc.allocate(u[:2])))
        out["solve_s"].append(diag["solve_s"])
        if log:
            log(t, x, u, info, diag)

    lost = out.pop("lost_at", None)
    hist = {k: np.array(v) for k, v in out.items()}
    hist["lost_at"] = lost
    return hist


def default_mpc(use_grad=True, **kw):
    """The controller as tuned, over the default coil array."""

    plant = Plant(quick_coils(), robot_params(), use_grad=use_grad)
    return SpatialMPC(plant, **kw)


def _self_check():
    tgt = np.array([0.0, 0.0, 0.0130])
    mpc = default_mpc(
        use_grad=False, ts=0.05, horizon=8, blocks=2, iters=1, substeps=3
    )
    p = mpc.plant.robot

    # 1. The allocation realizes the commanded axis exactly, and is circular.
    mpc._pinv = np.linalg.pinv(mpc.channel_basis([0.0, 0.0, 0.013]))
    for tilt in ([0.0, 0.0], [0.1, 0.0], [-0.05, 0.12]):
        z = mpc.allocate(np.array(tilt))
        u, v, _, _ = field_at(
            np.array([0.0, 0.0, 0.013]), mpc.plant.coils, phasor=z
        )
        want = np.array([tilt[0], tilt[1], 1.0])
        want /= np.linalg.norm(want)
        got = np.cross(u, v)
        got /= np.linalg.norm(got)
        a, b = ellipse_semiaxes(u, v)
        assert np.degrees(math.acos(min(1.0, got @ want))) < 1e-3, tilt
        assert abs(a - mpc.b_amp) < 1e-9 and abs(b - mpc.b_amp) < 1e-9, (a, b)
    print(
        f"allocation  : axis exact to <1e-3 deg, circular at {mpc.b_amp*1e3:.2f} mT, "
        f"lock bound f <= {mpc.f_max:.1f} Hz"
    )

    # 2. The linear model of the re-allocating loop. No unstable eigenvalue: the
    # violent instability belongs to the *fixed-current* plant, and re-solving
    # the currents for the measured position removes it. See `linearize`.
    mpc.design_lqr(tgt)
    assert max(v.real for v in mpc.open_poles) < 1e-6, mpc.open_poles
    assert max(v.real for v in mpc.lqr_poles) < -1.0, mpc.lqr_poles
    print(
        f"open poles  : max Re {max(v.real for v in mpc.open_poles):+.3f} /s "
        f"(marginal: three integrators, damped precession -0.78 +- 3.43j)"
    )
    print(f"lqr poles   : max Re {max(v.real for v in mpc.lqr_poles):+.2f} /s")

    # 3. Hold, from three displacements including one 13 mm out.
    for start in ([0.002, 0.001, 0.015], [0.006, -0.004, 0.018], [0.010, 0.008, 0.010]):
        run = closed_loop(mpc, make_state(start, s=(0.0, 0.0, 1.0)), lambda t: tgt, 6.0)
        assert run["lost_at"] is None, (start, run["lost_at"])
        err = np.linalg.norm(run["r"] - run["ref"], axis=1)
        print(
            f"hold        : from {np.linalg.norm(np.array(start)-tgt)*1e3:5.2f} mm -> "
            f"{err[-1]*1e6:6.1f} um in 6 s, peak |I| {run['amps'].max():.2f} A, "
            f"lock {run['ratio'].max():.3f}"
        )
        assert err[-1] < 5e-5, err[-1]
        assert run["ratio"].max() < 1.0 and run["amps"].max() < mpc.lim.i_max * 1.01

    # 4. Track a stepped target, and check the constraints and the budget.
    def steps(t):
        if t < 2.0:
            return tgt
        if t < 5.0:
            return tgt + np.array([0.006, 0.0, 0.0])
        return tgt + np.array([0.006, 0.0, 0.004])

    run = closed_loop(
        mpc, make_state([0.002, 0.001, 0.015], s=(0.0, 0.0, 1.0)), steps, 8.0
    )
    assert run["lost_at"] is None
    err = np.linalg.norm(run["r"] - run["ref"], axis=1)
    settled = [err[int(w / mpc.ts) - 1] for w in (2.0, 5.0, 8.0)]
    print(
        "step track  : settled " + ", ".join(f"{e*1e6:.1f} um" for e in settled)
    )
    assert max(settled) < 1e-4, settled

    acc = np.gradient(run["v"], mpc.ts, axis=0)
    jerk = np.gradient(acc, mpc.ts, axis=0)
    print(
        f"smoothness  : peak |a| {np.linalg.norm(acc,axis=1).max():.2f} m/s^2, "
        f"peak |j| {np.linalg.norm(jerk,axis=1).max():.1f} m/s^3"
    )
    assert np.linalg.norm(acc, axis=1).max() < 4.0 * GRAVITY

    solve = run["solve_s"]
    print(
        f"solve time  : median {np.median(solve)*1e3:.1f} ms, "
        f"p95 {np.percentile(solve,95)*1e3:.1f} ms, budget {mpc.ts*1e3:.0f} ms"
    )
    assert np.median(solve) < mpc.ts, np.median(solve)

    print("self-check PASS")


if __name__ == "__main__":
    _self_check()
