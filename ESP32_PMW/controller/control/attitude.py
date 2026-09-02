#!/usr/bin/env python3
"""Thrust-vector attitude: estimate where the thrust points, and steer it upright.

This is the attitude term of the 5-DOF controller, built at the seam that already exists.
`hover_controller_runner` forms a Cartesian lateral command `(ux, uy)` and only then
converts to the actuator's polar form. An attitude correction is another Cartesian term
added there, so position and attitude share one actuator mapping -- which is what makes
this extend to 5-DOF rather than sit beside it.

WHY THE SENSOR IS ACCELERATION, NOT THE POSE NORMAL
---------------------------------------------------
The estimator reports a rotor normal (`Pose.theta_deg`/`phi_deg`) and it is the obvious
signal to close on. Measured on a stationary robot it has **1 sigma = 9.5 deg against a
median tilt of 1.1-1.8 deg** -- noise 5-8x the signal. Averaging barely helps because the
defect is geometric, not statistical: `calibrate_zero` records tilt sensitivity going as
1/sin(theta), and at ~1.5 deg that is deep in the ill-conditioned regime.

    averaging window | from the pose normal | from lateral acceleration
        0.25 s       |       4.67 deg       |       0.49 deg
        0.50 s       |       3.38 deg       |       0.11 deg
        1.00 s       |       2.54 deg       |       0.02 deg

So the tilt is taken from the position track instead, which has 0.05 mm of scatter:

    tilt ~ atan( |a_lat| / (a_z + g) )

10-30x better, and it is the physically correct quantity: we steer the THRUST direction,
and `a_lat/(a_z+g)` is that direction rather than a proxy for it.

**Valid only while airborne.** Seated on the rod, the pad's reaction cancels lateral
acceleration and this reads ~0 whatever the true tilt is. Verified: over the 95-110 Hz band
of four flights the derived tilt vector is 0.006-0.054 deg, indistinguishable from zero,
because the robot was still on the pad. `ThrustVector.valid` is that gate, and it is not
advisory.

THE ROTATION THAT MUST BE MEASURED FIRST
----------------------------------------
`applyMixer`'s "Verify sign on rig" is still unverified. `theory.md` 20.5 prices the
consequence exactly: both lateral axes share one gain row, so a rotation between commanded
and realised direction is a complex loop gain, and the certified region has angular extent
**|psi| < 69.4 deg** -- collapsing fast, with the low-gain margin going from 21.9 dB to
0.9 dB by 67.5 deg. 12.8 measured the spatial model responding **72 deg** from the command,
outside that. So `TiltController.rot_deg` has NO default: the loop stays disabled until
`fit_rotation` returns a number. 69.4 deg is a loose target, which is why a coarse
in-flight identification suffices.

    uv run python controller/control/attitude.py
"""

from __future__ import annotations

import math

import numpy as np

G_MM_S2 = 9806.65

# Savitzky-Golay window for the second derivative. 0.25 s resolves 0.49 deg against a ~1 deg
# signal, and the drift being corrected is under 2 Hz (theory.md 18.14), so this is fast
# enough to act on it. Longer windows resolve better and lag more; the airborne window is
# only 1-2 s, so there is no room to spend.
WINDOW_S = 0.25
LOOP_HZ = 200.0                  # the grid the buffer is resampled onto

AIRBORNE_MM = 0.3                # z above pad before the sensor means anything
MIN_AZ_FRAC = 0.5                # |a_z + g| must exceed this * g, or the division is ill-posed

# Runaway guard. A wrong `rot_deg` makes the loop amplify; this catches that and falls back
# to the fixed feedforward trim rather than to nothing.
GUARD_S = 1.0                    # window over which tilt must not grow
GUARD_GROWTH = 1.3               # tilt this many times its starting value = runaway

MAX_ROT_ERR_DEG = 69.4           # theory.md 20.5: outside this the loop is not certifiable


def _wrap180(deg):
    return (deg + 180.0) % 360.0 - 180.0


class ThrustVector:
    """Thrust direction from the position track, as a tilt vector in the datum frame.

    Feed it every pose fix; ask for `tilt_xy` when `valid`. Degrees, (tx, ty), where the
    magnitude is the angle off vertical and the direction is the way the thrust leans.
    """

    def __init__(self, window_s=WINDOW_S, loop_hz=LOOP_HZ, z_pad_mm=0.0):
        self.window_s, self.loop_hz, self.z_pad_mm = window_s, loop_hz, z_pad_mm
        self._t: list[float] = []
        self._p: list[np.ndarray] = []
        self._tilt = None
        self._reason = "no samples yet"

    def update(self, xyz_mm, t):
        """One pose fix. Positions in mm, `t` the capture stamp in seconds."""

        if xyz_mm is None:
            return
        self._t.append(float(t))
        self._p.append(np.asarray(xyz_mm, dtype=float))
        # Keep a little more than the window so the SG fit has both edges.
        while self._t and self._t[-1] - self._t[0] > 1.6 * self.window_s:
            self._t.pop(0)
            self._p.pop(0)
        self._tilt, self._reason = self._solve()

    def _solve(self):
        if len(self._t) < 8:
            return None, "window not filled"
        span = self._t[-1] - self._t[0]
        if span < self.window_s:
            return None, "window not filled"
        # Resample onto a uniform grid: the fixes arrive at ~65 Hz but not evenly, and
        # savgol assumes a constant spacing. Interpolating here is what lets the same
        # filter run on a jittery source.
        from scipy.signal import savgol_filter

        ti = np.arange(self._t[0], self._t[-1], 1.0 / self.loop_hz)
        if ti.size < 9:
            return None, "window not filled"
        P = np.vstack([np.interp(ti, self._t, [p[k] for p in self._p]) for k in range(3)])
        w = min(int(self.window_s * self.loop_hz) // 2 * 2 + 1, (ti.size // 2) * 2 - 1)
        if w < 5:
            return None, "window not filled"
        acc = savgol_filter(P, w, 2, deriv=2, delta=1.0 / self.loop_hz, axis=1)
        ax, ay, az = (float(np.median(a)) for a in acc)
        # Airborne? On the pad the reaction cancels a_lat and this reads zero regardless.
        if float(np.median(P[2])) - self.z_pad_mm < AIRBORNE_MM:
            return None, "on the pad -- lateral acceleration is cancelled by the reaction"
        denom = az + G_MM_S2
        if denom < MIN_AZ_FRAC * G_MM_S2:
            return None, "a_z + g too small -- the tilt division is ill-conditioned"
        return (math.degrees(ax / denom), math.degrees(ay / denom)), ""

    @property
    def tilt_xy(self):
        """(tx, ty) in degrees, or None. Never a number when `valid` is False."""

        return self._tilt

    @property
    def valid(self):
        return self._tilt is not None

    @property
    def reason(self):
        """Why `tilt_xy` is None. Empty when it is not."""

        return self._reason


class TiltController:
    """Proportional on the thrust-vector error, with a runaway guard.

    Output is a Cartesian correction in the same frame and units as the position loop's
    `(ux, uy)`, so the runner adds the two before forming `az`/`mag`.

    `rot_deg` has NO default. Until `fit_rotation` measures it the loop refuses to act --
    theory.md 20.5 gives only 69.4 deg of tolerance and 12.8 measured 72 deg on a related
    model, so a guess is not a small risk here.
    """

    def __init__(self, gain=0.02, rot_deg=None, guard_s=GUARD_S, max_out=0.5):
        self.gain, self.rot_deg, self.guard_s, self.max_out = gain, rot_deg, guard_s, max_out
        self.enabled = rot_deg is not None
        self.tripped = False
        self._hist: list[tuple[float, float]] = []   # (t, |tilt|) while commanding

    def step(self, tilt_xy, t):
        """(cmd_x, cmd_y), or (0, 0) when disabled, tripped, or handed None."""

        if tilt_xy is None or not self.enabled or self.tripped:
            return 0.0, 0.0
        tx, ty = tilt_xy
        mag = math.hypot(tx, ty)

        # Runaway guard: a wrong `rot_deg` makes this loop amplify the very thing it is
        # correcting, and the failure looks exactly like "the trim did not help". Watch the
        # tilt while commanding and give up rather than push harder.
        self._hist.append((t, mag))
        self._hist = [(ti, m) for ti, m in self._hist if t - ti <= self.guard_s]
        if len(self._hist) > 4 and t - self._hist[0][0] >= self.guard_s * 0.9:
            if self._hist[-1][1] > GUARD_GROWTH * max(self._hist[0][1], 1e-6):
                self.tripped = True
                return 0.0, 0.0

        # Oppose the tilt, rotated into actuator coordinates by the MEASURED rotation.
        # MINUS, not plus: the mixer APPLIES `rot_deg`, so the command must be pre-rotated
        # backwards for the response to land opposite the tilt. `mixer_sign.py` prints the
        # same convention -- "a trim must use az = P + 180 - off". Getting this backwards
        # doubles the tilt, which is why the guard below exists.
        a = math.atan2(ty, tx) + math.pi - math.radians(self.rot_deg)
        out = min(self.gain * mag, self.max_out)
        return out * math.cos(a), out * math.sin(a)

    def reset(self):
        self.tripped = False
        self._hist.clear()


def fit_rotation(records):
    """Measure the mixer rotation from (commanded az, tilt-vector response) pairs.

    `records` is [(az_deg, dtx, dty), ...] -- the commanded weak direction and the CHANGE
    in the acceleration-derived tilt vector it produced. Returns (rot_deg, confidence),
    confidence in [0, 1] as the circular concentration of the per-sample estimates.

    **Refuses rather than guesses.** Returns (None, conc) when the responses are within the
    noise floor. The failure this exists to prevent is fitting a rotation to 0.01 deg of
    nothing, which is exactly what the existing flight data yields: over the 95-110 Hz band
    the measured response is 0.006-0.054 deg against a 0.49 deg resolution, so any angle
    read off it is noise wearing a number's clothes.
    """

    good = [(a, dx, dy) for a, dx, dy in records if math.hypot(dx, dy) > 0.49]
    if len(good) < 2:
        return None, 0.0
    ang = [math.radians(math.degrees(math.atan2(dy, dx)) - a) for a, dx, dy in good]
    c, s = float(np.mean(np.cos(ang))), float(np.mean(np.sin(ang)))
    conc = math.hypot(c, s)
    if conc < 0.5:
        return None, conc
    return math.degrees(math.atan2(s, c)) % 360.0, conc


class _Plant:
    """Toy rotor for the self-check: a tilt that grows with thrust, steered through an
    unknown rotation. Not a model of anything -- just enough to exercise the loop."""

    def __init__(self, rot_deg, bias=(2.0, 1.0), authority=8.0):
        self.rot, self.bias, self.authority = rot_deg, np.array(bias, float), authority
        self.tilt = self.bias.copy()

    def step(self, cmd_xy, dt):
        a = math.radians(self.rot)
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        self.tilt += dt * (self.bias - self.tilt) + self.authority * dt * (R @ np.array(cmd_xy))
        return tuple(self.tilt)


def demo():
    # --- the estimator refuses while seated, and answers once airborne ------------------
    tv = ThrustVector(z_pad_mm=0.0)
    for i in range(120):                     # parked on the pad: no motion at all
        tv.update(np.array([0.0, 0.0, 0.0]), i / 65.0)
    assert not tv.valid, "read a tilt from a robot sitting on the pad"
    assert "pad" in tv.reason, tv.reason

    tv = ThrustVector(z_pad_mm=0.0)
    ax_true, t = 0.079 * G_MM_S2, 0.0        # 0.079 g lateral = the measured 4.6 deg
    for i in range(160):
        t = i / 200.0
        tv.update(np.array([0.5 * ax_true * t * t, 0.0, 5.0]), t)
    assert tv.valid, tv.reason
    tx, ty = tv.tilt_xy
    assert abs(tx - 4.52) < 0.6, f"tilt {tx:.2f} deg, expected ~4.5"
    assert abs(ty) < 0.3, ty

    # --- the controller converges when the rotation is right ---------------------------
    for rot in (0.0, 90.0, 215.0):
        plant, ctl = _Plant(rot_deg=rot), TiltController(gain=0.05, rot_deg=rot)
        tilt = plant.tilt.copy()
        for i in range(400):
            tilt = plant.step(ctl.step(tuple(tilt), i / 200.0), 1 / 200.0)
        assert math.hypot(*tilt) < 0.9 * math.hypot(*plant.bias), (rot, tilt)

    # --- and the guard catches a rotation that is 180 deg wrong ------------------------
    plant = _Plant(rot_deg=0.0)
    ctl = TiltController(gain=0.05, rot_deg=180.0)          # exactly backwards
    tilt = plant.tilt.copy()
    for i in range(600):
        tilt = plant.step(ctl.step(tuple(tilt), i / 200.0), 1 / 200.0)
    assert ctl.tripped, "the guard let a 180 deg rotation error keep commanding"

    # --- fit_rotation refuses noise, and finds a real rotation -------------------------
    rot, conc = fit_rotation([(0.0, 0.01, 0.0), (90.0, 0.0, 0.02)])
    assert rot is None, f"fitted {rot} deg to responses below the noise floor"
    truth = 215.0
    recs = []
    for az in (0.0, 90.0, 180.0, 270.0):
        a = math.radians(az + truth)
        recs.append((az, 3.0 * math.cos(a), 3.0 * math.sin(a)))
    rot, conc = fit_rotation(recs)
    assert rot is not None and abs(_wrap180(rot - truth)) < 1.0, (rot, conc)
    assert conc > 0.99, conc
    assert abs(_wrap180(rot - truth)) < MAX_ROT_ERR_DEG, "outside theory.md 20.5's certified band"

    print(f"attitude: tilt {tx:.2f} deg from acceleration (truth 4.52); loop converges at "
          f"0/90/215 deg; guard trips on a 180 deg error; fit {rot:.0f} deg (conc {conc:.2f})"
          f"\n  ok")


if __name__ == "__main__":
    demo()


def fit_rotation_from_csv(paths):
    """`fit_rotation` over one or more flight CSVs written with the identification dither.

    Segments each run by the commanded `az`, takes the median thrust-tilt vector inside each
    segment (skipping the first 0.25 s, the estimator's own window, so a sample never
    straddles two commands), and regresses the CHANGE between consecutive segments against
    the change in command.
    """

    from controller.control import takeoff_report as R

    recs = []
    for p in paths:
        d = R.load(p)
        if "acc_tilt_x" not in d:
            print(f"  {p}: no acc_tilt columns -- logged before 2026-09-01, skipped")
            continue
        t, az = d["t"], d["az"]
        tx, ty = d["acc_tilt_x"], d["acc_tilt_y"]
        ok = np.isfinite(t) & np.isfinite(az) & np.isfinite(tx) & np.isfinite(ty)
        if ok.sum() < 20:
            print(f"  {p}: {int(ok.sum())} airborne samples with a command -- too few")
            continue
        t, az, tx, ty = t[ok], az[ok], tx[ok], ty[ok]
        edges = np.flatnonzero(np.diff(az) != 0) + 1
        segs = []
        for a, b in zip(np.r_[0, edges], np.r_[edges, len(t)]):
            m = t[a:b] - t[a] > WINDOW_S          # skip the estimator's own window
            if m.sum() >= 5:
                segs.append((float(az[a]), float(np.median(tx[a:b][m])),
                             float(np.median(ty[a:b][m]))))
        for (a0, x0, y0), (a1, x1, y1) in zip(segs, segs[1:]):
            if a1 != a0:
                recs.append((a1, x1 - x0, y1 - y0))
    if not recs:
        print("  no usable command changes -- the dither never ran while airborne")
        return None, 0.0
    rot, conc = fit_rotation(recs)
    if rot is None:
        print(f"  REFUSED: {len(recs)} response(s), all within the 0.49 deg noise floor "
              f"(concentration {conc:.2f}). Not fitting a rotation to nothing.")
    else:
        print(f"  rot_deg = {rot:.0f}  (concentration {conc:.2f}, {len(recs)} responses)"
              + ("" if conc > 0.7 else "  -- WEAK, do not close the loop on this"))
    return rot, conc
