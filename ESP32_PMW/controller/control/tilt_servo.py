#!/usr/bin/env python3
"""Visual-servo attitude trim on the mast rig, and the 30% drop as a fraction of it.

    uv run python controller/control/tilt_servo.py                    # self-check, synthetic plant
    uv run python controller/control/tilt_servo.py --identify --hold-hz 100
    uv run python controller/control/tilt_servo.py --run --freqs 60,100,140

The rotor leans 3-5 deg under drive, toward a field dipole the coils make (theory.md
18.18). This holds it at its REST attitude instead -- the mast's direction with the coils
off, measured at the start of every run -- by trimming the four carrier duties from the
stereo pose. It is a slow auto-trim, not a stabiliser: the loop closes near 0.3 Hz, under
the 0.8 Hz latency rule (19.11) and far below the 4 Hz wobble, which is averaged.

Then the experiment. With the loop converged and FROZEN, one coil is dropped to
`DROP_FRAC` of its trimmed duty. At a fixed frequency one coil's drive is a linear RLC,
so its current scales with its duty: 30% of the trimmed duty IS 30% of the balanced
current on that coil (the 39% spread of 23.1 is between coils, not within one). The
telemetry ratio is logged as the check. theory.md 24.

WHY THE SIGN PROBLEM GOES AWAY
------------------------------
The mixer's sign was never verified (18.16) because the seated and airborne windows did
not overlap. Here the robot is seated on purpose, so each coil is probed in turn -- duty
100 -> 80 for 0.6 s -- and the lean response IS that coil's Jacobian column, sign and
all. A column under 3x the pre-probe scatter refuses the run; that is the sign
certification, done on the rig instead of assumed.

WHAT STOPS THE COILS
--------------------
Nothing automatic, anywhere (theory.md 4.0), except what this file adds: a runaway guard
(|lean| growing for `RUNAWAY_S`), a lost-pose guard (`LOST_S`), and `stop` from the
`finally` on every exit including SIGINT. `duty=` is the firmware's per-channel override
(`main_flight.cpp`), counted as drive by `link._note_drive`.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

from controller.control import constants as C
from controller.control import ramp
from controller.control.tilt_report import rest_basis
from controller.pose import disc_pose

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT_ROOT = ROOT / "results" / "tilt_servo"

# ---- numbers --------------------------------------------------------------------------
#: Datum: coils-off window before the ramp, and the most swing the mast may show in it.
#: 5 deg is the figure `tilt_report.datum` names as "swinging on its wire".
DATUM_S, DATUM_MAX_SPREAD_DEG = 5.0, 5.0
#: Identification probe per coil: duty 100 -> 100-PROBE_DROP for PROBE_S, the lean read
#: over the last PROBE_AVG_S. 20% is two thirds of the sweep's 30% step, which moved the
#: axis 3-5 deg at 100-140 Hz (normal_angle_f*.png) against ~0.5 deg of 0.25 s noise.
PROBE_DROP_PCT, PROBE_S, PROBE_AVG_S, PROBE_MIN_SNR = 20.0, 0.6, 0.3, 3.0
#: Loop: lean averaged over AVG_S, integral gain for a LOOP_HZ closed loop. The seated
#: plant answers a duty step in ~0.3 s (the sweep's steps), fast against 0.3 Hz, so the
#: closed loop is the integrator alone: pole at 2*pi*LOOP_HZ.
AVG_S, LOOP_HZ = 0.25, 0.3
#: Duty limits and the most total drop the trim may spend, so collective thrust stays
#: within ~10% of the untrimmed case. Anti-windup is the clamp itself (no integrator
#: state beyond the duties).
DUTY_MIN, DUTY_MAX, MAX_TOTAL_DROP_PCT = 40.0, 100.0, 40.0
#: Converged when |lean_avg| < CONVERGED_DEG for CONVERGED_S, or give up at TRIM_MAX_S
#: and freeze wherever it is (logged either way).
CONVERGED_DEG, CONVERGED_S, TRIM_MAX_S = 1.0, 2.0, 8.0
#: Guards. Runaway: |lean_avg| above RUNAWAY_GROWTH x its value RUNAWAY_S earlier while
#: over CONVERGED_DEG -> stop. Same shape as `attitude.TiltController`'s.
RUNAWAY_S, RUNAWAY_GROWTH, LOST_S = 1.0, 1.3, 1.0
#: The experiment: coil DROP_COIL to DROP_FRAC of its trimmed duty, held HOLD_S, then a
#: 4 s down-ramp and OFF_S off, as `spiffs_data/tilt.json`.
DROP_COIL, DROP_FRAC, HOLD_S, DOWN_S, OFF_S = 0, 0.30, 5.0, 4.0, 10.0
#: `main_flight.cpp` States.
IDLE, SPINUP, FLIGHT, LANDING, OFF = 0, 1, 2, 3, 4
COIL = "ABCD"


# ---- the loop's arithmetic, testable without a rig -------------------------------------
class LeanFilter:
    """Lean vector (deg) of the fused axis about the datum, averaged over ``avg_s``."""

    def __init__(self, up, avg_s=AVG_S):
        self.up = np.asarray(up, float) / np.linalg.norm(up)
        self.e1, self.e2 = rest_basis(self.up)
        self.avg_s = avg_s
        self.hist = []          # (t, l1, l2)

    def push(self, axis, t):
        n = np.asarray(axis, float)
        n = n / np.linalg.norm(n)
        if float(n @ self.up) < 0:
            n = -n
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(n @ self.up)))))
        lat = n - float(n @ self.up) * self.up
        nl = np.linalg.norm(lat)
        u = lat / nl if nl > 1e-9 else np.zeros(3)
        l = (tilt * float(u @ self.e1), tilt * float(u @ self.e2))
        self.hist.append((t, *l))
        self.hist = [h for h in self.hist if h[0] >= t - self.avg_s]
        return l

    def mean(self):
        if not self.hist:
            return None
        a = np.array(self.hist)[:, 1:]
        return a.mean(0)

    def scatter(self):
        if len(self.hist) < 4:
            return float("nan")
        a = np.array(self.hist)[:, 1:]
        return float(np.sqrt(((a - a.mean(0)) ** 2).sum(1).mean()))


def identify(probe_responses, baseline, drop_pct=PROBE_DROP_PCT):
    """``J`` (2x4, deg per % of drop) from one lean vector per coil probed at ``drop_pct``."""

    J = np.zeros((2, 4))
    for i, resp in enumerate(probe_responses):
        J[:, i] = (np.asarray(resp) - np.asarray(baseline)) / drop_pct
    return J


class TrimLoop:
    """Integral trim on four duties: ``u <- u + Ki * J^+ * lean`` (drop <- drop - ...), clamped."""

    def __init__(self, J, loop_hz=LOOP_HZ, u0=None):
        self.J = np.asarray(J, float)
        self.Jp = np.linalg.pinv(self.J)                 # 4x2, drops in % per deg of lean
        self.ki = 2 * math.pi * loop_hz                  # per second
        self.u = np.full(4, DUTY_MAX) if u0 is None else np.asarray(u0, float).copy()
        self.frozen = False

    def step(self, lean, dt):
        if self.frozen or lean is None or dt <= 0:
            return self.u
        # J maps DROP (%) to lean: lean = rest + J drop, drop = 100 - u. Cancelling the
        # lean means drop <- drop - Ki dt J^+ lean, which in duty is u <- u + Ki dt J^+ lean.
        u = self.u + self.ki * dt * (self.Jp @ np.asarray(lean, float))
        u = np.clip(u, DUTY_MIN, DUTY_MAX)
        total = (DUTY_MAX - u).sum()
        if total > MAX_TOTAL_DROP_PCT:               # spend the cap proportionally
            u = DUTY_MAX - (DUTY_MAX - u) * (MAX_TOTAL_DROP_PCT / total)
        self.u = u
        return self.u


class Runaway:
    """``check(lean_mag, t)`` is True when the lean has grown by RUNAWAY_GROWTH over RUNAWAY_S."""

    def __init__(self):
        self.hist = []

    def check(self, mag, t):
        self.hist.append((t, mag))
        self.hist = [h for h in self.hist if h[0] >= t - RUNAWAY_S]
        old = self.hist[0]
        return (t - old[0] >= RUNAWAY_S * 0.9 and mag > CONVERGED_DEG
                and mag > RUNAWAY_GROWTH * max(old[1], CONVERGED_DEG))


class SimPlant:
    """Seated rotor for the self-check: lean = rest + J (100 - u) + wobble + noise, first
    order at ``tau``. Coil signs are whatever ``J`` says; the loop must not care."""

    def __init__(self, J, rest=(3.0, -2.0), tau=0.3, noise=0.8, seed=0):
        self.J, self.rest, self.tau, self.noise = np.asarray(J, float), np.asarray(rest, float), tau, noise
        self.lean = self.rest.copy()
        self.rng = np.random.default_rng(seed)

    def step(self, u, dt, t):
        target = self.rest + self.J @ (DUTY_MAX - np.asarray(u, float))
        self.lean += (target - self.lean) * min(1.0, dt / self.tau)
        wobble = 0.7 * np.array([math.sin(2 * math.pi * 4 * t), math.cos(2 * math.pi * 4 * t)])
        return self.lean + wobble + self.rng.normal(0, self.noise, 2)


# ---- the run ----------------------------------------------------------------------------
def _wait_state(link, want, timeout_s, feed=None):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        link.drain()
        if link.state == want:
            return True
        if feed is not None and feed.exc is not None:
            raise feed.exc
        time.sleep(0.02)
    return False


def _fused_lean(est, tick, lf):
    """Push this tick's fused axis into ``lf``; None when the frame gave no axis."""

    pose = tick.pose
    if pose is None:
        return None
    n_world = pose.extra.get("world", (None, pose.normal))[1]
    mast = est.mast_world()
    planes = []
    if mast is None:
        for cam in est.rig.cameras:
            m = est.last_mast.get(cam.name)
            if m is not None:
                planes.append(disc_pose.mast_plane((m[0], m[1]), cam))
    ratio = None
    if pose.ellipse and np.isfinite(pose.ellipse[1][0]) and pose.ellipse[1][0] > 0:
        ratio = pose.ellipse[1][1] / pose.ellipse[1][0]
    got = disc_pose.fused_axis(n_world, mast=mast, planes=planes, ratio_min=ratio, ref=lf.up)
    if got is None:
        return None
    return lf.push(got[0], tick.t)


def run(port=None, freqs=(100.0,), identify_only=False, hold_hz=100.0, out_dir=None,
        camera="camera:0,camera:1", width=640, height=400, fps=210, ramp_s=15.0):
    from controller.control.hover_controller_runner import CommandLink, _PoseFeed
    from controller.viz import live_viz as lv

    try:
        from ai.thermal import coil_thermal
    except ImportError:
        raise SystemExit("ai/thermal/coil_thermal.py missing -- refusing to arm (CLAUDE.md Safety)")

    freqs = [hold_hz] if identify_only else list(freqs)
    per_point = ramp_s + 4 * PROBE_S + TRIM_MAX_S + (0 if identify_only else HOLD_S) + DOWN_S
    coil_thermal.wait_until_safe(per_point * len(freqs))

    out_dir = Path(out_dir or OUT_ROOT / time.strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "sweep.log"
    link = CommandLink(port, dry_run=False, log_path=str(log_path))
    est = disc_pose.live_estimator()
    ticks = lv.stereo_frames(specs=camera, width=width, height=height, fps=fps,
                             label="tilt servo", record=str(out_dir / "flight"),
                             zero=None, est=est)
    feed = _PoseFeed(ticks, threaded=True)
    fh = open(out_dir / "servo.csv", "w", newline="")
    w = csv.writer(fh)
    w.writerow(["t", "freq_hz", "phase", "lean1_deg", "lean2_deg", "fused", "dA", "dB", "dC", "dD",
                "iA", "iB", "iC", "iD"])

    def label(name):
        # Same line shape the schedule firmware prints, so `body_angle.timeline` and
        # `tilt_report` read this run like a sweep take.
        link.log.write(f"[{time.monotonic():.3f}] <- label={name}\n")

    def send_duty(u):
        link.send("duty=" + ":".join(f"{x:.1f}" for x in u))

    n_seen = 0

    def next_lean(lf, timeout_s=2.0):
        """Block for the next fresh tick; returns (lean or None, tick) or raises on loss."""
        nonlocal n_seen
        t0 = time.monotonic()
        while True:
            link.drain()
            if feed.exc is not None:
                raise feed.exc
            tick, n = feed.read()
            if n != n_seen and tick is not None:
                n_seen = n
                return _fused_lean(est, tick, lf), tick
            if time.monotonic() - t0 > timeout_s:
                raise RuntimeError(f"no pose for {timeout_s} s")
            time.sleep(0.002)

    try:
        for f in freqs:
            # ---- datum: coils off, mast only ---------------------------------------
            link.send("stop")
            link.send("duty=off")
            vs, t0 = [], time.monotonic()
            while time.monotonic() - t0 < DATUM_S:
                _, tick = next_lean(LeanFilter([0, -1, 0]))
                m = est.mast_world()
                if m is not None:
                    vs.append(m)
            if len(vs) < 50:
                raise SystemExit(f"only {len(vs)} mast fixes in the datum window; no datum")
            V = np.array(vs)
            up = V.mean(0)
            up /= np.linalg.norm(up)
            spread = float(np.sqrt(np.mean(np.degrees(np.arccos(np.clip(V @ up, -1, 1))) ** 2)))
            print(f"datum: {len(vs)} mast fixes, up=[{up[0]:+.3f} {up[1]:+.3f} {up[2]:+.3f}], "
                  f"spread {spread:.2f} deg")
            if spread > DATUM_MAX_SPREAD_DEG:
                raise SystemExit(f"datum spread {spread:.1f} deg > {DATUM_MAX_SPREAD_DEG}; not arming")
            fh.write(f"# datum: {up[0]:.5f} {up[1]:.5f} {up[2]:.5f} spread {spread:.2f} n {len(vs)}\n")
            lf = LeanFilter(up)

            # ---- ramp to f --------------------------------------------------------
            segs = ((C.RAMP_START_HZ, f, ramp_s, ramp.EASE, C.RAMP_K),)
            ramp.check(segs)
            link.takeoff_cmd = list(ramp.seq_lines(segs))
            link.arm()
            label(f"FREQ_{int(f):03d}HZ")
            if not _wait_state(link, FLIGHT, ramp_s + 5.0, feed):
                raise RuntimeError("firmware never reached FLIGHT")

            def log_row(phase, lean, u, fused):
                cur = link.currents or (float("nan"),) * 4
                w.writerow([f"{time.monotonic():.4f}", f, phase] +
                           ([f"{lean[0]:.3f}", f"{lean[1]:.3f}"] if lean is not None else ["", ""]) +
                           [fused] + [f"{x:.2f}" for x in u] + [f"{x:.2f}" for x in cur])

            # ---- identify: settle, baseline, four probes ---------------------------
            u = np.full(4, DUTY_MAX)
            send_duty(u)
            t0 = time.monotonic()
            while time.monotonic() - t0 < 1.5:
                lean, _ = next_lean(lf)
                log_row("settle", lean, u, 1)
            baseline, noise = lf.mean().copy(), lf.scatter()
            resp = []
            for i in range(4):
                u = np.full(4, DUTY_MAX)
                u[i] = DUTY_MAX - PROBE_DROP_PCT
                send_duty(u)
                t0 = time.monotonic()
                while time.monotonic() - t0 < PROBE_S:
                    lean, _ = next_lean(lf)
                    log_row(f"probe{COIL[i]}", lean, u, 1)
                resp.append(lf.mean().copy())
                snr = np.linalg.norm(resp[-1] - baseline) / max(noise, 1e-6)
                print(f"probe {COIL[i]}: response {np.linalg.norm(resp[-1] - baseline):.2f} deg, "
                      f"snr {snr:.1f}")
                if snr < PROBE_MIN_SNR:
                    raise SystemExit(f"coil {COIL[i]} response under {PROBE_MIN_SNR}x the noise; "
                                     f"no Jacobian, not closing the loop")
                send_duty(np.full(4, DUTY_MAX))
                t0 = time.monotonic()
                while time.monotonic() - t0 < 0.4:
                    next_lean(lf)
            J = identify(resp, baseline)
            fh.write("# J (deg per % drop, rows lean1,lean2; cols A,B,C,D): "
                     + " ".join(f"{x:.4f}" for x in J.ravel()) + "\n")
            print("J =\n", np.round(J, 3))
            if identify_only:
                link.send("land")
                _wait_state(link, OFF, DOWN_S + 3.0, feed)
                continue

            # ---- trim -------------------------------------------------------------
            loop = TrimLoop(J)
            guard = Runaway()
            t_start = t_prev = time.monotonic()
            t_ok = None
            while True:
                lean, tick = next_lean(lf)
                now = time.monotonic()
                lm = lf.mean()
                u = loop.step(lm, now - t_prev)
                t_prev = now
                send_duty(u)
                log_row("trim", lean, u, 1)
                mag = float(np.linalg.norm(lm)) if lm is not None else float("nan")
                if guard.check(mag, now):
                    raise RuntimeError(f"runaway: lean {mag:.1f} deg and growing")
                if mag < CONVERGED_DEG:
                    t_ok = t_ok or now
                    if now - t_ok >= CONVERGED_S:
                        break
                else:
                    t_ok = None
                if now - t_start > TRIM_MAX_S:
                    print(f"trim: not converged in {TRIM_MAX_S} s, freezing at |lean| {mag:.1f}")
                    break
            loop.frozen = True
            d_bal = loop.u.copy()
            link.drain()
            i_bal = link.currents
            fh.write("# d_bal: " + " ".join(f"{x:.2f}" for x in d_bal) + "\n")
            fh.write("# I_bal: " + (" ".join(f"{x:.3f}" for x in i_bal) if i_bal else "none") + "\n")
            print(f"balanced: duties {np.round(d_bal, 1)}, currents {i_bal}")

            # ---- the drop, loop frozen ---------------------------------------------
            u = d_bal.copy()
            u[DROP_COIL] = DROP_FRAC * d_bal[DROP_COIL]
            send_duty(u)
            label(f"DROP_{int(f):03d}HZ")
            t0 = time.monotonic()
            while time.monotonic() - t0 < HOLD_S:
                lean, _ = next_lean(lf)
                log_row("drop", lean, u, 1)
            link.drain()
            if i_bal and link.currents:
                r = abs(link.currents[DROP_COIL]) / max(abs(i_bal[DROP_COIL]), 1e-6)
                fh.write(f"# I_drop/I_bal on {COIL[DROP_COIL]}: {r:.3f} (target {DROP_FRAC})\n")
                print(f"current ratio on {COIL[DROP_COIL]}: {r:.3f} (target {DROP_FRAC})")
            label(f"DOWN_{int(f):03d}HZ")
            link.send("land")
            _wait_state(link, OFF, DOWN_S + 3.0, feed)
            link.send("duty=off")
            t0 = time.monotonic()
            while time.monotonic() - t0 < OFF_S:
                next_lean(LeanFilter(up))
        label("TILT_OFF")
    finally:
        link.send("stop")
        fh.close()
        link.close()
        feed.close()
    print(f"wrote {out_dir}")
    return out_dir


# ---- self-check ----------------------------------------------------------------------------
def _self_check():
    """The loop against a synthetic seated rotor: identification recovers the plant's
    signs including a flipped coil, the trim converges, the drop scales the coil, and the
    runaway guard fires on a diverging lean."""

    rng = np.random.default_rng(1)
    J_true = np.array([[0.20, 0.00, -0.20, 0.00],
                       [0.00, -0.18, 0.00, 0.18]])      # coil B "wrong sign"
    plant = SimPlant(J_true)
    up = np.array([0.0, -1.0, 0.0])
    lf = LeanFilter(up)
    dt, t = 1 / 100.0, 0.0

    def axis_from_lean(l):
        e1, e2 = rest_basis(up)
        tilt = math.hypot(*l)
        if tilt < 1e-9:
            return up
        u = (l[0] * e1 + l[1] * e2) / tilt
        return math.cos(math.radians(tilt)) * up + math.sin(math.radians(tilt)) * u

    def run_for(u, s):
        nonlocal t
        for _ in range(int(s / dt)):
            t += dt
            lf.push(axis_from_lean(plant.step(u, dt, t)), t)
        return lf.mean().copy()

    # identify
    run_for(np.full(4, DUTY_MAX), 1.5)
    base, noise = lf.mean().copy(), lf.scatter()
    resp = []
    for i in range(4):
        u = np.full(4, DUTY_MAX)
        u[i] = DUTY_MAX - PROBE_DROP_PCT
        resp.append(run_for(u, PROBE_S))
        run_for(np.full(4, DUTY_MAX), 0.4)
    J = identify(resp, base)
    assert np.all(np.sign(J[np.abs(J_true) > 0.1]) == np.sign(J_true[np.abs(J_true) > 0.1])), J
    assert np.abs(J - J_true).max() < 0.06, J
    # trim
    loop, guard = TrimLoop(J), Runaway()
    converged = None
    for k in range(int(TRIM_MAX_S / dt)):
        t += dt
        lf.push(axis_from_lean(plant.step(loop.u, dt, t)), t)
        loop.step(lf.mean(), dt)
        mag = float(np.linalg.norm(lf.mean()))
        assert not guard.check(mag, t), "guard fired on a converging loop"
        if mag < CONVERGED_DEG:
            converged = converged or t
            if t - converged > CONVERGED_S:
                break
        else:
            converged = None
    assert converged is not None and t - converged > CONVERGED_S, "did not converge"
    assert np.all(loop.u >= DUTY_MIN) and np.all(loop.u <= DUTY_MAX)
    # rest lean (+3, -2): lean1 needs -3 = J[0,C]*drop -> ~15% off C; lean2 needs +2 =
    # J[1,D]*drop -> ~11% off D. A and B could only help by ADDING duty, which is clamped.
    assert 100 - loop.u[2] > 10 and 100 - loop.u[3] > 6 and 100 - loop.u[0] < 2, loop.u
    # drop: frozen, coil A to 30% of its trimmed duty
    loop.frozen = True
    u = loop.u.copy()
    u[0] = DROP_FRAC * loop.u[0]
    assert np.allclose(loop.step(np.array([5.0, 5.0]), dt), loop.u), "frozen loop moved"
    lean_drop = run_for(u, 1.0)
    assert np.linalg.norm(lean_drop) > 5, lean_drop          # a big, real response
    # runaway
    g = Runaway()
    fired = any(g.check(2.0 * 1.5 ** (i * dt), i * dt) for i in range(int(3.0 / dt)))
    assert fired, "runaway guard never fired on a doubling lean"
    print("tilt_servo: self-check passed (J recovered with a flipped coil, trim converged "
          f"to {np.round(loop.u, 1)}, frozen drop, runaway guard)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--identify", action="store_true", help="probe the four coils at --hold-hz, no trim")
    ap.add_argument("--hold-hz", type=float, default=100.0)
    ap.add_argument("--freqs", default="100", help="comma list, Hz")
    ap.add_argument("--port", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.run or a.identify:
        run(port=a.port, freqs=[float(x) for x in a.freqs.split(",")], identify_only=a.identify,
            hold_hz=a.hold_hz, out_dir=a.out)
    else:
        _self_check()
