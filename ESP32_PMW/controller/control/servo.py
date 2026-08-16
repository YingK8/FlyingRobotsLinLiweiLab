"""Visual-servo host: frame -> height -> commanded rotation frequency -> ESP32.

CAMERA-FREE BY DESIGN. Nothing here opens a VideoCapture, reads a frame or
shows a window: the notebook (servo.ipynb) owns the camera and the loop and
feeds frames in. That keeps this module importable and testable with no
hardware attached, and lets capture settings -- fourcc, resolution, gray
conversion -- be retuned in a cell without touching the control path.

Frames may be colour or single-channel; feeding gray costs one cvtColor and
makes MOG2 and both morphology passes run on 1 channel instead of 3.

    # You set the mask (THRESH, MIN_BLOB_AREA_PX, MORPH_KERNEL) and the robot's
    # diameter. Scale and datum come off the frames -- no calibration pass.
    step = altitude_hold(link, height_controller())
    with coils_on(link):            # the ONLY thing that guarantees coils off
        while ...:
            row, d = step(frame, z_ref_mm)   # setpoint is the loop's, per frame

takeoff is handled by accumulated steady-error in the incremental PID;
increasing frequency until liftoff.
"""

from __future__ import annotations

import math
import sys
import time
from collections import namedtuple
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
# Pipeline layering: a stage sees only the stages before it, so a forward import
# fails at once instead of quietly creating a cycle. control is stage 4 of 4.
sys.path[:0] = [str(HERE), str(HERE.parent / "pose"), str(HERE.parent / "calib"), str(HERE.parent / "camera")]
from link import SerialComm  # noqa: E402  (controller/control/link.py)
from z_track import TorqueLimits  # noqa: E402  (phase-lock torque budget)

# flight constraints:
F_MIN = 1.0
CMD_PERIOD_S = 0.02  # 50 Hz command rate

# --- Vision ------------------------------------------------------------------
MIN_BLOB_AREA_PX = 30  # reject specks / sensor noise
MORPH_KERNEL = 5  # px, open+close kernel to despeckle the mask
# Grey level splitting the white robot from the black ground. Fixed rather than
# Otsu: with the robot out of frame Otsu would happily threshold sensor noise
# into a "detection", whereas a fixed level just returns None. Tune it in the
# notebook by eye -- look at the mask, not at this number.
THRESH = 128

# Metric scale is MEASURED, not configured: the blob's extent along its first
# principal component is the robot's projected diameter, so
# px_per_mm = axis_px / DRONE_DIAMETER_MM. At 20 mm the pixel budget is tight --
# scale error is 1/axis_px per pixel of extent error, so a robot spanning 40 px
# turns one bad pixel into 2.5% of height. Frame the camera so it spans >= ~60 px.
DRONE_DIAMETER_MM = 20.0  # CALIBRATE: robot's physical diameter (mm)


def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


# =============================================================================
# SERIAL -- src/main_servo.cpp: F<hz> | A<ch>,<pct> | S
# =============================================================================


def connect(port: str | None = None, reset: bool = True) -> SerialComm:
    """Open the link. `reset` pulses EN so the board starts from a known state."""
    link = SerialComm(port)
    if reset:
        link.reset_device()
        time.sleep(2.0)  # ESP32 boot
    return link


def poll(link: SerialComm, out: str = "") -> dict | None:
    """Send `out`, drain every pending line, return the newest telemetry.

    Drains rather than taking one line per call: the firmware prints faster than
    a single-line read consumes, so the rest backs up and what you act on goes
    stale. Parses driveTelemetry's shared 2 Hz line; we only read freq and trip.
    """
    tel = None
    line = link.handle_serial_comm(out)
    while line is not None:
        if line.startswith("t="):
            tel = {}
            for part in line.replace("|", " ").split():
                k, _, v = part.partition("=")
                try:
                    tel[k] = float(v)
                except ValueError:
                    pass
        elif line.startswith(("!", "?")):
            print("firmware rejected:", line)
        line = link.handle_serial_comm()
    return tel


def set_carriers(link: SerialComm, duty=100.0):
    """One duty for all four coils, or a 4-list for per-coil (trim)."""
    duties = [duty] * 4 if np.isscalar(duty) else duty
    for ch, d in enumerate(duties):
        poll(link, f"A{ch},{clamp(d, 0.0, 100.0):.1f}")


def stop(link: SerialComm):
    """Coils off. Always reachable, including from a finally block."""
    poll(link, "S")


@contextmanager
def coils_on(link: SerialComm, duty=100.0):
    """Carriers up for the body of the `with`, and unconditionally down after.

    The firmware has no watchdog, so this is the only thing that brings the
    coils down when the loop exits -- including on a kernel interrupt. The loop
    lives in a notebook cell now, so KEEP THE WHOLE LOOP INSIDE ONE `with` IN
    ONE CELL: split across cells, interrupting leaves the coils energised with
    nothing driving them.
    """
    set_carriers(link, duty)
    try:
        yield
    finally:
        stop(link)


# =============================================================================
# VISION -- bright blob on a dark ground -> PCA ellipse -> height (mm)
#
# White robot, black background, mono camera. That premise is what keeps this
# simple: a fixed threshold segments it outright, so there is no background
# model, nothing to settle, and no requirement that the robot be MOVING to be
# seen. It can sit on the pad indefinitely and still be tracked -- which is what
# makes the datum meaningful.
# =============================================================================

# p0/p1 are the endpoints of the principal axis, computed once in detect() so
# annotate() never has to redo the PCA.
Det = namedtuple("Det", "u v axis_px p0 p1")

_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))


def detect(frame, thresh: int = THRESH, min_area: float = MIN_BLOB_AREA_PX) -> Det | None:
    """Brightest blob on a dark ground -> Det, or None.

    (u, v) is the CENTRE OF THE FITTED ELLIPSE -- the PCA mean of the contour
    points -- not the filled-region centroid. The two differ whenever the mask
    is asymmetric, and the ellipse centre is the one that stays put when a limb
    of the blob is clipped by threshold or frame edge, so it is the stabler
    thing to servo on.

    axis_px is the blob's extent along the first principal component: the
    robot's projected diameter, and the only thing that sets metric scale. It is
    tilt-robust, since a disk tilted by theta projects to an ellipse whose major
    axis is still the diameter and whose minor axis is diameter*cos(theta).

    Colour or single-channel frames both work; mono skips the conversion.
    """
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area or len(c) < 5:
        return None
    pts = c.reshape(-1, 2).astype(np.float64)
    mean, eigvec = cv2.PCACompute(pts, mean=None, maxComponents=1)
    axis_px = float(np.ptp((pts - mean) @ eigvec[0]))
    off = eigvec[0] * (axis_px / 2.0)
    return Det(mean[0, 0], mean[0, 1], axis_px, mean[0] - off, mean[0] + off)


def velocity(z, z_prev, dt):
    """Finite-difference vertical velocity, straight off the image stream.

    No filter and no state: every number the controller sees comes from a frame.
    The cost is real and lands on the D term -- kd multiplies a difference OF
    this difference, so centroid jitter enters the command squared. Measured on
    a 0.5 px centroid noise floor, this is ~10x the D-term jitter a Kalman
    estimate gives, so keep kd small (~0.02, not 0.15) and lean on kp/ki.
    """
    if z is None or z_prev is None or dt <= 0.0:
        return None
    return (z - z_prev) / dt


def _pt(p):
    return int(p[0]), int(p[1])


def annotate(frame, d: Det | None, z=None, zdot=None):
    """Draw the detection, its principal axis and the height readout.

    USE THE RETURN VALUE: a single-channel frame is converted to BGR first, so
    the coloured overlays survive, and that conversion is a copy -- the drawing
    does not land on the caller's array. `d=None` returns the frame undrawn, so
    a display loop can call this unconditionally.
    """
    vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame
    if d is None:
        return vis
    cv2.circle(vis, _pt((d.u, d.v)), 6, (0, 255, 0), 2)
    cv2.line(vis, _pt(d.p0), _pt(d.p1), (0, 200, 255), 2)
    txt = f"axis={d.axis_px:5.1f}px"
    if z is not None:
        # zdot is None on the first frame after any detection gap while z is
        # already a float, so it needs its own fallback -- formatting None as
        # a float raises.
        txt += f"  z={z:6.1f}mm  v={zdot or 0.0:6.1f}mm/s"
    cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return vis


# =============================================================================
# CONTROL
# =============================================================================


def height_controller(
    kp: float = 0.2,
    ki: float = 0.15,
    kd: float = 0.01,
    limits: "TorqueLimits" = None,
    f_start: float = F_MIN,
):
    """Incremental (velocity) PID -> update(z_ref, z, zdot, dt) -> freq (Hz).

    Commands frequency changes:

        u += kp*(err - err_prev) + ki*err*dt - kd*(zdot - zdot_prev)

    The hover frequency never appears, so nothing here has to know it -- it is
    implicit in the accumulation and the loop walks to whatever it actually is.

    TAKEOFF IS THE SAME MECHANISM. Sitting on the pad the robot has a large
    steady positive error, ki accumulates it, and u climbs until it lifts. There
    is no separate spin-up ramp, and no takeoff frequency to guess.

    THE ONLY LIMIT IS ON RATE, and it comes from the physics rather than a
    hand-set band: TorqueLimits.f_dot_max(f) is the fastest the field may change
    while the magnet's spin-up torque still fits under the torque budget. Exceed
    it and the magnet steps out of sync -- which is what an absolute frequency
    cap was really standing in for. It vanishes at f_ceiling (167.3 Hz), so the
    rate cap enforces the ceiling by itself: an aggressive demand ramps up and
    then eases to a halt just underneath rather than slamming into it. The floor
    is the one absolute that stays, and it isn't a tuning choice: the firmware
    reads f <= 0 as DC, which stops the field and drops the robot.

    ki must be non-zero -- in incremental form the other terms are pure
    differences, so a steady error produces no correction at all without it, and
    the robot would never leave the pad. kd is not optional either: magnetic
    hover is lightly damped and rings without it.

    A closure, not a class, so re-running a notebook cell starts from clean state.
    """
    lim = limits or TorqueLimits()
    ceiling = lim.f_ceiling()  # derived from the same torque budget, not hand-set
    u = clamp(f_start, F_MIN, ceiling)
    err_prev = zdot_prev = None

    def update(z_ref, z, zdot, dt):
        nonlocal u, err_prev, zdot_prev
        if z is None:
            return u  # lost the robot: hold what we had
        err, zdot = z_ref - z, (zdot or 0.0)
        if err_prev is None:  # first sample: no difference to take yet
            err_prev, zdot_prev = err, zdot
            return u
        du = kp * (err - err_prev) + ki * err * dt - kd * (zdot - zdot_prev)
        err_prev, zdot_prev = err, zdot
        # Asymmetric on purpose. Spinning the rotor UP has to fight drag, so the
        # budget is (s_lim*tau_max - k_drag*f^2); spinning it DOWN is HELPED by
        # drag, so the same torque buys (s_lim*tau_max + k_drag*f^2). At the
        # ceiling that is 0.08 Hz/s up against ~890 Hz/s down -- limiting both
        # by f_dot_max would strand the controller at the ceiling, unable to
        # descend, which is precisely where you most need it to.
        drag = lim.k_drag * u * u
        budget = lim.s_lim * lim.tau_max(u)
        span = 2.0 * math.pi * lim.i_robot / dt
        u = clamp(
            u + clamp(du, -(budget + drag) / span, (budget - drag) / span),
            F_MIN,
            ceiling,
        )
        return u

    return update


# =============================================================================
# RUNNER -- what the notebook drives
# =============================================================================


def altitude_hold(link, controller, diameter_mm: float = DRONE_DIAMETER_MM,
                  thresh: int = THRESH):
    """Closed loop, one frame at a time -> step(frame, z_ref_mm) -> (row, det).

    No calibration pass. Scale comes from each frame -- the robot's known
    diameter over its measured pixel extent -- and the datum latches on the
    first detection, so z = 0 is wherever the robot was when the loop started.
    Park it on the pad and heights read as height above the pad.

    THE NOTEBOOK OWNS THE LOOP AND THE SETPOINT. This owns only what has to
    survive between frames -- the datum, z_prev, the command clock, the run
    clock -- which as loose notebook variables would go stale the moment a cell
    is re-run, and silently: a wrong z_prev is a wrong velocity, not an
    exception. Passing z_ref_mm per call is what lets you walk the target, or
    command a step response, without rebuilding the closure and losing the PID's
    accumulated state with it.

    Drive it inside `with coils_on(link):` -- the firmware has no watchdog and
    that context manager is the only thing that brings the coils down on a
    kernel interrupt.

    `row` is a dict ready to append to a list and hand to a DataFrame; `det` is
    the detection, for annotate().
    """
    datum_v = None
    z_prev = None
    t0 = last_cmd = last_loop = time.monotonic()

    def step(frame, z_ref):
        nonlocal datum_v, z_prev, last_cmd, last_loop
        now = time.monotonic()
        dt, last_loop = now - last_loop, now

        d = detect(frame, thresh)
        if d is not None and datum_v is None:
            datum_v = d.v                      # first sighting defines z = 0
        # px_per_mm = axis_px / diameter_mm, so dividing by it is multiplying by
        # diameter_mm / axis_px. v grows downward, so a higher robot is a
        # smaller v -- negate.
        z = None if d is None else -(d.v - datum_v) * diameter_mm / d.axis_px
        zdot = velocity(z, z_prev, dt)
        z_prev = z

        u = controller(z_ref, z, zdot, dt)
        due = now - last_cmd >= CMD_PERIOD_S
        tel = poll(link, f"F{u:.2f}") if due else poll(link)
        if due:
            last_cmd = now

        return {
            "t": now - t0,
            "z_mm": z,
            "z_ref": z_ref,
            "zdot": zdot,
            "u_hz": u,
            "axis_px": None if d is None else d.axis_px,
            "freq_meas": (tel or {}).get("freq"),
        }, d

    return step
