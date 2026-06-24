"""Lateral position hold (v2): downward camera -> (x, y) -> 2-D tilt command.

A second, downward-looking camera sees the robot in the horizontal plane. We keep
it centred by TILTING the thrust vector: this module turns pixel position error
into a 2-D tilt command, which mixer.py then converts to per-coil carrier duties.

Lateral position is a DOUBLE INTEGRATOR (tilt -> horizontal force -> accel -> vel
-> pos), so each axis is PD -- the derivative (velocity) term is mandatory; a pure
proportional map oscillates and diverges. Control runs in PIXELS and the setpoint
is the frame centre (no metric scale needed to stay centred).

Detection reuses vision.BlobDetector (MOG2 background subtraction) and the 1-D
constant-acceleration vision.KalmanZ, run once per axis (its fields are duck-typed
against LateralKalman, which is tuned in pixels).
"""
import time

import cv2

from config import (
    CAMERA_INDEX_DOWN, DOWN_FRAME_WIDTH, DOWN_FRAME_HEIGHT, FPS,
    DEFAULT_LATERAL_GAINS, DEFAULT_LATERAL_KALMAN, LateralGains,
)
from vision import BlobDetector, KalmanZ


def open_down_camera(index: int = CAMERA_INDEX_DOWN) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DOWN_FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DOWN_FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open downward camera index {index}")
    return cap


class LateralTracker:
    """Downward camera -> filtered (x, y) position and (xdot, ydot), all in pixels."""

    def __init__(self, cap: cv2.VideoCapture, draw: bool = False,
                 detector: BlobDetector = None,
                 kalman_params=DEFAULT_LATERAL_KALMAN):
        self.cap = cap
        self.draw = draw
        self.detector = detector or BlobDetector()
        self.kf_x = KalmanZ(kalman_params)   # filters raw pixel u (no negation)
        self.kf_y = KalmanZ(kalman_params)   # filters raw pixel v
        self._last_t = time.monotonic()

    def step(self):
        """Return (x, xdot, y, ydot, raw_uv_or_None, frame). Positions are None
        until the first detection locks."""
        ok, frame = self.cap.read()
        if not ok:
            return None, None, None, None, None, None

        now = time.monotonic()
        dt = now - self._last_t
        self._last_t = now

        det = self.detector.detect(frame)
        xm = ym = None
        if det is not None:
            u, v, _area = det
            xm, ym = u, v

        x, xdot = self.kf_x.update(xm, dt)
        y, ydot = self.kf_y.update(ym, dt)

        if self.draw and det is not None:
            u, v, _ = det
            cv2.circle(frame, (int(u), int(v)), 6, (0, 255, 0), 2)
            if x is not None:
                cv2.putText(frame, f"x={x:6.1f} y={y:6.1f} px",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 2)
        return x, xdot, y, ydot, det, frame


class LateralController:
    """Per-axis PD (+ optional I) on pixel position error -> 2-D tilt command
    (in CAMERA axes; mixer.py applies the axis/sign calibration)."""

    def __init__(self, gains: LateralGains = DEFAULT_LATERAL_GAINS):
        self.g = gains
        self._ix = 0.0
        self._iy = 0.0

    def reset(self):
        self._ix = self._iy = 0.0

    @staticmethod
    def _clamp(v, lo, hi):
        return lo if v < lo else (hi if v > hi else v)

    def _axis(self, ref, pos, vel, dt, istate):
        g = self.g
        err = ref - pos
        deriv = -(vel if vel is not None else 0.0)   # damping (essential here)
        i_term = 0.0
        if g.ki > 0.0:
            istate += err * dt
            i_term = self._clamp(g.ki * istate, -g.i_clamp, g.i_clamp)
            istate = i_term / g.ki                    # back-calc anti-windup
        t = g.kp * err + g.kd * deriv + i_term
        return self._clamp(t, -g.max_tilt, g.max_tilt), istate

    def update(self, x_ref, y_ref, x, y, xdot, ydot, dt):
        """Return (tilt_x, tilt_y) in camera axes. No lock -> (0, 0) = level."""
        if x is None or y is None:
            return 0.0, 0.0
        tx, self._ix = self._axis(x_ref, x, xdot, dt, self._ix)
        ty, self._iy = self._axis(y_ref, y, ydot, dt, self._iy)
        return tx, ty


if __name__ == "__main__":
    # Standalone check: show the downward-camera centroid tracking. Move the robot
    # (or your hand) and confirm x/y follow. Needs a few still frames to settle.
    cap = open_down_camera()
    tracker = LateralTracker(cap, draw=True)
    print("Lateral vision standalone (downward cam). 'q' to quit.")
    try:
        while True:
            x, xdot, y, ydot, det, frame = tracker.step()
            if frame is None:
                continue
            cv2.imshow("visual_servo lateral", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
