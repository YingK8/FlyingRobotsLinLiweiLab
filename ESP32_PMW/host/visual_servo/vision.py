"""Vision front-end: camera frame -> marker height (mm) with a Kalman estimate.

Pipeline: capture -> HSV threshold a bright LED/reflective marker -> largest
contour centroid (u, v) -> convert vertical pixel to millimetres -> Kalman filter
(constant-acceleration z model) for a smoothed, latency-predictable (z, z_dot).

Single front camera => only the vertical (height) axis is observed here.
"""
import time

import cv2
import numpy as np

from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, FPS,
    HSV_LOWER, HSV_UPPER, MIN_BLOB_AREA_PX,
    PX_PER_MM, Z_REF_PX, DEFAULT_KALMAN, KalmanParams,
)


class KalmanZ:
    """Constant-acceleration 1-D Kalman filter on height z (mm)."""

    def __init__(self, params: KalmanParams = DEFAULT_KALMAN):
        self.p = params
        self.x = np.zeros((3, 1))          # [z, z_dot, z_ddot]
        self.P = np.eye(3) * 100.0
        self.H = np.array([[1.0, 0.0, 0.0]])
        self.R = np.array([[params.r_meas]])
        self.initialized = False

    def _Q(self, dt: float) -> np.ndarray:
        return np.diag([self.p.q_pos * dt,
                        self.p.q_vel * dt,
                        self.p.q_acc * dt])

    def update(self, z_meas, dt: float):
        """Advance one step. z_meas is mm or None (no detection -> predict only)."""
        if dt <= 0.0:
            dt = 1e-3
        F = np.array([[1.0, dt, 0.5 * dt * dt],
                      [0.0, 1.0, dt],
                      [0.0, 0.0, 1.0]])
        if not self.initialized:
            if z_meas is None:
                return None, None
            self.x[0, 0] = z_meas
            self.initialized = True
            return float(self.x[0, 0]), float(self.x[1, 0])

        # Predict
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)

        # Update (only when we actually saw the marker)
        if z_meas is not None:
            y = np.array([[z_meas]]) - self.H @ self.x
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y
            self.P = (np.eye(3) - K @ self.H) @ self.P

        return float(self.x[0, 0]), float(self.x[1, 0])

    def predict_ahead(self, latency_s: float):
        """Extrapolate height by `latency_s` to compensate camera/pipeline delay."""
        z, v, a = self.x[0, 0], self.x[1, 0], self.x[2, 0]
        return z + v * latency_s + 0.5 * a * latency_s * latency_s


def open_camera(index: int = CAMERA_INDEX) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {index}")
    return cap


def find_marker(frame, hsv_lower=HSV_LOWER, hsv_upper=HSV_UPPER,
                min_area=MIN_BLOB_AREA_PX):
    """Return (u, v, area) of the largest bright blob, or None if none found."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < min_area:
        return None
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    u = M["m10"] / M["m00"]
    v = M["m01"] / M["m00"]
    return u, v, area


def pixel_v_to_mm(v: float, z_ref_px: float = Z_REF_PX,
                  px_per_mm: float = PX_PER_MM) -> float:
    """Vertical pixel -> height in mm. Image v grows downward, so negate."""
    return -(v - z_ref_px) / px_per_mm


class HeightTracker:
    """Glue: pull a frame, detect marker, return filtered height (mm)."""

    def __init__(self, cap: cv2.VideoCapture, kalman: KalmanZ = None,
                 draw: bool = False):
        self.cap = cap
        self.kf = kalman or KalmanZ()
        self.draw = draw
        self._last_t = time.monotonic()

    def step(self):
        """Return (z_mm, zdot_mm_s, raw_uv_or_None, frame). z is None until first lock."""
        ok, frame = self.cap.read()
        if not ok:
            return None, None, None, None

        now = time.monotonic()
        dt = now - self._last_t
        self._last_t = now

        det = find_marker(frame)
        z_meas = None
        if det is not None:
            u, v, _area = det
            z_meas = pixel_v_to_mm(v)

        z, zdot = self.kf.update(z_meas, dt)

        if self.draw and det is not None:
            u, v, _ = det
            cv2.circle(frame, (int(u), int(v)), 6, (0, 255, 0), 2)
            if z is not None:
                cv2.putText(frame, f"z={z:6.1f}mm v={zdot:6.1f}mm/s",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 2)
        return z, zdot, det, frame


if __name__ == "__main__":
    # Standalone check: show the detected marker + height. Move it past a ruler.
    cap = open_camera()
    tracker = HeightTracker(cap, draw=True)
    print("Vision standalone. 'q' to quit.")
    try:
        while True:
            z, zdot, det, frame = tracker.step()
            if frame is None:
                continue
            cv2.imshow("visual_servo vision", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
