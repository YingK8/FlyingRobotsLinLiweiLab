"""Shared configuration for the visual-servoing height-hold host.

Everything tunable lives here so the control loop, vision, and serial link agree
on one set of numbers. Values marked CALIBRATE must be measured per setup with
calibrate.py before closed-loop flight.
"""
from dataclasses import dataclass, field


# --- Serial link -----------------------------------------------------------
SERIAL_PORT = "/dev/tty.usbserial-0001"  # CHANGE: see `ls /dev/tty.*` (mac) or COMx (win)
BAUD = 115200

# Firmware safety envelope -- MUST match the constants in src/main_servo.cpp.
F_MIN = 60.0    # Hz, min spin to stay aloft
F_MAX = 320.0   # Hz, hard ceiling
F_LAND = 60.0   # Hz, descent target

# Watchdog in firmware is CMD_TIMEOUT_MS = 250 ms; send commands faster than that.
CMD_PERIOD_S = 0.02   # 50 Hz command/heartbeat rate (also feeds the watchdog)


# --- Vision ----------------------------------------------------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 60

# Bright-marker threshold (HSV). Defaults target a saturated LED/reflective dot.
# Tune with calibrate.py --tune-threshold. (H 0-179, S/V 0-255.)
HSV_LOWER = (0, 0, 220)      # near-white / bright blob
HSV_UPPER = (179, 80, 255)
MIN_BLOB_AREA_PX = 30        # reject specks / sensor noise

# Image V axis grows downward, so a HIGHER marker = SMALLER pixel-v. We negate
# in vision.py so that `z` increases upward (intuitive for control).
PX_PER_MM = 2.0              # CALIBRATE: pixels per millimetre at the hover plane
Z_REF_PX = float(FRAME_HEIGHT) / 2.0  # pixel-v treated as z = 0 (set in calibrate)


# --- Control ---------------------------------------------------------------
@dataclass
class ControlGains:
    f_hover: float = 150.0   # CALIBRATE: equilibrium frequency where lift = weight
    kp: float = 0.8          # Hz per mm of height error
    kd: float = 0.15         # Hz per (mm/s) -- damping; essential, lightly damped plant
    ki: float = 0.0          # add only if steady-state droop persists
    f_min: float = F_MIN
    f_max: float = F_MAX
    max_slew_hz_s: float = 300.0   # host-side rate limit (firmware also slews)
    i_clamp: float = 20.0          # anti-windup: |integral term| <= this many Hz


@dataclass
class KalmanParams:
    # Constant-acceleration model in z (mm). Process/measurement noise in mm.
    q_pos: float = 1.0
    q_vel: float = 25.0
    q_acc: float = 400.0
    r_meas: float = 4.0      # marker centroid noise (mm^2)


DEFAULT_GAINS = ControlGains()
DEFAULT_KALMAN = KalmanParams()
