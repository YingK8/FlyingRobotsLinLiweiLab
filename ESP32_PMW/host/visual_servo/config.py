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

# Detection is MOG2 background subtraction (the robot is the moving foreground
# against a static rig). No marker/threshold tuning needed; the legacy HSV
# bright-marker path below is kept only for calibrate.py's `threshold` tool.
MIN_BLOB_AREA_PX = 30        # reject specks / sensor noise

# Background-subtraction detector (shared by the height + lateral trackers).
BG_HISTORY = 500             # MOG2 frames of background memory
BG_VAR_THRESHOLD = 40.0      # MOG2 sensitivity (lower = more sensitive)
MORPH_KERNEL = 5             # px, open+close kernel to despeckle the foreground mask

# Legacy HSV bright-marker bounds (only used by calibrate.py `threshold`).
HSV_LOWER = (0, 0, 220)      # near-white / bright blob
HSV_UPPER = (179, 80, 255)

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


# --- Lateral position hold (v2): downward camera ---------------------------
# A second, downward-looking camera observes the robot's (x, y) in the horizontal
# plane; the goal is to keep it centred. Lateral position is a DOUBLE INTEGRATOR
# (tilt -> horizontal force -> accel -> vel -> pos), so the controller MUST have a
# derivative (velocity) term -- a pure proportional map oscillates/diverges.
# Control runs in PIXELS (no metric scale needed to stay centred); the setpoint
# defaults to the frame centre.
CAMERA_INDEX_DOWN = 1        # downward camera index (front/height cam is CAMERA_INDEX)
DOWN_FRAME_WIDTH = 1280
DOWN_FRAME_HEIGHT = 720


@dataclass
class LateralGains:
    kp: float = 0.02         # tilt-units per pixel of position error
    kd: float = 0.012        # tilt-units per (pixel/s) -- damping; ESSENTIAL here
    ki: float = 0.0          # add only for slow steady drift
    i_clamp: float = 0.3     # anti-windup: |integral term| <= this (tilt units)
    max_tilt: float = 1.0    # clamp on |tilt command| per axis (tilt units)


@dataclass
class LateralKalman:
    # Constant-acceleration model per axis, in PIXELS.
    q_pos: float = 1.0
    q_vel: float = 100.0
    q_acc: float = 2000.0
    r_meas: float = 4.0      # centroid measurement noise (px^2)


# tilt-command -> per-coil carrier-duty mixer (see mixer.py).
CARRIER_NOMINAL = 100.0      # level full-on duty (MUST match firmware carrier_duty)
MIXER_GAIN = 40.0            # % carrier-duty swing per unit tilt command
# 2x2 axis/sign calibration: maps controller tilt in CAMERA axes -> ROBOT tilt
# axes. Identity until you run `calibrate.py lateral`. WRONG SIGN => divergence.
LATERAL_CAL = ((1.0, 0.0),
               (0.0, 1.0))

DEFAULT_LATERAL_GAINS = LateralGains()
DEFAULT_LATERAL_KALMAN = LateralKalman()
