# visual_servo — camera height-hold for the magnetic flying robot

A host-side closed loop that hovers the magnet-equipped robot at a target height.
A front-facing camera measures the robot's height from a bright marker; a PD +
gravity-feedforward controller turns height error into a **rotation frequency**,
which is streamed to the ESP32 over USB serial. **Frequency is the throttle**
(lift scales with the rotating-field spin rate); per-coil amplitudes are held
fixed in this 1-D height mode.

```
camera ─▶ vision.py (marker → height mm, Kalman) ─▶ controller.py (PD+FF → Hz)
                                                          │
                                              serial_link.py  F<hz>\n
                                                          ▼
                                  ESP32  src/main_servo.cpp  (setGlobalFrequency)
```

## Firmware side

Flash the `servo` environment (it excludes the two characterization mains):

```
~/.platformio/penv/bin/pio run -e servo -t upload
```

On boot it does a one-shot eased spin-up to `F_MIN`, then enters command mode and
listens for these `\n`-terminated commands (see [src/main_servo.cpp](../../src/main_servo.cpp)):

| Send          | Effect                                            |
|---------------|---------------------------------------------------|
| `F<hz>`       | set target frequency (clamped to `[F_MIN,F_MAX]`, slew-limited) |
| `A<ch>,<duty>`| set one carrier duty % (reserved for attitude)    |
| `S`           | safe descent (ramp to `F_LAND`)                   |
| `P`           | ping → immediate telemetry line                   |

Telemetry out (~50 Hz): `T,<millis>,<freq>,<dutyA>,<dutyB>,<dutyC>,<dutyD>`.

**Watchdog:** if no command arrives for ~250 ms, the firmware ramps the frequency
down to `F_LAND` — a controlled descent, never a hard cut. The host sends a
command at 50 Hz to keep it fed, and sends `S` automatically on exit.

> The safety constants `F_MIN / F_MAX / F_LAND` exist in **both**
> [src/main_servo.cpp](../../src/main_servo.cpp) and [config.py](config.py).
> Keep them in sync.

## Host setup

```
cd host/visual_servo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Set `SERIAL_PORT` and `CAMERA_INDEX` in [config.py](config.py) (`ls /dev/tty.*`
on macOS, `COMx` on Windows).

## Calibration order (do this before closed-loop flight)

1. **Marker mask** — `python calibrate.py threshold` → copy `HSV_LOWER/UPPER` into config.
2. **Pixel scale** — `python calibrate.py scale --mm 100` (click two points 100 mm
   apart) → copy `PX_PER_MM` and `Z_REF_PX` into config.
3. **Vision check** — `python vision.py`; move the marker past a ruler and confirm
   the reported `z` (mm) tracks reality.
4. **Lift-vs-frequency sweep** — `python calibrate.py sweep --out sweep.csv`.
   **Confirm `z` rises monotonically with frequency** over your hover band. If it
   peaks then falls you are past electrical resonance (~190 Hz) and the control
   sign inverts — keep the band on the rising side and lower `F_MAX`.
5. **Find F_HOVER** — `python calibrate.py hover`; ramp until it floats, press
   SPACE → copy the printed value into `ControlGains.f_hover`.

## Closed-loop run

```
python run_servo.py --z-ref 80 --f-hover 160 --kp 0.8 --kd 0.15
```

Logs to `servo_log.csv` (`t, z, z_ref, zdot, u_cmd, freq_meas, duty[4]`). Tune
from the step response: raise `Kp` toward criticality, add `Kd` for damping, add a
small `Ki` only if there's steady-state droop. `'q'` in the window or Ctrl-C stops
and triggers a controlled descent.

## Safety drills (run once)

- **Keyboard kill** — press `q`; confirm the craft descends, not drops.
- **Unplug serial** — confirm the firmware watchdog ramps to `F_LAND` within ~250 ms.

## Limits

- Single front camera observes **height only** (toward/away is unobserved). Full
  3-D needs an ArUco/AprilTag fiducial or a second top-down camera.
- Control bandwidth is limited by camera latency; a global-shutter camera at
  60–120 fps helps most. The Kalman `--latency` flag predicts ahead to compensate.

## Files

| File              | Role                                                      |
|-------------------|-----------------------------------------------------------|
| `config.py`       | all tunables + safety envelope (mirror of firmware)       |
| `vision.py`       | camera → marker centroid → height (mm) + Kalman filter    |
| `controller.py`   | PD + gravity-FF height controller → frequency             |
| `serial_link.py`  | threaded pyserial link (commands out, telemetry in)       |
| `run_servo.py`    | closed-loop main + CSV logging                            |
| `calibrate.py`    | scale / hover / sweep / threshold calibration tools       |
