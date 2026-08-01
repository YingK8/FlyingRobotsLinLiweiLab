# visual_servo — camera height-hold for the magnetic flying robot

A host-side closed loop that hovers the magnet-equipped robot. A front-facing
camera measures **height** (robot found by **MOG2 background subtraction** — no
marker needed; it's the moving foreground against the static rig); a PD +
gravity-feedforward controller turns height error into a **rotation frequency**.
**(v2)** A second, downward camera measures **lateral position (x, y)** and a
per-axis PD tilts the thrust (via per-coil carrier duties) to keep the craft
centred. **Frequency is the throttle** (lift scales with spin rate); **carrier
asymmetry is the tilt** (steers the thrust direction).

```
front cam ─▶ vision.py   (bg-sub → height mm, Kalman) ─▶ controller.py (PD+FF → Hz) ─┐
down  cam ─▶ lateral.py  (bg-sub → x,y px, Kalman) ─▶ LateralController (PD → tilt) ─┤
                                                       mixer.py (tilt → 4 carrier %) ─┤
                                                                       serial_link.py ▼
                                       ESP32 src/main_servo.cpp   F<hz>  +  A<ch>,<duty>
```

> **v1 = height only** (`run_servo.py`). **v2 = height + lateral centring**
> (`run_pos_hold.py`, needs the second camera). Bring v1 up first.

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
down to `F_LAND` — a controlled descent, never a hard cut — **and levels all
carrier duties** so a host crash mid-tilt drops the craft straight down, not
sideways. The host sends commands at 50 Hz to keep it fed, and sends `S` on exit.

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

1. **Background settle** — keep the rig still for a second or two so MOG2 learns
   the background; the robot then shows up as the moving blob. (No marker/threshold
   step — `calibrate.py threshold` is legacy and only applies to the old HSV path.)
2. **Pixel scale** — `python calibrate.py scale --mm 100` (click two points 100 mm
   apart) → copy `PX_PER_MM` and `Z_REF_PX` into config.
3. **Vision check** — `python vision.py`; move the robot past a ruler and confirm
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

## Lateral centring (v2)

Once height-hold is solid, add the downward camera and centre the craft.

1. **Set the down camera** — put `CAMERA_INDEX_DOWN` (and frame size) in config;
   check tracking with `python lateral.py` (move the robot, confirm x/y follow).
2. **Axis/sign calibration** — `python calibrate.py lateral --f-hover 160 --z-ref 80`.
   With the craft hovering, press SPACE; it pokes +x then +y tilt, measures the
   drift on the down camera, and prints a 2×2 `LATERAL_CAL`. Paste it into config.
   **This fixes the camera↔robot rotation/sign — skipping it risks divergence.**
3. **Close the loop** — `python run_pos_hold.py --z-ref 80 --f-hover 160`. Lateral
   starts **disarmed**; once it hovers, press **SPACE** to arm centring. Tune
   `--lkd` (damping) first, then `--lkp`. `'q'` exits with a controlled descent.

Logs to `pos_hold_log.csv` (`t, z, …, x, y, x_ref, y_ref, tilt_x, tilt_y, armed,
duty[4]`). The lateral loop **requires the derivative term** — position is a double
integrator, so a P-only map oscillates/diverges.

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
| `vision.py`       | front cam → MOG2 blob → height (mm) + Kalman; `BlobDetector` |
| `controller.py`   | PD + gravity-FF height controller → frequency             |
| `lateral.py`      | (v2) down cam → (x,y) tracker + per-axis PD → tilt        |
| `mixer.py`        | (v2) tilt command → 4 carrier duties (+ `LATERAL_CAL`)    |
| `serial_link.py`  | threaded pyserial link (commands out, telemetry in)       |
| `run_servo.py`    | v1 height-only closed-loop main + CSV logging            |
| `run_pos_hold.py` | (v2) height + lateral closed-loop main + CSV logging      |
| `calibrate.py`    | scale / hover / sweep / threshold / lateral calibration   |
