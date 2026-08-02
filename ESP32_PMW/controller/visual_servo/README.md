# visual_servo — camera height-hold for the magnetic flying robot

A host-side closed loop that hovers the magnet-equipped robot. A side-on camera
measures **height** (the robot is found by **MOG2 background subtraction** — no
marker needed; it is the moving foreground against the static rig), and an
**incremental PID** turns height error into a commanded **rotation frequency**.
**Frequency is the throttle**: lift scales with spin rate.

```
notebook (servo.ipynb)          servo.py                     ESP32 (main_servo.cpp)
  camera + the loop  ──frame──▶ detect  → height mm
                                velocity → mm/s
                                height_controller (PID → Hz) ──F<hz>──▶ PwmController
                                coils_on / stop ─────────────A<ch>,<pct> | S──▶
                                                  ◀──2 Hz telemetry──
```

Two files, and the split is deliberate: **`servo.py` never touches the camera.**
It opens no `VideoCapture`, reads no frame and shows no window, so it imports and
runs with no hardware attached. The notebook owns the camera and the loop and
feeds frames in.

## Firmware side

Flash the `servo` environment (it builds `main_servo.cpp` alone):

```
~/.platformio/penv/bin/pio run -e servo -t upload
```

`main_servo.cpp` is a **dumb shim** — its own header says "No state machine, no
ramp, no watchdog: the host owns all of that." It executes one line at a time:

| Send           | Effect                                                     |
|----------------|------------------------------------------------------------|
| `F<hz>`        | `setGlobalFrequency` — **no clamp, no slew limit** in firmware |
| `A<ch>,<pct>`  | carrier duty for ch 0..3 → A,B,C,D, constrained to 0–100    |
| `S`            | all carriers to 0 (coils off)                              |

There is no ping command, and `S` is an immediate cut, not a ramped descent.

`enableCurrentBalance()` is on, so **`A<ch>` sets a per-channel *ceiling*** that
the balancer works under, rather than a duty applied directly. Current sense
trips at 10 A.

Telemetry is `driveTelemetry()`'s shared line at **2 Hz** (500 ms throttle),
which is what `poll()` parses:

```
t=<millis> freq=<hz> | <per-channel current + duty> | spread=<A> bal=<0|1> trip=<0|1>
```

Only `freq` and `trip` are read. At 2 Hz it is a health indicator, **not** a
feedback signal — the loop closes on the camera.

> ### ⚠ There is no watchdog. The host is the only safety.
>
> If the host stops sending, the coils **stay energised at whatever they were
> last told**. Nothing on the board notices, ramps down, or levels the duties.
> `servo.coils_on(link)` is the entire protection: its `finally` sends `S` when
> the loop exits, including on a kernel interrupt. **Keep the whole flight loop
> inside one `with coils_on(link):` in one notebook cell.** Split it across cells
> and interrupting leaves the coils live with nothing driving them.

## Host setup

```
cd ESP32_PMW/controller/visual_servo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # opencv-python, numpy, pyserial
jupyter lab servo.ipynb
```

There is no config file and no CLI entry point. Serial port discovery lives in
`ai/serial_comm.py` (`servo.connect()` also pulses EN so the board boots from a
known state); the camera is selected by **name substring** in the notebook, since
`/dev/video*` minor numbers shift on every replug.

## Workflow

The notebook cells run in order:

1. **Camera.** `open_camera()` matches by name, pins `CAP_V4L2`, and prints the
   mode actually negotiated — `cap.set()` returning `True` does not mean the
   camera granted it.
2. **`measure_fps(cap)`.** Times real reads. The driver's claimed FPS is
   routinely not what the loop sees.
3. **Serial.** `link = servo.connect()`.
4. **Calibration.** Coils off, robot moved **by hand** for the whole pass: MOG2
   only sees moving foreground, and that same pass settles the background model.
   `scale_from_detections()` returns `px_per_mm` and the `z = 0` datum row.
5. **Hold.** Inside `with servo.coils_on(link):`. Start with a short duration.
6. **Plot.** `rows` → height and commanded frequency against time.

### How scale is measured

`px_per_mm` is **measured, not configured**: the blob's extent along its first
principal component is the robot's projected diameter, so
`px_per_mm = axis_px / DRONE_DIAMETER_MM`. Set `DRONE_DIAMETER_MM` to your
robot's real diameter.

At 20 mm the pixel budget is tight — one pixel of segmentation error is
`1/axis_px` of height, so a robot spanning 40 px turns one bad pixel into 2.5%.
Frame it to span **≥ 60 px**; the calibration prints a warning below that. Fix it
by moving the camera closer, never by raising the resolution.

Because it works from apparent diameter, this measures **distance to camera**,
not height. For a fixed side-on camera that is ~constant, which is exactly why it
works — and also why the scale only holds while the robot stays in that plane.

## Control design

`height_controller()` is an **incremental (velocity-form) PID**:

```
u += kp*(err - err_prev) + ki*err*dt - kd*(zdot - zdot_prev)
```

- **The hover frequency never appears.** It is implicit in the accumulation, so
  nothing has to know or calibrate it — there is no feedforward and no `f_hover`.
- **Takeoff is the same mechanism.** On the pad the robot holds a large steady
  positive error; `ki` accumulates it and `u` climbs until it lifts. No spin-up
  ramp, no takeoff frequency to guess. This is why **`ki` must be non-zero** — in
  incremental form the other terms are pure differences, so a steady error would
  produce no correction at all and the robot would never leave the pad.
- **The only limit is on rate**, and it comes from physics rather than a hand-set
  band: `TorqueLimits.f_dot_max(f)` is the fastest the field may change while the
  magnet's spin-up torque still fits the budget. It vanishes at
  **`f_ceiling` = 167.3 Hz**, so the rate cap enforces the ceiling by itself —
  an aggressive demand ramps up and eases to a halt just underneath. The limit is
  **asymmetric on purpose**: spinning up fights drag, spinning down is helped by
  it, so capping both equally would strand the controller at the ceiling unable
  to descend.
- **The floor is absolute.** `F_MIN = 1.0`; the firmware reads `f <= 0` as DC,
  which stops the field and drops the robot.

Velocity is a plain finite difference off the image stream — no Kalman filter, no
hidden state, so every number the controller sees comes from a frame. The cost
lands on the D term: `kd` multiplies a difference *of* a difference, so centroid
jitter enters the command squared. **Keep `kd` small** (default `0.01`; `0.15`
would be far too large here) and lean on `kp`/`ki`. Defaults are
`kp=0.2, ki=0.15, kd=0.01`.

Pass `z_ref=` to `step()` to command a step response without rebuilding the
closure and losing the PID's accumulated state.

## Camera and frame rate

**Know the ceiling before trying to raise it.** Logitech webcams expose only
`YUYV` and `MJPG` over UVC — **there is no mono format to stream** — and the
consumer models (C270, C920, C930e) are capped at 30 fps. Only the C922
(720p60) and Brio (1080p60) reach 60.

Measured on the C270 on this rig, requesting 120 fps at each size:

| Resolution | Measured | | Resolution | Measured |
|---|---|---|---|---|
| 160x120 | 28.9 fps | | 800x600  | 27.9 fps |
| 320x240 | 28.0 fps | | 960x720  | 27.9 fps |
| 640x480 | 28.0 fps | | 1280x720 | **27.9 fps** |

**Resolution is free — so run 1280x720.** The rate is flat within 3% from
160x120 to full 720p, so lowering it sacrifices `px_per_mm` precision and buys
nothing. Requesting 60 or 120 fps clamps to 30; requesting 15 does yield 15,
which proves the camera honours the property and the 30 cap is the hardware,
not OpenCV ignoring the request.

A useful side effect: 1280x720 sustaining 28 fps is itself proof the stream is
already MJPG-compressed, since raw `YUYV` at that size cannot fit a USB 2.0
pipe (below).

What actually helps, and is done in the notebook:

- **Set the `MJPG` fourcc, before the frame size** (or the driver re-negotiates).
  Without it V4L2 picks `YUYV`; 1280x720 `YUYV` at 30 fps needs ~442 Mbit/s,
  beyond the ~320 Mbit/s a USB 2.0 isochronous pipe delivers in practice (480
  theoretical), so the camera drops to a rate that fits. Use `measure_fps` with
  `mjpg=True`/`False` to see what it is worth on your host — macOS AVFoundation
  ignores the fourcc and negotiates MJPG on its own, so this matters on Linux.
- **`GRAY = True`**, converting once on the host so MOG2 and both morphology
  passes run on 1 channel instead of 3. Does not lift the 30 fps ceiling, but
  keeps the processing loop from becoming the bottleneck.
- **`CAP_PROP_BUFFERSIZE = 1`**, so latency does not pile up in the driver queue.

## Safety drills

- **Keyboard stop** — press `q` in the window; confirm the coils go off (the loop
  breaks, leaves the `with`, and `stop()` fires).
- **Kernel interrupt** — interrupt the hold cell mid-flight; confirm the coils go
  off. This is the drill that matters, because it is the one the firmware cannot
  help with.
- **Know your power cut.** With no firmware watchdog, a host crash or a pulled USB
  cable leaves the coils energised. Have the bench supply within reach.

## Limits

- A single side-on camera observes **height only**; toward/away is unobserved and
  is what the fixed-plane assumption above rests on. Full 3-D needs a fiducial or
  a second camera.
- **Lateral centring is not implemented.** The per-axis PD, the tilt-to-duty mixer
  and the camera↔robot calibration were removed as unwired; recover them from git
  history if you add a second camera.
- Control bandwidth is limited by camera latency and, on this rig, by a hard
  30 fps sensor cap.

## Files

| File           | Role                                                              |
|----------------|-------------------------------------------------------------------|
| `servo.py`     | camera-free library: serial, MOG2 detection, scale, PID, `altitude_hold` step closure, `coils_on` |
| `servo.ipynb`  | owns the camera and the loop: capture, fps check, calibration, flight, plots |
| `requirements.txt` | opencv-python, numpy, pyserial                                |

Shared dependencies live in `ESP32_PMW/ai/`: `serial_comm.py` (`SerialComm`) and
`z_track.py` (`TorqueLimits`, the phase-lock torque budget).
