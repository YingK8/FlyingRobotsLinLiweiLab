# ESP32_PMW

A magnetically driven sub-centimetre flying robot. The robot carries no electronics: an
ESP32 drives four coils (per-channel phase PWM + a 20 kHz carrier for current control) and
the rotating field spins and tilts the rotor. A Python host closes the loop around it from
a two-camera stereo rig.

| where | what |
|---|---|
| `src/main_flight.cpp` | the flight firmware. One `main_*.cpp` per experiment, one PlatformIO env each |
| `lib/` | firmware libraries (`PwmController`, `SerialComm`, ...) |
| `controller/` | the live host pipeline: `camera/` -> `calib/` -> `pose/` -> `control/`, plus `viz/`. A real Python package -- import by full path, `from controller.pose import stereo` |
| `controller/native/` | `pmw_pose`: the C++ port of everything `StereoPoseEstimator.update` does per frame (`pose/theory.md` 21), plus the live capture and the interleaved tracker (22). Built by `uv sync --extra native`; `pose/stereo_native.py` wraps the estimator, `pose/tracker.py` the tracker, and `live_viz._stereo_estimator` picks the native core when it is importable. The Python estimator stays as the reference and `pose/native_parity.py` holds the two together |
| `ai/` | gitignored scratchpad: bench harnesses and offline design tooling. `ai/thermal/coil_thermal.py` is load-bearing (see Safety) |
| `controller/report.py` | one command for the whole offline pass on a take: solve, mast, angles, the command record, the plots and the overlay video, all into `<take>/report/`. The stages are the modules above; this only orders them |
| `controller/run.ipynb` | the operator notebook, one cell per stage; cell 12 flies |
| `results/`, `data/` | captures and outputs; each has a README mapping files to the script that made them |

## Untouchable

The GPIO map in `src/constants.h` is golden. Never do arithmetic on a pin constant, never
renumber or "tidy" the A/B/C/D blocks, never assume PWM and carrier pins are adjacent.
They are wired that way on the board and nowhere else.

## Where the reasoning lives

`controller/camera/theory.md`, `calib/theory.md`, `pose/theory.md`, `control/theory.md` are
the authoritative write-ups, one chapter per stage. Derivations for anything implemented or
investigated go there -- including the negative results, which is most of their value. Cite
a section from code (`see theory.md 6.2`) instead of re-deriving it in a comment.

Code comments explain *why*, name the measurement a number came from, and say plainly when
a number is a guess. A constant with no provenance is a bug waiting for a session.

## Do not trust

- `docs/PCB_Design_Documentation.md` -- last touched 2026-06-14, documents PWM_amp Rev 1
  (sheet dated 2026-03-04) as drawn. The bench has moved since. Check `hw_references/` and
  the board itself before believing a pin.
- `docs/pose_localization_project_context.md` -- the pre-build design record. Its
  arguments were superseded by `controller/pose/theory.md` and `calib/theory.md`; its
  RealSense plan was never built. Kept only because live code still cites its section
  numbers. Read the chapters instead.
- anything that says `ai/` holds the old sweep / sysid / validation tree. That tree was
  deleted 2026-08-31. `ai/` is now a gitignored scratchpad holding `thermal/`, `spinup/`,
  `design/` and `matlab/`. The flight path depends on exactly one file in it,
  `ai/thermal/coil_thermal.py`, and refuses to arm without it.

## Safety

**Nothing stops the coils automatically.** There is no firmware watchdog and no host
watchdog. Both were stripped on 2026-08-29 at the operator's instruction -- the firmware's
500 ms silence watchdog was reported not to work, and the host-side lands were removed by
the same request. This is an operating decision, not a finding that they were harmful; the
argument for putting them back is in `controller/control/theory.md` 4.0. The kills are:

1. the **stop** button in the viser panel (the only software kill during a run),
2. `uv run python controller/control/safe_off.py`,
3. the **GPIO14** button on the board (`src/reset_button.h`).

**Coils overheat before anything else breaks.** Measured: +1 C per 2 s of drive, and 80 C
reached after four ramps. There is no temperature sensor, so `ai/thermal/coil_thermal.py`
models it (heat rate, Newton cooling, a 70 C ceiling) and `link.SerialComm` stamps every
energised interval on `close()`. **Drive the coils only through `SerialComm`** -- it is the
one chokepoint every path goes through, which is why the stamping lives there and not in
the caller. `fly()` calls `coil_thermal.wait_until_safe()` for you and **refuses to arm if
that file is missing**, because `ai/` is gitignored and a fresh clone does not have it.

Keep a single ramp under a minute. `ramp.check` refuses a profile over `MAX_RAMP_S` rather
than clamping it, since the segments go to the firmware verbatim.

SIGINT, an exception, or a clean exit also de-energise through the runner's `finally`. A
crashed kernel does not. Always end a session with `safe_off.py`.

## Ramp profiles live on the host, not in firmware

`PwmSequencer` takes arbitrary ramp tasks, so a spin-up profile is data the host sends --
never a shape compiled in behind a reflash:

```
seq=clear                              # ALL THREE are IDLE-only. `clear` mid-SPINUP used
seq=ramp:<from>:<to>:<ms>:<mode>:<k>   #   to empty the running queue, which made isDone()
seq=go                                 #   true and jumped the firmware to FLIGHT with the
                                       #   coils still energised.
                                       # mode 0=POLYNOMIAL (k=1 linear, 2 quadratic),
                                       #      1=EASE, 2=EXPONENTIAL
```

**`controller/control/ramp.py` is the only place a ramp is defined.** It owns the shape,
the validation and the `seq=` encoding; the numbers it is built from are in
`constants.py`. `RunConfig.segments` defaults to `ramp.DEFAULT`, and a profile is a tuple
of `(from_hz, to_hz, seconds, mode, k)` -- pass another to try a different shape.

`ramp.check()` **refuses and never clamps**: a total over `MAX_RAMP_S`, a zero-length
segment, or a **gap between segments** (`seq=go` compiles the queue at the first segment's
start only, so a gap is a commanded frequency step). The segments reach the firmware
verbatim, so nothing here can be clamped for you.

Every run stamps `# ramp: <label>` as the first line of its CSV, so an attempt records the
profile it flew and `takeoff_report.compare()` labels curves by shape, not by clock time.

**`seq=` is the only way to spin the coils up.** The firmware's own `takeoff` command and
its `takeoff=<start>:<end>:<ms>:<k>` parser were deleted 2026-08-31, along with
`HOVER_HZ` / `CLIMB_MS` / `RAMP_START_HZ` -- a second ramp path with its own copies of the
numbers, which had drifted to a linear curve that never captured the rotor. The board can
no longer be spun up from a serial monitor without the Python host. **Do not add a second
path back**; if you ever do, `link._note_drive` must count it as drive or its heat goes
unaccounted. The reasoning is in `controller/control/theory.md` 18.7.

That rule is about `main_flight`. The one other firmware that spins the coils up is
`-e tilt` (`src/main_tilt.cpp`), restored 2026-09-01 for the channel-0 amplitude sweep.
It runs a SPIFFS schedule, **parses no serial**, and starts on reset -- so it energises
with nothing passing through `handle_serial_comm`, and its heat is counted only because
`controller/control/tilt_sweep.py` brackets the run with `SerialComm.note_external_drive`.
Any other schedule-driven env must do the same. Its only software kill is GPIO14; there
is no `stop` it can hear. See `control/theory.md` 23.

Empirically best so far: **30 s EASE (k=2) to 210 Hz** (`ramp.DEFAULT`). Coil current peaks
at resonance and falls above it, so torque margin is worse up there -- but lift goes as f^2
and wins anyway until step-out. Optimise for lift, not for torque margin.

**Where resonance IS, is unknown.** The 174 Hz in `F_RESONANCE_HZ` was measured on the old
capacitor bank; the bank went to 800 uF per coil array on 2026-09-01 and $f_0 \propto
C^{-1/2}$, so that number is stale and must not be rescaled arithmetically -- the old $C$
never agreed with itself (334 uF fitted vs 500 uF selected scale it to 112 or 138 Hz). The
nominal is 150 Hz. Re-measure with `coil_phase.py --measure`; `theory.md` 22.3 has the
argument, 18.6 the history.

**`probe=` is a drive path.** It energises, so `link._note_drive` counts it and
`coil_phase.measure` gates it on `coil_thermal.wait_until_safe()`. It is IDLE-only and
blocks for its burst, during which neither the GPIO14 button nor a host `stop` is serviced
-- which is why it is capped at 2 s and cuts the coils the instant it returns.

**`duty=A:B:C:D` is a drive path too.** Per-channel carrier ceilings that replace the az/mag
mixer while set (`duty=off` clears). It exists for `controller/control/tilt_servo.py`, which
needs four independent amplitudes; `link._note_drive` counts it. It scales `collective`, so
`throttle=` and the landing ramp behave the same either way.

## Commands

```bash
pio run -e flight                  # build the flight firmware (default env)
pio run -e flight -t upload        # flash it
pio run -e <env> -t uploadfs       # schedule-driven envs only; flight takes its commands
                                   #   over serial and reads no SPIFFS schedule
uv sync --extra native                          # build pmw_pose (needs cmake + Homebrew opencv, eigen)
uv run python controller/pose/native_parity.py  # hold the C++ core to the Python reference
uv run python controller/pose/tracker.py        # pairing, skew guard, view-cache exactness (no camera)
uv run python controller/control/z_track.py     # self-checks: run the module
uv run python controller/control/coil_phase.py  # per-channel current phase; --measure drives
uv run python controller/report.py results/tilt_sweep/<take>   # the whole offline pipeline
uv run python ai/spinup/detector.py --all       # did the rotor turn on each take?
```

Self-checks are a `demo()` or `_self_check()` under `if __name__ == "__main__"`, using plain
`assert`s. There is no pytest here; do not add one. New non-trivial modules ship with one.

## Loop rate

The control loop runs at **500 Hz** on its own clock; the pose pipeline delivers
**~90-100 Hz** (640x400) on the Python capture path, or roughly **4x that** through
`stereo_frames(tracker=True)` -- see "Two cameras, two rates" below. They are decoupled:
`stereo_frames` runs on a producer thread behind a drop-oldest slot and the controller
propagates with `predictor.StatePredictor` between fixes. `hover_controller.json` is designed at `rate_hz=500`, and the closed-loop
poles are invariant in that number (`control/theory.md` 19.8) -- raising the clock changes
`K` by ~12% and nothing else.

**The clock is the small prize and always was.** 19.1: against the 0.78 Hz closed loop,
200 Hz of command cost 0.7 deg of phase lag and the pose pipeline cost 4.3. Re-derive that
table before spending a session on the clock again. The two things that did pay were the
**analytic Jacobian** (19.12, solve 6.9 -> 4.1 ms) and the **serial baud** (19.13, a
command's wire time 2.5 -> 0.31 ms, which was a larger delay than the whole clock change).

The host and firmware baud MUST match -- `link.SerialComm.BAUD` and `src/constants.h`'s
`SERIAL_BAUD`, both 921600. There is no handshake and no ack, so a mismatch is **silent**:
the firmware never parses a command and the coils hold their last value.

Two rules that are not optional, both in `controller/control/theory.md` 19.6:

- **Anything that consumes a frame is gated on `fresh`; anything that commands the coils
  runs on the clock.** Ungate a frame consumer and one held fix satisfies `PRIME_FIXES`
  in 30 ms and ramps the coils on a single frame.
- **The control loop stays on the main thread.** `signal` can only be delivered there, and
  a worker mid-`link.send` during SIGINT's `land()` half-writes a `stop`.

**Smoothness is a separate axis from accuracy, and the static metrics cannot see it.**
Score a fitting change by the **second difference** of the trajectory (real motion at
100+ Hz has small acceleration, noise does not), carrying `discrepancy_mm` /
`refine_rms_px` / `union_coverage` / solve count as guards -- jitter alone is minimised by
an estimator that has stopped listening to the image. That is how
`REFINE_TOL_ANALYTIC` went 1e-3 -> 5e-4 (angular jitter 0.45x, every guard flat or
better). `pose/theory.md` 16.29. **The jitter floor and the `native_parity` floor are the
same floor**: past 5e-4 the solve runs below its Jacobian's forward-difference noise and
the two cores diverge.

Benchmark any pipeline change before claiming it:

```bash
uv run python -c "from controller.viz import live_viz; live_viz.from_recording(
    'results/flights/New Folder With Items/2026-08-29_231418', viz=live_viz.NullViz(),
    speed=0, zero=None, max_frames=250, scale=0.5, csv_out='results/bench/after.csv')"
```

It prints median segment / estimate / wall ms per pair. 250 frames of that recording solve
246; if a change drops that, it bought its speed by losing the robot. `scale=0.5` is the
640x400 the loop flies at -- without it the replay runs at the recording's native
1280x800 and none of the 19.x numbers apply. **The pose core in force is printed**
(`pose core: native (pmw_pose)` or `python`); pass `native=False` to
`_stereo_estimator` to bench the reference. On this recording the two solve the same 246
frames and agree to 1e-6 mm at p95 (`pose/theory.md` 21.3): the port is a port, and a
speed-up that changes an answer is a bug in it.

## Two cameras, two rates

**One ELP delivers ~208 fps at 640x400 and nothing on the host changes that** (five
flights' `meta.json`, `dropped` zero, so it is the camera and not the consumer).
Downsampling happens after the USB transfer, so it buys compute -- which is not short --
and `pose/theory.md` 19.3 measured a 2x2 bin at roughly double the position bias for 6 Hz.
The faster sensor modes are **crops**, needing their own calibration. Do not go looking for
rate in the resolution; the argument is `pose/theory.md` 22.1 and it has been made twice.

The rate is in the **pairing**. `sources.StereoCamera.read` consumes each camera's slot
and waits for a fresh frame from both, so a pair costs the slower camera's period.
`controller/pose/tracker.py` never consumes its slot and fires a pose on every frame from
*either* camera, paired with the other's newest -- about twice the observations, same
resolution, same FOV, no recalibration. `live_viz.stereo_frames(tracker=True)` is the
switch; the yields are identical, so `_PoseFeed`, the control loop's `fresh` gating and
the notebook cannot tell.

**`max_skew_s` is not optional and is not a tuning knob.** A slot that is never consumed
has a failure mode the consuming one does not: a camera that stops delivering would pair
its frozen last frame forever, at full rate, with full confidence. It is what replaces
`MonoCamera.read`'s 2 s timeout.

**A constant in frames is a duration in disguise.** Three separate numbers broke when the
pose rate tripled, all of them correct at the rate they were written for and none of them
failing loudly: `stereo.WINDOW_FRAMES` (a quarter second only at 60 fps),
`background.RunningPlate.step` (counts per frame, meaning counts per second), and a `lost`
counter that could not move. `pose/tracker.py` now derives the first two from the measured
rate -- `PLATE_STEP_REF_HZ`, `cfg["window_frames"]`. Anything else counted in frames wants
the same treatment before the rate moves again. `pose/theory.md` 22.8.

**The frame rate you can use is set by the LIGHT, not just the sensor.** At 210 fps the
exposure available is 4.76 ms, and if the scene does not fill it the ELP returns *empty
buffers* at full rate rather than slowing down -- measured 53%/40% empty at 640x400@210
against 10.8%/6.5% at 1280x800@120 on the same scene. Healthy here is frame mean 59/82 with
max 235; at mean 20 it solves nothing. `scratchpad/light.py`-style mean/max readout first.

**When the pipeline stops solving, check the frame means before anything else.** A healthy
lit scene here reads mean 59/82 with max 235; at mean 20 and max 165 the segmenter loses the
rim, the two views disagree, and `n_rejected` (the discrepancy gate) eats every frame. That
is what a threefold drop in bench lighting looks like from the software side, and it is
indistinguishable from a dozen code faults unless you look. `trk.stats()` carries
`n_detected` / `n_rejected` / `n_rejected_fit` / `n_rejected_mono` and `age_ms` per camera
for exactly this. `pose/theory.md` 22.8.

**A live preview on a still robot stopped solving after ~4-5 minutes; earlier instances UNEXPLAINED.** It is a
cliff, not a slope: `lost` sits at the 4 plate-warmup frames for 60,000+ ticks and then
loses every frame at once, with both cameras still delivering (frames 1-4 ms old) and the
skew guard responsible for a tenth of it. The obvious reading -- `RunningPlate` walking onto
a subject that stopped moving -- is **not established**: correcting the plate's step for the
pose rate changed nothing (63,322 ticks against 73,724). Do not repeat that guess without
the evidence. `stats()` now carries `n_rejected` / `n_rejected_fit` / `n_rejected_mono` and
`live_viz._why_no_pose` differences them across the silence, which is the measurement to
read next. `pose/theory.md` 22.8. Saved plates are not a control -- the ones on disk are
from 2026-08-29 and the scene is twice as bright now.

Capture is on **AVFoundation directly**, not OpenCV `videoio`: Homebrew's `videoio` links
ffmpeg 7 against an installed ffmpeg 8 and does not load, and `brew reinstall opencv`
would pull OpenCV 5.0 -- the version bump `pose/theory.md` 21.2 records as moving
`fitEllipseDirect` and `remap`. **Do not "fix" the OpenCV install to get `videoio` back.**

**Every published segmentation timing before 2026-09-03 ran a path that bailed early.**
`from_recording` uses running plates, on which `segment()` returns `None` on 100% of this
recording's frames and the pose comes from the tracked-ellipse seed; with the saved rig
plates the mask path runs and the fit-quality gate then rejects every frame (view B's hull
rms is 1.6% of its major axis against the 1.2% gate). Both cores reproduce both
behaviours; neither has yet been made to segment this take. `pose/theory.md` 21.1.
