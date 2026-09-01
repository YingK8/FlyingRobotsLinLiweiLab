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
| `ai/` | gitignored scratchpad: bench harnesses and offline design tooling. `ai/thermal/coil_thermal.py` is load-bearing (see Safety) |
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

Empirically best so far: **30 s EASE (k=2) to 210 Hz** (`ramp.DEFAULT`). Note that coil
current peaks at the ~174 Hz resonance and falls above it, so torque margin is worse up
there -- but lift goes as f^2 and wins anyway until step-out. Optimise for lift, not for
torque margin.

## Commands

```bash
pio run -e flight                  # build the flight firmware (default env)
pio run -e flight -t upload        # flash it
pio run -e <env> -t uploadfs       # schedule-driven envs only; flight takes its commands
                                   #   over serial and reads no SPIFFS schedule
uv run python controller/control/z_track.py     # self-checks: run the module
uv run python ai/spinup/detector.py --all       # did the rotor turn on each take?
```

Self-checks are a `demo()` or `_self_check()` under `if __name__ == "__main__"`, using plain
`assert`s. There is no pytest here; do not add one. New non-trivial modules ship with one.

## Loop rate

The control loop runs at **500 Hz** on its own clock; the pose pipeline delivers
**~90-100 Hz** (640x400). They are decoupled: `stereo_frames` runs on a producer thread
behind a drop-oldest slot and the controller propagates with `predictor.StatePredictor`
between fixes. `hover_controller.json` is designed at `rate_hz=500`, and the closed-loop
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

Benchmark any pipeline change before claiming it:

```bash
uv run python -c "from controller.viz import live_viz; live_viz.from_recording(
    'results/flights/2026-08-29_231418', viz=live_viz.NullViz(), speed=0,
    zero=None, max_frames=250, csv_out='results/bench/after.csv')"
```

It prints median segment / estimate / wall ms per pair. 250 frames of that recording solve
246; if a change drops that, it bought its speed by losing the robot.
