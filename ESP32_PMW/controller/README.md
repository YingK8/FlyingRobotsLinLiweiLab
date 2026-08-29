# `controller/`: the vision → control pipeline

Four stages. Each consumes the previous stage's output and nothing else; each has
a theory chapter that walks its own code.

```
camera  ──frames──▶  calib  ──K, rig, datum──▶  pose  ──(t, x, z)──▶  control  ──▶  coils
```

| stage | folder | produces | chapter |
|---|---|---|---|
| 1 | [`camera/`](camera/) | timestamped frames; honest rate numbers | [The sensor](camera/theory.md) |
| 2 | [`calib/`](calib/) | intrinsics, `stereo_rig.json`, tilt/centre corrections, the datum | [Calibration](calib/theory.md) |
| 3 | [`pose/`](pose/) | a 5-DOF `Pose` per frame | [Pose](pose/theory.md) |
| 4 | [`control/`](control/) | field frequency and lateral commands over serial | [Control](control/theory.md) |

Plus [`viz/`](viz/) (`live_viz.py`), which is not a pipeline stage: it is the live
viser view of the estimated pose. It lives here rather than in `../ai/` because
`control/hover_controller_runner.py` imports it to show what the loop is doing.

The layering is enforced, not merely intended: each module's `sys.path` bootstrap
adds only the stages *before* it, so a forward import fails immediately instead of
quietly creating a cycle. Stages 1-3 hold to this strictly. The two exceptions are
`viz/`, which reaches back to `camera`, `calib` and `pose` to draw them, and
`control/hover_controller_runner.py`, which adds `../viz` to show the loop live.

Everything that is not pipeline or view code, tests, sweeps, one-shot
validation, instrumentation, sysID, plotting, lives in [`../ai/`](../ai/), which
may depend on all four stages.

## Running it

[`run.ipynb`](run.ipynb) is the driver, and the one to start from. One cell per
stage, the pipeline end to end: calibrate (`calib/calibrate.py`), measure the static
noise (`pose/noise.py`), record a flight (`camera/record.py`), estimate and view the
pose (`pose/background.py`, `viz/live_viz.py`), then fly it
(`control/hover_controller_runner.py`).

```bash
cd ESP32_PMW
uv run jupyter lab controller/run.ipynb
```

The per-stage entry points below are what `run.ipynb` calls, and are useful when
you want one stage on its own:

```bash
cd ESP32_PMW

# 1. camera: which physical camera is at which index right now?
uv run python controller/camera/identify.py

# 2. calibrate: board -> intrinsics -> extrinsics -> stereo_rig.json
uv run jupyter lab controller/calib/calibrate_camera.ipynb

# 3. pose: the live loop, with overlay (ellipse, rotor axis, ignored area, fps)
uv run jupyter lab controller/pose/online_camera.ipynb

# 4. control: closed loop off the camera. No CLI; drive it from Python.
#    StereoRig.sources() resolves A and B to whatever indices they hold today.
uv run python -c "import sys; sys.path[:0] = ['controller/control', 'controller/calib']; \
  import hover_controller_runner as r; from rig import StereoRig; \
  r.fly(source='camera', camera=StereoRig.load().sources(), dry_run=True)"
```

Two things worth knowing before the first run:

- **The coils have no firmware watchdog.** Nothing on the robot turns them off
  by itself, so the host must, on every exit path including an exception. See
  [chapter 4 §4.0](control/theory.md).
- **`POSE_APPEARANCE` must be set before importing the pose package.**
  `estimator.RADIUS_MM` binds at import, so setting it afterwards silently leaves
  the previous appearance's radius in force. `bright` (default), `dark` (the mono
  ELP rig), or `red`.

## Tests

```bash
uv run python ai/tests/run_tests.py
```

`stereo::test_speed` is expected to fail and is left visible on purpose: the full
two-view solve takes ~3.4 ms of a 4.17 ms budget at 240 Hz, and segmentation needs
the rest. `segment.py` documents it.
