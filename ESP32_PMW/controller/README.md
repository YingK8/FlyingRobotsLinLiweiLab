# `controller/` — the vision → control pipeline

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

The layering is enforced, not merely intended: each module's `sys.path` bootstrap
adds only the stages *before* it, so a forward import fails immediately instead of
quietly creating a cycle.

Everything that is not front-end pipeline code — tests, sweeps, one-shot
validation, instrumentation, sysID, plotting — lives in [`../ai/`](../ai/), which
may depend on all four stages.

## Running it

```bash
cd ESP32_PMW

# 1. camera: what modes does this sensor really deliver?
uv run python controller/camera/modes.py --index 0

# 2. calibrate: board -> intrinsics -> extrinsics -> stereo_rig.json
uv run jupyter lab controller/calib/calibrate_camera.ipynb

# 3. pose: the live loop, with overlay (ellipse, rotor axis, ignored area, fps)
uv run jupyter lab controller/pose/online_camera.ipynb

# 4. control: closed loop off the camera
uv run python controller/control/hover_controller_runner.py --source camera --dry-run
```

Two things worth knowing before the first run:

- **The coils have no firmware watchdog.** `control/servo.py`'s `coils_on()`
  context manager is the only thing that guarantees they turn off. See
  [chapter 4 §4.0](control/theory.md).
- **`POSE_APPEARANCE` must be set before importing the pose package.**
  `estimator.RADIUS_MM` binds at import, so setting it afterwards silently leaves
  the previous appearance's radius in force. `bright` (default), `dark` (the mono
  ELP rig), or `red`.

## Tests

```bash
uv run python ai/tests/run_tests.py
```

8/9 suites pass. The one failure, `stereo::test_speed`, is left visible on
purpose: the full two-view solve is ~3.4 ms against a 4.17 ms budget at 240 Hz and
segmentation needs the rest. `segment.py` documents it.
