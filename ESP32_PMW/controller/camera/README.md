# `controller/elp` — the camera, as a measured object

The ELP global-shutter module (VID `0x32E4`, PID `0x9281`, OV9281-class mono
sensor) is the camera the rig now runs on. This package is small on purpose: it
holds only the things that are about **the camera itself** and that nothing else
in the repo already does.

| file | what it is |
|---|---|
| `elp_camera.json` | the device profile: modes, measured rates, verdicts, provenance |
| `camera.py` | opening the ELP with the mode *checked*, and one call that opens either one camera or a pair |
| `modes.py` | the live mode sweep — what frame rate each mode actually delivers, and why some do not |
| `background.py` | the empty-rig frame that `segment.background_mask` subtracts |

## What is deliberately NOT here

Almost everything else, because it already exists and duplicating it would create
two versions to keep in agreement:

- **Capture, grabbing, threading, skew** — `controller/pose/sources.py`.
  `CameraSource` is the threaded drop-oldest grabber; `StereoSource` reads a pair
  and *measures* the capture skew rather than assuming it. Nothing here
  reimplements any of it.
- **Stereo calibration, end to end** — `controller/pose/stereo_calibration.ipynb`.
  Board generation and print-scale check, detection, paired capture, per-camera
  intrinsics, `cv2.stereoCalibrate`, acceptance, `stereo_rig.json`, a synthetic
  self-test and an intrinsics regression. If you are calibrating, go there.
- **The live pose loop** — `controller/pose/online_camera.ipynb`.

## The mode sweep, and why it exists

`stereo_calibration.ipynb` §3 carries a measured mode table, and so did the
planning notes for this package. **They disagree**, which is the reason `modes.py`
exists as a script rather than a table someone typed:

| mode | notebook §3 | planning sweep |
|---|---|---|
| 1280×800 | 119 fps | 121.4 |
| 640×400 | 217 | 271.3 |
| 640×480 | 107 | 209.9 |
| 320×240 | 333 | 421.7 |

Neither run recorded enough about itself to say which is right. Both used short
samples; at least one converted to grayscale on the grabber thread, inside the
timing. `modes.py` fixes that by construction — warmup by *time* rather than frame
count, 600 frames by default, `--repeat` for run-to-run spread — and by separating
the two numbers that were being conflated:

- **`fps_grabbed`** = `n_grabbed / elapsed`, what the grabber thread pulled off the
  device. This is the camera.
- **`fps_consumed`**, what a single-threaded loop sustained. This is your control
  loop, and it is the smaller of the two whenever compute is binding.

`CameraSource.n_dropped` counts frames the *consumer* never saw, not camera loss.
Reading it as camera loss is exactly how a Python-loop ceiling gets blamed on
hardware — which is the leading explanation for the 160×120 mode measuring 285 fps
against 640 asked.

`--probe-formats` asks ffmpeg which modes AVFoundation reports as genuinely
native, through a path that shares no code with the sweep, and so tests the
remaining hypothesis: that a mode measuring oddly is one the sensor does not have
and the driver is synthesising.

```bash
uv run python controller/elp/modes.py --index 0                    # the profile's modes
uv run python controller/elp/modes.py --probe-formats              # what is really native
uv run python controller/elp/modes.py --frames 100 --warmup-s 0    # reproduce the old run
uv run python controller/elp/modes.py --write-config               # update elp_camera.json
```

Results go to `results/elp/modes_<timestamp>.csv`, with a `# key, value`
provenance block, read back with `pandas.read_csv(path, comment="#")`.

## The background frame

`segment.py`'s `dark` appearance needs to know where the robot may be. Everything
that is not the robot — coils, wires, the support box, the room beyond the
backdrop, the backdrop itself — is **fixed to the rig**, so one subtraction
removes all of it:

```bash
uv run python controller/elp/background.py --index 0 --frames 30   # robot out of frame
uv run python controller/elp/background.py --check                 # is it still valid?
```

This is worth a bench step because of what it costs at runtime. Measured on a
1280×800 frame:

| method | per frame |
|---|---|
| background subtraction | **0.056 ms** |
| backdrop finder, ¼ resolution | 2.43 ms |
| backdrop finder, full resolution | 48.4 ms |

At 1280×800 the camera period is 8.3 ms and `segment()` alone already costs 7.9 ms
on one core, so this is the difference between a loop that keeps up with the
camera and one that drops every fourth frame.

Without a background frame `segment.valid_region` falls back to the backdrop
finder, which works — it is what the ELP capture tests run against — but costs
40× more and depends on the backdrop staying the brightest smooth thing in view.

**A stale background is silently wrong.** Moving the camera, refocusing or
changing the lighting invalidates it, and nothing downstream can tell: the
subtraction simply starts reporting shifted edges as robot. `--check` measures how
much of the frame currently differs and says whether it is still the same scene.

## A note on indices

AVFoundation exposes no device names through OpenCV, so cameras are addressed by
index alone, and USB devices enumerate *before* the built-in FaceTime — the ELP is
normally 0. `camera.probe_indices()` lists what actually streams. This is also why
`open_elp` checks the granted mode against the requested one by default: a
`VideoCapture.set` that silently substitutes a mode changes the intrinsics' scale,
and every distance downstream is then wrong by a factor nothing else can detect.
