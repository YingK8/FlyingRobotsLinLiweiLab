# Setup and calibration: `sync_check_20261106_192101_055744`

Note: flew other way into backing

## Caveats (read first)

- **Calibration UNVERIFIED.** It has not passed a held-out verification; geometry from it is not accepted for measurement.
- Board square size **20.32 mm is nominal, not caliper-measured**. A scale error is invisible to every reprojection check and scales all distances.
- Stereo fit from **6 pairs**.
- World origin: **mounting-boss axis, UNSURVEYED**; physical accuracy validated: False.
- Times are host capture-receipt times (not exposure); the LED alignment does not prove when coil current started.

## Camera intrinsics

Image 1280x800 px; rotate180 [False, False]; device indices [1, 0]; distortion model OpenCV (k1, k2, p1, p2, k3).

| camera | fx | fy | cx | cy | k1, k2, p1, p2, k3 | rms px |
|---|---|---|---|---|---|---|
| A | 1303.37 | 1299.37 | 637.52 | 439.73 | [-0.4019,  0.2052,  0.0003,  0.0007, -0.0186] | 0.336 |
| B | 1272.70 | 1273.81 | 724.36 | 419.93 | [-0.407 ,  0.2651,  0.0016,  0.0003, -0.1506] | 0.293 |

## Extrinsics (camera B relative to camera A)

Baseline **517.6 mm**, optical axes **99.6 deg** apart, stereo rms 0.298 px.

`T_camA_from_camB` (mm):

```
[[ -0.1671,   0.0023,  -0.9859, 331.1855],
 [  0.002 ,   1.    ,   0.002 ,  -2.7044],
 [  0.9859,  -0.0016,  -0.1671, 397.7781],
 [  0.    ,   0.    ,   0.    ,   1.    ]]
```

## Setup (world) frame

plumb-up z; x_hint projected perpendicular to z; y=z cross x. Units mm. Fitted by `viser_app_rope_x_holder`, fixture rms 0.76 px, holder tilt from rope 1.29 deg.

`R_camA_from_world`:

```
[[ 0.92565, -0.09083,  0.36731],
 [ 0.27072, -0.51921, -0.81064],
 [ 0.26435,  0.84981, -0.45602]]
```

World origin in camera A: [-77.32,  40.13, 390.91] mm.  Map: p_camA = R_camA_from_world @ p_world + origin_in_camA.

| camera | position in world (mm) | looks down by (deg) |
|---|---|---|
| A | [ -42.6, -318.4,  239.2] | 27.1 |
| B | [368.4,  -9. , 181.6] | 16.7 |

## Frequency ramp

`1->195Hz/20s/POLY/k3`, 20.0 s of drive (`ramp.json`). t = 0 is the START LED; the ramp starts there.

2026-11-05 operator request: one cubic ramp, no linear segments, no hold. f = 1 + 194*(t/20)^3 over 20 s (POLYNOMIAL k=3). Starts at zero slope: ~5.4 s below the ~4.9 Hz capture window; slope peaks at 29.1 Hz/s at the end. Host sends stop at 20 s. Not bench-validated.

| segment | from Hz | to Hz | seconds | mode | k |
|---|---|---|---|---|---|
| 1 | 1 | 195 | 20 | POLY | 3 |

Sent to the ESP32: `seq=clear`, `seq=ramp:1:195:20000:0:3`, `seq=go`
Planned f(t): `command_planned.csv`; device-reported frequency: `current.csv` (`frequency_reported_hz` vs `t_s`).

## Randomised coil duty

No `run_parameters.json`: recorded before per-run duty randomisation (2026-11-06); duty was unrandomised 100 % on every channel.

| channel | planned % | device-applied % (median while driven) | powered (peak A) | effective % |
|---|---|---|---|---|
| A | 100 | n/a | **NO** (0.00) | 0 |
| B | 100 | n/a | yes (2.32) | 100 |
| C | 100 | n/a | **NO** (0.00) | 0 |
| D | 100 | n/a | yes (2.19) | 100 |

**Unpowered coils** (operator, 2026-11-06: 'the earlier recordings had no current on some coils because they were not powered on'): their effective duty is 0, whatever was planned.

Applied = the ramp's carrier duty times planned/100, as the device reports it; the current that flowed is `current.csv` `current_*_A`. Duty is not current.

## LED and clock sync

LED ROIs [x, y, w, h]: A [249, 228, 57, 57], B [815, 245, 51, 51].
Method **led**: operator-given START LED ROIs; the firmware-declared pulse pair was decoded in both cameras and matched to the sole device START and STOP.
- `camera_A`: t_exp = 1.000014476 * t + (-175972.134993) s
- `camera_B`: t_exp = 1.000320665 * t + (-176026.017318) s
- `esp`: t_exp = 1.000000000 * t + (-6.210451) s

## Files

- `A.mp4`, `B.mp4`: every recorded frame, as captured (118.92 fps measured; the mp4 header rate is for playback only, use `frames.csv` for time).
- `frames.csv`: one row per capture; `mp4_frame` is the frame number in both mp4s (-1 = dropped from the video); `t_exp_*_s` is the LED-synced experiment time.
- `ramp.json`, `host_commands.log`: the ramp flown and every command the host sent.
- `run_parameters.json`: written before START -- duty draw, seed, distribution, ramp, camera, calibration, time anchor, software. `telemetry.jsonl`: every serial line, host-stamped. `manifest.json`, `config.json`, `capture_meta.json`: run and capture records.
- other `*.csv`: copies of `processed/` tables; `setup.json`: every number above, with sources.

## Sources

- `stereo_rig`: `/Users/meli/Desktop/Kevin/UCB/FlyingRobotsLinLiweiLab/ESP32_PMW/LLM/runs/sessions/20261105_133801_240e2b/calibration_append/fit_20261105_213821_458879_770999f9/stereo_rig.json` (sha256 c64c94d061b8)
- `sync`: `LLM/runs/sync_check_20261106_192101_055744/processed/sync.json` (sha256 eefc2e217df5)
- `video_A`: `LLM/runs/sync_check_20261106_192101_055744/raw/camera/A/A.mp4` (sha256 d564ee24e1af)
- `video_B`: `LLM/runs/sync_check_20261106_192101_055744/raw/camera/B/B.mp4` (sha256 c69f061ee52d)
- `telemetry.jsonl`: `LLM/runs/sync_check_20261106_192101_055744/raw/telemetry.jsonl` (sha256 f063ccbdf82f)
- `manifest.json`: `LLM/runs/sync_check_20261106_192101_055744/manifest.json` (sha256 f383932d0a9d)
- `config.json`: `LLM/runs/sync_check_20261106_192101_055744/config.json` (sha256 289c77ada326)
- `capture_meta.json`: `LLM/runs/sync_check_20261106_192101_055744/raw/camera/meta.json` (sha256 6ed0aaee01c0)
- `ramp`: `LLM/runs/sync_check_20261106_192101_055744/ramp.json` (sha256 7fab89212865)
