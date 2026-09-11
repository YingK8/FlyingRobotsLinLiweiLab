# Alignment-rate campaign -- design B (half outer ring), 2026-09-10

Same experiment as `../20260909_205843` (design A): ramp to a drive frequency, hold level,
cut coils A and C (channels 0, 2), film the axis realigning. See `controller/control/theory.md`
24 for the experiment and **24.12 for what changed for this robot**.

## What differs from design A -- read before comparing

| | design A (`20260909_205843`) | design B (this folder) |
|---|---|---|
| drop hold | 5 s -- settled window truncated to 4-5 s | **15 s** -- full 4-6 s window (theory.md 25.5) |
| spin-down | 4 s ramp | **100 ms** |
| repeats | 5 (more at some points) | **10** at 10-80 Hz, **5** at 90-120 Hz |
| cooling | thermal model, then none | **by the clock**: 20 s per predicted C, 2x between frequencies, operator releases early |
| ramp | 3.5 Hz/s below 60 Hz, 2.8 above | unchanged |

Coning above ~104 Hz drive is not measurable at this capture mode (Nyquist, theory.md 25.14):
110 and 120 Hz give an alignment rate but no cone numbers.

## Where things are

| path | what |
|---|---|
| `NNNhz/index.csv` | one row per take: outcome, absolute paths to the flight and its `sweep.log`, measured I^2 heat |
| `NNNhz/tilt.json` | the exact schedule that chunk flew (SPIFFS is overwritten per frequency) |
| `sweep_index.csv` | every chunk's rows together |
| `campaign.log` | the runner's full console output |
| `overlays/050hz_<take>_overlay.mp4` | `controller/pose/disc_video.render_with_angles(..., window_s=(-2, 15.5))`: both cameras with the segmentation mask, fitted ellipse and reprojected axis, over a radial/azimuth panel keyed to the cut. `002719` stops at ~7 s, `003049` at ~11.7 s, `003756` never -- watch the disc turn into four still blades |
| `/Volumes/UBUNTU 24_0/ESP32_PMW_flights/droneB/<stamp>/` | **the video**: `A/A.mp4`, `B/B.mp4`, `frames.csv`, `meta.json` (`"drone": "B"`) |
| `results/tilt_sweep/<stamp>_droneB/sweep.log` | firmware labels + telemetry per take |

**The video lives on the USB stick, and `index.csv` points at it by absolute path.** Plug it in
before analysing. **Never delete video** -- zip it (as `results/flights/Archive.zip` was) and
keep the zip.

## Run / resume

```bash
uv run python -u controller/control/tilt_run.py --sweep \
  --freqs 10,20,30,40,50,60,70,80,90,100,110,120 \
  --repeats 10 --repeats-high 5 --drone B \
  --flights-root "/Volumes/UBUNTU 24_0/ESP32_PMW_flights/droneB" \
  --out results/alignment_rate/20260910_droneB \
  --no-cool --force-hot 2>&1 | tee -a results/alignment_rate/20260910_droneB/campaign.log
```

**After the 100 C stop (2026-09-11) every take at >= 70 Hz is gated on a measured coil
temperature** -- add `--gate-above-hz 70 --gate-below-c 45` to the command above. No take starts,
and no chunk is flashed, until a reading under 45 C comes in (`echo 38 > ai/thermal/operator_reading`,
or type it at a terminal). There is no timeout: unattended, the campaign pauses. `go` does not
open the gate. Where the gate is active it replaces the clock waits.

Re-running the identical command **resumes**: it counts `ok` takes per frequency and records
only the shortfall. At a `MEASURE THE COILS` prompt, type the temperature (or Enter), or from
elsewhere `echo 45 > ai/thermal/operator_reading` (or `echo go`). Readings >= 70 C are recorded
but do not release the wait. Kills: GPIO14 on the board; the runner parks the board on exit.

Rate-swap control (theory.md 25.14), after the campaign, into its own folder:

```bash
uv run python -u controller/control/tilt_run.py --sweep --freqs 50 --repeats 5 --seg2-rate 2.8 \
  --drone B --flights-root "/Volumes/UBUNTU 24_0/ESP32_PMW_flights/droneB" \
  --out results/alignment_rate/20260910_droneB_rateswap/050hz_at_2p8 --no-cool --force-hot
uv run python -u controller/control/tilt_run.py --sweep --freqs 60 --repeats 5 --seg2-rate 3.5 \
  --drone B --flights-root "/Volumes/UBUNTU 24_0/ESP32_PMW_flights/droneB" \
  --out results/alignment_rate/20260910_droneB_rateswap/060hz_at_3p5 --no-cool --force-hot
```

## Analysis (after solving each take's axis)

```bash
uv run python controller/control/alignment_rate.py --campaign results/alignment_rate/20260910_droneB
uv run python controller/control/alignment_rate.py --settle   results/alignment_rate/20260910_droneB
```

`--campaign` first: `settle_report` reads `report/campaign.csv` for one panel.

## Run log

| when | what |
|---|---|
| 2026-09-10 23:19 | campaign started; disk check 23.1 GB free vs ~3.25 GB needed |
| 2026-09-10 23:20 | first take verified: 9698 frames, 0 dropped, 209 fps, `"drone": "B"`, full label set |
| 2026-09-10 23:20 | **`heat_c` in `index.csv` is inflated ~20x for every take recorded by the first process.** Coils A and C read a negative sense offset (-1.8 to -2.2 A, held at 0% duty) and the old integral summed its magnitude. Fixed in `tilt_run._drive_sum`; take 1 recomputes 1.66 -> 0.10 C. The column is recomputed from each `sweep.log` after the campaign. Waits were never affected -- they use the prediction. |
| 2026-09-11 ~00:04 | after the 10th 30 Hz take, `park()` failed once on an esptool connect glitch (`IndexError` in `loader.py`), so the board rebooted into the 30 Hz schedule **unfilmed** until a manual park succeeded first try. No take lost; heat at 30 Hz is negligible. `park()` now retries 4x. The runner was stopped in the between-chunk wait (board parked, 30/30 takes on disk) and resumed with the same command: it skipped 10/20/30 Hz and continued at 40 Hz. |
| 2026-09-11 01:30 | **STOPPED: operator measured the coils at 100 C** while 70 Hz take 9 was being FILMED (it started 01:28:34 -- not in the wait, as first logged). The runner was killed mid-take, so its own `finally`/park never ran; the coils were cut seconds later by an explicit `park()` (bootloader, app off), and 100 C was written to the thermal stamp. That partial take, `2026-09-11_012834` on the stick (A/B video, frames.csv, meta.json, 13.5 MB), is in NO index and is not analysed. It is kept, not deleted. The model at that moment read **63 C** (corrected integral) / **75 C** (raw integral) -- it under-counted heat by roughly 2x, so the 20 s/C clock waits (sized from it) were far too short from ~60 Hz up. **68 good takes are safe**: 10-60 Hz complete, 70 Hz 8/10. Remaining: 70 Hz x2, 80 Hz x10, 90-120 Hz x5 each. Takes after this point may not be comparable to earlier ones if the heat reached the robot's magnets -- re-check one low-frequency point after cool-down before trusting a resumed chunk. |

## Analysis status (2026-09-11)

All 68 good takes solved at stride 1 (`disc_axis`, now writing `.partial` and renaming on
success). `report/campaign.csv` and `report/campaign_trials.csv` come from
`alignment_rate.campaign`; the comparison with design 1 is in `../compare_1_vs_B/` and the
write-up in `controller/control/theory.md` 24.13. Five `*.orphan` files on the stick (takes
`2026-09-11_000651` ... `_001243`) are leftovers from solver workers that outlived their parent;
those takes were re-solved cleanly and the orphans are kept, not deleted.
