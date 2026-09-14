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
110 Hz gives an alignment rate but no cone numbers. **120 Hz gives neither** (measured
2026-09-13): its takes solved at 229-232 Hz, under the 240 Hz that `alignment_rate`'s Nyquist
gate needs for the rate as well, so all five are refused. They stay on disk and in `index.csv`.

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

| 2026-09-12 18:20 | before resuming, 20 and 50 Hz were re-recorded as the low-frequency check asked for above -- into their own folder, `../20260912_droneB_rerun` (20/20 ok) |
| 2026-09-12 19:15 | 70 Hz finished (10/10) with no temperature gate, at the operator's instruction |
| 2026-09-12 19:23 | 80 Hz take `2026-09-12_192214` was cut at its `KILL` label by a mistimed SIGINT meant for the wait after it: indexed `timeout`, not analysed, video kept |
| 2026-09-12 19:29-19:33 | four 80 Hz takes run back to back with no cooling (operator's choice), then stopped for cooling. **Cooling rule changed** (`tilt_run.HEAT_SCALE = 2`, `COOL_TO_C = 70`): each take charges 2x its measured I^2 heat to the stamp, and `cool_for` waits Newton time (tau 25 min) until the next take would end below 70 C, including before each chunk's first take. The stamp was re-anchored at 2x the modelled rise (60.7 -> 99.4 C). |
| 2026-09-13 00:00-06:16 | the rest, unattended: `--force-hot --stop-after-h 8` under `caffeinate -i`. Waits ran 4-8 min at 80 Hz, ~12 at 90, ~16 at 100-110, 14-18 at 120. One `park()` esptool failure after 110 Hz take 4 ("chip stopped responding"); attempt 2 parked it. |
| 2026-09-13 06:16 | **campaign complete: 100 good takes** -- 10 at each of 10-80 Hz, 5 at each of 90-120 Hz. Two non-ok rows stay indexed (40 Hz GPIO14 abort, the cut 80 Hz take). Measured heat per take at 90-120 Hz: 9.2-11.1 C. No coil temperature was measured at any point after 2026-09-11 -- every temperature above is the model. Takes from 70 Hz take 9 on are not yet solved. |
| 2026-09-13 ~12:00 | **14 takes at 80-110 Hz excluded and queued for re-recording** at the operator's instruction (videos kept; `outcome` in each `index.csv`). The operator suspected the outliers spun CW. Test (`ai/alignment/spin_sense.py` -> `report/spin_sense.csv`): demodulate the axis wobble against the drive phase, +f vs -f, on the SPIN-UP while the drive passes 15-35 Hz -- a locked rotor cannot reverse, so that sense holds to the cut. The same test on the pre-cut hold is noise from 50 Hz up (log10 power ratio within +/-0.1) and was not used. Validated at 20-40 Hz, where ramp and hold agree on all 29 takes (27 "+", 2 CW: 20 Hz r2, 40 Hz r7). Result at 80-110 Hz: **CW** (ratio <= -0.35) 90 Hz r2 r3, 100 Hz r2, 110 Hz r4 -> `excluded:cw_spin`; **no coherent spin** during the ramp (\|ratio\| < 0.15) 80 Hz r2 r4 r10, 90 Hz r1, 110 Hz r2 -> `excluded:no_coherent_spin`; plus the remaining **non-responders** (swing < 5 deg) that clearly spun CCW, 80 Hz r6 r8 r9, 100 Hz r1 r5 -> `excluded:no_response`. So CW explains only part of it: three CW takes realigned normally, and five CCW takes did not respond. |
| 2026-09-13 ~13:30 | **THE ENTRY ABOVE IS WRONG, AND SO IS EVERY TIMED RESULT ON A TAKE WITH DROPPED FRAMES.** The operator, who watched the runs, said they were all fine. Cause: `FlightWriter.add` writes a `frames.csv` row for frames it drops from the mp4 (queue full, 64 deep), and `disc_axis` pairs mp4 frame i with row i, so frame times are compressed and t = 0 lands seconds after the real cut. Proof: `frames.csv` rows - mp4 frames == `meta.json` `dropped` exactly (2026-09-13_101717: 12734 - 10973 = 1761). 91 of the campaign's and rerun's 125 takes have drops (up to 4761). The "non-responders", early rotor stops and spin-sense verdicts were artefacts; the "responders" were drop-free takes. **All 14 exclusions restored to `ok`**; the 2026-09-13 replacements are kept (80 Hz 16, 90 Hz 8, 100 Hz 8, 110 Hz 7 good takes). `report/`, `spin_sense*.csv`, `rerecord_vs_before.*` and `../compare_1_vs_B` must be regenerated after the timing repair. |
| 2026-09-13 ~14:00 | **Recorder fixed**: `frames.csv` gains a `written` column and `record.read_index` keeps written rows only (`controller/camera/theory.md` §1.6). Every take recorded from here is timed exactly even when frames drop -- and they still drop (43-5585 a take with nothing else running). Pairing test on every pre-flag take (`ai/alignment/align_pairing_test.py`, onset of the swing within 0.04 s of that frequency's drop-free lag): 43 drop-free, 31 exact pairing from the end, 20 exact by the original pairing, 39 neither. The 39 were marked `excluded:mistimed_drops` (videos kept). |
| 2026-09-13 14:29-16:20 | Re-record of 38 of the 39, lowest frequency first, **nothing else running on the machine**, fixed recorder, 2x-heat Newton cooling. **Stopped by the operator** in the cooling wait before the first 90 Hz take (board parked, nothing energised): 31 recorded, all `ok` -- 10 Hz x1, 20 x1, 30 x1, 40 x3, 50 x3, 60 x7, 70 x5, 80 x5, plus the rerun's 20 x3 and 50 x2. Not recorded: 90 x3, 100 x3, 110 x1, so those keep 5, 5 and 6 good takes. |
| 2026-09-14 02:51-03:57 | **120 Hz x5 more** (operator), nothing in parallel; 10/10 good. Drops 5585, 1377, 315, 43, 415 -- flagged, so timed exactly, but the first lost 39% of its frames. 120 Hz stays refused by the 2x-drive Nyquist gate at this camera mode regardless. The one remaining mistimed 120 Hz take (`2026-09-13_055548`) excluded: 120 Hz has 9 good. Board parked and verified afterwards. |
| 2026-09-14 ~04:10 | **51 pre-flag takes lined up at the cut** (`ai/alignment/retime_takes.py`, list in `ai/alignment/retime_categories_2026-09-14.csv`): 31 rewritten by pairing from the end (`axis.csv`, `axis_minor.csv`, `tilt_A.csv`, `tilt_B.csv`; originals kept as `*.forward_t.csv`), 20 left as recorded. Each take has a `retime.json` with the method and `post_cut_exact` (false for the 20). Checked: the swing onset in the rewritten files sits +0.07 s after `KILL`, the drop-free lag. |

## Analysis status (2026-09-14, after the timing repair)

`report/campaign*.csv`, `report/settling*.csv` and the `report/*.png` were re-run on 2026-09-14
from the repaired takes (retimed where needed, re-recorded takes solved with the `written`
flag). Design B responds to the cut at every drive from 10 to 110 Hz:

| drive (Hz) | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 | 110 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| analysed repeats | 9 | 10 | 9 | 9 | 7 | 8 | 9 | 14 | 5 | 4 | 6 |
| modal | 7 | 5 | 7 | 7 | 5 | 6 | 6 | 14 | 2 | 3 | 5 |
| hinge rate (deg/s, median) | 443 | 1112 | 554 | 546 | 651 | 633 | 984 | 1391 | 1383 | 899 | 636 |
| swing (deg, settle) | 3.8 | 5.8 | 7.1 | 8.4 | 9.6 | 10.3 | 11.0 | 11.7 | 11.7 | 10.5 | 9.8 |
| settled / analysed | 0/9 | 4/10 | 6/9 | 6/10 | 8/10 | 7/9 | 8/10 | 14/14 | 3/5 | 5/5 | 3/6 |
| cone decay tau (s) | 0.12 | 0.29 | 0.64 | 0.74 | 0.96 | 1.11 | 1.32 | 1.27 | 1.44 | 0.76 | 0.51 |

Not measured: 120 Hz (all 9, Nyquist) and 5 takes with no usable kill (10 Hz x1, 30 x1, 60 x1,
80 x2). **Superseded, kept only as a record of the mistake:** `report/spin_sense.csv`,
`report/spin_sense_after_rerecord.csv`, `report/rerecord_vs_before.{png,csv}` and the
`overlays/*hz_r*_overlay.mp4` names that carry a spin verdict -- all built on mistimed frames.
The 20 takes retimed as `forward` have exact timing at the cut but approximate seconds after
it (`retime.json` `post_cut_exact: false`); settling and cone decay on them are approximate.

## Analysis status (2026-09-11, superseded)

All 68 good takes solved at stride 1 (`disc_axis`, now writing `.partial` and renaming on
success). `report/campaign.csv` and `report/campaign_trials.csv` come from
`alignment_rate.campaign`; the comparison with design 1 is in `../compare_1_vs_B/` and the
write-up in `controller/control/theory.md` 24.13. Five `*.orphan` files on the stick (takes
`2026-09-11_000651` ... `_001243`) are leftovers from solver workers that outlived their parent;
those takes were re-solved cleanly and the orphans are kept, not deleted.

**2026-09-13:** the remaining 32 good takes (70 Hz takes 9-10, 80-120 Hz) solved at stride 1 by
a scratchpad batch over `disc_axis.solve` (4 workers, ~270 s a take; `solve.log`), then
`--campaign` and `--settle` re-run over all 100. Per-take rates of the older takes are
byte-identical to the previous run, but at **70 Hz the `modal` label flipped**: with takes 9-10
added, the three repeats that give a rate (1760-2793 deg/s) are no longer the modal population,
so 70 Hz now pools no rate (was 3/7). Nothing else from 10-60 Hz changed. Takes left out:
120 Hz x5 (Nyquist, above), and the 14 "no usable kill" takes at 30-70 Hz listed in
`settle_analysis.log`.
