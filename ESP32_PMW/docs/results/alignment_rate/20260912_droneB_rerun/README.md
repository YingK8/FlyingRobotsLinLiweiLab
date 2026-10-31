# Design-B rerun at 20 and 50 Hz, 2026-09-12

The same experiment and schedule as `../20260910_droneB` (see its README), repeated at 20 and
50 Hz. The original campaign was stopped on 2026-09-11 when the coils measured 100 C. Its
README asks for a low-frequency point to be re-checked after cool-down before resumed takes
are trusted, in case the rotor magnets were heated. These 20 takes are that check. Compare
them with `../20260910_droneB/020hz` and `050hz`.

| path | what |
|---|---|
| `NNNhz/index.csv` | one row per take: outcome, absolute paths to the flight and its `sweep.log`, measured I^2 heat |
| `NNNhz/tilt.json` | the schedule that chunk flew |
| `sweep_index.csv` | both chunks' rows together |
| `campaign.log` | runner console output, from both processes |
| `/Volumes/UBUNTU 24_0/ESP32_PMW_flights/droneB/2026-09-12_*/` | the video (never delete it) |

## How it was run

Take 1 at 20 Hz was recorded with every take gated on a coil reading (`--gate-above-hz 0`;
the operator measured 20 C). The operator then dropped the gate and asked for no wait between
low-frequency takes. The process was stopped while it waited at the take-2 gate (board
parked, no take running) and resumed:

```bash
uv run python -u controller/control/tilt_run.py --sweep --freqs 20,50 --repeats 10 --drone B \
  --flights-root "/Volumes/UBUNTU 24_0/ESP32_PMW_flights/droneB" \
  --out results/alignment_rate/20260912_droneB_rerun --wait-s 0
```

## Result of the run

- 20/20 takes `ok`, no GPIO14 aborts.
- Measured heat: 0.35-0.48 C per take at 20 Hz (4.2 C for the chunk), 2.29-2.64 C per take
  at 50 Hz (24.8 C). The model's stamp read 49.4 C at 18:47 after the last take. The coils
  were NOT measured at the end, and the model under-read about 2x at >= 60 Hz on 2026-09-11.
- Every take dropped frames (108 to 3234). The original campaign dropped frames on 41 of its
  logged takes too, up to 4761, and those were solved, so this does not make the takes
  incomparable. Check the solve rate before trusting that.
- Solved 2026-09-13 at stride 1 (`../20260910_droneB/solve.log`); `--campaign` and `--settle`
  run into `report/` (`analysis.log`). 5 takes give no usable kill (20 Hz x3, 50 Hz x2).

> **2026-09-14 timing repaired** (see `../20260910_droneB/README.md`, 2026-09-13 ~14:00 onward):
> 5 takes whose drops fell on both sides of the cut were excluded (`excluded:mistimed_drops`)
> and re-recorded with the fixed recorder (20 Hz x3, 50 Hz x2, 2026-09-13 ~14:33-14:51); the
> others were lined up at the cut where needed (`retime.json` in each take). The analysis below
> predates the repair and is re-run from the repaired takes.
>
> **Re-run 2026-09-14** (`analysis.log`, `report/`): 20 Hz -- 10 analysed, 3 modal, hinge rate
> 1485 deg/s, swing 6.3 deg, 3/8 settled, cone tau 0.54 s (2 takes no usable kill); 50 Hz -- 9
> analysed, 8 modal, 976 deg/s, swing 9.9 deg, 4/10 settled, cone tau 0.94 s. The main campaign's
> repaired 50 Hz reads 651 deg/s, swing 9.6 deg, tau 0.96 s: the two sessions now agree on swing
> and coning, and the "robot did not come back the same at 50 Hz" finding below was the timing
> error.
>
> **2026-09-13: everything below this line is unreliable.** 15 of these 20 takes dropped
> frames, and a dropped frame shifts every later frame's time (`frames.csv` keeps its row,
> the mp4 does not; see `../20260910_droneB/README.md`, 2026-09-13 ~13:30). The 50 Hz
> "difference" and the spin-sense reading are artefacts until the timing is repaired.

## The check against the original (2026-09-10/11) chunks

| | original 20 Hz | rerun 20 Hz | original 50 Hz | rerun 50 Hz |
|---|---|---|---|---|
| median rotation after the cut | 40.9 deg | 22.3 deg | -1.2 deg | **30.7 deg** |
| modal repeats | 3/10 | 2/10 | 6/7 | 6/8 |
| hinge rate | 1285 deg/s | 460 deg/s | none | **976 deg/s** |
| swing | 5.7 deg | 6.2 deg | 1.4 deg | **9.6 deg** |
| cone decay tau | 0.30 s | 0.54 s | 0.90 s | 0.92 s |

(original columns from `../20260910_droneB/report` after its 2026-09-13 re-analysis)

**The robot did not come back the same at 50 Hz, and the change is not a loss.** The original
50 Hz chunk barely responded to the cut (swing 1.4 deg, no rate); the rerun responds like a
healthy point (swing 9.6 deg, 976 deg/s). The original 50 Hz takes all spun CCW at spin-up
(`../20260910_droneB/report/spin_sense.csv`), so spin direction does not explain the original
non-response. 20 Hz is too scattered on both days (2-3 modal of 10) to call. What changed
between the two sessions is not known; candidates are the rotor magnets after the 100 C stop,
the rod or mounting, and coil temperature at the time of the take.

**Spin sense of the rerun** (`report/spin_sense.csv`, `ai/alignment/spin_sense.py` with `ROOT`
pointed here; ramp test, see `../20260910_droneB/README.md` 2026-09-13 ~12:00): all 50 Hz
takes spun CCW except r9 (no coherent spin). The two 50 Hz takes that did not respond (swing
0.16 deg) are r9 and **r6, which spun clearly CCW**. At 20 Hz, **r4 spun CW and responded
normally** (swing 6.2 deg); r8 spun CCW and did not respond; r3 and r6 gave no ramp reading.
Same picture as 80-110 Hz: a CW spin does not by itself make a take an outlier, and a missing
response is not always a direction problem.
