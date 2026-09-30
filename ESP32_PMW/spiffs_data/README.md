# spiffs_data/

SPIFFS-uploaded task-sequence payloads for the JSON-driven experiment firmwares.

**Not** the top-level `data/` directory — that one holds captured experiment
results (CSVs, plots, logs; see `data/README.md`). PlatformIO's default SPIFFS
source directory is also named `data/`, which is exactly the collision this
rename avoids: `platformio.ini` sets `data_dir = spiffs_data`.

Each JSON-loading firmware opens its own file by name; there is no swappable
`experiment.json`. Upload the payloads once, then the firmware:

```bash
pio run -e <env> --target uploadfs   # packs spiffs_data/*.json to flash
pio run -e <env> --target upload     # builds + flashes the firmware
```

## Files

- `takeoff_upside_down.json` — CW 1→190 Hz ramp. Loaded by `[env:takeoff_upside_down]`.
- `tilt.json` — **generated, not hand-written.** One point of a sweep, written by
  `controller/control/tilt_schedule.py --freq <f> --write` for `[env:tilt]`, and by
  `ai/make_tilt.py --hz <f> --n <n> <design>` for the rim experiment. Both are
  overwritten per frequency, so the copy in git is only the last one written and is
  almost always stale — regenerate it rather than trusting it. The runner copies each
  flashed schedule beside its takes (`results/alignment_rate/*/NNNhz/tilt.json`), and that
  copy is the record of what actually ran.

The other payloads (`ceiling_sweep`, `takeoff`, `carrier_ramp`, `comp_test`,
`coupling_*`, `dc_calibration`) are absent. Their firmwares were restored from `main`
on 2026-09-27 (`[env:takeoff]`, `[env:ceiling]`, `[env:carrier_ramp]`, `[env:current_pid]`),
so those envs currently build but have no schedule to load — recover the payload from
git history before running one. `tilt`, `takeoff_upside_down` and the calibration envs
are the ones with payloads present.

Schedule format: [`lib/JsonPwmSequencer/README.md`](../lib/JsonPwmSequencer/README.md).
