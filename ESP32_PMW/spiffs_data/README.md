# spiffs_data/

SPIFFS-uploaded task-sequence payloads for the JSON-driven firmware. **Not** the
top-level `data/` directory — that one holds captured experiment results (CSVs,
plots, logs; see `data/README.md`). PlatformIO's default SPIFFS source dir is
also named `data/`, which is exactly the collision this rename avoids:
`[env:swim]` points `board_build.filesystem_dir` at `spiffs_data/`.

## One firmware per task

Each task is its own firmware entry point (`src/main_<name>.cpp` +
`[env:<name>]` in `platformio.ini`) that opens **its own** JSON file from SPIFFS.
There is no swappable `experiment.json` to copy over. `uploadfs` uploads every
JSON in this directory at once; each firmware just opens the filename it was
built for.

```
# once (uploads every JSON in spiffs_data/ to the device filesystem):
~/.platformio/penv/bin/pio run -e swim -t uploadfs
# then the firmware itself:
~/.platformio/penv/bin/pio run -e swim -t upload
```

The firmware plays the JSON's commutation (frequency / phase / direction) and —
for lift tasks — overlays the current-balance PI loop, which is folded into
`PwmController` and opted into with `enableCurrentBalance()` in the main (there
is no `piEnabled` flag in the JSON). With balance on, the JSON's commanded
carrier duty is the PI's per-channel **ceiling**; the loop keeps the four channel
currents within ~0.4 A of each other beneath it, so per-channel balance trims are
found automatically rather than hand-tuned. Omit the call and the sequencer's
carrier drives verbatim — the right choice for anything that deliberately drives
channels unequally, since balancing would erase what it measures.

Telemetry is the shared line format emitted by `driveTelemetry()` in
`src/drive_common.h`, parseable by the log plotters in `ai/` (see README).

## Files

- `swim.json` — the swim task: 1→30 Hz linear ramp over 20 s at 100% carrier,
  then a 30↔22 Hz undulation ×5 (1 s each way), then coils off. CCW, PI-balanced.
  Loaded by `[env:swim]`. **Generated — do not hand-edit**; regenerate it with
  `ai/gen_swim_experiment.py` on the `drone-swimming` branch (`ai/` is gitignored
  on main). Each phase carries a `SWIM_*` label for log segmentation.
