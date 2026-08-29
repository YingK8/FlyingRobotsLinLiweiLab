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

The other payloads (`tilt`, `ceiling_sweep`, `takeoff`, `carrier_ramp`,
`comp_test`, `coupling_*`, `dc_calibration`) were removed in `5866dd5` along
with the firmwares that loaded them. Recover from git history if a sweep needs
rerunning.

Schedule format: [`lib/JsonPwmSequencer/README.md`](../lib/JsonPwmSequencer/README.md).
