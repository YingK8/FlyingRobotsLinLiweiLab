# spiffs_data/

Schedule files for the JSON-driven firmwares, uploaded to the ESP32's SPIFFS
filesystem. Each `src/main_<name>.cpp` opens one of these by name at boot and
plays it. `platformio.ini` sets `data_dir = spiffs_data`, which is why this
directory is not called `data/` (PlatformIO's default name); captured experiment
output lives in `results/`.

```bash
# once -- uploads EVERY json in this directory to the device:
~/.platformio/penv/bin/pio run -e tilt -t uploadfs
# then the firmware that reads one of them:
~/.platformio/penv/bin/pio run -e tilt -t upload
```

Parser: [`lib/JsonPwmSequencer`](../lib/JsonPwmSequencer/). `// line` and
`/* block */` comments are allowed in these files
(`-D ARDUINOJSON_ENABLE_COMMENTS=1` in `platformio.ini`).

---

## JSON file shape

```json
{
  "resolution_ms": 25,
  "initial_freq": 0.0,
  "initial_duty": [50, 50, 50, 50],
  "direction": "CCW",
  "schedule": [
    { "method": "...", "channels": 0, "value": 0.0 }
  ]
}
```

A bare top-level array is also accepted — it is treated as `schedule` with every
setting at its default.

## Top-level settings

All optional. Channel order is always **A, B, C, D** = index 0, 1, 2, 3.

| Key | Type | Default | What it does |
| --- | --- | --- | --- |
| `resolution_ms` | int | `25` | Timestep the schedule is compiled to. Smaller = smoother ramps, more queue entries and more RAM. |
| `initial_freq` | float | `0.0` | Global commutation frequency (Hz) at t=0. `0` = DC: the field is energized but stationary, nothing spins until a ramp raises it. |
| `initial_duty` | float[4] | `[50,50,50,50]` | Starting **commutation** duty per channel, % — the square wave that spins the field. Not the carrier. 50 is the standard drive. |
| `direction` | `"CW"` / `"CCW"` | `"CCW"` | Seeds all four phases from the project convention. See below. |
| `initial_phase` | float[4] | (from `direction`) | Explicit per-channel phase in degrees. Applied *after* `direction`, so it overrides it. Entries past index 3 are ignored. |
| `schedule` | array | (empty) | The steps, run in array order. |

### `direction` — what CW and CCW actually do

`direction` picks one of two hard-coded phase arrays and writes it to all four
channels. That is the whole mechanism — there is no separate "reverse" flag:

| Setting | Phases written to A, B, C, D | Rotation order |
| --- | --- | --- |
| `"CCW"` *(default)* | `{90, 270, 180, 0}` | A → C → B → D |
| `"CW"` | `{270, 90, 180, 0}` | A → D → B → C |

Only channels **A and B** differ; C and D are 180° and 0° either way. The arrays
are `PHASES_CW` / `PHASES_CCW` in
[`lib/JsonPwmSequencer/JsonPwmSequencer.cpp:11-12`](../lib/JsonPwmSequencer/JsonPwmSequencer.cpp)
(the library keeps its own copy; `src/drive_common.h:13-14` holds an identical
pair used by the `main_*.cpp` constructors).

Two things to watch:

- **The default is CCW, not CW.** The parser seeds the phases from `PHASES_CCW`
  and only switches when the string matches `"cw"` (case-insensitive). Anything
  else — a missing key, `"clockwise"`, a typo — silently leaves it CCW.
- **This overrides the firmware's constructor.** `main_tilt.cpp` builds its
  controller with `PHASES_CCW`, but `tilt.json` says `"direction": "CW"`, so the
  rig runs **CW**. Same story for `hover_zigzag` and `takeoff_upside_down`, which
  are constructed CW and run CCW. The JSON always wins — read it, not the `.cpp`,
  to know which way a rig will spin.

## Step fields

Every entry in `schedule` is an object. `method` is required; every other field
is optional and defaults as shown. Unused fields are ignored.

| Field | Type | Default | Used by |
| --- | --- | --- | --- |
| `method` | string | — | all — required; an unrecognized name is printed to serial at load and skipped |
| `channels` | int **or** int array | none | the per-channel methods. `0` targets one channel, `[0, 3]` targets several **in one step** so they change simultaneously. Indices outside 0–3 are dropped. |
| `mask` | int (0–15) | `0` | `activateChannels` only — bit *i* = channel *i* |
| `value` | float or string | `0.0` | the instant setters, `setDirection`, `activateChannels`, and `label` (string) |
| `from` | float | `0.0` | every ramp — the starting value |
| `to` | float | `0.0` | every ramp — the ending value |
| `duration_ms` | int | `0` | every ramp and `addWaitTask` |
| `shape` | float | per-mode | every ramp — bends the curve. See [Ramp shapes](#ramp-shapes). |

> A per-channel method whose `channels` resolves to nothing does **not** run and
> is reported as an unknown method. `{"method": "addPhaseTask", "value": 90}` with
> no `channels` key silently does nothing.

## Methods

### Instant setters — take effect in one step, no duration

| Method | Fields | Effect |
| --- | --- | --- |
| `addDutyCycleTask` | `channels`, `value` | Sets **commutation** duty, % (clamped 0–100). Changes the drive waveform's on-time, not the current. |
| `addCarrierDutyCycleTask` | `channels`, `value` | Sets **carrier** duty, % (clamped 0–100) — the H-bridge current knob, i.e. coil strength. With `enableCurrentBalance()` on, this is the *ceiling* the PI loop works beneath, not the delivered duty. |
| `addPhaseTask` | `channels`, `value` | Sets phase in degrees. **Not** clamped or wrapped — pass what you mean. Use this to set a rotation offset the CW/CCW convention doesn't cover. |
| `activateChannels` | `mask`, `value` | Carrier duty = `value` for every channel whose bit is set in `mask`, and **`0` for every channel that is not**. `mask: 15` = all four on, `mask: 0` = all off. The one-line way to gate coils on and off. |
| `setDirection` | `value` | `0` → CW phases, any non-zero → CCW phases; written to all four channels at once. **`value` defaults to `0`, so omitting it means CW** — the opposite of the top-level `direction` default. Wipes out any per-channel `addPhaseTask` offsets. |

### Waits

| Method | Fields | Effect |
| --- | --- | --- |
| `addWaitTask` | `duration_ms` | Hold the current state. Nothing changes; this is how you get a dwell between steps. |

### Frequency ramps — global commutation frequency, Hz

All three take `from`, `to`, `duration_ms`, `shape`. They differ only in the
curve between `from` and `to`.

| Method | Curve |
| --- | --- |
| `addLinearRampTask` | Power ramp `t^p`. With no `shape` this is a straight line. |
| `addEaseRampTask` | Symmetric S-curve — eases in *and* out around the midpoint. The usual choice for spin-up. |
| `addExponentialRampTask` | `(e^(kt)−1)/(e^k−1)` — slow start, fast finish (or reversed with a negative `shape`). |

`from` is taken literally; it is **not** read from the current frequency. If a
ramp's `from` disagrees with where the previous step left off, the frequency
jumps at that boundary.

### Carrier ramps — carrier duty %, all four channels together

Same three curves, same fields. These ignore `channels` and write all four.

| Method | Curve |
| --- | --- |
| `addCarrierRampTask` | Power ramp `t^p` (straight line by default) |
| `addCarrierEaseRampTask` | S-curve |
| `addCarrierExponentialRampTask` | Exponential |

### Phase ramp — per channel

| Method | Fields | Effect |
| --- | --- | --- |
| `addPhaseRampTask` | `channels`, `from`, `to`, `duration_ms`, `shape` | Sweeps the named channel(s) from `from` to `to` degrees. Channels not listed are left alone. Always uses the S-curve, regardless of name. |

### Labels

| Method | Fields | Effect |
| --- | --- | --- |
| `label` | `value` (string) | Tags every following step until the next `label`, retrievable as `labelForStep()` and printed in telemetry so a log can be segmented per phase. No hardware effect and it does not consume a queue slot. |

## Ramp shapes

`shape` is one number whose meaning depends on the ramp's curve. Omit it for the
default.

| Curve | Used by | `shape` | Default | Effect |
| --- | --- | --- | --- | --- |
| Power `t^p` | `addLinearRampTask`, `addCarrierRampTask` | `p` | `1` (straight line) | `p > 1` slow start / fast finish; `0 < p < 1` fast start / slow finish. `p ≤ 0` falls back to 1. |
| S-curve | `addEaseRampTask`, `addCarrierEaseRampTask`, `addPhaseRampTask` | `k` | `2` | `k = 1` is linear; larger `k` sharpens the middle and flattens both ends. Values below 1 are clamped to 1. |
| Exponential | `addExponentialRampTask`, `addCarrierExponentialRampTask` | `k` | `2` | `k > 0` slow start / fast finish; `k < 0` the reverse; `k ≈ 0` degenerates to linear. |

## Limits

- **No loop or repeat.** The queue is flat. Anything repetitive has to be
  unrolled by whatever generates the file (see `ai/gen_coupling_experiment.py`
  for the coupling sweeps).
- **No absolute timestamps.** Steps run strictly in array order; timing comes
  from `duration_ms` accumulating.
- **Four channels, fixed.** Indices 0–3 only.
- Channels whose carrier duty is never commanded stay untouched rather than being
  forced to 0.

---

## Which firmware balances

Balance is a property of the `main_<name>.cpp`, not of the JSON — it is simply
whether that file calls `PwmController::enableCurrentBalance()`. When it is on,
the JSON's commanded carrier duty becomes each channel's **ceiling** and the PI
loop holds the four measured currents within ~0.4 A of each other beneath it, so
per-channel trims are found automatically instead of hand-tuned.

- **Balanced** — `tilt`, `takeoff_upside_down`, `hover_zigzag`, `carrier_ramp`,
  `flight`.
- **Passthrough** (commanded carrier drives verbatim) — `comp_test`,
  `coupling_test`, `dc`, `ceiling`. These deliberately drive channels unequally
  to measure something, so balancing them would erase the effect.
- **`takeoff` is currently passthrough too** — `main_takeoff.cpp:14` has
  `// ctl.enableCurrentBalance();` commented out, despite the file's own header
  comment saying "PI-balanced". Uncomment it to restore balancing.

## Files

| File | Loaded by | What it does |
| --- | --- | --- |
| `tilt.json` | `[env:tilt]` | CW. EASE 1→150 Hz over 30 s at 100% carrier, then steps channels 0 and 3 (A and D) down 90%→10% in 10% steps, 2.5 s each, then off. Each step commands a uniform ceiling on the pair; the PI produces the trims live. |
| `ceiling_sweep.json` | `[env:ceiling]` | CCW. All four at 100% carrier (`activateChannels` mask 15) across an EASE 1→210 Hz ramp, open-loop. Gives `Ceiling_i(f)`. |
| `takeoff.json` | `[env:takeoff]` | CCW, then explicit phases `{0,180,90,270}`. EASE 1→200 Hz over 40 s at 100% carrier, 5 s hold, off. |
| `takeoff_upside_down.json` | `[env:takeoff_upside_down]` | CCW. EASE 1→200 Hz over 25 s, a short 200→210→190 Hz overshoot, then a long 100%-carrier hold. |
| `hover_zigzag.json` | `[env:hover_zigzag]` | CCW. Linear 1→160 Hz over 15 s at 100% carrier, then a 160↔140 Hz zigzag ×5 (1 s down, 2 s up). |
| `carrier_ramp.json` | `[env:carrier_ramp]` | CCW, then explicit phases `{0,180,90,270}`. Carrier 0→100% over 10 s at a fixed 190 Hz. |
| `comp_test.json` | `[env:comp_test]` | CCW. BASELINE (equal 50%) → GAP → TRIMMED per-channel A/B comparison. Passthrough. |
| `coupling_cw.json` / `coupling_ccw.json` | `[env:coupling_test]` | Coil-coupling sweep: solo + pairwise + all-4 at several current levels. Generated by `ai/gen_coupling_experiment.py` — regenerate, don't hand-edit. The env loads the CW file; edit the `loadFromJsonFile` line in `src/calibration/main_coupling_test.cpp` for CCW. |
| `dc_calibration.json` | `[env:dc]` | CCW. 100% commutation + 100% carrier parks every pin HIGH through the driver path for a DC current-sense capture; latches off after its window. This is what produces the `SENS` gains in `src/drive_common.h`. |
| `experiment.json` / `test_experiment.json` | — | Legacy generic payloads, kept for reference. No env loads them. |

Two firmwares load no JSON at all: `[env:flight]` (`main_flight.cpp`) takes live
serial commands, and `[env:current_pid]` (`main_current_pid.cpp`) builds its
sequence in code.
