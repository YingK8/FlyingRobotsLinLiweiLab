# JsonPhaseSequencer

A `PhaseSequencer` that loads a schedule from a JSON file (SPIFFS) and compiles
it into a runnable queue. The file is an **object** carrying the initial state
plus a `schedule` array of method-call objects, executed in order — not a
schedule of absolute timestamps.

## Schema

The top-level object is:

```json
{
  "resolution_ms": 25,
  "initial_freq": 190.0,
  "initial_duty": [50, 50, 50, 50],
  "direction": "CCW",
  "schedule": [ /* method-call objects, see below */ ]
}
```

The config keys seed the compiled queue's initial state — they replace the
arguments the loader used to take, so the call site is just
`seq.loadFromJsonFile("/experiment.json")`. All are optional:

| key | default | meaning |
|---|---|---|
| `resolution_ms` | `25` | compile step resolution (ms) |
| `initial_freq` | `0.0` | starting global drive frequency (Hz); `0` = DC/stationary until the schedule ramps it up |
| `initial_duty` | `[50,50,50,50]` | starting commutation duty per channel (A,B,C,D) |
| `direction` | `"CCW"` | seeds all four phases from the project `CW {270,90,180,0}` / `CCW {90,270,180,0}` convention |
| `initial_phase` | (from `direction`) | optional explicit `float[4]` phase override, per channel |

A bare top-level **array** is still accepted — it is treated as the `schedule`
with every config key at its default. Each schedule entry is an object:

```json
{ "method": "...", "channel": 0, "mask": 0, "value": 0.0,
  "from": 0.0, "to": 0.0, "duration_ms": 0 }
```

All fields are optional except `method`; unused fields default to `0`
(`channel` defaults to `-1`, meaning "no channel"). Unknown `method` strings
are logged as a warning at load time but don't abort the rest of the file.

`method` is one of:

| method | fields used | effect |
|---|---|---|
| `addDutyCycleTask` | `channel`, `value` | instantly set the channel's commutation duty (0-100%) |
| `addPhaseTask` | `channel`, `value` | instantly set the channel's phase (degrees) |
| `addCarrierDutyCycleTask` | `channel`, `value` | instantly set the channel's carrier duty (0-100%) |

For the three per-channel setters above, `channel` may be a **single int**
(`"channel": 0`) or an **int array** (`"channel": [0, 3]`). The array form
applies the same `value` to every listed channel in one queue step, so they
change *simultaneously* — unlike two consecutive single-channel calls, which
produce two steps a compile tick apart. Out-of-range indices are dropped.
| `addWaitTask` | `duration_ms` | hold the current state for this long |
| `addLinearRampTask` | `from`, `to`, `duration_ms` | linear ramp of the **global** drive frequency (Hz) |
| `addEaseRampTask` | `from`, `to`, `duration_ms` | cubic ease-in-out ramp of the global drive frequency |
| `addCarrierRampTask` | `from`, `to`, `duration_ms` | linear ramp of carrier duty, all channels identically |
| `addCarrierEaseRampTask` | `from`, `to`, `duration_ms` | ease ramp of carrier duty, all channels identically |
| `addPhaseRampTask` | `channel`, `from`, `to`, `duration_ms` | ramp one channel's phase, others unchanged |
| `setDirection` | `value` (0=CW, 1=CCW) | instantly set all 4 channels' phase to the project's CW `{270,90,180,0}` or CCW `{90,270,180,0}` convention |
| `activateChannels` | `mask` (0-15 bitmask), `value` (ON carrier duty %) | instantly set carrier duty to `value` for masked channels, `0` for the rest |
| `label` | `value` (string) | tags every step from here until the next `label`, for telemetry correlation (`labelForStep()`); no hardware effect, does not advance the queue |

There is **no loop/repeat primitive** — an experiment that needs repeats
(e.g. a coupling sweep across several current levels) must be unrolled into
the flat array by whatever generates the JSON (see
`tools/gen_coupling_experiment.py`), keeping the on-device queue linear.

## Example

```json
{
  "resolution_ms": 25,
  "initial_freq": 190.0,
  "initial_duty": [50, 50, 50, 50],
  "direction": "CW",
  "schedule": [
    { "method": "setDirection", "value": 0 },
    { "method": "label", "value": "CW_I100_SOLO_A" },
    { "method": "activateChannels", "mask": 1, "value": 100.0 },
    { "method": "addWaitTask", "duration_ms": 3000 },
    { "method": "activateChannels", "mask": 0, "value": 0.0 },
    { "method": "addWaitTask", "duration_ms": 2000 }
  ]
}
```

See `EXAMPLES.md` for more, and `example_profile.json` for a minimal file you
can load as-is.

## Usage

```cpp
JsonPhaseSequencer seq(&controller);
seq.loadFromJsonFile("/experiment.json");  // parses + compiles
seq.start();
// in loop(): seq.run();  isDone() reports queue exhaustion.
```
