# JsonPhaseSequencer

A `PhaseSequencer` that loads a schedule from a JSON file (SPIFFS) and compiles
it into a runnable queue. The file is a **flat JSON array** of method-call
objects, executed in order — not a schedule of absolute timestamps.

## Schema

Each array entry is an object:

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
| `addDutyCycleTask` | `channel`, `value` | instantly set one channel's commutation duty (0-100%) |
| `addPhaseTask` | `channel`, `value` | instantly set one channel's phase (degrees) |
| `addCarrierDutyCycleTask` | `channel`, `value` | instantly set one channel's carrier duty (0-100%) |
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
[
  { "method": "setDirection", "value": 0 },
  { "method": "label", "value": "CW_I100_SOLO_A" },
  { "method": "activateChannels", "mask": 1, "value": 100.0 },
  { "method": "addWaitTask", "duration_ms": 3000 },
  { "method": "activateChannels", "mask": 0, "value": 0.0 },
  { "method": "addWaitTask", "duration_ms": 2000 }
]
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
