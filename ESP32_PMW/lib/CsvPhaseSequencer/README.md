# JsonPhaseSequencer

A scheduler for PhaseSequencer that loads a JSON file describing time-stamped control profiles (set, ramp, easeRamp) and executes them at the correct time. Supports linear and polynomial (ease) interpolation.

## Example JSON

```
[
  { "time_ms": 0, "command": "set", "channel": 0, "parameter": "frequency", "value": 100.0 },
  { "time_ms": 100, "command": "ramp", "channel": 0, "parameter": "duty", "from": 50.0, "to": 80.0, "duration_ms": 500 },
  { "time_ms": 700, "command": "easeRamp", "channel": 1, "parameter": "phase", "from": 0.0, "to": 90.0, "duration_ms": 300 }
]
```

- `set`: Directly sets a parameter at a timestamp.
- `ramp`: Linearly interpolates from `from` to `to` over `duration_ms`.
- `easeRamp`: Uses cubic ease-in-out interpolation from `from` to `to` over `duration_ms`.

## Usage
- Call `loadFromJsonFile("/path/to/file.json")` at startup.
- Call `run()` in your main loop.
