# JsonPhaseSequencer Example JSON Schedules

## Example 1: Set Carrier Duty Cycle
```json
[
  { "method": "addCarrierDutyCycleTask", "channel": 0, "value": 75.0 }
]
```

## Example 2: Sequence Multiple Carrier Duty Changes
```json
[
  { "method": "addCarrierDutyCycleTask", "channel": 0, "value": 50.0 },
  { "method": "addWaitTask", "duration_ms": 500 },
  { "method": "addCarrierDutyCycleTask", "channel": 0, "value": 80.0 }
]
```

## Example 3: Mix with Other PWM Tasks
```json
[
  { "method": "addDutyCycleTask", "channel": 1, "value": 60.0 },
  { "method": "addCarrierDutyCycleTask", "channel": 2, "value": 90.0 },
  { "method": "addWaitTask", "duration_ms": 1000 },
  { "method": "addCarrierDutyCycleTask", "channel": 2, "value": 10.0 }
]
```

## Example 4: Coupling-characterization segment (direction + combo + label)

One current level, one direction, one solo + one pair — the pattern
`tools/gen_coupling_experiment.py` repeats across all 11 combos (4 solos + 6
pairs + ALL) and every current level, for both CW and CCW:

```json
[
  { "method": "setDirection", "value": 0 },
  { "method": "label", "value": "CW_I100_SOLO_A" },
  { "method": "activateChannels", "mask": 1, "value": 100.0 },
  { "method": "addWaitTask", "duration_ms": 3000 },
  { "method": "activateChannels", "mask": 0, "value": 0.0 },
  { "method": "addWaitTask", "duration_ms": 2000 },
  { "method": "label", "value": "CW_I100_PAIR_AB" },
  { "method": "activateChannels", "mask": 3, "value": 100.0 },
  { "method": "addWaitTask", "duration_ms": 3000 },
  { "method": "activateChannels", "mask": 0, "value": 0.0 },
  { "method": "addWaitTask", "duration_ms": 2000 }
]
```

`mask` is a 4-bit channel bitmask (bit 0=A, 1=B, 2=C, 3=D), so `1`=solo A,
`3`=A+B, `15`=ALL. The firmware prints `t=.. label=.. mask=..` whenever the
label changes — that's what `tools/coupling_matrix.py --segments-log` uses to
find each segment's window in a paired PicoScope capture, instead of assuming
a fixed segment count/duration.

---

See DOCS.md for more details and main.cpp for integration examples.
