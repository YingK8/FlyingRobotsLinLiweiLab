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

---

See DOCS.md for more details and main.cpp for integration examples.
