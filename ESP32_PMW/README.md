# ESP32 PWM Coil Controller

Controls a 4-channel phased PWM coil array to spin and tilt a magnetic disk. Each channel has an independent phase offset and a carrier PWM for H-bridge current control.

---

## Upload

Each experiment is a self-contained firmware that loads its schedule from SPIFFS
and **runs on boot** (no arming). Upload the JSON payloads once, then the firmware:

```bash
pio run -e <env> --target uploadfs   # packs spiffs_data/*.json to flash (once)
pio run -e <env> --target upload     # builds + flashes the firmware
```

| Environment | Source file | Balance | Purpose |
|---|---|---|---|
| `flight` *(default)* | `main_flight.cpp` | PI | live PC-commanded takeoff → hover → directional accel |
| `servo` | `main_servo.cpp` | PI | dumb serial shim; the host owns the control loop |
| `takeoff_upside_down` | `main_takeoff_upside_down.cpp` | PI | CW, EASE 1→200 Hz over 25 s at 100% carrier, then overshoot to 210 Hz and settle at 190 |
| `comp_test` | `calibration/main_comp_test.cpp` | passthrough | per-channel trim A/B test |
| `coupling_test` | `calibration/main_coupling_test.cpp` | passthrough | coil-coupling sweep |
| `dc` | `calibration/main_dc.cpp` | passthrough | DC current-sense calibration |
| `serialcomm_demo` | `examples/main_serialcomm_demo.cpp` | — | `lib/SerialComm` echo demo, not a flight target |

All seven envs above build. `main_flight.cpp` was briefly missing the
declarations for its state variables (`state`, `collective`, `azSet`, `magSet`,
`clampf`, the mode enum, `SPINUP_THROTTLE`, `FREQ_MIN/MAX`); that block is back,
and `pio run -e flight` links at 6.8% RAM / 27.0% flash.

`platformio.ini` used to declare `tilt`, `takeoff`, `carrier_ramp`, `ceiling`,
`current_pid`, and `hover_zigzag` as well. Their sources were deleted in
`5866dd5`, so those env blocks are gone too. The three `calibration/` envs build
but need their JSON payloads, which were removed in the same commit -- recover
from git history before rerunning one.

"PI" = the current-balance loop (folded into `PwmController`, opt-in via
`enableCurrentBalance()`) rebalances the four channels beneath the schedule's
carrier ceiling. "passthrough" = the commanded carriers drive verbatim.

Monitor serial after upload:
```bash
pio device monitor -e <env>   # 115200 baud
```

Or chain both:
```bash
pio run -e flight --target upload && pio device monitor -e flight
```

The host-side Python is in two places: the vision → control pipeline that flies
the robot is in [`controller/`](controller/README.md), and the sweeps, sysID,
validation and plotting are in `ai/` (run with `uv run python ai/<script>.py`).

---

## Channel Map

| Index | Name | PWM pin | Carrier pin | ADC pin | Phase (CCW) |
|---|---|---|---|---|---|
| 0 | A | GPIO 32 | GPIO 33 | GPIO 36 | 90° |
| 1 | B | GPIO 25 | GPIO 26 | GPIO 39 | 270° |
| 2 | C | GPIO 18 | GPIO 19 | GPIO 34 | 180° |
| 3 | D | GPIO 22 | GPIO 23 | GPIO 35 | 0° |

Transcribed from the `#else` (classic ESP32) branch of `src/constants.h`, which
is the authority. GPIO 14 is `RESET_BUTTON_PIN`, not a coil pin.

Rotation order (CCW): A → C → B → D

---

## PwmController

Controls PWM frequency, duty cycle, phase, and carrier duty per channel. The
onboard current-sense reader and the current-balance PI loop are folded in as
opt-in capabilities (formerly a separate `CurrentBalanceController` + framework).

```cpp
#include "PwmController.h"

PwmController* controller = new PwmController(PWM_PINS, INITIAL_PHASES, INITIAL_DUTY_CYCLES, NUM_CHANNELS);
controller->begin(190.0f);   // pass a real freq; begin(0) divides by zero
controller->initCarrierPWM(CARRIER_PINS, PWM_FREQ, INITIAL_CARRIER_DUTY_CYCLES);

// Opt in to current sensing (+ overcurrent latch) and, for lift experiments,
// the balance loop. Call with coils OFF so the ADC zero seeds cleanly.
controller->enableCurrentSense(ADC_PINS, SENS, /*tripA*/10.0f);
controller->enableCurrentBalance();   // omit for open-loop passthrough

// In loop():
controller->run();   // drift-compensate + sample current + trip + balance
```

Key methods:

```cpp
controller->setGlobalFrequency(float hz);          // change freq on all channels (phase-continuous)
controller->setDutyCycle(int ch, float pct);        // 0–100%
controller->setPhase(int ch, float degrees);        // 0–360°
controller->setCarrierDutyCycle(int ch, float pct); // 0–100% H-bridge current; = the CEILING when balance is on
controller->getFrequency();                         // returns current freq (Hz)
controller->measuredCurrents();                     // float[4] sensed current (A), or nullptr
controller->overcurrentTripped();                   // true once the latch fired
```

Most experiment mains don't call these directly — `src/drive_common.h`
(`driveBoot` / `driveTelemetry`) wraps the boilerplate
so each `main_*.cpp` stays a short, explicit `setup()` + `loop()`.

---

## PwmSequencer

Queues time-based tasks (ramps, waits, phase snaps) and executes them against a PwmController.

```cpp
#include "PwmSequencer.h"

PwmSequencer* seq = new PwmSequencer(controller);

// One addRampTask for every quantity; TaskType picks it, TaskMode the curve
seq->addRampTask(1.0f, 200.0f, 15000, TaskType::PWM_FREQ, TaskMode::EASE);   // 1→200 Hz ease
seq->addRampTask(1.0f, 200.0f, 15000, TaskType::PWM_FREQ, TaskMode::LINEAR); // linear
seq->addWaitTask(3000);                                  // pause 3 s
// Per-channel ramp: NAN entries leave that channel unchanged
seq->addRampTask(startPhases, endPhases, NUM_CHANNELS, durationMs, TaskType::PWM_PHASE);

// Instant full-state set: build a TRAJECTORY_POINT task by hand and push it
SequenceTask snap = {};
snap.type = TaskType::TRAJECTORY_POINT;
snap.startFreq = snap.endFreq = 200.0f;
for (int i = 0; i < NUM_CHANNELS; i++) {
  snap.dutyCycles[i] = dutyCycles[i];
  snap.startPhases[i] = phases[i];
  snap.carrierDuties[i] = carriers[i];
}
seq->addSequenceTask(snap);

seq->compile(25, 1.0f, INITIAL_DUTY_CYCLES, INITIAL_PHASES); // 25 ms resolution
seq->start();

// In loop():
seq->run();
bool done = seq->isDone();
```

`compile()` must be called before `start()`. `resolutionMs` (first arg) is the trajectory timestep, 25 ms is typical.

---

## Scheduling Tasks

### Code-based (PwmSequencer)

Build the sequence in `setup()`, then execute it in `loop()`:

```cpp
// Ramp to speed, hold, reduce carrier, then stop
seq->addRampTask(1.0f, 200.0f, 15000, TaskType::PWM_FREQ, TaskMode::EASE); // 1→200 Hz ease
seq->addWaitTask(5000);                        // hold 5 s

seq->addRampTask(100.0f, 50.0f, 2000, TaskType::CARRIER_DUTY); // ramp all carriers to 50%

seq->addWaitTask(3000);
seq->compile(25, 1.0f, INITIAL_DUTY_CYCLES, INITIAL_PHASES);
seq->start();
```

Check completion and react:
```cpp
void loop() {
  controller->run();
  seq->run();
  if (seq->isDone()) { /* sequence finished */ }
}
```

---

### JSON file (JsonPwmSequencer)

Upload a `.json` file to SPIFFS, then load it at startup. The file is an object
carrying the initial state plus a `schedule` array, each entry naming a
`PwmSequencer` method and its arguments, applied in array order (not by a
timestamp field):

```json
{
  "resolution_ms": 25,
  "initial_freq": 190.0,
  "direction": "CCW",
  "schedule": [
    { "method": "addDutyCycleTask",       "channels": 0, "value": 60.0 },
    { "method": "addPhaseRampTask",       "channels": 1, "from": 0.0, "to": 90.0, "duration_ms": 500 },
    { "method": "addCarrierDutyCycleTask","channels": 0, "value": 75.0 },
    { "method": "addWaitTask",            "duration_ms": 3000 }
  ]
}
```

A bare top-level array is still accepted, with every config key at its default.

`method` is one of: `addDutyCycleTask` / `addPhaseTask` / `addCarrierDutyCycleTask`
(instant, per-channel set), `addWaitTask`, `addLinearRampTask` /
`addEaseRampTask` / `addExponentialRampTask` (global frequency ramp),
`addCarrierRampTask` / `addCarrierEaseRampTask` / `addCarrierExponentialRampTask`
(all channels), `addPhaseRampTask` (per-channel), `setDirection`,
`activateChannels`, or `label`. Unrecognized methods are skipped and logged to
serial.

Every ramp method takes an optional `"shape"` that tunes its curve.
`addLinearRampTask` / `addCarrierRampTask` are power ramps `t^p` — `shape` is
the power `p>0`, defaulting to 1, i.e. the straight line they have always
produced. See `lib/JsonPwmSequencer/README.md` for the full table.

```cpp
#include "JsonPwmSequencer.h"
JsonPwmSequencer* seq = new JsonPwmSequencer(controller);
seq->loadFromJsonFile("/schedule.json");
seq->start();
// call seq->run() in loop()
```

---

## Further Reading

- Firmware libraries: [`lib/PwmController/`](lib/PwmController/README.md),
  [`lib/PwmSequencer/`](lib/PwmSequencer/README.md),
  [`lib/JsonPwmSequencer/`](lib/JsonPwmSequencer/README.md),
  [`lib/SerialComm/`](lib/SerialComm/README.md)
- The vision → control pipeline that drives this firmware:
  [`controller/README.md`](controller/README.md)
- Schedule payloads: [`spiffs_data/README.md`](spiffs_data/README.md)
- Captured experiment data: [`data/README.md`](data/README.md)
- Board / power stage: [`docs/PCB_Design_Documentation.md`](docs/PCB_Design_Documentation.md)
