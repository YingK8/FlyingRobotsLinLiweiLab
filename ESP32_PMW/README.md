# ESP32 PWM Coil Controller

Controls a 4-channel phased PWM coil array to spin and tilt a magnetic disk. Each channel has an independent phase offset and a carrier PWM for H-bridge current control.

---

## Upload

```bash
pio run -e <env> --target upload
```

| Environment | Source file | Purpose |
|---|---|---|
| `tilt` | `main_tilt.cpp` | Full carrier sweep 100% → 0%, channel D only |
| `tilt1` | `main_tilt1.cpp` | High-half sweep 100% → 50%, channels B + D |
| `tilt2` | `main_tilt2.cpp` | Low-half sweep 50% → 0%, channels A + C |
| `takeoff` | `main_takeoff.cpp` | Takeoff sequence |

Monitor serial after upload:
```bash
pio device monitor -e <env>   # 115200 baud
```

Or chain both:
```bash
pio run -e tilt1 --target upload && pio device monitor -e tilt1
```

---

## Channel Map

| Index | Name | PWM pin | Carrier pin | Phase |
|---|---|---|---|---|
| 0 | A | GPIO 32 | GPIO 33 | 0° |
| 1 | B | GPIO 25 | GPIO 26 | 180° |
| 2 | C | GPIO 27 | GPIO 14 | 90° |
| 3 | D | GPIO 23 | GPIO 13 | 270° |

Rotation order (CCW): A → C → B → D

---

## PhaseController

Controls PWM frequency, duty cycle, phase, and carrier duty per channel.

```cpp
#include "PhaseController.h"

PhaseController* controller = new PhaseController(PWM_PINS, INITIAL_PHASES, INITIAL_DUTY_CYCLES, NUM_CHANNELS);
controller->begin();
controller->initCarrierPWM(CARRIER_PINS, PWM_FREQ, INITIAL_CARRIER_DUTY_CYCLES);

// In loop():
controller->run();
```

Key methods:

```cpp
controller->setGlobalFrequency(float hz);          // change freq on all channels (phase-continuous)
controller->setDutyCycle(int ch, float pct);        // 0–100%
controller->setPhase(int ch, float degrees);        // 0–360°
controller->setCarrierDutyCycle(int ch, float pct); // 0–100%, controls H-bridge current
controller->getFrequency();                         // returns current freq (Hz)
```

---

## PhaseSequencer

Queues time-based tasks (ramps, waits, phase snaps) and executes them against a PhaseController.

```cpp
#include "PhaseSequencer.h"

PhaseSequencer* seq = new PhaseSequencer(controller);

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

## Carrier Sweep Pattern (tilt1 / tilt2)

Both files ramp to 200 Hz with all channels at 100% carrier, then:

- **tilt1**: steps B + D down from 100% → 50% in −10% steps, 3 s per step
- **tilt2**: steps A + C down from 50% → 0% in −10% steps, 3 s per step (B + D stay at 100%)

LED on GPIO 2 blinks each step. Run **tilt2 first** (lower power, safer).

---

## Scheduling Tasks

### Code-based (PhaseSequencer)

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

### JSON file (JsonPhaseSequencer)

Upload a `.json` file to SPIFFS, then load it at startup. It's an array of
entries, each naming a `PhaseSequencer` method and its arguments, applied in
array order (not by a timestamp field):

```json
[
  { "method": "addDutyCycleTask",       "channel": 0, "value": 60.0 },
  { "method": "addPhaseRampTask",       "channel": 1, "from": 0.0, "to": 90.0, "duration_ms": 500 },
  { "method": "addCarrierDutyCycleTask","channel": 0, "value": 75.0 },
  { "method": "addWaitTask",            "duration_ms": 3000 }
]
```

`method` is one of: `addDutyCycleTask` / `addPhaseTask` / `addCarrierDutyCycleTask`
(instant, per-channel set), `addWaitTask`, `addLinearRampTask` / `addEaseRampTask`
(global frequency ramp), `addCarrierRampTask` / `addCarrierEaseRampTask` (all
channels), or `addPhaseRampTask` (per-channel). Unrecognized methods are
skipped and logged to serial.

```cpp
#include "JsonPhaseSequencer.h"
JsonPhaseSequencer* seq = new JsonPhaseSequencer(controller);
seq->loadFromJsonFile("/schedule.json");
seq->start();
// call seq->run() in loop()
```

---

### CSV waypoints (CsvPhaseSequencer)

Define channel states at each timestamp; the sequencer interpolates between them:

```csv
# channels, 4
# step_size_ms, 25
# interpolation, linear
time,channel,duty,carrier_duty,frequency,phase
0,0,50,100,10,0
0,1,50,100,10,90
0,2,50,100,10,180
0,3,50,100,10,270
500,0,60,80,15,0
500,1,60,80,15,90
500,2,60,80,15,180
500,3,60,80,15,270
```

Interpolation modes: `linear`, `hermite` / `ease` (smoothstep).

```cpp
#include "CsvPhaseSequencer.h"
CsvPhaseSequencer* seq = new CsvPhaseSequencer(controller);
seq->loadFromCSVFile("/profile.csv");
seq->start();
// call seq->run() in loop()
```

Upload the data file to SPIFFS with: `pio run -e <env> --target uploadfs`

---

## Further Reading

- Full API docs: [`DOCS.md`](DOCS.md)
- Library details: [`lib/PhaseController/README.md`](lib/PhaseController/README.md), [`lib/PhaseSequencer/README.md`](lib/PhaseSequencer/README.md)
