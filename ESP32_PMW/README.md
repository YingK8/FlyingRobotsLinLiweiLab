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

seq->addEaseRampTask(1.0f, 200.0f, 15000);              // cubic ease ramp, 1→200 Hz over 15 s
seq->addLinearRampTask(1.0f, 200.0f, 15000);            // linear ramp
seq->addWaitTask(3000);                                  // pause 3 s
seq->addDutyCycleTask(dutyCycles, NUM_CHANNELS);         // snap duty cycles
seq->addPhaseTask(phases, NUM_CHANNELS);                 // snap phases
seq->addCarrierDutyCycleTask(carriers, NUM_CHANNELS);   // snap carrier duties
seq->addPhaseRampTask(startPhases, endPhases, durationMs);

seq->compile(25, 1.0f, INITIAL_DUTY_CYCLES, INITIAL_PHASES); // 25 ms resolution
seq->start();

// In loop():
seq->run();
bool done = seq->isDone();
```

`compile()` must be called before `start()`. `resolutionMs` (first arg) is the trajectory timestep — 25 ms is typical.

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
seq->addEaseRampTask(1.0f, 200.0f, 15000);   // 1→200 Hz, 15 s ease ramp
seq->addWaitTask(5000);                        // hold 5 s

float carriers[4] = {50.0, 50.0, 50.0, 50.0};
seq->addCarrierDutyCycleTask(carriers, 4);     // snap all carriers to 50%

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

Upload a `.json` file to SPIFFS, then load it at startup. Each entry fires at `time_ms`:

```json
[
  { "time_ms": 0,    "command": "set",      "channel": 0, "parameter": "frequency", "value": 100.0 },
  { "time_ms": 100,  "command": "ramp",     "channel": 0, "parameter": "duty",      "from": 50.0, "to": 80.0, "duration_ms": 500 },
  { "time_ms": 700,  "command": "easeRamp", "channel": 1, "parameter": "phase",     "from": 0.0,  "to": 90.0, "duration_ms": 300 }
]
```

Commands: `set` (instant), `ramp` (linear), `easeRamp` (cubic ease-in-out).  
Parameters: `frequency`, `duty`, `phase`, `carrier_duty`.

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
