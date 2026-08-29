# ESP32 PWM Coil Controller

Controls a 4-channel phased PWM coil array to spin and tilt a magnetic disk. Each channel has an independent phase offset and a carrier PWM for H-bridge current control.

---

## Upload

The firmware is self-contained: it loads its schedule from SPIFFS and **runs on
boot** (no arming). Upload the JSON payload once, then the firmware:

```bash
pio run -e swim --target uploadfs   # packs spiffs_data/*.json to flash (once)
pio run -e swim --target upload     # builds + flashes the firmware
```

| Environment | Source file | Schedule | Balance | Purpose |
|---|---|---|---|---|
| `swim` | `main_swim.cpp` | `spiffs_data/swim.json` | PI | 1→30 Hz ramp over 20 s, then 30↔22 Hz undulation ×5 |

"PI" = the current-balance loop (folded into `PwmController`, opt-in via
`enableCurrentBalance()`) rebalances the four channels beneath the schedule's
carrier ceiling. Omitting `enableCurrentBalance()` gives "passthrough", where the
commanded carriers drive verbatim.

Monitor serial after upload:
```bash
pio device monitor -e swim   # 115200 baud
```

Or chain both:
```bash
pio run -e swim --target upload && pio device monitor -e swim
```

Or drive the whole loop — uploadfs, flash, reset, capture — from the host:

```bash
uv run python ai/run_swim.py --capture-s 40 --log swim_run.log
```

Analysis / automation Python lives in `ai/` (run with `uv run python ai/<script>.py`).

---

## Channel Map

Authoritative source: [`src/constants.h`](src/constants.h). Two maps, selected by
`SWIM_SETUP`.

**Swim rig — DRV8874 on an Adafruit ESP32 Feather V2, `SWIM_SETUP=1`,
`GATE_ACTIVE_LOW=0`:**

| Index | Name | PH pin (direction) | EN pin (carrier) | CCW phase |
|---|---|---|---|---|
| 0 | A | GPIO 14 (D14) | GPIO 26 (A0) | 90° |
| 1 | B | GPIO 32 (D32) | GPIO 25 (A1) | 270° |
| 2 | C | GPIO 33 (D33) | GPIO 21 (MISO) | 180° |
| 3 | D | GPIO 27 (D27) | GPIO 4 (A5) | 0° |

Two Feather traps the `isOutputCapable` `static_assert` in `constants.h` now
catches at compile time: the **A2/A3/A4** labels are GPIO 34/39/36, which are
input-only on the ESP32 and cannot drive anything, and GPIO 5/12/15 are
boot-strap pins — a boot-time high on an `EN` line energizes that coil before
`setup()` runs.

The DRV8874 runs in **PH/EN mode**, which needs `PMODE` tied logic low and
`nSLEEP` high; `PMODE` is latched on the `nSLEEP` rising edge, so it cannot be
changed while running. `PWM_PINS` drive `PH` (direction), `CARRIER_PINS` drive
`EN` at `PWM_FREQ` (5 kHz) for amplitude, and the driver gates the two together
in silicon:

| nSLEEP | EN | PH | OUT1 | OUT2 | |
|---|---|---|---|---|---|
| 1 | 0 | X | L | L | Brake — **0 % carrier is a real off** |
| 1 | 1 | 0 | L | H | Reverse |
| 1 | 1 | 1 | H | L | Forward |

`EN/IN1` and `PH/IN2` carry internal 100 kΩ pulldowns, so the outputs stay
Hi-Z until the ESP32 configures its pins — boot is safe. `IPROPI` reports
450 µA/A with no shunt resistor, which is how current sense (and with it the PI
balance loop, inert on swim today) would come back: pick `RIPROPI`, route it to
an ADC pin, and set `ADC_PINS` / `SENS`.

**Flight rig — VNH5019 + NC7SVU04, `SWIM_SETUP=0`, `GATE_ACTIVE_LOW=1`:**

| Index | Name | PWM pin | Carrier pin | Current-sense ADC | CCW phase |
|---|---|---|---|---|---|
| 0 | A | GPIO 32 | GPIO 33 | GPIO 36 | 90° |
| 1 | B | GPIO 25 | GPIO 26 | GPIO 39 | 270° |
| 2 | C | GPIO 18 | GPIO 19 | GPIO 34 | 180° |
| 3 | D | GPIO 22 | GPIO 23 | GPIO 35 | 0° |

Here the phase pin feeds an inverter before `INA`/`INB`, so energizing a coil
means driving the pin **low** — that is what `GATE_ACTIVE_LOW` selects. Set it
wrong and every channel sits 180° from its commanded phase: the field still
rotates, so it fails quietly.

Rotation order (CCW): A → C → B → D. CW swaps A and B to 270°/90°; both phase
sets live in `src/drive_common.h` as `PHASES_CCW` / `PHASES_CW`.

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

`main_swim.cpp` doesn't call most of these directly — `src/drive_common.h`
(`driveBoot` / `driveTelemetry`, plus the shared `PHASES_*` / `INITIAL_DUTY` /
`CARRIER_ZERO` / `SENS` constants) wraps the boilerplate so the main stays a
short, explicit `setup()` + `loop()`.

---

## PhaseSequencer

Queues time-based tasks (ramps, waits, phase snaps) and executes them against a PwmController.

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

## The swim task

`main_swim.cpp` is 20 lines: boot, configure the carrier, enable current sense
and the balance loop, load `/swim.json`, run. All the motion lives in the JSON.

The schedule ramps the global commutation frequency 1 → 30 Hz over 20 s at 100%
carrier, then undulates 30 ↔ 22 Hz five times (1 s each way) to produce a stroke,
then cuts the coils. 30 Hz is well below the 150–210 Hz flight regime, so the
stroke bounds are seeds to tune on hardware, not derived values.

Regenerate the schedule rather than hand-editing it:

```bash
uv run python ai/gen_swim_experiment.py                        # the committed defaults
uv run python ai/gen_swim_experiment.py --strokes 3 --ramp-mode ease
uv run python ai/gen_swim_experiment.py --spinup-hz 25 --stroke-low-hz 18
```

The generator only emits methods the on-device parser already understands — its
job is unrolling the stroke cycles, since repeat is not a queue primitive. Each
phase is preceded by a `label` (`SWIM_SPINUP_1_30HZ`, `SWIM_STROKE_01_DOWN`, …,
`SWIM_OFF`) so `labelForStep()` and `ai/plot_serial_log.py` can segment a run.

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
  { "method": "addDutyCycleTask",       "channels": 0, "value": 60.0 },
  { "method": "addPhaseRampTask",       "channels": 1, "from": 0.0, "to": 90.0, "duration_ms": 500 },
  { "method": "addCarrierDutyCycleTask","channels": 0, "value": 75.0 },
  { "method": "addWaitTask",            "duration_ms": 3000 }
]
```

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
produced. See `lib/JsonPhaseSequencer/README.md` for the full table.

```cpp
#include "JsonPhaseSequencer.h"
JsonPhaseSequencer* seq = new JsonPhaseSequencer(controller);
seq->loadFromJsonFile("/schedule.json");
seq->start();
// call seq->run() in loop()
```

Upload the data file to SPIFFS with: `pio run -e swim --target uploadfs`

---

## Further Reading

- Full API docs: [`DOCS.md`](DOCS.md)
- Library details: [`lib/PwmController/README.md`](lib/PwmController/README.md), [`lib/PhaseSequencer/README.md`](lib/PhaseSequencer/README.md)
