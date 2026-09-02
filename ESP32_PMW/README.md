# ESP32 PWM Coil Controller

Controls a 4-channel phased PWM coil array to spin and tilt a magnetic disk. Each channel has an independent phase offset and a carrier PWM for H-bridge current control.

---

## Upload

Every `src/main_*.cpp` defines its own `setup()`/`loop()`, so exactly one may be
compiled per build — hence one PlatformIO env per experiment, each selecting its
sketch with `build_src_filter`. Most firmwares load their schedule from SPIFFS and
**run on boot** (no arming).

```bash
pio run -e tilt --target uploadfs   # packs ALL of spiffs_data/*.json to flash (once)
pio run -e tilt --target upload     # builds + flashes the firmware
```

`uploadfs` uploads the whole `spiffs_data/` directory regardless of which env you
name, so you only need it once after editing any JSON — then just re-`upload` the
firmware you want to run.

| Environment | Source file | Schedule | Balance | Purpose |
| --- | --- | --- | --- | --- |
| `flight` *(default)* | `main_flight.cpp` | none — live serial commands | PI | PC-commanded takeoff → hover → directional accel (`SerialComm`) |
| `takeoff` | `main_takeoff.cpp` | `takeoff.json` | **off** (commented out) | EASE 1→200 Hz / 40 s at 100% carrier |
| `takeoff_upside_down` | `main_takeoff_upside_down.cpp` | `takeoff_upside_down.json` | PI | inverted-rig takeoff ramp, 100% carrier |
| `tilt` | `main_tilt.cpp` | `tilt.json` | PI | EASE 1→150 Hz, then steps channels A+D down 90%→10% in 10% steps |
| `hover_zigzag` | `main_hover_zigzag.cpp` | `hover_zigzag.json` | PI | 1→160 Hz, then 160↔140 Hz zigzag ×5 at 100% carrier |
| `ceiling` | `main_ceiling.cpp` | `ceiling_sweep.json` | passthrough | all 4 at 100% carrier over an EASE 1→210 Hz ramp — gives `Ceiling_i(f)` |
| `carrier_ramp` | `main_carrier_ramp.cpp` | `carrier_ramp.json` | PI | carrier 0→100% over 10 s at a fixed 190 Hz |
| `coupling_test` | `calibration/main_coupling_test.cpp` | `coupling_cw.json` | passthrough | coil-coupling sweep: solo + pairwise + all-4 |
| `comp_test` | `calibration/main_comp_test.cpp` | `comp_test.json` | passthrough | BASELINE → GAP → TRIMMED per-channel A/B |
| `dc` | `calibration/main_dc.cpp` | `dc_calibration.json` | passthrough | pins parked HIGH for a DC current-sense calibration capture |
| `current_pid` | `main_current_pid.cpp` | none — built in code | own PI | balance-loop tuning rig; gains adjustable at runtime |
| `serialcomm_demo` | `examples/main_serialcomm_demo.cpp` | none | n/a | `lib/SerialComm` echo demo, not a flight experiment |

"PI" = the current-balance loop (folded into `PwmController`, opt-in via
`enableCurrentBalance()`) rebalances the four channels beneath the schedule's
carrier ceiling. Omitting `enableCurrentBalance()` gives "passthrough", where the
commanded carriers drive verbatim. The calibration rigs are passthrough on purpose:
they deliberately drive the channels unequally, so balancing them would erase the
effect being measured.

> `main_takeoff.cpp:14` has `enableCurrentBalance()` **commented out** even though
> its header comment still says "PI-balanced". The table above reports what the code
> does, not what the comment claims. Uncomment it to restore balancing.

Monitor serial after upload:
```bash
pio device monitor -e tilt   # 115200 baud
```

Or chain both:
```bash
pio run -e tilt --target upload && pio device monitor -e tilt
```

Host-side tooling — the schedule generators, the flash-and-capture runners, and the
log plotters — lives in `ai/`, which `.gitignore` keeps out of the repo on purpose.
It only exists in working copies of the experiment branches.

---

## Channel Map

Authoritative source: [`src/constants.h`](src/constants.h).

| Index | Name | PWM pin | Carrier pin | Current-sense ADC | CCW phase |
| --- | --- | --- | --- | --- | --- |
| 0 | A | GPIO 32 | GPIO 33 | GPIO 36 | 90° |
| 1 | B | GPIO 25 | GPIO 26 | GPIO 39 | 270° |
| 2 | C | GPIO 18 | GPIO 19 | GPIO 34 | 180° |
| 3 | D | GPIO 22 | GPIO 23 | GPIO 35 | 0° |

Rotation order (CCW): A → C → B → D. CW swaps A and B to 270°/90°.

This is the `SWIM_SETUP=0` map — the build flag every env inherits from `[env]` in
`platformio.ini`. `-D SWIM_SETUP=1` selects a different board's PWM pins
(27/12/15/33) with no carrier or ADC pins at all (`constants.h:18-32`). No env sets
it today, so that branch is currently dead code.

---

## Where configuration lives

Four files own four different things. Knowing which is which saves a lot of
"why didn't my change take effect".

| File | Owns | Read by |
| --- | --- | --- |
| [`src/constants.h`](src/constants.h) | **board/hardware**: `PWM_PINS`, `CARRIER_PINS`, `ADC_PINS`, `PWM_FREQ` (20 kHz carrier), `LED_PIN`, `NUM_CHANNELS`, `RESET_BUTTON_PIN` | everything, via `drive_common.h` |
| [`src/drive_common.h`](src/drive_common.h) | **drive constants + the boot/telemetry idiom**: `PHASES_CW`, `PHASES_CCW`, `INITIAL_DUTY`, `CARRIER_ZERO`, `SENS`, `driveBoot()`, `driveTelemetry()` | every `main_*.cpp` except `main_current_pid.cpp` |
| [`spiffs_data/*.json`](spiffs_data/) | **the motion**: frequency/carrier/phase schedule, direction, labels | the matching firmware at boot, via `JsonPwmSequencer` |
| [`platformio.ini`](platformio.ini) | **which sketch builds** (`build_src_filter`) and the shared build flags | the build |

### `src/drive_common.h`

One header, included by nine of the eleven firmwares. It transitively pulls in
`constants.h`, `reset_button.h`, `safety_startup.h`, `telemetry.h`,
`PwmController.h` and `JsonPwmSequencer.h`, so `#include "drive_common.h"` is
normally the only include a `main_*.cpp` needs.

```cpp
static const float PHASES_CW[NUM_CHANNELS]   = {270.0f,  90.0f, 180.0f, 0.0f};
static const float PHASES_CCW[NUM_CHANNELS]  = { 90.0f, 270.0f, 180.0f, 0.0f};
static const float INITIAL_DUTY[NUM_CHANNELS]= { 50.0f,  50.0f,  50.0f, 50.0f};
static const float CARRIER_ZERO[NUM_CHANNELS]= {  0.0f,   0.0f,   0.0f,  0.0f};
static const float SENS[NUM_CHANNELS] = {15.26f, 15.28f, 15.57f, 15.34f};
```

- `PHASES_CW` / `PHASES_CCW` — the project's A/B/C/D phase convention per rotation
  direction. CW and CCW differ only in channels A and B.
- `INITIAL_DUTY` — **commutation** duty (the square wave that spins the field), not
  carrier duty. 50% is the standard drive.
- `CARRIER_ZERO` — passed to `initCarrierPWM()` so the coils come up **off**; the
  schedule (or `setCarrierDutyCycle()`) raises them.
- `SENS` — VNH5019 current-sense gain, A per V, measured per board. Change these
  only after re-running the `dc` calibration; they scale every current reading and
  therefore the balance loop and the overcurrent trip.

Two helpers keep the mains short:

- `driveBoot()` — serial at 115200, `forceAllGatesLow()` **before any driver
  exists**, reset button, LED on, SPIFFS mount. Call it first in `setup()`, before
  `ctl.begin()`, so the coils can't glitch on and the ADC zero captured by
  `enableCurrentSense()` is taken against a true-off baseline.
- `driveTelemetry(ctl)` — polls the block/restart button every loop, then prints the
  2 Hz line the log parsers expect:
  `t=.. freq=.. | I[A]: .. | duty[%]: .. | spread=.. bal=.. trip=..`

### Where CW/CCW actually gets decided

Phases can be set in four places, and **later ones win**:

1. **The constructor**, from `drive_common.h` — sets the boot/idle phases:
   `PwmController ctl(PWM_PINS, PHASES_CCW, INITIAL_DUTY, NUM_CHANNELS);`
2. **The JSON's top-level `"direction"`** (`"CW"` / `"CCW"`, default CCW) — reseeds
   all four phases when the schedule is compiled
   ([`JsonPwmSequencer.cpp:69-78`](lib/JsonPwmSequencer/JsonPwmSequencer.cpp)).
3. **The JSON's optional `"initial_phase": [4]`** — explicit per-channel override of
   step 2.
4. **Mid-schedule steps** — `{"method": "setDirection", "value": 0|1}` (0 = CW,
   non-zero = CCW) snaps all four back to the convention; `addPhaseTask` /
   `addPhaseRampTask` set individual channels to arbitrary angles.

**The JSON overrides the constructor.** Several firmwares are constructed with one
direction and then run in the other, which is not a bug but is easy to misread:

| Firmware | Constructor | JSON `direction` | Actually runs |
| --- | --- | --- | --- |
| `main_tilt.cpp` | `PHASES_CCW` | `"CW"` | **CW** |
| `main_hover_zigzag.cpp` | `PHASES_CW` | `"CCW"` | **CCW** |
| `main_takeoff_upside_down.cpp` | `PHASES_CW` | `"CCW"` | **CCW** |

`takeoff.json` and `carrier_ramp.json` go a step further: after declaring
`"direction": "CCW"` they use `addPhaseTask` to set `{0, 180, 90, 270}` — the CCW
set rotated by −90°, so same rotation, different starting angle. If you want to know
which way a rig will spin, read the JSON, not the constructor.

No JSON currently uses mid-schedule `setDirection`.

### The three copies of `PHASES_*`

Same values, three definitions — worth knowing before you edit one:

| Location | Linkage | Used by |
| --- | --- | --- |
| `src/drive_common.h:13-14` | `static` (per translation unit) | the `main_*.cpp` files, at construction |
| `lib/JsonPwmSequencer/JsonPwmSequencer.cpp:11-12` | library globals | the `direction` / `setDirection` path, at compile-schedule time |
| `src/main_current_pid.cpp:29-31` | file-local | that sketch only — it does **not** include `drive_common.h` |

Changing the convention means changing all three. `main_current_pid.cpp` also
re-declares `INITIAL_DUTY_CYCLES` and its own `SENS`; it is the one firmware that
does not share the common header.

---

## Anatomy of a firmware

[`src/main_tilt.cpp`](src/main_tilt.cpp) is the canonical JSON-driven firmware — 16
lines of setup, all the motion in the JSON:

```cpp
#include "drive_common.h"

PwmController ctl(PWM_PINS, PHASES_CCW, INITIAL_DUTY, NUM_CHANNELS);
JsonPwmSequencer seq(&ctl);

void setup() {
  driveBoot();                                          // serial, gates LOW, LED, SPIFFS
  ctl.begin();                                          // DC; the schedule sets the freq
  ctl.initCarrierPWM(CARRIER_PINS, PWM_FREQ, CARRIER_ZERO);  // coils up OFF
  ctl.enableCurrentSense(ADC_PINS, SENS);               // + overcurrent latch
  ctl.enableCurrentBalance();                           // omit for passthrough
  seq.loadFromJsonFile("/tilt.json");
  seq.start();
}

void loop() {
  seq.run();
  ctl.run();
  driveTelemetry(ctl);   // 2 Hz log line + reset-button poll
}
```

Every other JSON-driven env is this file with a different phase array, JSON
filename, and `enableCurrentBalance()` present or absent.

[`src/main_flight.cpp`](src/main_flight.cpp) is the variant with no JSON: it takes
live newline-terminated commands over serial (`takeoff`, `throttle=`, `az=`, `mag=`,
`hover`, `land`, `stop`, `freq=`) and mixes a thrust vector by dropping the
az-facing coils' carrier ceilings beneath the balance loop.

---

## PwmController

Controls PWM frequency, duty cycle, phase, and carrier duty per channel. The
onboard current-sense reader and the current-balance PI loop are folded in as
opt-in capabilities (formerly a separate `CurrentBalanceController` + framework).

```cpp
#include "drive_common.h"   // brings in PwmController.h + all the constants below

PwmController ctl(PWM_PINS, PHASES_CCW, INITIAL_DUTY, NUM_CHANNELS);
ctl.begin(190.0f);   // pass a real freq; begin(0) is DC (stationary field)
ctl.initCarrierPWM(CARRIER_PINS, PWM_FREQ, CARRIER_ZERO);

// Opt in to current sensing (+ overcurrent latch) and, for lift experiments,
// the balance loop. Call with coils OFF so the ADC zero seeds cleanly.
ctl.enableCurrentSense(ADC_PINS, SENS, /*tripA*/10.0f);
ctl.enableCurrentBalance();   // omit for open-loop passthrough

// In loop():
ctl.run();   // drift-compensate + sample current + trip + balance
```

Key methods:

```cpp
ctl.setGlobalFrequency(float hz);          // change freq on all channels (phase-continuous)
ctl.setDutyCycle(int ch, float pct);        // 0–100% commutation duty
ctl.setPhase(int ch, float degrees);        // 0–360°
ctl.setCarrierDutyCycle(int ch, float pct); // 0–100% H-bridge current; = the CEILING when balance is on
ctl.getFrequency();                         // returns current freq (Hz)
ctl.measuredCurrents();                     // float[4] sensed current (A), or nullptr
ctl.overcurrentTripped();                   // true once the latch fired
```

Most firmwares call almost none of these directly — `drive_common.h` wraps the
boilerplate (see [Where configuration lives](#where-configuration-lives)) so each
main stays a short, explicit `setup()` + `loop()`.

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

seq->compile(25, 1.0f, INITIAL_DUTY, PHASES_CCW); // 25 ms resolution
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
seq->compile(25, 1.0f, INITIAL_DUTY, PHASES_CCW);
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

Upload a `.json` file to SPIFFS, then load it at startup. The file is either a bare
array of steps, or an object with the run's initial conditions plus a `schedule`
array. Steps are applied in array order (not by a timestamp field):

```json
{
  "resolution_ms": 25,
  "initial_freq": 190.0,
  "initial_duty": [50, 50, 50, 50],
  "direction": "CCW",
  "schedule": [
    { "method": "addDutyCycleTask",        "channels": 0, "value": 60.0 },
    { "method": "addPhaseRampTask",        "channels": 1, "from": 0.0, "to": 90.0, "duration_ms": 500 },
    { "method": "addCarrierDutyCycleTask", "channels": 0, "value": 75.0 },
    { "method": "addWaitTask",             "duration_ms": 3000 }
  ]
}
```

The top-level keys are the run's starting state — `direction` (and the optional
`initial_phase` array) is where a schedule's rotation direction is set, overriding
the phase array the firmware was constructed with. `// line` and `/* block */`
comments are allowed, courtesy of `-D ARDUINOJSON_ENABLE_COMMENTS=1`.

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

Upload the data files to SPIFFS with: `pio run -e tilt --target uploadfs`

---

## Further Reading

- Full API docs: [`DOCS.md`](DOCS.md)
- Schedule payloads: [`spiffs_data/README.md`](spiffs_data/README.md)
- Library details: [`lib/PwmController/README.md`](lib/PwmController/README.md), [`lib/PwmSequencer/README.md`](lib/PwmSequencer/README.md), [`lib/JsonPwmSequencer/README.md`](lib/JsonPwmSequencer/README.md), [`lib/SerialComm/README.md`](lib/SerialComm/README.md)
