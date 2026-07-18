# PhaseSequencer

A high-level interface for sequencing PWM phase, frequency, and duty cycle behaviors over time, built on top of PwmController for the ESP32 platform.

---

## 1. Introduction

**PhaseSequencer** enables you to create complex, time-varying PWM patterns and behaviors for multi-channel systems. It is ideal for robotics, experimental setups, and research where you need to ramp, sequence, or synchronize PWM outputs in a repeatable and programmable way.

**Key Features:**
- Sequence ramp-up, ramp-down, and hold behaviors
- One `addRampTask` for every quantity (frequency, duty, carrier duty, phase), linear or eased
- Full (all channels) **and** per-channel control on every builder
- Integrates directly with PwmController
- Designed for PlatformIO and ESP-IDF/Arduino environments

---

## 2. Tutorial: Getting Started

### Prerequisites
- ESP32 development board
- PlatformIO (recommended) or ESP-IDF/Arduino
- Basic C++ knowledge
- Working PwmController setup

### Installation
1. Add the `PhaseSequencer` and `PwmController` source files to your PlatformIO project.
2. Ensure all dependencies are available.

### Basic Usage Example
```cpp
#include "PwmController.h"
#include "PhaseSequencer.h"

const int NUM_CHANNELS = 4;
const gpio_num_t PWM_PINS[NUM_CHANNELS] = {GPIO_NUM_19, GPIO_NUM_33, GPIO_NUM_27, GPIO_NUM_32};
const float INITIAL_PHASES[NUM_CHANNELS] = {0.0, 90.0, 180.0, 270.0};
const float INITIAL_DUTY_CYCLES[NUM_CHANNELS] = {50.0, 50.0, 50.0, 50.0};

PwmController controller(PWM_PINS, INITIAL_PHASES, INITIAL_DUTY_CYCLES, NUM_CHANNELS);
PhaseSequencer seq(&controller);

void setup() {
    controller.begin(100.0f);
    // Ramp frequency 1Hz -> 100Hz over 10s with an S-curve
    seq.addRampTask(1.0f, 100.0f, 10000, TaskType::PWM_FREQ, TaskMode::EASE);
    seq.compile(25, 1.0f, INITIAL_DUTY_CYCLES, INITIAL_PHASES);
    seq.start();
}

void loop() {
    controller.run();
    seq.run();
}
```

### Building and Flashing
- Use PlatformIO's build and upload commands:
  - `pio run`
  - `pio upload`

---

## 3. How-to Guides

### How to Sequence a Ramp (any quantity)
- Use the single `addRampTask(...)`. Pick the quantity with `TaskType`
  (`PWM_FREQ`, `PWM_DUTY`, `CARRIER_DUTY`, `PWM_PHASE`) and the curve with
  `TaskMode` (`LINEAR`, `EASE`, or `EXPONENTIAL`). The optional final `shape`
  argument tunes EASE (S-curve sharpness k≥1) and EXPONENTIAL (exponent
  multiplier k); omit it (NAN) for the per-mode default of 2:
  ```cpp
  seq.addRampTask(1.0f, 100.0f, 10000, TaskType::PWM_FREQ, TaskMode::EASE);
  seq.addRampTask(1.0f, 100.0f, 10000, TaskType::PWM_FREQ, TaskMode::EASE, 4.0f); // sharper S
  seq.addRampTask(1.0f, 100.0f, 10000, TaskType::PWM_FREQ, TaskMode::EXPONENTIAL, 3.0f); // hard ease-in
  seq.addRampTask(0.0f, 100.0f, 2000, TaskType::CARRIER_DUTY, TaskMode::LINEAR);
  ```
- Call `compile(...)` then `start()`.

### How to Add Waits and Holds
- Use `addWaitTask(durationMs)` to insert a pause in the sequence.

### How to Control One Channel vs All Channels
- **Full** (all channels): pass a scalar, e.g.
  `addRampTask(0, 100, 2000, TaskType::PWM_PHASE)`.
- **Per-channel**: pass a `float[4]`; a channel whose value is `NAN` is left
  unchanged, so you can drive any single channel while the rest hold:
  ```cpp
  float starts[4] = {NAN, 0.0f, NAN, 0.0f};   // only ch1 & ch3
  float ends[4]   = {NAN, 90.0f, NAN, 90.0f};
  seq.addRampTask(starts, ends, 4, 2000, TaskType::PWM_PHASE, TaskMode::LINEAR);
  ```

### How to Instantly Set the Full State
- Build a `SequenceTask{type = TaskType::TRAJECTORY_POINT, ...}` by hand
  (fill `startFreq`/`endFreq`, `dutyCycles`, `startPhases`, `carrierDuties` for
  all 4 channels) and push it with `addSequenceTask(task)`. One task, one
  hardware sync: this is what CSV/JSON import use. Unlike a ramp, a
  trajectory point has no NAN-skip. Every channel must be given explicitly.

### How to Integrate with PwmController
- Pass a pointer to your PwmController instance when constructing PhaseSequencer.
- Call `controller.run()` and `seq.run()` in your main loop.

---

## 4. Reference

### Class: PhaseSequencer

#### Constructor
```cpp
PhaseSequencer(PwmController* phaseCtrl);
```
- `phaseCtrl`: Pointer to an initialized PwmController

#### Methods
Every ramp builder has a **full** (scalar → all channels) and a
**per-channel** (`float[4]`, `NAN` = leave channel unchanged) form.
```cpp
void reserve(size_t size);
void addSequenceTask(SequenceTask task); // generic; used for TRAJECTORY_POINT

void addWaitTask(uint32_t durationMs);

// Ramps: one builder for freq / duty / carrier / phase
void addRampTask(float start, float end, uint32_t durationMs,
                 TaskType type = TaskType::PWM_FREQ,
                 TaskMode ramp_mode = TaskMode::LINEAR);           // full
void addRampTask(const float* starts, const float* ends, int numChannels,
                 uint32_t durationMs, TaskType type = TaskType::PWM_FREQ,
                 TaskMode ramp_mode = TaskMode::LINEAR);           // per-channel

void compile(uint32_t resolutionMs, float initialFreq,
             const float* initialDuty, const float* initialPhase);
void start();
void run();
bool isDone() const;
```

#### Enums
- `TaskType`: `PWM_DUTY`, `PWM_FREQ`, `PWM_PHASE`, `CARRIER_DUTY`, `WAIT`,
  `TRAJECTORY_POINT`. Scoped (`enum class`) to avoid colliding with
  per-sketch constants like `const int PWM_FREQ = 15000`, so always qualify:
  `TaskType::PWM_FREQ`.
- `TaskMode`: `LINEAR`, `EASE`, `EXPONENTIAL`, the interpolation curve for a
  ramp (the last two take an optional `shape` parameter).

Note: `PWM_FREQ` is a single global frequency; per-channel ramps ignore all but
channel 0 for it. Duty/carrier tasks are clamped to 0–100%.

---

## 5. Explanation

### How It Works
- Maintains a queue of tasks (ramps, waits, phase changes)
- Compiles tasks into a time-based trajectory for the controller
- Calls PwmController methods to update outputs in real time

### Advantages
- Enables complex, repeatable PWM behaviors for experiments
- Abstracts timing and sequencing logic from main application
- Easy to extend with new task types

### Limitations
- Resolution and timing depend on compile settings and MCU load
- Requires careful setup of initial conditions for smooth transitions

### Troubleshooting
- If sequence does not run as expected, check task order and parameters
- Ensure `compile()` is called before `start()`
- Use serial prints or logic analyzer to debug timing

---

## 6. Further Reading
- [PwmController documentation](../PwmController/README.md)
- [PlatformIO documentation](https://docs.platformio.org/)
- Example projects in this repository

