# PhaseSequencer

A high-level interface for sequencing PWM phase, frequency, and duty cycle behaviors over time, built on top of PhaseController for the ESP32 platform.

---

## 1. Introduction

**PhaseSequencer** enables you to create complex, time-varying PWM patterns and behaviors for multi-channel systems. It is ideal for robotics, experimental setups, and research where you need to ramp, sequence, or synchronize PWM outputs in a repeatable and programmable way.

**Key Features:**
- Sequence ramp-up, ramp-down, and hold behaviors
- Supports linear and eased ramps for frequency, phase, and duty cycle
- Integrates directly with PhaseController
- Designed for PlatformIO and ESP-IDF/Arduino environments

---

## 2. Tutorial: Getting Started

### Prerequisites
- ESP32 development board
- PlatformIO (recommended) or ESP-IDF/Arduino
- Basic C++ knowledge
- Working PhaseController setup

### Installation
1. Add the `PhaseSequencer` and `PhaseController` source files to your PlatformIO project.
2. Ensure all dependencies are available.

### Basic Usage Example
```cpp
#include "PhaseController.h"
#include "PhaseSequencer.h"

const int NUM_CHANNELS = 4;
const gpio_num_t PWM_PINS[NUM_CHANNELS] = {GPIO_NUM_19, GPIO_NUM_33, GPIO_NUM_27, GPIO_NUM_32};
const float INITIAL_PHASES[NUM_CHANNELS] = {0.0, 90.0, 180.0, 270.0};
const float INITIAL_DUTY_CYCLES[NUM_CHANNELS] = {50.0, 50.0, 50.0, 50.0};

PhaseController controller(PWM_PINS, INITIAL_PHASES, INITIAL_DUTY_CYCLES, NUM_CHANNELS);
PhaseSequencer seq(&controller);

void setup() {
    controller.begin(100.0f);
    seq.addEaseRampTask(1.0f, 100.0f, 10000); // Ramp from 1Hz to 100Hz over 10s
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

### How to Sequence a Frequency Ramp
- Use `addLinearRampTask(startHz, endHz, durationMs)` or `addEaseRampTask(...)` to add a ramp.
- Call `compile(...)` to generate the trajectory.
- Call `start()` to begin the sequence.

### How to Add Waits and Holds
- Use `addWaitTask(durationMs)` to insert a pause in the sequence.

### How to Change Duty Cycle or Phase Over Time
- Use `addDutyCycleTask(dutyCycles, numChannels)` or `addPhaseTask(phases, numChannels)`.
- For smooth transitions, use `addPhaseRampTask(startPhases, endPhases, durationMs)`.

### How to Integrate with PhaseController
- Pass a pointer to your PhaseController instance when constructing PhaseSequencer.
- Call `controller.run()` and `seq.run()` in your main loop.

---

## 4. Reference

### Class: PhaseSequencer

#### Constructor
```cpp
PhaseSequencer(PhaseController* phaseCtrl);
```
- `phaseCtrl`: Pointer to an initialized PhaseController

#### Methods
- `void reserve(size_t size);` // Reserve space for tasks
- `void addDutyCycleTask(const float* dutyCycles, int numChannels);`
- `void addPhaseTask(const float* phases, int numChannels);`
- `void addWaitTask(uint32_t durationMs);`
- `void addLinearRampTask(float startHz, float endHz, uint32_t durationMs);`
- `void addEaseRampTask(float startHz, float endHz, uint32_t durationMs);`
- `void addPhaseRampTask(const float* startPhases, const float* endPhases, uint32_t durationMs);`
- `void compile(uint32_t resolutionMs, float initialFreq, const float* initialDuty, const float* initialPhase);`
- `void start();`
- `void run();`
- `bool isDone() const;`

#### Task Types
- Set duty cycles, set phases, wait, linear ramp, ease ramp, phase ramp

---

## 5. Explanation

### How It Works
- Maintains a queue of tasks (ramps, waits, phase changes)
- Compiles tasks into a time-based trajectory for the controller
- Calls PhaseController methods to update outputs in real time

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
- [PhaseController documentation](../PhaseController/README.md)
- [PlatformIO documentation](https://docs.platformio.org/)
- Example projects in this repository

