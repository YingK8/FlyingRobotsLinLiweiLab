# ESP32 PWM Sequencer Library Documentation

## 1. Tutorial: Getting Started

### Introduction
This tutorial will guide you through setting up and using the PhaseController, PhaseSequencer, and JsonPhaseSequencer libraries to control PWM outputs on the ESP32. You will learn how to blink an LED, ramp PWM, and schedule carrier PWM changes using a JSON file.

### Prerequisites
- ESP32 development board
- PlatformIO or Arduino IDE
- Basic C++ knowledge

### Step 1: Hardware and Software Setup
- Connect your ESP32 to your computer.
- Create a new PlatformIO or Arduino project.
- Add the library source files to your project.

### Step 2: Basic PWM Output
```cpp
#include "PhaseController.h"

const int NUM_CHANNELS = 4;
const gpio_num_t PWM_PINS[NUM_CHANNELS] = {GPIO_NUM_19, GPIO_NUM_33, GPIO_NUM_27, GPIO_NUM_32};
const float INITIAL_PHASES[NUM_CHANNELS] = {0.0, 90.0, 180.0, 270.0};
const float INITIAL_DUTY_CYCLES[NUM_CHANNELS] = {50.0, 50.0, 50.0, 50.0};

PhaseController controller(PWM_PINS, INITIAL_PHASES, INITIAL_DUTY_CYCLES, NUM_CHANNELS);

void setup() {
    controller.begin(100.0f); // Start at 100 Hz
}

void loop() {
    controller.run(); // Call regularly for drift compensation
}
```

### Step 3: Sequencing PWM with PhaseSequencer
```cpp
#include "PhaseSequencer.h"
// ...existing code...
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

### Step 4: Scheduling Carrier PWM with JSON
```cpp
#include "JsonPhaseSequencer.h"
// ...existing code...
JsonPhaseSequencer seq(&controller);

void setup() {
    controller.begin(100.0f);
    seq.loadFromJsonFile("/schedule.json");
    seq.start();
}

void loop() {
    controller.run();
    seq.run();
}
```

---

## 2. How-to Guides

### How to Set a Duty Cycle or Phase
- Use `setDutyCycle(channel, value)` and `setPhase(channel, degrees)` on your PhaseController instance.

### How to Schedule a Carrier PWM Change
- Use `addCarrierDutyCycleTask(carrierDutyCycles, numChannels)` in PhaseSequencer, or the corresponding method in your JSON schedule.

### How to Load and Run a JSON Schedule
- Use `JsonPhaseSequencer::loadFromJsonFile("/path/to/file.json")` and call `run()` in your main loop.

---

## 3. Reference

### PhaseController
- See header file for full docstrings.
- Key methods: `begin`, `run`, `setGlobalFrequency`, `setDutyCycle`, `setPhase`, `initCarrierPWM`, `setCarrierDutyCycle`.

### PhaseSequencer
- See header file for full docstrings.
- Key methods: `addDutyCycleTask`, `addCarrierDutyCycleTask`, `addPhaseTask`, `addWaitTask`, `addLinearRampTask`, `addEaseRampTask`, `addPhaseRampTask`, `compile`, `start`, `run`, `isDone`.

### JsonPhaseSequencer
- See header file for full docstrings.
- Key method: `loadFromJsonFile`.

---

## 4. Explanation

### What is a Sequencer?
A sequencer allows you to define a series of PWM changes (frequency, duty, phase, carrier) over time, so you can automate complex behaviors for robotics, experiments, or actuation.

### What is Carrier PWM?
Carrier PWM is a high-frequency PWM signal (often used for modulation or power electronics) that can be independently controlled per channel, in addition to the main PWM signal.

### How Does the JSON Scheduler Work?
The JsonPhaseSequencer loads a schedule from a JSON file, where each entry specifies a method (e.g., `addCarrierDutyCycleTask`) and its parameters. The sequencer compiles these into a trajectory and executes them at the correct time, calling the appropriate hardware methods.

---

For more details, see the header files for docstrings and method descriptions.
