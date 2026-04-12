# PhaseController

A low-level, software-controlled PWM output library for ESP32, designed for flexible phase and duty cycle control, with synchronization support for multi-board setups.

---

## 1. Introduction

**PhaseController** provides precise, independent control of PWM signals on any GPIO pin of the ESP32. It is ideal for research, robotics, and experimental setups where phase, frequency, and duty cycle must be dynamically adjusted, and where synchronization between multiple boards is required.

**Key Features:**
- Independent phase and duty cycle control per channel
- Any GPIO pin can be used for PWM output
- Multi-channel support
- Board-to-board frequency synchronization
- Designed for PlatformIO and ESP-IDF/Arduino environments

---

## 2. Tutorial: Getting Started

### Prerequisites
- ESP32 development board
- PlatformIO (recommended) or ESP-IDF/Arduino
- Basic C++ knowledge

### Installation
1. Add the `PhaseController` source files to your PlatformIO project (or clone this repo).
2. Ensure the `ledc` driver is available (included with ESP-IDF/Arduino).

### Basic Usage Example
```cpp
#include "PhaseController.h"

const int NUM_CHANNELS = 4;
const gpio_num_t PWM_PINS[NUM_CHANNELS] = {GPIO_NUM_19, GPIO_NUM_33, GPIO_NUM_27, GPIO_NUM_32};
const float INITIAL_PHASES[NUM_CHANNELS] = {0.0, 90.0, 180.0, 270.0};
const float INITIAL_DUTY_CYCLES[NUM_CHANNELS] = {50.0, 50.0, 50.0, 50.0};

PhaseController controller(PWM_PINS, INITIAL_PHASES, INITIAL_DUTY_CYCLES, NUM_CHANNELS);

void setup() {
    controller.begin(100.0f); // Start at 100 Hz
    // Optionally, enable sync and carrier PWM per channel
}

void loop() {
    controller.run(); // Call regularly for drift compensation
}
```

### Building and Flashing
- Use PlatformIO's build and upload commands:
  - `pio run`
  - `pio upload`

---

## 3. How-to Guides

### How to Set Up Multiple PWM Channels
- Pass arrays of pins, phases, and duty cycles to the constructor.
- Use `setDutyCycle(channel, value)` and `setPhase(channel, degrees)` to adjust at runtime.

### How to Synchronize Multiple Boards
- Use `enableSync(syncPin)` on all boards.
- Designate one board as master (output sync), others as clients (input sync).

### How to Change Frequency or Duty Cycle Dynamically
- Call `setGlobalFrequency(newHz)` to change all channels.
- Use `setDutyCycle(channel, value)` for per-channel updates.

### How to Use Carrier PWM
- Call `initCarrierPWM(channel, pin, freq, duty)` for each channel.
- Adjust with `setCarrierDutyCycle(channel, duty)`.

---

## 4. Reference

### Class: PhaseController

#### Constructor
```cpp
PhaseController(const gpio_num_t* pins, const float* phaseOffsetsDegrees, const float* dutyCycles, int numChannels);
```
- `pins`: Array of GPIO pins
- `phaseOffsetsDegrees`: Array of initial phase offsets (degrees)
- `dutyCycles`: Array of initial duty cycles (%)
- `numChannels`: Number of channels

#### Methods
- `void begin(float initialFreqHz);`
- `void run();`  // Call in main loop
- `void setGlobalFrequency(float newHz);`
- `void setFrequency(int channel, float newHz);`
- `void setDutyCycle(int channel, float dutyPercent);`
- `void setPhase(int channel, float degrees);`
- `float getFrequency(int channel) const;`
- `float getPhase(int channel) const;`
- `float getDutyCycle(int channel) const;`
- `void enableSync(gpio_num_t syncPin);`
- `void initCarrierPWM(int channel, gpio_num_t pin, float freqHz, float dutyPercent);`
- `void setCarrierDutyCycle(int channel, float dutyPercent);`

#### Configuration
- All channels are independent.
- Any GPIO can be used (subject to ESP32 hardware constraints).

---

## 5. Explanation

### How It Works
- Uses ESP32's hardware timers and LEDC driver for precise PWM.
- Software logic allows phase and duty cycle to be changed on the fly.
- Synchronization is achieved via a shared sync pin and timer interrupts.

### Advantages
- Highly flexible: any pin, any phase, any duty
- Multi-board sync for distributed robotics/experiments
- Designed for research and rapid prototyping

### Limitations
- Maximum number of channels is limited by available timers and hardware resources
- Precise timing depends on system load and interrupt latency
- Not a drop-in replacement for all hardware PWM use cases

### Troubleshooting
- If PWM output is not as expected, check pin assignments and ensure no conflicts
- For sync, ensure only one master and all others are clients
- Use logic analyzer to verify timing if needed

---

## 6. Further Reading
- [ESP-IDF LEDC documentation](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/peripherals/ledc.html)
- [PlatformIO documentation](https://docs.platformio.org/)
- Example projects in this repository