# PwmController

A low-level, software-controlled PWM output library for ESP32, designed for flexible phase and duty cycle control, with synchronization support for multi-board setups.

---

## 1. Introduction

**PwmController** provides precise, independent control of PWM signals on any GPIO pin of the ESP32. It is ideal for research, robotics, and experimental setups where phase, frequency, and duty cycle must be dynamically adjusted, and where synchronization between multiple boards is required.

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
1. Add the `PwmController` source files to your PlatformIO project (or clone this repo).
2. Ensure the `ledc` driver is available (included with ESP-IDF/Arduino).

### Basic Usage Example
```cpp
#include "PwmController.h"
#include "constants.h" // NUM_CHANNELS and PWM_PINS -- the pin map lives there

const float INITIAL_PHASES[NUM_CHANNELS] = {0.0, 90.0, 180.0, 270.0};
const float INITIAL_DUTY_CYCLES[NUM_CHANNELS] = {50.0, 50.0, 50.0, 50.0};

PwmController controller(PWM_PINS, INITIAL_PHASES, INITIAL_DUTY_CYCLES, NUM_CHANNELS);

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
- Call `initCarrierPWM(pins, freqHz, dutyPercents)` **once**, with arrays for all
  channels and a single shared carrier frequency.
- Adjust one channel afterwards with `setCarrierDutyCycle(channel, duty)`.
- With the balance loop on, `setCarrierDutyCycle` sets that channel's *ceiling*,
  not its duty.

---

## 4. Reference

### Class: PwmController

#### Constructor
```cpp
PwmController(const gpio_num_t* pins, const float* phaseOffsetsDegrees, const float* dutyCycles, int numChannels);
```
- `pins`: Array of GPIO pins
- `phaseOffsetsDegrees`: Array of initial phase offsets (degrees)
- `dutyCycles`: Array of initial duty cycles (%)
- `numChannels`: Number of channels

#### Methods

Lifecycle:
- `void begin(float initialFreqHz = 0.0f);` — default `0` starts in DC mode: the
  field is held static rather than rotating.
- `void run();` — drift compensation, current sampling, trip check, balance. Call
  every `loop()`.
- `void shutdown(unsigned long rampMs = 2000);` — ramp the coils down and stop.
- `bool rampDownStep(float stepPct);` — one step of a manual ramp-down.

Drive:
- `void setGlobalFrequency(float newHz);`
- `void setFrequency(int channel, float newHz);`
- `void setDutyCycle(int channel, float dutyPercent);`
- `void setPhase(int channel, float degrees);`
- `float getFrequency() const;` — global, no argument; `0` in DC mode.
- `bool isDC() const;`
- `float getPhase(int channel) const;`
- `float getDutyCycle(int channel) const;`
- `void enableSync(gpio_num_t syncPin);`

Carrier:
- `void initCarrierPWM(const gpio_num_t *pins, float freqHz, const float *dutyPercents);`
- `void setCarrierDutyCycle(int channel, float dutyPercent);`
- `float getCarrierDutyCycle(int channel) const;`

Current sense and balance (both opt-in, see §4a):
- `void enableCurrentSense(const gpio_num_t *adcPins, const float *sensPerVolt, float overcurrentTripA = 0.0f);`
- `void enableCurrentBalance(const BalanceConfig &cfg = BalanceConfig(), float startDuty = 50.0f);`
- `void setBalanceGains(float kp, float ki, float kd);`
- `void setBalanceRamp(float pctPerMs);`
- `const float *measuredCurrents() const;` — `float[4]` in amps, or `nullptr`.
- `float carrierCeiling(int channel) const;`
- `bool balanceActive() const;` / `bool currentSenseActive() const;`
- `bool overcurrentTripped() const;` — true once the latch has fired.

### 4a. Current sense and the balance loop

Both were folded into `PwmController` (they used to be a separate
`CurrentBalanceController` plus framework). Neither runs unless you opt in.

```cpp
// Call with the coils OFF so the ADC zero seeds cleanly.
controller.enableCurrentSense(ADC_PINS, SENS, /*tripA=*/10.0f);
controller.enableCurrentBalance();   // omit for open-loop passthrough
```

With balance on, each channel's commanded carrier duty becomes its **ceiling**;
the PI loop equalizes the four measured currents beneath those ceilings, so
per-channel trims are found at runtime instead of hand-tuned. A *differential*
set of ceilings therefore still tilts the rotor.

`BalanceConfig` (`src/CurrentBalanceController.h`) carries the tuning; the
defaults are the converged values and every field is overridable:

| field | default | meaning |
|---|---|---|
| `kp` / `ki` / `kd` | 2.2 / 0.10 / 0.15 | duty % per amp of error |
| `dutyMin` / `dutyMax` | 5 / 100 | hard clamp, before the per-channel ceiling |
| `iMax` | 12.0 | per-channel overcurrent backoff level (A) |
| `overcurrentBackoffPct` | 5.0 | duty backed off per tick above `iMax` |
| `nominalTickMs` | 2.0 | gains are scaled by `dt/nominalTickMs`, so they are rate-independent |
| `minSwitchMarginA` | 0.3 | hysteresis before the latched-minimum channel reassigns |
| `minRampPctPerMs` | 0.05 | ramp rate of the minimum channel toward its ceiling |

`CurrentSense` (`src/current_sense.h`) does the ADC reads and the zero-offset
calibration behind `enableCurrentSense`.

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