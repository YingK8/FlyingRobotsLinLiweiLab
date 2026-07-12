#pragma once

#include "driver/gpio.h"
#include "driver/ledc.h"
#include "esp_timer.h"
#include <Arduino.h>

#define FREQ_FILTER_SIZE 5

struct PhaseParams {
  unsigned long startUs;
  unsigned long endUs;
  bool wraps;
};

class PWMController {
public:
  // pins/phaseOffsetsDegrees/dutyCycles: numChannels-long, one entry per PWM
  // channel. phaseOffsetsDegrees in [0,360), dutyCycles in [0,100]%.
  PWMController(const gpio_num_t *pins, const float *phaseOffsetsDegrees,
                  const float *dutyCycles, int numChannels);
  ~PWMController();

  // Configures pins and starts the phase timer at initialFreqHz.
  void begin(float initialFreqHz=0.0f);

  // Drift-compensation tick. Call regularly in loop().
  void step();

  // Configuration
  void setGlobalFrequency(float newHz);
  void setFrequency(int channel, float newHz);
  void setDutyCycle(int channel, float dutyPercent); // 0-100
  void setPhase(int channel, float degrees);         // 0-360

  // Getters
  float getFrequency() const;
  float getPhase(int channel) const;
  float getDutyCycle(int channel) const;

  void enableSync(gpio_num_t syncPin);

  // Carrier PWM (multi-channel)
  void initCarrierPWM(const gpio_num_t *pins, float freqHz,
                      const float *dutyPercents);
  void setCarrierDutyCycle(int channel, float dutyPercent); // 0-100

  // 0-100, or 0 if carrier PWM isn't initialized / channel out of range.
  float getCarrierDutyCycle(int channel) const;

  // De-energize all coils: ramp every carrier duty to 0 over rampMs, then
  // stop the phase timer (output halts; controller/timer stay allocated,
  // unlike the destructor). BLOCKING (delay()); for non-blocking, use
  // rampDownStep().
  void shutdown(unsigned long rampMs = 2000);

  // One non-blocking ramp-down step: subtracts stepPct (>0) from each
  // channel's CURRENT carrier duty, so the commanded duty only ever
  // decreases (no target to overshoot toward). Call once per control tick
  // until it returns true (every channel at 0).
  bool rampDownStep(float stepPct);

private:
  // Minimum on/off period constraint: 0.0 ms
  const float MIN_ON_OFF_MS = 0.0f;

  // Internal methods
  static void IRAM_ATTR _timerCallback(void *arg);
  static void IRAM_ATTR _onSyncInterrupt();
  void updatePhaseParams(int channel);

  // Hardware
  int _numChannels;
  gpio_num_t *_pins;
  esp_timer_handle_t _periodicTimer;
  gpio_num_t _syncPin;

  // Carrier PWM (multi-channel)
  gpio_num_t *_carrierPinsArray = nullptr;
  float _carrierFreqHz;
  float *_carrierDutyCyclePct = nullptr;
  bool *_carrierLedcConfigured = nullptr;   // LEDC channel attached to the pin?
  uint32_t *_carrierLastDutyTicks = nullptr; // last duty written, in timer ticks
  ledc_mode_t _carrierSpeedMode = LEDC_LOW_SPEED_MODE;
  ledc_timer_t _carrierTimer = LEDC_TIMER_0;
  uint8_t _carrierDutyResolutionBits = 10;

  // State Arrays
  float *_phaseOffsetsPct;
  float *_dutyCycles;
  PhaseParams *_params;
  float _globalFreqHz; // New global frequency variable

  // Sync State
  int64_t _lastSyncTimeUs;
  int64_t _averagedPeriodUs;
  int64_t _periodBuffer[FREQ_FILTER_SIZE];
  int _filterIdx;

  // ISR variables
  static PWMController *_isrInstance;
  int64_t _lastIsrTimeUs;
  bool _firstSyncReceived;

  // Concurrency
  portMUX_TYPE _spinlock = portMUX_INITIALIZER_UNLOCKED;
};
