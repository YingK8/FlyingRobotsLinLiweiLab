#pragma once

#include "constants.h"
#include <Arduino.h>

// Self-calibrating zero-offset + EMA filter over the VNH5019 CS pins
// (constants.h's ADC_PINS). Two quirks in current_sense.cpp are hard-won on
// this hardware; don't simplify them away.
class CurrentSense {
public:
  // sensPerVolt[i]: A/V, per-board VNH5019 CS calibration. tauFilterMs: EMA
  // time constant; 50ms is the confirmed best tradeoff on this rig (less
  // filtering reintroduces argmin flapping in main_current_pid.cpp).
  explicit CurrentSense(const float sensPerVolt[NUM_CHANNELS],
                        float tauFilterMs = 50.0f);

  // Call once at boot with coils confirmed OFF: seeds the filter and the
  // zero-offset from the true floating baseline.
  void seed();

  // Call at your own pace (ESP32 ADC needs real settling time between
  // conversions -- pace this separately from any fast control loop). Updates
  // the filtered raw mV and i_meas[].
  void update(float dtMs);

  // Re-zero against the current filtered reading. Call throughout ARMING
  // (coils confirmed off) so the zero tracks warm-up drift; stop calling it
  // once the run starts to freeze the calibration.
  void recalibrateZero();

  float i_meas[NUM_CHANNELS];

private:
  float _sensPerVolt[NUM_CHANNELS];
  float _tauFilterMs;
  float _csMv[NUM_CHANNELS];
  float _adcZeroMv[NUM_CHANNELS];
};
