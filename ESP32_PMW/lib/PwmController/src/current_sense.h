#pragma once

#include <Arduino.h>
#include "driver/gpio.h"

// Self-calibrating zero-offset + EMA filter over the VNH5019 CS pins. Two quirks
// in current_sense.cpp are hard-won on this hardware; don't simplify them away.
// ADC pins are passed in by the caller (constants.h::ADC_PINS) to keep the
// library self-contained.
class CurrentSense {
public:
  static const int N = 4; // 4-channel rig (matches CurrentBalanceController)

  // adcPins[N]: VNH5019 CS ADC pins (constants.h::ADC_PINS).
  // sensPerVolt[N]: per-board CS calibration (A/V).
  // tauFilterMs: EMA constant; 50ms is best on this rig (less reintroduces
  //   argmin flapping in the balance loop).
  CurrentSense(const gpio_num_t adcPins[N], const float sensPerVolt[N],
               float tauFilterMs = 50.0f);

  // Call once at boot, coils confirmed OFF: seeds filter + zero-offset from the
  // floating baseline.
  void seed();

  // Updates filtered mV and i_meas[]. Pace separately from any fast control
  // loop: the ESP32 ADC needs real settling time between conversions.
  void update(float dtMs);

  // Re-zero against the current reading (coils OFF) to track warm-up drift; stop
  // once the run starts to freeze calibration.
  void recalibrateZero();

  float i_meas[N];

private:
  gpio_num_t _adcPins[N];
  float _sensPerVolt[N];
  float _tauFilterMs;
  float _csMv[N];
  float _adcZeroMv[N];
};
