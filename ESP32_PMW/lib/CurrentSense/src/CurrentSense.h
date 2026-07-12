#pragma once

#include "driver/gpio.h"
#include <Arduino.h>

// Self-calibrating zero-offset + EMA filter over the VNH5019 CS pins.
// Two quirks in CurrentSense.cpp are hard-won on this hardware; don't
// simplify them away.
class CurrentSense {
public:
  static const int MAX_CHANNELS = 8;

  // adcPins/sensPerVolt: numChannels-long, one entry per current-sense
  // channel. sensPerVolt is A/V, per-board VNH5019 CS calibration.
  // tauFilterMs: EMA time constant; 50ms is the confirmed best tradeoff on
  // this rig (less filtering reintroduces argmin flapping in main_current_pid.cpp).
  CurrentSense(const gpio_num_t *adcPins, const float *sensPerVolt,
               int numChannels, float tauFilterMs = 50.0f);

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

  float i_meas[MAX_CHANNELS];

private:
  int _numChannels;
  gpio_num_t _adcPins[MAX_CHANNELS];
  float _sensPerVolt[MAX_CHANNELS];
  float _tauFilterMs;
  float _csMv[MAX_CHANNELS];
  float _adcZeroMv[MAX_CHANNELS];
};
