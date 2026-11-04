#pragma once

#include <Arduino.h>
#include "driver/gpio.h"

class CurrentSense {
public:
  static const int N = 4;

  // deadMv: per channel, added to any raw sample above the zero (null = none)
  CurrentSense(const gpio_num_t adcPins[N], const float sensPerVolt[N],
                const float *deadMv = nullptr, float tauFilterMs = 50.0f);

  void seed();

  void update(float dtMs);

  float i_meas[N];

private:
  gpio_num_t _adcPins[N];
  float _sensPerVolt[N];
  float _deadMv[N];
  float _tauFilterMs;
  float _csMv[N];
  float _adcZeroMv[N];
};
