#pragma once

#include "constants.h"
#include <Arduino.h>

// gates must be LOW before any driver exists
inline void forceAllGatesLow() {
  const gpio_num_t ALL_PINS[] = {A_PWM_PIN, B_PWM_PIN, C_PWM_PIN, D_PWM_PIN,
                                      A_CARRIER_PIN, B_CARRIER_PIN, C_CARRIER_PIN,
                                      D_CARRIER_PIN};
  for (gpio_num_t p : ALL_PINS) {
    pinMode(p, OUTPUT);
    digitalWrite(p, LOW);
  }
}
