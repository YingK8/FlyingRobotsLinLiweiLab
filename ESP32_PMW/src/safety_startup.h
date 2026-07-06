#pragma once

#include "constants.h"
#include <Arduino.h>

// Holds every gate-driver input LOW before a PhaseController exists, so the
// coils can't glitch on during boot/LEDC attach even if the coil supply is
// already live. Call first in setup(), before `new PhaseController(...)`.
inline void forceAllGatesLow() {
  const gpio_num_t ALL_PINS[] = {A_PWM_PIN, B_PWM_PIN, C_PWM_PIN, D_PWM_PIN,
                                 A_CARRIER_PIN, B_CARRIER_PIN, C_CARRIER_PIN,
                                 D_CARRIER_PIN};
  for (gpio_num_t p : ALL_PINS) {
    pinMode(p, OUTPUT);
    digitalWrite(p, LOW);
  }
}
