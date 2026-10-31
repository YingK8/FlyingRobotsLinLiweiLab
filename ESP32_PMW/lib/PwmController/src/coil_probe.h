#pragma once

#include "driver/gpio.h"
#include <Arduino.h>

// CS is unsigned, so the signal is at 2f: lock in at 2f and halve the angle.
struct ProbeResult {
  float amp[4];
  float phaseDeg[4];
  float coherence;
  uint32_t n;
};

bool coilProbe(const gpio_num_t *adcPins, int nCh, float fHz,
                         uint32_t ms, const float *commandedPhaseDeg,
                         ProbeResult &out);
