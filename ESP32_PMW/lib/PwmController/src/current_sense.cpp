#include "current_sense.h"
#include <math.h>

static const float DEAD_THRESH_MV = 5.0f;

CurrentSense::CurrentSense(const gpio_num_t adcPins[N],
                             const float sensPerVolt[N], const float *deadMv,
                             float tauFilterMs)
    : _tauFilterMs(tauFilterMs) {
  for (int i = 0; i < N; i++) {
    _adcPins[i] = adcPins[i];
    _sensPerVolt[i] = sensPerVolt[i];
    _deadMv[i] = deadMv ? deadMv[i] : 0.0f;
    _csMv[i] = 0.0f;
    _adcZeroMv[i] = 0.0f;
    i_meas[i] = 0.0f;
  }
}

void CurrentSense::seed() {
  analogReadResolution(12);
  for (int i = 0; i < N; i++) {
    analogSetPinAttenuation(_adcPins[i], ADC_11db);
    // throwaway read: the ADC mux needs settling after a pin switch.
    analogReadMilliVolts(_adcPins[i]);
    _csMv[i] = analogReadMilliVolts(_adcPins[i]);
    _adcZeroMv[i] = _csMv[i];
  }
}

void CurrentSense::update(float dtMs) {
  float alpha = 1.0f - expf(-dtMs / _tauFilterMs);
  for (int i = 0; i < N; i++) {
    // throwaway read: without it two channels read ~0 whatever the current.
    analogReadMilliVolts(_adcPins[i]);
    float mv = analogReadMilliVolts(_adcPins[i]);
    // per sample, before the filter: under AC the clipping happens sample by sample.
    // DEAD_THRESH_MV is a guess (idle noise not measured; idle read 0.00 A in cs_cal).
    if (mv - _adcZeroMv[i] > DEAD_THRESH_MV) mv += _deadMv[i];
    _csMv[i] += alpha * (mv - _csMv[i]);
  }
  for (int i = 0; i < N; i++)
    i_meas[i] = _sensPerVolt[i] * (_csMv[i] - _adcZeroMv[i]) / 1000.0f;
}
