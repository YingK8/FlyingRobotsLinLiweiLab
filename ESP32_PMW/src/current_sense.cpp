#include "current_sense.h"
#include <math.h>

CurrentSense::CurrentSense(const float sensPerVolt[NUM_CHANNELS],
                           float tauFilterMs)
    : _tauFilterMs(tauFilterMs) {
  for (int i = 0; i < NUM_CHANNELS; i++) {
    _sensPerVolt[i] = sensPerVolt[i];
    _csMv[i] = 0.0f;
    _adcZeroMv[i] = 0.0f;
    i_meas[i] = 0.0f;
  }
}

void CurrentSense::seed() {
  analogReadResolution(12);
  for (int i = 0; i < NUM_CHANNELS; i++) {
    analogSetPinAttenuation(ADC_PINS[i], ADC_11db); // ~0..3.1V
    analogReadMilliVolts(ADC_PINS[i]); // throwaway: let the ADC mux/S&H settle
    _csMv[i] = analogReadMilliVolts(ADC_PINS[i]);
    _adcZeroMv[i] = _csMv[i];
  }
}

void CurrentSense::update(float dtMs) {
  float alpha = 1.0f - expf(-dtMs / _tauFilterMs);
  for (int i = 0; i < NUM_CHANNELS; i++) {
    // Throwaway read: the ESP32 ADC needs to settle after the mux switches
    // pins. Without it, two channels stuck at ~0 regardless of real current.
    analogReadMilliVolts(ADC_PINS[i]);
    _csMv[i] += alpha * (analogReadMilliVolts(ADC_PINS[i]) - _csMv[i]);
  }
  for (int i = 0; i < NUM_CHANNELS; i++)
    i_meas[i] = _sensPerVolt[i] * (_csMv[i] - _adcZeroMv[i]) / 1000.0f;
}

void CurrentSense::recalibrateZero() {
  for (int i = 0; i < NUM_CHANNELS; i++)
    _adcZeroMv[i] = _csMv[i];
}
