#include "coil_probe.h"

#include <math.h>

bool coilProbe(const gpio_num_t *adcPins, int nCh, float fHz, uint32_t ms,
               const float *commandedPhaseDeg, ProbeResult &out) {
  if (!adcPins || !commandedPhaseDeg || nCh <= 0 || nCh > 4) return false;
  if (!(fHz >= 1.0f) || ms == 0) return false;

  analogReadResolution(12);
  for (int c = 0; c < nCh; c++) analogSetPinAttenuation(adcPins[c], ADC_11db);

  // double: float loses the low bits the variance, and so the coherence, needs.
  double si[4] = {0}, sq[4] = {0}, s1[4] = {0}, s2[4] = {0};
  uint32_t n[4] = {0};

  const double w2 = 4.0 * M_PI * (double)fHz;
  // timestamp at the conversion: the mux visit is sequential, not simultaneous.
  const int64_t t0 = esp_timer_get_time();
  const int64_t endUs = t0 + (int64_t)ms * 1000;

  while (esp_timer_get_time() < endUs) {
    for (int c = 0; c < nCh; c++) {
      // throwaway read: the ADC mux needs settling after a pin switch.
      analogRead(adcPins[c]);
      const int64_t ts = esp_timer_get_time();
      const float v = (float)analogRead(adcPins[c]);
      const double a = w2 * (double)(ts - t0) * 1e-6;
      si[c] += v * cos(a);
      sq[c] += v * sin(a);
      s1[c] += v;
      s2[c] += (double)v * v;
      n[c]++;
    }
  }

  out.n = n[0];
  out.coherence = 1.0f;
  for (int c = 0; c < nCh; c++) {
    if (n[c] < 16) return false;
    const double inv = 1.0 / (double)n[c];
    const double re = si[c] * inv, im = sq[c] * inv;
    const double amp = 2.0 * sqrt(re * re + im * im);
    out.amp[c] = (float)amp;

    double psi = 0.5 * (atan2(im, re) - M_PI);
    double th = psi * 180.0 / M_PI - (double)commandedPhaseDeg[c];
    th = fmod(th, 180.0);
    if (th > 90.0) th -= 180.0;
    if (th <= -90.0) th += 180.0;
    out.phaseDeg[c] = (float)th;

    const double mean = s1[c] * inv;
    const double var = s2[c] * inv - mean * mean;
    const double coh = (var > 1e-9) ? (0.5 * amp * amp) / var : 0.0;
    const float cf = (float)(coh > 1.0 ? 1.0 : coh);
    if (cf < out.coherence) out.coherence = cf;
    if (n[c] < out.n) out.n = n[c];
  }
  return true;
}
