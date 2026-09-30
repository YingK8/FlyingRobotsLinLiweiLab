#include "coil_probe.h"

#include <math.h>

bool coilProbe(const gpio_num_t *adcPins, int nCh, float fHz, uint32_t ms,
               const float *commandedPhaseDeg, ProbeResult &out) {
  if (!adcPins || !commandedPhaseDeg || nCh <= 0 || nCh > 4) return false;
  if (!(fHz >= 1.0f) || ms == 0) return false;   // !(>=) also catches NaN

  analogReadResolution(12);
  for (int c = 0; c < nCh; c++) analogSetPinAttenuation(adcPins[c], ADC_11db);

  // Accumulators. double for the sums of squares: at ~2 kHz for a second these run to
  // ~1e10 in raw counts squared, where a float has already lost the low bits that the
  // variance -- and therefore the coherence -- is computed from.
  double si[4] = {0}, sq[4] = {0}, s1[4] = {0}, s2[4] = {0};
  uint32_t n[4] = {0};

  // The lock-in reference is 2f (see the header: CS is unsigned, so the signal is at
  // twice the drive rate). `t0` is an arbitrary origin -- every channel shares it, and a
  // common phase offset cancels when the host references the four-channel mean.
  const double w2 = 4.0 * M_PI * (double)fHz;
  const int64_t t0 = esp_timer_get_time();
  const int64_t endUs = t0 + (int64_t)ms * 1000;

  while (esp_timer_get_time() < endUs) {
    for (int c = 0; c < nCh; c++) {
      // Throwaway read: the ADC mux needs to settle after switching pins, or channels
      // read as ~0 regardless of real current (current_sense.cpp says the same).
      analogRead(adcPins[c]);
      const int64_t ts = esp_timer_get_time();
      const float v = (float)analogRead(adcPins[c]);
      // Timestamp taken at the conversion, so the sequential mux visit carries no
      // systematic phase error into the accumulators.
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
    if (n[c] < 16) return false;         // nothing usable; do not report a number
    const double inv = 1.0 / (double)n[c];
    // Quadrature amplitude of the 2f component: the factor 2 undoes the 1/2 from
    // averaging a product of two sinusoids.
    const double re = si[c] * inv, im = sq[c] * inv;
    const double amp = 2.0 * sqrt(re * re + im * im);
    out.amp[c] = (float)amp;

    // atan2(im, re) is 2*psi + pi: |sin| expands with a NEGATIVE first harmonic
    // (|sin x| = 2/pi - (4/3pi) cos 2x + ...), so the reference sits half a turn from
    // the current. Undo that, halve, then subtract what this channel was commanded.
    double psi = 0.5 * (atan2(im, re) - M_PI);
    double th = psi * 180.0 / M_PI - (double)commandedPhaseDeg[c];
    // The 2f lock-in only ever knew the phase modulo 180 deg. Wrap to the branch the
    // spread actually lives in; a channel genuinely half a turn out is an amplitude
    // fault, which coil_balance.py is the tool for.
    th = fmod(th, 180.0);
    if (th > 90.0) th -= 180.0;
    if (th <= -90.0) th += 180.0;
    out.phaseDeg[c] = (float)th;

    // Coherence: how much of the AC power actually sits at 2f. A burst with the coils
    // off, or one swamped by the 20 kHz carrier chopping the CS pin, scores near zero
    // and the host drops the point rather than fitting noise.
    const double mean = s1[c] * inv;
    const double var = s2[c] * inv - mean * mean;
    const double coh = (var > 1e-9) ? (0.5 * amp * amp) / var : 0.0;
    const float cf = (float)(coh > 1.0 ? 1.0 : coh);
    if (cf < out.coherence) out.coherence = cf;
    if (n[c] < out.n) out.n = n[c];
  }
  return true;
}
