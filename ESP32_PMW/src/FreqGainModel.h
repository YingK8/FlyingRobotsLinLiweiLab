#pragma once
// Corrects the feedforward's static (DC, omega=0) gain assumption for the
// actual drive frequency -- see progress.md Section 5 for the full
// derivation, state_space_constants.h's FREQ_MODEL_VALID_HZ comment for how
// this complements (not replaces) the integral gate.
//
// Bd/FF (tools/build_state_space_model.py) assume a plant gain of Kv/R at
// every frequency -- exactly true only at each channel's LC resonance f0
// (where the series R-L-C's impedance |Z(f0)|=R, its minimum). Away from
// f0, |Z(f)|>R strictly, so the true required duty for a given target
// current is u_ff_static * (|Z(f)|/R) -- ALWAYS >= the static estimate,
// never less. (An earlier draft of this fix multiplied by R/|Z(f)|<=1
// instead -- backwards; caught during derivation, see progress.md 5.5.)

#include <math.h>
#include "constants.h"
#include "rlc_gain_model.h"
#include "state_space_constants.h"

inline float freqGainCorrection(int ch, float freq_hz) {
  float f = fmaxf(freq_hz, FREQ_GAIN_MIN_HZ);  // avoid 1/(2*pi*f*C) blowup near f=0
  float omega = 2.0f * (float)M_PI * f;
  float x = omega * RLC_L[ch] - 1.0f / (omega * RLC_C[ch]);
  float z = sqrtf(RLC_R[ch] * RLC_R[ch] + x * x);  // |Z(f)|, always >= R
  float correction = z / RLC_R[ch];                // >= 1.0 by construction
  return fminf(correction, FREQ_GAIN_CORRECTION_MAX);
}
