#include "DutyAllocator.h"

static inline float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

DutyAllocator::DutyAllocator(const float Bd[NUM_CHANNELS][NUM_CHANNELS], float r_duty,
                              float duty_min, float duty_max)
    : _r_duty(r_duty), _duty_min(duty_min), _duty_max(duty_max) {
  float frob_sq = 0.0f;
  for (int i = 0; i < NUM_CHANNELS; i++) {
    for (int j = 0; j < NUM_CHANNELS; j++) {
      _Bd[i][j] = Bd[i][j];
      frob_sq += Bd[i][j] * Bd[i][j];
    }
  }
  float L = 2.0f * frob_sq + 2.0f * r_duty;
  _alpha = 1.0f / L;
}

void DutyAllocator::allocate(const float x_target[NUM_CHANNELS], const float u_ff[NUM_CHANNELS],
                              const float duty_prev[NUM_CHANNELS], float duty_out[NUM_CHANNELS]) const {
  float u[NUM_CHANNELS];
  for (int i = 0; i < NUM_CHANNELS; i++) u[i] = clampf(duty_prev[i], _duty_min, _duty_max);

  for (int iter = 0; iter < N_ITERS; iter++) {
    float resid[NUM_CHANNELS];
    for (int i = 0; i < NUM_CHANNELS; i++) {
      resid[i] = -x_target[i];
      for (int j = 0; j < NUM_CHANNELS; j++) resid[i] += _Bd[i][j] * u[j];
    }
    float grad[NUM_CHANNELS];
    for (int i = 0; i < NUM_CHANNELS; i++) {
      float bt_resid = 0.0f;
      for (int j = 0; j < NUM_CHANNELS; j++) bt_resid += _Bd[j][i] * resid[j]; // Bd^T @ resid
      grad[i] = 2.0f * bt_resid + 2.0f * _r_duty * (u[i] - u_ff[i]);
    }
    for (int i = 0; i < NUM_CHANNELS; i++)
      u[i] = clampf(u[i] - _alpha * grad[i], _duty_min, _duty_max);
  }
  for (int i = 0; i < NUM_CHANNELS; i++) duty_out[i] = u[i];
}
