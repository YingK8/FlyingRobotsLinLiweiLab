#pragma once
// Fixed-iteration projected-gradient constrained duty allocator -- see
// tools/validate_duty_allocator.py for the offline-validated algorithm this
// mirrors exactly (matched the true constrained optimum to 0.000% gap on
// the known CCW channel-C-saturation scenario, beat naive per-channel
// clamping by ~3.7x in cost).
//
// Solves: min ||Bd@u - x_target||^2 + r_duty*||u-u_ff||^2
//         s.t. DUTY_MIN <= u_i <= DUTY_MAX
// This is what lets correction redistribute across coupled channels when
// one saturates (e.g. CCW channel C needing ~90-100% duty), instead of
// just clipping the saturated channel in isolation.

#include "constants.h"

class DutyAllocator {
public:
  DutyAllocator(const float Bd[NUM_CHANNELS][NUM_CHANNELS], float r_duty,
                float duty_min, float duty_max);

  // x_target: the effective target this tick (x_target_eff = Bd @ u_desired,
  // folding feedforward+feedback into one QP target -- see caller).
  // u_ff: feedforward duty, used only as the effort-penalty anchor.
  // duty_prev: previous tick's applied duty, used to warm-start iteration.
  void allocate(const float x_target[NUM_CHANNELS], const float u_ff[NUM_CHANNELS],
                const float duty_prev[NUM_CHANNELS], float duty_out[NUM_CHANNELS]) const;

private:
  static const int N_ITERS = 15;
  float _Bd[NUM_CHANNELS][NUM_CHANNELS];
  float _r_duty;
  float _duty_min, _duty_max;
  float _alpha; // = 1/L, L = 2*||Bd||_F^2 + 2*r_duty, computed once at construction
};
