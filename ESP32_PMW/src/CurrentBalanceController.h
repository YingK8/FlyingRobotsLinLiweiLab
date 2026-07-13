#pragma once
// The higher-level control law (core 1): coupling-aware feedforward
// (FF = Bd^-1), LQID feedback (K_x/K_d/K_z, direction-selected), a
// constrained duty allocator (DutyAllocator), per-channel anti-windup, and
// the reference governor. Pure computation -- no hardware I/O -- so it's
// directly testable and independent of which core it happens to run on.
//
// Anti-windup is PER-CHANNEL (freeze only the channels actually pinned at
// DUTY_MAX, or all channels on a global overcurrent trip), not global.
// This matters now that the coupling-aware feedforward can legitimately
// saturate one channel (e.g. CCW's channel C, or CW's channel D) from
// feedforward alone -- a global freeze would starve the OTHER channels'
// still-needed correction the instant any one channel hits its ceiling
// (confirmed as a real regression on hardware before this fix: HOLD-phase
// spread jumped from ~0.08A to ~0.45A because one saturated channel froze
// every channel's integrator).

#include "constants.h"
#include "DutyAllocator.h"

class CurrentBalanceController {
public:
  CurrentBalanceController();

  // Full control-law update for one tick. Must be called at exactly
  // CONTROL_TICK_MS intervals -- the Ts the offline model (Ad/Bd, hence
  // FF/K_x/K_d/K_z) was fit for. Returns the updated r_target (reference
  // governor's ramp/backoff decision). freq_hz gates integral accumulation
  // below FREQ_MODEL_VALID_HZ -- see state_space_constants.h.
  float computeTick(const float i_meas[NUM_CHANNELS], float r_target, float r_max,
                     float r_ramp_per_ms, bool directionIsCcw, float freq_hz,
                     float duty_out[NUM_CHANNELS]);

  // Reset z_integral/err_prev/derr_filt/duty_prev warm-start -- call once
  // on observing the ARMING->RAMP_UP transition (see caller).
  void reset();

private:
  float _duty_prev[NUM_CHANNELS];
  float _z_integral[NUM_CHANNELS];
  float _err_prev[NUM_CHANNELS];
  float _derr_filt[NUM_CHANNELS];

  DutyAllocator _allocatorCw;
  DutyAllocator _allocatorCcw;
};
