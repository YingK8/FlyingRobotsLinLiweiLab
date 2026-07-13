#include "CurrentBalanceController.h"
#include "state_space_constants.h"
#include "feedforward_cw.h"
#include "feedforward_ccw.h"
#include "lqr_gains_cw.h"
#include "lqr_gains_ccw.h"
#include "FreqGainModel.h"

static inline float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

CurrentBalanceController::CurrentBalanceController()
    : _allocatorCw(BD_CW, ALLOCATOR_R_DUTY, DUTY_MIN, DUTY_MAX),
      _allocatorCcw(BD_CCW, ALLOCATOR_R_DUTY, DUTY_MIN, DUTY_MAX) {
  reset();
  for (int i = 0; i < NUM_CHANNELS; i++) _duty_prev[i] = START_DUTY;
}

void CurrentBalanceController::reset() {
  for (int i = 0; i < NUM_CHANNELS; i++) {
    _z_integral[i] = 0.0f;
    _err_prev[i] = 0.0f;
    _derr_filt[i] = 0.0f;
  }
}

float CurrentBalanceController::computeTick(const float i_meas[NUM_CHANNELS], float r_target,
                                             float r_max, float r_ramp_per_ms, bool directionIsCcw,
                                             float freq_hz, float duty_out[NUM_CHANNELS]) {
  bool model_valid = freq_hz >= FREQ_MODEL_VALID_HZ;
  const float (*KX)[NUM_CHANNELS] = directionIsCcw ? LQR_KX_CCW : LQR_KX_CW;
  const float (*KD)[NUM_CHANNELS] = directionIsCcw ? LQR_KD_CCW : LQR_KD_CW;
  const float (*KZ)[NUM_CHANNELS] = directionIsCcw ? LQR_KZ_CCW : LQR_KZ_CW;
  const float (*FF)[NUM_CHANNELS] = directionIsCcw ? FEEDFORWARD_CCW : FEEDFORWARD_CW;
  const float (*BD)[NUM_CHANNELS] = directionIsCcw ? BD_CCW : BD_CW;
  const DutyAllocator &allocator = directionIsCcw ? _allocatorCcw : _allocatorCw;

  float err[NUM_CHANNELS];
  bool any_overcurrent = false;
  for (int i = 0; i < NUM_CHANNELS; i++) {
    err[i] = i_meas[i] - r_target;
    // I_SOFT_LIMIT_A, not I_MAX_A -- this soft backoff must trip below the
    // hard cap (PwmActuator::checkOvercurrentTrip, core 0) or it would never
    // get a chance to act: core 0 samples ~50x faster than this control
    // tick, so an equal threshold means the hard trip always wins the race.
    if (i_meas[i] > I_SOFT_LIMIT_A) any_overcurrent = true;
  }

  // Filtered discrete derivative of tracking error (see state_space_constants.h
  // for why this is filtered rather than a raw one-sample difference).
  for (int i = 0; i < NUM_CHANNELS; i++) {
    float derr_raw = (err[i] - _err_prev[i]) / CONTROL_TS_S;
    _derr_filt[i] += DERIV_ALPHA * (derr_raw - _derr_filt[i]);
  }

  // Coupling-aware feedforward: u_ff = FF @ x_target, x_target[j] == r_target
  // for every channel (shared target). Kept as an explicit matvec rather
  // than hand-optimized to row-sums -- trivial cost, stays correct if
  // targets are ever made non-uniform.
  //
  // FF/Bd are a STATIC (DC, omega=0) gain model, exactly correct only at
  // each channel's LC resonance f0 (~153-156Hz) -- see FreqGainModel.h and
  // progress.md Section 5 for the full derivation. freqGainCorrection()
  // scales u_ff UP (>=1x) to account for the true, higher impedance away
  // from resonance; this is independent of and complementary to
  // FREQ_MODEL_VALID_HZ's integral gate below, which only stops the
  // integral from accumulating on top of a wrong feedforward -- it doesn't
  // fix the feedforward's own magnitude.
  float u_ff[NUM_CHANNELS];
  for (int i = 0; i < NUM_CHANNELS; i++) {
    u_ff[i] = 0.0f;
    for (int j = 0; j < NUM_CHANNELS; j++) u_ff[i] += FF[i][j] * r_target;
    u_ff[i] *= freqGainCorrection(i, freq_hz);
  }

  // LQID feedback correction, added on top of feedforward -- u_desired is
  // the UNCONSTRAINED ideal duty this tick wants.
  float u_desired[NUM_CHANNELS];
  for (int i = 0; i < NUM_CHANNELS; i++) {
    float correction = 0.0f;
    for (int j = 0; j < NUM_CHANNELS; j++)
      correction -= KX[i][j] * err[j] + KD[i][j] * _derr_filt[j] + KZ[i][j] * _z_integral[j];
    u_desired[i] = u_ff[i] + correction;
  }

  // Constrained allocation: fold feedforward+feedback into one QP target
  // (x_target_eff = Bd @ u_desired -- "what current response did the
  // unconstrained controller want") and find the closest FEASIBLE duty that
  // reproduces it, letting non-saturated channels pick up a saturated
  // channel's slack optimally instead of just clipping it in isolation.
  // Two real bugs were found and fixed here on hardware before this worked
  // correctly: (1) ALLOCATOR_R_DUTY was inherited from
  // design_lqr_gains.py's offline --r-duty default (0.1), a completely
  // different cost formulation/scale -- at that value, a mere 10-point duty
  // deviation cost more than the entire current-tracking term, so duty
  // barely moved away from feedforward despite large persistent error (see
  // state_space_constants.h). (2) the anti-windup freeze below was
  // originally unconditional at DUTY_MAX, which could lock in an excessive
  // correction and never let it unwind after an overshoot -- see that
  // comment for the confirmed-on-hardware failure mode.
  float x_target_eff[NUM_CHANNELS];
  for (int i = 0; i < NUM_CHANNELS; i++) {
    x_target_eff[i] = 0.0f;
    for (int j = 0; j < NUM_CHANNELS; j++) x_target_eff[i] += BD[i][j] * u_desired[j];
  }
  allocator.allocate(x_target_eff, u_ff, _duty_prev, duty_out);

  // Per-channel anti-windup: freeze only the channel(s) actually pinned at
  // DUTY_MAX (or all channels on a global overcurrent trip) -- see this
  // file's header comment for why this must NOT be a global freeze anymore.
  // Never freeze on hitting DUTY_MIN (a separate, hard-won lesson from an
  // earlier hardware deadlock: low-clamping just means duty undershot the
  // floor, and the integral must keep accumulating to escape it).
  //
  // DIRECTIONAL freeze, not unconditional: freezing at DUTY_MAX must only
  // block the integral from growing FURTHER in the direction that deepens
  // saturation (err[i] < 0, i.e. still undershooting) -- it must NOT block
  // it from unwinding back down when the situation reverses (err[i] > 0,
  // i.e. now overshooting). An unconditional freeze locks in whatever
  // excessive correction caused the saturation and can never release it
  // even after current subsequently overshoots -- confirmed as a real
  // hardware bug (CCW: duty stuck at [77,73,100,100], current climbed to
  // ~4.5A against a 1.8A target, entirely unable to correct because the
  // integral was frozen solid). This is the same class of bug as the
  // DUTY_MIN lesson above, just the mirror-image direction.
  //
  // Also gated on model_valid (freq_hz >= FREQ_MODEL_VALID_HZ, see
  // state_space_constants.h): below that frequency the coupling model this
  // whole control law is fit against doesn't apply yet, so tracking error
  // there is a low-frequency-regime artifact, not something to integrate
  // against. Confirmed as a real hardware bug: channel D pinned at DUTY_MAX
  // for 13+s through RAMP_UP while its own error crossed from undershoot
  // into sustained overshoot, because correction accumulated during the
  // invalid-model region never unwound once the model became valid.
  bool any_saturated_high = false;
  for (int i = 0; i < NUM_CHANNELS; i++) {
    bool this_saturated_high = (duty_out[i] >= DUTY_MAX);
    any_saturated_high = any_saturated_high || this_saturated_high;
    bool would_deepen_saturation = this_saturated_high && (err[i] < 0.0f);
    if (!would_deepen_saturation && !any_overcurrent && model_valid)
      _z_integral[i] += CONTROL_TS_S * err[i];
  }

  for (int i = 0; i < NUM_CHANNELS; i++) {
    _err_prev[i] = err[i];
    _duty_prev[i] = duty_out[i];
  }

  // Reference governor: only raise r while there's headroom (no channel
  // saturated high); back off hard on a real overcurrent trip. This stays
  // a GLOBAL decision (unlike anti-windup) -- if any channel is already
  // maxed, raising the shared target further only deepens that channel's
  // saturation without helping the others hit a target they can't share.
  if (any_overcurrent) {
    r_target = clampf(r_target - OVERCURRENT_BACKOFF_A, 0.0f, r_max);
  } else if (!any_saturated_high) {
    r_target = clampf(r_target + r_ramp_per_ms * (float)CONTROL_TICK_MS, 0.0f, r_max);
  }
  return r_target;
}
