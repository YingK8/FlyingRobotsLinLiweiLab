#include "RatioCurrentController.h"

#include <math.h>

static inline float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

RatioCurrentController::RatioCurrentController(const Config &cfg) : _cfg(cfg) {
  reset();
}

void RatioCurrentController::reset() {
  // Shared-constraint mode: 50.0f mirrors main_current_pid.cpp's
  // START_DUTY ("all channels start equal; PI does the rest") --
  // regression-locked, must not change (see the class header's balance-
  // mode equivalence claim).
  //
  // Independent mode: magnitude ALSO starts at 0 (below), so warm-starting
  // duty at 50.0f while every target starts near 0 creates a large,
  // slow-to-resolve mismatch (confirmed via native test: with a tight
  // magnitudeSettleTolA, convergence stalled for thousands of ticks
  // because duty had to decay from 50% down to near its true near-zero
  // target before maxAbsErr could ever clear the settle gate). Starting
  // at dutyMin instead keeps duty and target starting close together, so
  // they can ramp up in step.
  float startDuty = _cfg.sharedConstraint ? 50.0f : _cfg.dutyMin;
  for (int i = 0; i < kNumChannels; i++) {
    _integrator[i] = startDuty;
    _dutyOut[i] = startDuty;
    _lastErr[i] = 0.0f;
  }
  _idxAnchor = 0;
  _magnitude = 0.0f;  // independent mode ramps up from 0
}

float RatioCurrentController::pidStep(int i, float target, float i_meas_i, float rateScale,
                                        bool *railLocked) {
  // Identical to main_current_pid.cpp's non-anchor PI+D branch (lines
  // 125-146): asymmetric anti-windup -- freeze the integrator ONLY when the
  // error is pushing FURTHER into the same saturation the candidate is
  // already past, never merely because the candidate exceeds the clamp.
  float err = target - i_meas_i;
  float derivative = (rateScale > 0.0f) ? (err - _lastErr[i]) / rateScale : 0.0f;
  _lastErr[i] = err;
  float candidate = _integrator[i] + _cfg.kp * err + _cfg.kd * derivative;
  float duty = clampf(candidate, _cfg.dutyMin, _cfg.dutyMax);
  bool pushingIntoHighSat = (candidate > _cfg.dutyMax) && (err > 0.0f);
  bool pushingIntoLowSat = (candidate < _cfg.dutyMin) && (err < 0.0f);
  if (!pushingIntoHighSat && !pushingIntoLowSat) _integrator[i] += _cfg.ki * err * rateScale;
  if (railLocked) *railLocked = pushingIntoHighSat || pushingIntoLowSat;
  return duty;
}

void RatioCurrentController::tickSharedConstraint(const float i_meas[kNumChannels], float dt_ms,
                                                     float duty_out[kNumChannels]) {
  float rateScale = dt_ms / _cfg.nominalTickMs;

  // Ratio-normalized argmin -- generalizes main_current_pid.cpp's raw-
  // current argmin (i_meas[i]). With ratios=[1,1,1,1] this is bit-for-bit
  // the same comparison (division by 1.0f is exact).
  float nrm[kNumChannels];
  for (int i = 0; i < kNumChannels; i++) nrm[i] = i_meas[i] / _cfg.ratios[i];

  int trueIdxAnchor = 0;
  for (int i = 1; i < kNumChannels; i++)
    if (nrm[i] < nrm[trueIdxAnchor]) trueIdxAnchor = i;
  if (nrm[trueIdxAnchor] < nrm[_idxAnchor] - _cfg.minSwitchMarginA) _idxAnchor = trueIdxAnchor;

  // Normalized magnitude every channel's target is scaled from -- with
  // ratios=[1,1,1,1] this equals i_meas[_idxAnchor] exactly, matching
  // main_current_pid.cpp's i_min.
  float magnitude = nrm[_idxAnchor];

  for (int i = 0; i < kNumChannels; i++) {
    float duty;
    if (i_meas[i] > _cfg.iMaxA) {
      _integrator[i] -= _cfg.overcurrentBackoffPct;
      duty = clampf(_integrator[i], _cfg.dutyMin, _cfg.dutyMax);
    } else if (i == _idxAnchor) {
      // Ramped, not snapped: gives the argmin re-evaluation above time to
      // reassign _idxAnchor to a genuinely different, still-weaker channel
      // before this one overshoots the group.
      duty = clampf(_dutyOut[i] + _cfg.rampPctPerMs * dt_ms, _cfg.dutyMin, _cfg.dutyMax);
      _integrator[i] = duty;  // continuity for when this channel later falls
                               // back under normal PI control
      _lastErr[i] = 0.0f;     // target == i_meas[_idxAnchor] while forced --
                               // keep the derivative term fresh for when it exits
    } else {
      duty = pidStep(i, magnitude * _cfg.ratios[i], i_meas[i], rateScale);
    }
    duty_out[i] = duty;
    _dutyOut[i] = duty;
  }
}

void RatioCurrentController::tickIndependent(const float i_meas[kNumChannels], float dt_ms,
                                                float duty_out[kNumChannels]) {
  float rateScale = dt_ms / _cfg.nominalTickMs;
  bool anySaturatedHigh = false;
  bool anyOvercurrent = false;
  float maxAbsErr = 0.0f;

  for (int i = 0; i < kNumChannels; i++) {
    float target = _magnitude * _cfg.ratios[i];
    float duty;
    bool railLocked = false;
    if (i_meas[i] > _cfg.iMaxA) {
      anyOvercurrent = true;
      _integrator[i] -= _cfg.overcurrentBackoffPct;
      duty = clampf(_integrator[i], _cfg.dutyMin, _cfg.dutyMax);
    } else {
      duty = pidStep(i, target, i_meas[i], rateScale, &railLocked);
    }
    // A rail-locked channel (anti-windup frozen, doing everything
    // physically possible) is EXCLUDED from the settle check -- its error
    // is a saturation artifact, not evidence the group hasn't converged.
    // Without this, a channel whose early (small) target sits below what
    // dutyMin can physically produce would permanently deadlock the ramp:
    // duty pinned at dutyMin, error can never shrink, magnitude can never
    // advance to raise the target past the floor (confirmed via native
    // test -- this was a real bug, not just a theoretical one).
    if (!railLocked) {
      float absErr = fabsf(target - i_meas[i]);
      if (absErr > maxAbsErr) maxAbsErr = absErr;
    }
    if (duty >= _cfg.dutyMax) anySaturatedHigh = true;
    duty_out[i] = duty;
    _dutyOut[i] = duty;
  }

  // Reference governor on the SHARED magnitude, a global (not per-channel)
  // decision -- see this class's header comment for why, including why the
  // ramp-up branch also requires maxAbsErr < magnitudeSettleTolA (not just
  // !anySaturatedHigh) -- that gate is what stops magnitude from racing
  // ahead of what the lagged PI loop has actually achieved.
  if (anyOvercurrent) {
    float backedOff = _magnitude - _cfg.overcurrentBackoffPct;
    _magnitude = backedOff < 0.0f ? 0.0f : backedOff;
  } else if (!anySaturatedHigh && maxAbsErr < _cfg.magnitudeSettleTolA) {
    _magnitude += _cfg.rampPctPerMs * dt_ms;
  }
}

void RatioCurrentController::computeTick(const float i_meas[kNumChannels], float dt_ms,
                                           float duty_out[kNumChannels]) {
  if (_cfg.sharedConstraint) tickSharedConstraint(i_meas, dt_ms, duty_out);
  else tickIndependent(i_meas, dt_ms, duty_out);
}

void RatioCurrentController::computeTickWithReference(const float i_meas[kNumChannels], float dt_ms,
                                                         float reference,
                                                         float duty_out[kNumChannels]) {
  if (_cfg.sharedConstraint) {
    tickSharedConstraint(i_meas, dt_ms, duty_out);
    return;
  }
  float rateScale = dt_ms / _cfg.nominalTickMs;
  for (int i = 0; i < kNumChannels; i++) {
    float target = reference * _cfg.ratios[i];
    float duty;
    if (i_meas[i] > _cfg.iMaxA) {
      _integrator[i] -= _cfg.overcurrentBackoffPct;
      duty = clampf(_integrator[i], _cfg.dutyMin, _cfg.dutyMax);
    } else {
      duty = pidStep(i, target, i_meas[i], rateScale);
    }
    duty_out[i] = duty;
    _dutyOut[i] = duty;
  }
}
