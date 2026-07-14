#include "CurrentBalanceController.h"
#include <math.h>

static inline float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

CurrentBalanceController::CurrentBalanceController(const BalanceConfig &cfg)
    : _cfg(cfg) {
  reset(cfg.dutyMin);
}

void CurrentBalanceController::reset(float startDuty) {
  _idxMin = 0;
  for (int i = 0; i < N; i++) {
    _integrator[i] = startDuty;
    _dutyOut[i] = startDuty;
    _lastErr[i] = 0.0f;
  }
}

void CurrentBalanceController::step(const float *iMeas, float dtMs,
                                    const float *ceiling, float *dutyOut) {
  const float rateScale = dtMs / _cfg.nominalTickMs;

  // A channel is "active" only if the schedule commands a (non-NAN) carrier
  // level for it. Untouched channels stay NAN (see JsonPhaseSequencer) -> park
  // them off and exclude from the balance math.
  bool active[N];
  int firstActive = -1;
  for (int i = 0; i < N; i++) {
    active[i] = ceiling && !isnan(ceiling[i]);
    if (active[i] && firstActive < 0)
      firstActive = i;
  }
  if (firstActive < 0) {
    for (int i = 0; i < N; i++) {
      dutyOut[i] = 0.0f;
      _dutyOut[i] = 0.0f;
    }
    return;
  }

  // Argmin over active channels, with the same latch + hysteresis as the
  // original inline controller (don't re-latch unless a different active
  // channel reads minSwitchMarginA lower).
  int trueIdxMin = firstActive;
  for (int i = firstActive + 1; i < N; i++)
    if (active[i] && iMeas[i] < iMeas[trueIdxMin])
      trueIdxMin = i;
  if (!active[_idxMin])
    _idxMin = trueIdxMin; // latched channel turned off -> follow the argmin
  if (iMeas[trueIdxMin] < iMeas[_idxMin] - _cfg.minSwitchMarginA)
    _idxMin = trueIdxMin;
  const float iMin = iMeas[_idxMin]; // anchor level every other channel targets

  for (int i = 0; i < N; i++) {
    if (!active[i]) {
      dutyOut[i] = 0.0f;
      _dutyOut[i] = 0.0f;
      continue;
    }

    // Per-channel bounds: the schedule's commanded carrier is the ceiling the
    // loop regulates beneath. When the ceiling drops below dutyMin (e.g. the
    // tail of a step-down toward 0%), collapse the whole range to it so the
    // channel can actually reach 0 instead of being pinned at dutyMin.
    const float hi = clampf(ceiling[i], 0.0f, _cfg.dutyMax);
    const float lo = (hi < _cfg.dutyMin) ? hi : _cfg.dutyMin;

    float duty;
    if (iMeas[i] > _cfg.iMax) {
      // Overcurrent backoff.
      _integrator[i] -= _cfg.overcurrentBackoffPct;
      duty = clampf(_integrator[i], lo, hi);
    } else if (i == _idxMin) {
      // Anchor: ramp the bottleneck channel toward its ceiling (was a hard
      // 100% in the original; now the schedule-commanded level). Ramped, not
      // snapped, so the argmin above can reassign mid-ramp.
      duty = clampf(_dutyOut[i] + _cfg.minRampPctPerMs * dtMs, lo, hi);
      _integrator[i] = duty; // continuity for when it later falls back to PI
      _lastErr[i] = 0.0f;    // keep the derivative fresh for the handoff
    } else {
      // PI(+D) toward the anchor current, with directional anti-windup: freeze
      // the integrator only when the error pushes FURTHER into saturation.
      const float err = iMin - iMeas[i];
      const float derivative =
          (rateScale > 0.0f) ? (err - _lastErr[i]) / rateScale : 0.0f;
      _lastErr[i] = err;
      const float candidate = _integrator[i] + _cfg.kp * err + _cfg.kd * derivative;
      duty = clampf(candidate, lo, hi);
      const bool pushingIntoHighSat = (candidate > hi) && (err > 0.0f);
      const bool pushingIntoLowSat = (candidate < lo) && (err < 0.0f);
      if (!pushingIntoHighSat && !pushingIntoLowSat)
        _integrator[i] += _cfg.ki * err * rateScale;
    }

    dutyOut[i] = duty;
    _dutyOut[i] = duty;
  }
}
