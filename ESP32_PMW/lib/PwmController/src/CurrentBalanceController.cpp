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
  _holdFrozen = false;
  _holdTarget = 0.0f;
  for (int i = 0; i < N; i++) {
    _integrator[i] = startDuty;
    _dutyOut[i] = startDuty;
    _lastErr[i] = 0.0f;
  }
}

void CurrentBalanceController::step(const float *iMeas, float dtMs,
                             const float *ceiling, float *dutyOut) {
  const float rateScale = dtMs / _cfg.nominalTickMs;

  bool active[N];
  int firstActive = -1;
  float maxCeil = 0.0f;
  for (int i = 0; i < N; i++) {
    // NAN ceiling parks the channel off and excludes it from the anchor.
    active[i] = ceiling && !isnan(ceiling[i]);
    if (active[i]) {
      if (firstActive < 0)
        firstActive = i;
      if (ceiling[i] > maxCeil)
        maxCeil = ceiling[i];
    }
  }
  if (firstActive < 0 || maxCeil <= 0.0f) {
    for (int i = 0; i < N; i++) {
      dutyOut[i] = 0.0f;
      _dutyOut[i] = 0.0f;
    }
    return;
  }

  float ratio[N];
  float nrm[N];
  bool isRef[N];
  for (int i = 0; i < N; i++) {
    ratio[i] = active[i] ? (ceiling[i] / maxCeil) : NAN;
    nrm[i] = (active[i] && ratio[i] > 1e-6f) ? (iMeas[i] / ratio[i]) : INFINITY;
    isRef[i] = active[i] && (ceiling[i] >= maxCeil - _cfg.refBandPct);
  }

  int trueIdxMin = -1;
  for (int i = 0; i < N; i++)
    if (isRef[i] && (trueIdxMin < 0 || nrm[i] < nrm[trueIdxMin]))
      trueIdxMin = i;
  if (trueIdxMin < 0) {
    for (int i = 0; i < N; i++)
      if (active[i] && (trueIdxMin < 0 || nrm[i] < nrm[trueIdxMin]))
        trueIdxMin = i;
  }
  if (!active[_idxMin] || !isRef[_idxMin])
    _idxMin = trueIdxMin;
  if (nrm[trueIdxMin] < nrm[_idxMin] - _cfg.minSwitchMarginA)
    _idxMin = trueIdxMin;
  const float magnitude = nrm[_idxMin];

  bool hasFollower = false;
  for (int i = 0; i < N; i++)
    if (active[i] && !isRef[i]) {
      hasFollower = true;
      break;
    }
  // freeze at the first follower so the held current stops drifting up.
  if (hasFollower && !_holdFrozen) {
    _holdTarget = magnitude;
    _holdFrozen = true;
  }
  const float mag = _holdFrozen ? _holdTarget : magnitude;

  const bool coRamp = !_holdFrozen && (magnitude < _cfg.minSignalA);

  for (int i = 0; i < N; i++) {
    if (!active[i]) {
      dutyOut[i] = 0.0f;
      _dutyOut[i] = 0.0f;
      continue;
    }

    const float hi = clampf(ceiling[i], 0.0f, _cfg.dutyMax);
    const float lo = (hi < _cfg.dutyMin) ? hi : _cfg.dutyMin;

    float duty;
    if (iMeas[i] > _cfg.iMax) {
      _integrator[i] -= _cfg.overcurrentBackoffPct;
      duty = clampf(_integrator[i], lo, hi);
    } else if (!_holdFrozen && (coRamp || i == _idxMin)) {
      duty = clampf(_dutyOut[i] + _cfg.minRampPctPerMs * dtMs, lo, hi);
      _integrator[i] = duty;
      _lastErr[i] = 0.0f;
    } else {
      const float target = mag * ratio[i];
      const float err = target - iMeas[i];
      const float derivative =
          (rateScale > 0.0f) ? (err - _lastErr[i]) / rateScale : 0.0f;
      _lastErr[i] = err;
      const float candidate = _integrator[i] + _cfg.kp * err + _cfg.kd * derivative;
      duty = clampf(candidate, lo, hi);
      // freeze the integrator only when the error pushes further into saturation.
      const bool pushingIntoHighSat = (candidate > hi) && (err > 0.0f);
      const bool pushingIntoLowSat = (candidate < lo) && (err < 0.0f);
      if (!pushingIntoHighSat && !pushingIntoLowSat)
        _integrator[i] += _cfg.ki * err * rateScale;
    }

    dutyOut[i] = duty;
    _dutyOut[i] = duty;
  }
}
