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

  // A channel is "active" only if the schedule commands a (non-NAN) carrier
  // level for it. Untouched channels stay NAN (see JsonPhaseSequencer) -> park
  // them off and exclude from the balance math. maxCeil = the largest commanded
  // carrier, used to normalize the per-channel ratios below.
  bool active[N];
  int firstActive = -1;
  float maxCeil = 0.0f;
  for (int i = 0; i < N; i++) {
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

  // Per-channel current RATIO = commanded carrier relative to the largest
  // commanded carrier. A channel commanded at the top level (ratio == 1) is a
  // "reference" channel that HOLDS thrust; a channel the schedule stepped BELOW
  // it (tilt: coils 0 & 3 dropped to 90/80/...%) is a "follower" with ratio < 1.
  // nrm[] is the ratio-normalized current. isRef[] marks the reference channels.
  float ratio[N];
  float nrm[N];
  bool isRef[N];
  for (int i = 0; i < N; i++) {
    ratio[i] = active[i] ? (ceiling[i] / maxCeil) : NAN;
    nrm[i] = (active[i] && ratio[i] > 1e-6f) ? (iMeas[i] / ratio[i]) : INFINITY;
    // Within refBandPct of the top commanded carrier => reference/held channel.
    isRef[i] = active[i] && (ceiling[i] >= maxCeil - _cfg.refBandPct);
  }

  // Anchor = argmin(nrm) over the REFERENCE channels only. This is the crux of
  // the tilt: the held channels set the magnitude and stay balanced to their
  // own weakest (thrust regulation), while the stepped-down followers are NEVER
  // the anchor -- so their (nonlinearly) lower current can't drag the reference
  // channels down. Each follower is then PI'd to magnitude * ratio[i], i.e. a
  // clean fraction of the held level. When nothing is stepped down (spin-up,
  // uniform carrier sweep) every active channel is a reference => this reduces
  // to the original global argmin / pure equal-balance, unchanged. Latch +
  // hysteresis (minSwitchMarginA) as before.
  int trueIdxMin = -1;
  for (int i = 0; i < N; i++)
    if (isRef[i] && (trueIdxMin < 0 || nrm[i] < nrm[trueIdxMin]))
      trueIdxMin = i;
  if (trueIdxMin < 0) {
    // No reference channel (all active channels stepped down equally). Fall back
    // to the plain active argmin so there is always a valid anchor.
    for (int i = 0; i < N; i++)
      if (active[i] && (trueIdxMin < 0 || nrm[i] < nrm[trueIdxMin]))
        trueIdxMin = i;
  }
  if (!active[_idxMin] || !isRef[_idxMin])
    _idxMin = trueIdxMin; // latched anchor turned off / stepped down -> re-pick
  if (nrm[trueIdxMin] < nrm[_idxMin] - _cfg.minSwitchMarginA)
    _idxMin = trueIdxMin;
  const float magnitude = nrm[_idxMin]; // normalized anchor level; each active
                                        // channel targets magnitude * ratio[i]

  // Freeze-at-end-of-spin-up: the first tick the schedule commands a DIFFERENTIAL
  // (a follower channel with ratio < 1 appears, i.e. a tilt step begins), latch
  // the balanced level the anchor-ramp discovered during spin-up as a CONSTANT
  // hold target. Thereafter every channel current-regulates to _holdTarget *
  // ratio[i] (held channels to _holdTarget itself) and the anchor no longer
  // ramps. This keeps the held channels' CURRENT flat: as the followers drop and
  // free shared-supply headroom, the loop pulls the held channels' DUTY down to
  // hold _holdTarget, instead of leaving them pinned at duty-ceiling while their
  // current drifts up. A uniform-carrier schedule never produces a follower, so
  // it never freezes and keeps the original anchor-ramp balance behavior.
  bool hasFollower = false;
  for (int i = 0; i < N; i++)
    if (active[i] && !isRef[i]) {
      hasFollower = true;
      break;
    }
  if (hasFollower && !_holdFrozen) {
    _holdTarget = magnitude;
    _holdFrozen = true;
  }
  const float mag = _holdFrozen ? _holdTarget : magnitude;

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
    } else if (!_holdFrozen && i == _idxMin) {
      // Anchor (spin-up only): ramp the bottleneck channel toward its ceiling to
      // discover the balanced level. Ramped, not snapped, so the argmin above
      // can reassign mid-ramp. Once the hold target is frozen this branch is
      // disabled and the former anchor current-regulates like everyone else.
      duty = clampf(_dutyOut[i] + _cfg.minRampPctPerMs * dtMs, lo, hi);
      _integrator[i] = duty; // continuity for when it later falls back to PI
      _lastErr[i] = 0.0f;    // keep the derivative fresh for the handoff
    } else {
      // PI(+D) toward this channel's ratio-scaled share of the hold level, with
      // directional anti-windup: freeze the integrator only when the error
      // pushes FURTHER into saturation. (target == mag for ratio == 1, so
      // uniform-carrier balance is unchanged.)
      const float target = mag * ratio[i];
      const float err = target - iMeas[i];
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
