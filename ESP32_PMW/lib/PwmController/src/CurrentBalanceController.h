#pragma once

#include <Arduino.h>

// Onboard current-balance PI controller; every experiment firmware shares this
// one tuned loop.
//
// Policy (global, no board grouping): each tick, ratio[i] = ceiling[i] / max_j
// ceiling[j]. The ratio-normalized lowest channel (argmin iMeas[i]/ratio[i]) is
// ramped toward its ceiling duty (push the bottleneck as hard as allowed);
// every other channel runs PI(+D) toward magnitude*ratio[i], so the four
// currents converge to that ratio. Ceiling is per-channel, supplied every tick
// (PhaseSequencer::getCommandedCarrier).
//   - UNIFORM carrier => all ratios 1 => equal-balance (below 100% still
//     equalizes, capped at that duty).
//   - DIFFERENTIAL carrier (tilt: some coils stepped down) makes those channels
//     track a proportionally lower current: closed-loop tilt, thrust still
//     regulated, no per-channel hand trims.
// NAN ceiling => channel parked off (duty 0), excluded from the argmin (matches
// how the JSON sequencer leaves untouched channels NAN).

struct BalanceConfig {
  // Converged tuning (KP=2.2, KI=0.10, KD=0.15); runtime-tunable via setters.
  float kp = 2.2f;                  // duty % per A of error
  float ki = 0.10f;                 // duty % per A of error, per nominal tick
  float kd = 0.15f;
  float dutyMin = 5.0f;
  float dutyMax = 100.0f;           // hard clamp; per-channel ceiling applied on
                                    // top each tick
  float iMax = 12.0f;               // hard per-channel overcurrent backoff level
  float overcurrentBackoffPct = 5.0f;
  // Gains tuned at a fixed 2ms step; KI/derivative scale by dt/nominalTickMs so
  // their meaning is rate-independent.
  float nominalTickMs = 2.0f;
  // Hysteresis on the latched-minimum channel: switch only if another reads at
  // least this much lower, so noise near convergence doesn't flap the min.
  float minSwitchMarginA = 0.3f;
  // Ramp rate of the latched-minimum channel toward its ceiling (duty %/ms at
  // nominalTickMs). Ramping, not snapping, lets the argmin reassign to a
  // genuinely weaker channel.
  float minRampPctPerMs = 0.05f;

  // Below this anchor current, co-ramp all active channels together instead of
  // only the anchor. At low drive frequency every channel reads ~0 A, so a lone
  // anchor would run to 100% while the rest sit parked; co-ramping holds them
  // equal until the current is measurable, then the PI takes over.
  float minSignalA = 0.25f;

  // Carrier within this % of the top carrier => a "reference" (held) channel,
  // eligible as anchor; anything stepped lower is a tilt follower. Only
  // reference channels set the balance magnitude, so followers drop to
  // magnitude*ratio without dragging the held channels down.
  float refBandPct = 0.5f;
};

class CurrentBalanceController {
public:
  static const int N = 4;

  explicit CurrentBalanceController(const BalanceConfig &cfg = BalanceConfig());

  // Reset integrator/duty/error state; call before a run with the duty every
  // channel starts equal at (e.g. 50%).
  void reset(float startDuty);

  // One control step (no I/O; caller applies dutyOut to hardware).
  //   iMeas[N]   filtered per-channel current (A)
  //   dtMs       real interval since the previous call
  //   ceiling[N] per-channel max duty (%) the schedule commands (NAN => parked)
  //   dutyOut[N] written: new per-channel carrier duty
  void step(const float *iMeas, float dtMs, const float *ceiling,
            float *dutyOut);

  // Runtime tuning (serial kp=/ki=/kd=/ramp=).
  void setGains(float kp, float ki, float kd) {
    _cfg.kp = kp;
    _cfg.ki = ki;
    _cfg.kd = kd;
  }
  void setRamp(float pctPerMs) { _cfg.minRampPctPerMs = pctPerMs; }
  const BalanceConfig &config() const { return _cfg; }
  int latchedMinIndex() const { return _idxMin; }
  bool holdFrozen() const { return _holdFrozen; }
  float holdTarget() const { return _holdTarget; }

private:
  BalanceConfig _cfg;
  float _integrator[N];
  float _dutyOut[N];
  float _lastErr[N];
  int _idxMin; // latched; persists across ticks, see minSwitchMarginA
  // Freeze-at-end-of-spin-up: latched true the first tick a follower (ratio < 1,
  // a tilt step) appears. _holdTarget = balanced current captured then, held
  // constant after.
  bool _holdFrozen;
  float _holdTarget;
};
