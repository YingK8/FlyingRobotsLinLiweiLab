#pragma once

#include <Arduino.h>

// Onboard current-balance PI controller, extracted verbatim (behavior-wise)
// from the inline runControlTick() that lived in src/main_current_pid.cpp so
// every experiment firmware can share one tuned balance loop.
//
// Policy (global, no board grouping): each tick, per-channel RATIOS are derived
// from the commanded carrier relative to the largest commanded carrier
// (ratio[i] = ceiling[i] / max_j ceiling[j]). The ratio-normalized LOWEST
// channel (argmin iMeas[i]/ratio[i]) is ramped toward its ceiling duty (push
// the bottleneck as hard as allowed); every other channel is driven by PI(+D)
// toward magnitude * ratio[i], so the four channel currents converge to that
// ratio. The ceiling is per-channel and supplied every tick (see
// PhaseSequencer::getCommandedCarrier).
//   - UNIFORM commanded carrier => all ratios == 1 => pure equal-balance, exactly
//     the original controller (main_current_pid.cpp: flat 100% => push to max and
//     equalize). A uniform carrier below 100% still equalizes, capped at that duty.
//   - DIFFERENTIAL carrier (e.g. tilt: coils 0 & 3 stepped below 100%) makes those
//     channels track a proportionally lower current -- a closed-loop tilt with
//     thrust still regulated -- instead of hand-tuning per-channel trims.
//
// A NAN ceiling entry means "this channel isn't commanded on" -> it is parked
// off (duty 0) and excluded from the argmin, matching how the JSON sequencer
// leaves untouched channels NAN.

struct BalanceConfig {
  // Converged tuning from main_current_pid.cpp's history: KP=2.2, KI=0.10,
  // KD=0.15. Runtime-tunable via the setters below.
  float kp = 2.2f;                  // duty % per A of error
  float ki = 0.10f;                 // duty % per A of error, per nominal tick
  float kd = 0.15f;
  float dutyMin = 5.0f;
  float dutyMax = 100.0f;           // hard clamp; the per-channel ceiling is
                                    // applied on top of this each tick
  float iMax = 12.0f;               // hard per-channel overcurrent backoff level
  float overcurrentBackoffPct = 5.0f;
  // Gains were tuned assuming a fixed 2ms step; KI/derivative are scaled by
  // (actual dt / nominalTickMs) so their real-world meaning is rate-independent.
  float nominalTickMs = 2.0f;
  // Hysteresis on WHICH channel is latched as the forced-minimum: only switch
  // if a DIFFERENT channel reads at least this much lower, so measurement noise
  // near convergence doesn't flap the min identity and undo progress.
  float minSwitchMarginA = 0.3f;
  // Rate limit on how fast the latched-minimum channel ramps toward its ceiling
  // (duty %/ms at nominalTickMs). Ramping (not snapping) gives the argmin
  // re-evaluation time to reassign the min to a genuinely weaker channel.
  float minRampPctPerMs = 0.05f;

  // A channel whose commanded carrier is within this many percent of the top
  // commanded carrier counts as a "reference" (held) channel and is eligible to
  // be the anchor; anything stepped further down (a tilt follower) is not. Only
  // reference channels set the balance magnitude, so tilt followers can drop to
  // magnitude*ratio without dragging the held channels down with them.
  float refBandPct = 0.5f;
};

class CurrentBalanceController {
public:
  static const int N = 4;

  explicit CurrentBalanceController(const BalanceConfig &cfg = BalanceConfig());

  // Reset integrator/duty/error state; call once before a run starts, with the
  // duty every channel begins equal at (e.g. 50%).
  void reset(float startDuty);

  // One control step. iMeas[N] = filtered per-channel current (A); dtMs = real
  // interval since the previous call; ceiling[N] = per-channel max duty (%) the
  // schedule currently commands (NAN => channel parked off). Writes the new
  // per-channel carrier duty into dutyOut[N]. The caller applies dutyOut to the
  // hardware (this class does no I/O).
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
  int _idxMin; // latched -- persists across ticks, see minSwitchMarginA
  // Freeze-at-end-of-spin-up hold target: latched true the first tick a follower
  // (ratio < 1, i.e. a tilt step) appears; _holdTarget is the balanced current
  // level captured at that instant, held constant thereafter.
  bool _holdFrozen;
  float _holdTarget;
};
