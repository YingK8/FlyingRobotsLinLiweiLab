#pragma once
// Generalizes main_current_pid.cpp's proven per-channel PI(+D) control law
// (0.447A true whole-run peak spread, the best-performing controller built
// on this rig -- see progress.md) from "equalize all channels" to
// "track arbitrary per-channel current RATIOS", so the same control law
// serves both a balance experiment (ratios=[1,1,1,1]) and a tilt experiment
// (arbitrary ratios, e.g. [1.0,0.5,0.3,0.8]) without duplicating the PI+D/
// anti-windup logic. Pure computation -- no hardware I/O -- so it's directly
// testable (see test/native_pid/) independent of which firmware embeds it.
//
// Two modes, chosen via Config::sharedConstraint:
//
// - Shared-constraint mode (true): direct generalization of
//   main_current_pid.cpp's anchor-channel policy. That policy exists to
//   maximize achievable current under a SHARED supply constraint --
//   pushing the ratio-normalized weakest channel toward DUTY_MAX is the
//   only way to raise the whole group's ceiling. With ratios=[1,1,1,1]
//   this reproduces main_current_pid.cpp's exact formulas (see
//   test/native_pid/ for the regression-equivalence tests that pin this).
//
// - Independent mode (false): for rigs where each coil has its own supply/
//   driver headroom (no shared constraint) -- e.g. a tilt experiment. No
//   anchor/argmin at all; every channel independently tracks
//   magnitude * ratios[i] via the same PI+D + anti-windup formula. The
//   shared scalar `magnitude` is a reference-governor ramp: it only rises
//   while no channel is saturated high or over the overcurrent limit, AND
//   every channel is already tracking its current target within
//   Config::magnitudeSettleTolA -- it backs off (globally) the instant any
//   channel trips overcurrent. Two deliberate design choices here, not
//   incidental: (1) backing off the shared magnitude (rather than letting
//   one channel silently clip while others keep climbing) keeps the
//   commanded ratio EXACT at all times -- the physically-meaningful
//   invariant for a tilt torque profile; (2) the settle-tolerance gate on
//   ramping up exists because, UNLIKE the shared-constraint anchor (which
//   ramps its OWN duty directly, inherently self-limiting since duty is
//   clamped every tick), this mode ramps a virtual CURRENT target that
//   must pass through a lagged PI loop before duty catches up -- without
//   this gate, magnitude races many steps ahead of what the loop has
//   actually achieved before a channel's clamped duty ever reads back
//   >=dutyMax, overshooting the sustainable ceiling and permanently
//   saturating every channel (confirmed via native test: with an ungated
//   ramp, magnitude overshot to ~10.5 against a ~3.0 sustainable ceiling).
//   Gating the ramp on "has the group actually settled near its current
//   target" keeps magnitude from ever getting more than one step ahead of
//   what's been verified achievable.

// Deliberately does NOT include constants.h -- that header pulls in
// ESP-IDF/Arduino headers (driver/gpio.h, <Arduino.h>), which would make
// this class impossible to compile/test on a native (host) target. This
// class owns its own channel-count constant instead, matching the "pure
// computation, no hardware I/O" design goal stated above.
constexpr int kNumChannels = 4;

class RatioCurrentController {
public:
  struct Config {
    float ratios[kNumChannels];  // k_i, target current PROPORTIONS (not
                                  // required to sum to 1 -- only relative
                                  // magnitude matters)
    bool sharedConstraint;       // true: anchor-channel mode. false: independent mode.
    float kp, ki, kd;            // duty %, per A of error (and per A/tick, per A*tick)
    float rampPctPerMs;          // anchor-channel ramp rate (shared-constraint mode)
                                  // / magnitude ramp rate (independent mode)
    float minSwitchMarginA;      // anchor hysteresis, shared-constraint mode only
    float dutyMin, dutyMax;
    float iMaxA;                 // hard per-channel overcurrent trip
    float overcurrentBackoffPct; // integrator/magnitude step-back on trip
    float nominalTickMs;         // KI/KD rate-normalization reference (see
                                  // main_current_pid.cpp's NOMINAL_TICK_MS)
    float magnitudeSettleTolA;   // independent mode only -- see tickIndependent()'s
                                  // header comment for why this exists
  };

  explicit RatioCurrentController(const Config &cfg);

  // One control tick. dt_ms is the actual elapsed time since the previous
  // call (KI/KD are rate-normalized against Config::nominalTickMs, exactly
  // as main_current_pid.cpp's runControlTick() does).
  void computeTick(const float i_meas[kNumChannels], float dt_ms,
                    float duty_out[kNumChannels]);

  // Independent-mode variant: same PI+D/anti-windup/overcurrent-backoff
  // control law as computeTick(), but `reference` (this tick's shared
  // target scale, i.e. each channel tracks reference * ratios[i]) is
  // supplied by the caller instead of self-governed by the internal
  // magnitude ramp. For callers whose own schedule already dictates the
  // reference's timing (e.g. main_experiment.cpp driving it from a JSON
  // task queue's commanded carrier duty) and only want this class's
  // ratio-tracking control law layered on top. Does not read or advance
  // the internal _magnitude ramp -- computeTick() and
  // computeTickWithReference() are independent entrypoints; don't mix
  // calls to both within one reset() epoch. Falls back to the normal
  // shared-constraint tick if Config::sharedConstraint is true, since that
  // mode's "reference" is a live measurement (the anchor channel), not a
  // governed scalar an external caller could meaningfully replace.
  void computeTickWithReference(const float i_meas[kNumChannels], float dt_ms,
                                 float reference, float duty_out[kNumChannels]);

  // Reset all internal state (integrators, duty warm-start, anchor/
  // magnitude) -- call once on entering a fresh run (ARMING->RAMP_UP).
  void reset();

  void setGains(float kp, float ki, float kd) { _cfg.kp = kp; _cfg.ki = ki; _cfg.kd = kd; }
  void setRampPctPerMs(float r) { _cfg.rampPctPerMs = r; }
  const Config &config() const { return _cfg; }
  float magnitude() const { return _magnitude; }  // independent mode diagnostics
  float integrator(int i) const { return _integrator[i]; }

private:
  Config _cfg;
  float _integrator[kNumChannels];
  float _dutyOut[kNumChannels];
  float _lastErr[kNumChannels];
  int _idxAnchor;    // shared-constraint mode only
  float _magnitude;  // independent mode only -- the shared ramping r(t)

  void tickSharedConstraint(const float i_meas[kNumChannels], float dt_ms,
                              float duty_out[kNumChannels]);
  void tickIndependent(const float i_meas[kNumChannels], float dt_ms,
                         float duty_out[kNumChannels]);
  // Shared PI+D + asymmetric anti-windup formula for one non-anchor/
  // independent channel, tracking `target`. Identical to
  // main_current_pid.cpp's non-anchor branch (lines 125-146). If
  // `railLocked` is non-null, set to true when anti-windup froze the
  // integrator this tick (i.e. the channel is doing everything physically
  // possible and its error is a saturation artifact, not something more
  // ramp/target movement would fix) -- see tickIndependent()'s use of this.
  float pidStep(int i, float target, float i_meas_i, float rateScale,
                 bool *railLocked = nullptr);
};
