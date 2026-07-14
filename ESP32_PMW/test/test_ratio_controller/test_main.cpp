// Offline (native, no ESP32/coils) test suite for RatioCurrentController --
// see the plan ("Generalize the PI current-balance controller to ratio-
// tracking profiles") Section 4. Scope note: the JSON profile loader
// (src/PiProfile.cpp) is NOT covered here -- it depends on SPIFFS/Serial
// (ESP32-only) and wraps ArduinoJson purely for I/O, not logic worth a
// native port; its schema is instead validated offline by
// tools/validate_pi_profile.py (plain Python, no native-build risk).
// This suite focuses on the actually novel/risky part: the control law.
//
// Run with: pio test -e native_pid

#include <math.h>
#include <stdio.h>
#include <unity.h>

#include "RatioCurrentController.h"

// ============================================================
// Reference implementation: main_current_pid.cpp's runControlTick(),
// transcribed with IDENTICAL operation order/grouping so IEEE-754 results
// are expected to match RatioCurrentController bit-for-bit when
// ratios=[1,1,1,1]. This is the regression-safety pin for Phase 1's
// "generalizing the balance policy must not change balance behavior."
// ============================================================
namespace ref {
const float KP = 2.2f, KI = 0.10f, KD = 0.15f;
const float OVERCURRENT_BACKOFF_PCT = 5.0f;
const float DUTY_MIN = 5.0f, DUTY_MAX = 100.0f;
const float I_MAX_A = 12.0f;
const float NOMINAL_TICK_MS = 2.0f;
const float MIN_SWITCH_MARGIN_A = 0.3f;
const float MIN_RAMP_PCT_PER_MS = 0.05f;

static inline float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

struct State {
  float integrator[4] = {50, 50, 50, 50};
  float duty_out[4] = {50, 50, 50, 50};
  float last_err[4] = {0, 0, 0, 0};
  int idx_min = 0;
};

void runControlTick(State &s, const float i_meas[4], float dt_ms) {
  float rate_scale = dt_ms / NOMINAL_TICK_MS;
  int true_idx_min = 0;
  for (int i = 1; i < 4; i++)
    if (i_meas[i] < i_meas[true_idx_min]) true_idx_min = i;
  if (i_meas[true_idx_min] < i_meas[s.idx_min] - MIN_SWITCH_MARGIN_A) s.idx_min = true_idx_min;
  float i_min = i_meas[s.idx_min];

  for (int i = 0; i < 4; i++) {
    float duty;
    if (i_meas[i] > I_MAX_A) {
      s.integrator[i] -= OVERCURRENT_BACKOFF_PCT;
      duty = clampf(s.integrator[i], DUTY_MIN, DUTY_MAX);
    } else if (i == s.idx_min) {
      duty = clampf(s.duty_out[i] + MIN_RAMP_PCT_PER_MS * dt_ms, DUTY_MIN, DUTY_MAX);
      s.integrator[i] = duty;
      s.last_err[i] = 0.0f;
    } else {
      float err = i_min - i_meas[i];
      float derivative = (rate_scale > 0.0f) ? (err - s.last_err[i]) / rate_scale : 0.0f;
      s.last_err[i] = err;
      float candidate = s.integrator[i] + KP * err + KD * derivative;
      duty = clampf(candidate, DUTY_MIN, DUTY_MAX);
      bool pushingIntoHighSat = (candidate > DUTY_MAX) && (err > 0.0f);
      bool pushingIntoLowSat = (candidate < DUTY_MIN) && (err < 0.0f);
      if (!pushingIntoHighSat && !pushingIntoLowSat) s.integrator[i] += KI * err * rate_scale;
    }
    s.duty_out[i] = duty;
  }
}
}  // namespace ref

// Deterministic synthetic i_meas generator: sinusoidal per-channel drift
// with occasional overcurrent spikes and crossovers, designed to exercise
// anchor switching (hysteresis), saturation, and overcurrent backoff --
// not just steady-state convergence.
void syntheticIMeas(int tick, float out[4]) {
  float t = (float)tick;
  out[0] = 2.0f + 1.5f * sinf(t * 0.02f);
  out[1] = 2.0f + 1.3f * sinf(t * 0.021f + 1.0f);
  out[2] = 2.0f + 1.4f * sinf(t * 0.019f + 2.0f);
  out[3] = 2.0f + 1.2f * sinf(t * 0.023f + 3.0f);
  if (tick % 137 == 0) out[2] += 11.0f;  // periodic overcurrent spike (> I_MAX_A=12)
  for (int i = 0; i < 4; i++)
    if (out[i] < 0.0f) out[i] = 0.0f;
}

RatioCurrentController::Config balanceConfig() {
  RatioCurrentController::Config cfg;
  cfg.ratios[0] = cfg.ratios[1] = cfg.ratios[2] = cfg.ratios[3] = 1.0f;
  cfg.sharedConstraint = true;
  cfg.kp = ref::KP;
  cfg.ki = ref::KI;
  cfg.kd = ref::KD;
  cfg.rampPctPerMs = ref::MIN_RAMP_PCT_PER_MS;
  cfg.minSwitchMarginA = ref::MIN_SWITCH_MARGIN_A;
  cfg.dutyMin = ref::DUTY_MIN;
  cfg.dutyMax = ref::DUTY_MAX;
  cfg.iMaxA = ref::I_MAX_A;
  cfg.overcurrentBackoffPct = ref::OVERCURRENT_BACKOFF_PCT;
  cfg.nominalTickMs = ref::NOMINAL_TICK_MS;
  cfg.magnitudeSettleTolA = 0.2f;  // unused in shared-constraint mode, set for determinism
  return cfg;
}

void test_balance_mode_matches_reference_steady_climb() {
  RatioCurrentController ctrl(balanceConfig());
  ref::State refState;
  float dutyOut[4];

  for (int tick = 0; tick < 2000; tick++) {
    float i_meas[4];
    syntheticIMeas(tick, i_meas);
    float dt_ms = (tick % 5 == 0) ? 5.0f : 2.0f;  // vary dt to exercise rate-normalization

    ctrl.computeTick(i_meas, dt_ms, dutyOut);
    ref::runControlTick(refState, i_meas, dt_ms);

    for (int i = 0; i < 4; i++) {
      char msg[64];
      snprintf(msg, sizeof(msg), "tick=%d channel=%d", tick, i);
      TEST_ASSERT_FLOAT_WITHIN_MESSAGE(1e-4f, refState.duty_out[i], dutyOut[i], msg);
    }
  }
}

void test_balance_mode_ratios_all_one_is_noop_generalization() {
  // Sanity check on the generalization itself, independent of the
  // reference impl: with ratios=[1,1,1,1], the normalized argmin/magnitude
  // must equal the raw argmin/magnitude every tick (the algebraic identity
  // the whole regression-safety argument rests on).
  RatioCurrentController::Config cfg = balanceConfig();
  RatioCurrentController ctrl(cfg);
  float dutyOut[4];
  float i_meas[4] = {1.5f, 3.0f, 0.5f, 2.0f};
  ctrl.computeTick(i_meas, 2.0f, dutyOut);
  // Channel 2 (index 2, current=0.5, the minimum) should be the one ramping
  // from its 50.0 start toward DUTY_MAX -- i.e. its duty should have
  // increased by exactly rampPctPerMs*dt_ms from the 50.0 reset default.
  TEST_ASSERT_FLOAT_WITHIN(1e-5f, 50.0f + cfg.rampPctPerMs * 2.0f, dutyOut[2]);
}

void test_independent_mode_maintains_ratio_once_converged() {
  RatioCurrentController::Config cfg;
  cfg.ratios[0] = 1.0f; cfg.ratios[1] = 0.5f; cfg.ratios[2] = 0.3f; cfg.ratios[3] = 0.8f;
  cfg.sharedConstraint = false;
  cfg.kp = 2.2f; cfg.ki = 0.5f; cfg.kd = 0.0f;  // higher ki for fast test convergence
  cfg.rampPctPerMs = 0.05f;
  cfg.minSwitchMarginA = 0.3f;
  cfg.dutyMin = 5.0f; cfg.dutyMax = 100.0f;
  cfg.iMaxA = 12.0f;
  cfg.overcurrentBackoffPct = 5.0f;
  cfg.nominalTickMs = 2.0f;
  // Tight settle tolerance (vs. the 0.2A untuned-placeholder default used
  // elsewhere/in the example profile) -- this test checks the MECHANISM
  // achieves good ratio precision when configured for it; a looser
  // tolerance trades convergence precision for ramp speed, a real but
  // separate tuning question.
  cfg.magnitudeSettleTolA = 0.02f;

  RatioCurrentController ctrl(cfg);
  float dutyOut[4];
  // Symmetric plant model: current_i = duty_i * gain (same gain all
  // channels), simulated in closed loop -- if the controller correctly
  // tracks magnitude*ratios[i] per channel, converged currents should be
  // in the same 1:0.5:0.3:0.8 proportion as the ratios.
  const float gain = 0.03f;  // A per duty%
  float i_meas[4] = {0, 0, 0, 0};

  for (int tick = 0; tick < 3000; tick++) {
    ctrl.computeTick(i_meas, 2.0f, dutyOut);
    for (int i = 0; i < 4; i++) i_meas[i] = dutyOut[i] * gain;
  }

  // Assert converged currents preserve the ratio (normalize by channel 0).
  TEST_ASSERT_GREATER_THAN(0.01f, i_meas[0]);
  float ratioB = i_meas[1] / i_meas[0], ratioC = i_meas[2] / i_meas[0], ratioD = i_meas[3] / i_meas[0];
  TEST_ASSERT_FLOAT_WITHIN(0.02f, 0.5f, ratioB);
  TEST_ASSERT_FLOAT_WITHIN(0.02f, 0.3f, ratioC);
  TEST_ASSERT_FLOAT_WITHIN(0.02f, 0.8f, ratioD);
}

void test_independent_mode_backs_off_globally_on_overcurrent() {
  RatioCurrentController::Config cfg;
  cfg.ratios[0] = 1.0f; cfg.ratios[1] = 1.0f; cfg.ratios[2] = 1.0f; cfg.ratios[3] = 1.0f;
  cfg.sharedConstraint = false;
  cfg.kp = 2.2f; cfg.ki = 0.10f; cfg.kd = 0.0f;
  cfg.rampPctPerMs = 1.0f;  // fast ramp so magnitude climbs quickly in the test
  cfg.minSwitchMarginA = 0.3f;
  cfg.dutyMin = 5.0f; cfg.dutyMax = 100.0f;
  cfg.iMaxA = 5.0f;  // low trip so the test can reach it quickly
  cfg.overcurrentBackoffPct = 2.0f;
  cfg.nominalTickMs = 2.0f;
  cfg.magnitudeSettleTolA = 0.2f;

  RatioCurrentController ctrl(cfg);
  float dutyOut[4];

  // Drive one channel (index 1) artificially over iMaxA every tick,
  // regardless of duty (simulates a hardware fault/short) -- the OTHER
  // three channels' measured current tracks duty normally.
  const float gain = 0.05f;
  float i_meas[4] = {0, 0, 0, 0};
  float lastDuty0 = -1.0f;
  bool sawBackoff = false;

  for (int tick = 0; tick < 500; tick++) {
    ctrl.computeTick(i_meas, 2.0f, dutyOut);
    i_meas[0] = dutyOut[0] * gain;
    i_meas[1] = 999.0f;  // permanently overcurrent
    i_meas[2] = dutyOut[2] * gain;
    i_meas[3] = dutyOut[3] * gain;
    if (lastDuty0 >= 0.0f && dutyOut[0] < lastDuty0 - 1e-6f) sawBackoff = true;
    lastDuty0 = dutyOut[0];
  }
  // Channel 0 (a healthy, non-overcurrent channel) must have backed off at
  // some point due to the GLOBAL magnitude governor reacting to channel
  // 1's overcurrent -- proving the backoff isn't merely local to the
  // faulted channel (which would silently distort the ratio).
  TEST_ASSERT_TRUE(sawBackoff);
}

void test_anti_windup_prevents_runaway_on_saturated_channel() {
  // In shared-constraint mode, the anchor is always the CURRENT minimum,
  // so every non-anchor channel's error (target=i_min minus its own
  // current) is <=0 once idx_min correctly tracks the true minimum --
  // meaning a non-anchor channel can only ever saturate LOW (duty pinned
  // at DUTY_MIN), never sustain a HIGH saturation (that's structurally
  // impossible in this policy, not merely untested). This is the actual,
  // historically-significant anti-windup lesson this project documents
  // (see RatioCurrentController.cpp's pidStep() comment and the earlier
  // hardware deadlock it references): exercise the LOW-saturation-then-
  // recover path.
  RatioCurrentController::Config cfg = balanceConfig();
  RatioCurrentController ctrl(cfg);
  float dutyOut[4];

  // Channel 0 is the anchor (lowest, target=0.1 for everyone else).
  // Channels 1-3 sit far ABOVE that target for many ticks -- duty (and,
  // once anti-windup's low-saturation freeze engages, the integrator)
  // should pin at DUTY_MIN.
  float i_meas[4] = {0.1f, 5.0f, 5.0f, 5.0f};
  for (int tick = 0; tick < 200; tick++) ctrl.computeTick(i_meas, 2.0f, dutyOut);
  TEST_ASSERT_FLOAT_WITHIN(1e-3f, cfg.dutyMin, dutyOut[3]);  // saturated low

  // Now channel 3's current drops back near the (still-low) anchor
  // target -- duty must recover UP within a handful of ticks, not stay
  // pinned at DUTY_MIN due to a runaway-low (over-wound-down) integrator.
  i_meas[3] = 0.05f;
  for (int tick = 0; tick < 20; tick++) ctrl.computeTick(i_meas, 2.0f, dutyOut);
  TEST_ASSERT_GREATER_THAN(cfg.dutyMin + 1.0f, dutyOut[3]);
}

void test_reset_clears_state() {
  RatioCurrentController::Config cfg = balanceConfig();
  RatioCurrentController ctrl(cfg);
  float dutyOut[4];
  float i_meas[4] = {0.1f, 5.0f, 5.0f, 5.0f};
  for (int tick = 0; tick < 200; tick++) ctrl.computeTick(i_meas, 2.0f, dutyOut);

  ctrl.reset();
  ctrl.computeTick(i_meas, 2.0f, dutyOut);
  // Immediately after reset, this is tick-1 behavior again: channel 0 (the
  // anchor) should be at 50.0 + one ramp step, matching a fresh run.
  TEST_ASSERT_FLOAT_WITHIN(1e-4f, 50.0f + cfg.rampPctPerMs * 2.0f, dutyOut[0]);
}

void test_with_reference_tracks_externally_supplied_reference() {
  RatioCurrentController::Config cfg;
  cfg.ratios[0] = 1.0f; cfg.ratios[1] = 0.5f; cfg.ratios[2] = 0.3f; cfg.ratios[3] = 0.8f;
  cfg.sharedConstraint = false;
  cfg.kp = 2.2f; cfg.ki = 0.5f; cfg.kd = 0.0f;  // higher ki for fast test convergence
  cfg.rampPctPerMs = 0.05f;  // irrelevant here -- computeTickWithReference never
                             // touches the internal magnitude ramp this drives
  cfg.minSwitchMarginA = 0.3f;
  cfg.dutyMin = 5.0f; cfg.dutyMax = 100.0f;
  cfg.iMaxA = 12.0f;
  cfg.overcurrentBackoffPct = 5.0f;
  cfg.nominalTickMs = 2.0f;
  cfg.magnitudeSettleTolA = 0.02f;

  RatioCurrentController ctrl(cfg);
  float dutyOut[4];
  const float gain = 0.03f;  // A per duty%, same symmetric plant model as the
                             // computeTick() ratio-tracking test above
  float i_meas[4] = {0, 0, 0, 0};
  // Reference driven directly by the "caller's schedule" every tick, exactly
  // as main_experiment.cpp reads it back from the JSON-commanded carrier
  // duty -- fixed here for a converged-ratio check. Chosen small enough that
  // even the highest-ratio channel (A, 1.0) doesn't saturate duty at gain
  // 0.03 A/duty% (target 2.0A -> ~67% duty, well under dutyMax=100).
  const float reference = 2.0f;

  for (int tick = 0; tick < 3000; tick++) {
    ctrl.computeTickWithReference(i_meas, 2.0f, reference, dutyOut);
    for (int i = 0; i < 4; i++) i_meas[i] = dutyOut[i] * gain;
  }

  TEST_ASSERT_GREATER_THAN(0.01f, i_meas[0]);
  float ratioB = i_meas[1] / i_meas[0], ratioC = i_meas[2] / i_meas[0], ratioD = i_meas[3] / i_meas[0];
  TEST_ASSERT_FLOAT_WITHIN(0.02f, 0.5f, ratioB);
  TEST_ASSERT_FLOAT_WITHIN(0.02f, 0.3f, ratioC);
  TEST_ASSERT_FLOAT_WITHIN(0.02f, 0.8f, ratioD);
  // Confirms the internal magnitude self-governor was never touched --
  // computeTickWithReference() is a fully independent entrypoint from
  // computeTick()/tickIndependent().
  TEST_ASSERT_EQUAL_FLOAT(0.0f, ctrl.magnitude());
}

int main() {
  UNITY_BEGIN();
  RUN_TEST(test_balance_mode_matches_reference_steady_climb);
  RUN_TEST(test_balance_mode_ratios_all_one_is_noop_generalization);
  RUN_TEST(test_independent_mode_maintains_ratio_once_converged);
  RUN_TEST(test_independent_mode_backs_off_globally_on_overcurrent);
  RUN_TEST(test_anti_windup_prevents_runaway_on_saturated_channel);
  RUN_TEST(test_reset_clears_state);
  RUN_TEST(test_with_reference_tracks_externally_supplied_reference);
  return UNITY_END();
}
