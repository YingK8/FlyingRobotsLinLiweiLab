// Onboard current-balance PI controller: fully autonomous, bounded run.
//
// Policy (global, no board grouping): each tick, find the channel with the
// LOWEST current -- its target becomes the current MAX (push it up); every
// other channel's target becomes the current MIN (pull it down). No coupling
// model, no feedforward, no interactive serial commands -- just measurement,
// PI, duty.
//
// Bounded/autonomous: boots -> arms (3s, coils off, self-calibrate ADC zero)
// -> ramps drive frequency 1->210Hz over 30s while the PI runs -> holds at
// 210Hz for 20s -> ramps duty down -> latches off forever. Serial telemetry
// (one line per ~500ms) is the only serial I/O -- capture it with
// tools/trigger_reset_log.py for post-run analysis (tools/plot_pid_log.py).
//
// Validation bar: max(i_meas) - min(i_meas) < 0.1A, sustained through HOLD.
// KP/KI/OVERCURRENT_BACKOFF_PCT are untuned placeholders -- tune on hardware.

#include <Arduino.h>
#include "PhaseController.h"
#include "PhaseSequencer.h"
#include "constants.h"

static inline float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// ============================== DRIVE ==============================
// rotation is counter-clockwise: A -> C -> B -> D (CCW diagnostic test --
// was {270,90,180,0} CW; flipped to check whether the persistent D-weakest
// pattern is coupling-driven (should flip/change with direction, since
// coupled real-power ~ sin(dphi) is odd) or a static hardware property
// (should stay the same regardless of direction))
const float INITIAL_PHASES[NUM_CHANNELS] = {90.0, 270.0, 180.0, 0.0};
const float INITIAL_DUTY_CYCLES[NUM_CHANNELS] = {50.0, 50.0, 50.0, 50.0};
const float INITIAL_CARRIER_DUTY_CYCLES[NUM_CHANNELS] = {0, 0, 0, 0};
const float START_DUTY = 50.0f;  // all channels start equal; PI does the rest

const float start_freq = 1.0f;
const float end_freq = 210.0f;
const unsigned long ramp_duration_ms = 20000;

// ============================== CURRENT SENSE ==============================
const int   ADC_PINS[NUM_CHANNELS] = {36, 39, 34, 35};              // A,B,C,D
const float SENS[NUM_CHANNELS]     = {15.26, 15.28, 15.57, 15.34};  // A per V

const float TAU_FILTER_MS = 50.0f;  // ~10 commutation periods at 210Hz --
                                     // tried lowering to 20ms to cut lag, but
                                     // that REINTRODUCED argmin flapping
                                     // during the ramp (noise + fast-rising
                                     // real signal combine to still cross
                                     // MIN_SWITCH_MARGIN_A) and made resonance
                                     // spread slightly worse -- 50ms confirmed
                                     // as the better tradeoff, reverted

float cs_mv[NUM_CHANNELS] = {0, 0, 0, 0};
float i_meas[NUM_CHANNELS] = {0, 0, 0, 0};       // amps
float adc_zero_mv[NUM_CHANNELS] = {0, 0, 0, 0};  // self-calibrated during ARMING

void seedAdcFilter() {
  for (int i = 0; i < NUM_CHANNELS; i++) {
    analogSetPinAttenuation((gpio_num_t)ADC_PINS[i], ADC_11db);  // ~0..3.1V
    analogReadMilliVolts(ADC_PINS[i]);  // throwaway: let the ADC mux/S&H settle after switching pins
    cs_mv[i] = analogReadMilliVolts(ADC_PINS[i]);
    adc_zero_mv[i] = cs_mv[i];
  }
}

void updateAdcFilter(float dt_ms) {
  float alpha = 1.0f - expf(-dt_ms / TAU_FILTER_MS);
  for (int i = 0; i < NUM_CHANNELS; i++) {
    // Throwaway read: the ESP32 ADC needs a moment to settle after the mux
    // switches to a new pin. With the control loop now unthrottled (calling
    // this every loop() iteration, back-to-back across 4 pins), skipping
    // this caused two channels to read a stuck ~0 regardless of real current
    // -- not a rate limit, a real hardware settling requirement.
    analogReadMilliVolts(ADC_PINS[i]);
    cs_mv[i] += alpha * (analogReadMilliVolts(ADC_PINS[i]) - cs_mv[i]);
  }
}

void updateMeasuredCurrents() {
  for (int i = 0; i < NUM_CHANNELS; i++)
    i_meas[i] = SENS[i] * (cs_mv[i] - adc_zero_mv[i]) / 1000.0f;
}

// ============================== PI CONTROLLER ==============================
// Placeholders -- tune on hardware (flash -> log -> plot -> adjust -> reflash).
const float KP = 2.2f;                    // duty % per A of error
const float KI = 0.10f;                   // duty % per A of error, per tick
const float KD = 0.15f;
const float OVERCURRENT_BACKOFF_PCT = 5.0f;
const float DUTY_MIN = 5.0f;
const float DUTY_MAX = 100.0f;
const float I_MAX_A  = 12.0f;             // hard per-channel safety cap

// Runs every loop() iteration now (no fixed tick gate -- loop() is fast and
// there's no reason to throttle it). KP/KI/KD were tuned assuming a fixed
// 2ms step, so KI/derivative are scaled by (actual dt / NOMINAL_TICK_MS)
// to keep the gains' real-world meaning rate-independent instead of
// silently integrating/differentiating faster just because loop() sped up.
const float NOMINAL_TICK_MS = 2.0f;

// Hysteresis on WHICH channel is currently forced to 100%: without this, as
// the group converges (the actual goal), channels' readings get close
// together and measurement noise causes the identity of "the minimum" to
// flap between channels tick to tick -- every flap resets a channel to a
// fresh 100%-duty episode, undoing nearly-converged progress right when it's
// closest to succeeding (confirmed in testing: convergence to ~0.32A spread
// during RAMP_UP was wiped out by a flap right as HOLD began). Only switch
// which channel is latched as the forced-100% one if a DIFFERENT channel
// reads at least this much lower than the currently latched one.
const float MIN_SWITCH_MARGIN_A = 0.3f;

float integrator[NUM_CHANNELS] = {START_DUTY, START_DUTY, START_DUTY, START_DUTY};
float duty_out[NUM_CHANNELS]   = {START_DUTY, START_DUTY, START_DUTY, START_DUTY};
float last_err[NUM_CHANNELS] = {0, 0, 0, 0};
int idx_min = 0;  // latched -- persists across ticks, see MIN_SWITCH_MARGIN_A

PhaseController *controller;
PhaseSequencer *seq;

// Global min/max policy (literal, no board grouping) + plain PI, every channel,
// every tick. See the plan's "control policy" section for the full derivation.
void runControlTick(float dt_ms) {
  float rate_scale = dt_ms / NOMINAL_TICK_MS;
  int true_idx_min = 0;
  for (int i = 1; i < NUM_CHANNELS; i++)
    if (i_meas[i] < i_meas[true_idx_min]) true_idx_min = i;
  if (i_meas[true_idx_min] < i_meas[idx_min] - MIN_SWITCH_MARGIN_A)
    idx_min = true_idx_min;
  float i_min = i_meas[idx_min];  // the anchor level every other channel targets

  for (int i = 0; i < NUM_CHANNELS; i++) {
    float duty;
    if (i_meas[i] > I_MAX_A) {
      integrator[i] -= OVERCURRENT_BACKOFF_PCT;
      duty = clampf(integrator[i], DUTY_MIN, DUTY_MAX);
    } else if (i == idx_min) {
      // Literal spec: the minimum-current channel is always pulled straight
      // to 100% duty -- no PID tracking, no gradual approach. This is what
      // anchors the group to a real driven current level; without it, every
      // channel can drift toward zero together (min/max only equalizes
      // channels RELATIVE to each other, it has no absolute setpoint).
      duty = 100.0f;
      integrator[i] = duty;  // continuity for when this channel later falls
                              // back under normal PI control (target=i_min)
      last_err[i] = 0.0f;    // i_meas[idx_min] == i_min while forced -- keep
                              // the derivative term fresh for when it exits
    } else {
      // Anti-windup: freeze the integrator ONLY when the error is pushing
      // FURTHER into saturation (candidate beyond the clamp AND err has the
      // same sign as that direction). Freezing merely because the raw
      // candidate exceeds the clamp -- regardless of err's sign -- traps any
      // channel whose integrator started high (e.g. just finished being the
      // forced-100% min channel) permanently AT the clamp: a negative err
      // trying to pull it back down would never actually update the
      // integrator, since candidate stays above the clamp for many ticks in
      // a row while KP*err alone is too small to cross it in one step.
      float err = i_min - i_meas[i];
      // Rate-independent derivative: normalize the raw per-call delta back
      // to "per NOMINAL_TICK_MS" terms so KD's meaning doesn't change when
      // dt_ms varies (this now runs every loop() iteration, not a fixed tick).
      float derivative = (rate_scale > 0.0f) ? (err - last_err[i]) / rate_scale : 0.0f;
      last_err[i] = err;
      float candidate = integrator[i] + KP * err + KD * derivative;
      duty = clampf(candidate, DUTY_MIN, DUTY_MAX);
      bool pushingIntoHighSat = (candidate > DUTY_MAX) && (err > 0.0f);
      bool pushingIntoLowSat  = (candidate < DUTY_MIN) && (err < 0.0f);
      if (!pushingIntoHighSat && !pushingIntoLowSat) integrator[i] += KI * err * rate_scale;
    }
    controller->setCarrierDutyCycle(i, duty);
    duty_out[i] = duty;
  }
}

// ============================== STATE MACHINE ==============================
enum Phase { ARMING, RAMP_UP, HOLD, ENDING, STOPPED };
Phase phase = ARMING;
unsigned long phase_start = 0;

const unsigned long ARM_MS = 3000;
const unsigned long HOLD_MS = 5000;
const unsigned long ramp_tick_ms = 20;
const float end_step_pct = 2.0f;
unsigned long last_ramp_ms = 0;

void allCoilsOff() {
  for (int i = 0; i < NUM_CHANNELS; i++) controller->setCarrierDutyCycle(i, 0.0f);
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  // SAFE STARTUP: hold every gate-driver input LOW before constructing
  // anything, so the coils cannot energize during boot / LEDC attach if the
  // supply is already live.
  const gpio_num_t ALL_PINS[] = {A_PWM_PIN, B_PWM_PIN, C_PWM_PIN, D_PWM_PIN,
                                 A_CARRIER_PIN, B_CARRIER_PIN, C_CARRIER_PIN,
                                 D_CARRIER_PIN};
  for (gpio_num_t p : ALL_PINS) { pinMode(p, OUTPUT); digitalWrite(p, LOW); }
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  analogReadResolution(12);
  seedAdcFilter();

  controller = new PhaseController(PWM_PINS, INITIAL_PHASES, INITIAL_DUTY_CYCLES, NUM_CHANNELS);
  seq = new PhaseSequencer(controller);

  controller->begin(start_freq);
  controller->initCarrierPWM(CARRIER_PINS, PWM_FREQ, INITIAL_CARRIER_DUTY_CYCLES);
  allCoilsOff();

  seq->addRampTask(start_freq, end_freq, ramp_duration_ms, TaskType::PWM_FREQ, TaskMode::EASE);
  seq->compile(25, 1.0f, INITIAL_DUTY_CYCLES, INITIAL_PHASES);

  phase = ARMING;
  phase_start = millis();
  Serial.println("current_pid: arming 3s, then autonomous 1->210Hz ramp + PI, bounded run");
}

void loop() {
  unsigned long now = millis();  // state-machine timing (seconds-scale, ms is plenty)

  // ADC sampling is rate-limited -- NOT a software throttle, a real ESP32
  // ADC hardware constraint. Tried removing this entirely (call every
  // loop() iteration): two channels first read a stuck 0 (mux/S&H not
  // settled), then after adding a throwaway read, TWO channels read wildly
  // non-physical values (-12A range) -- the hardware just can't sustain
  // reliable conversions at loop()'s unthrottled rate. So ADC sampling is
  // paced at ADC_SAMPLE_MS; the control COMPUTATION still runs every
  // loop() iteration (unthrottled) against whatever the latest valid
  // sample is -- these are two different rates now, tracked separately.
  const unsigned long ADC_SAMPLE_MS = 1;
  static unsigned long last_adc_us = micros();
  unsigned long now_us = micros();
  float dt_since_adc_ms = (float)(now_us - last_adc_us) / 1000.0f;
  if (dt_since_adc_ms >= (float)ADC_SAMPLE_MS) {
    updateAdcFilter(dt_since_adc_ms);
    updateMeasuredCurrents();
    last_adc_us = now_us;
  }

  // dt for the control computation itself (KI/KD rate-scaling) -- this is
  // the real per-loop-iteration interval, separate from ADC_SAMPLE_MS above.
  static unsigned long last_control_us = now_us;
  float control_dt_ms = (float)(now_us - last_control_us) / 1000.0f;
  if (control_dt_ms <= 0.0f) control_dt_ms = 0.001f;  // guard div-by-zero only
  last_control_us = now_us;

  controller->run();
  if (phase == RAMP_UP || phase == HOLD) seq->run();

  switch (phase) {
    case ARMING:
      allCoilsOff();
      digitalWrite(LED_PIN, (now / 150) & 1);  // fast blink = arming
      // Coils confirmed off here -- self-calibrate the zero-current offset.
      for (int i = 0; i < NUM_CHANNELS; i++) adc_zero_mv[i] = cs_mv[i];
      if (now - phase_start >= ARM_MS) {
        for (int i = 0; i < NUM_CHANNELS; i++) {
          integrator[i] = START_DUTY;
          controller->setCarrierDutyCycle(i, START_DUTY);
        }
        idx_min = 0;
        seq->start();
        phase = RAMP_UP;
        phase_start = now;
        Serial.println("ARMED -> ramping");
      }
      break;

    case RAMP_UP: {
      runControlTick(control_dt_ms);
      if (controller->getFrequency() >= end_freq - 0.5f) {
        phase = HOLD;
        phase_start = now;
      }
      break;
    }

    case HOLD: {
      runControlTick(control_dt_ms);
      if (now - phase_start >= HOLD_MS) {
        phase = ENDING;
        last_ramp_ms = now;
      }
      break;
    }

    case ENDING:
      if (now - last_ramp_ms >= ramp_tick_ms) {
        last_ramp_ms = now;
        if (controller->rampDownStep(end_step_pct)) {
          digitalWrite(LED_PIN, LOW);
          phase = STOPPED;
          Serial.println("coils off");
        }
      }
      break;

    case STOPPED:
      allCoilsOff();
      digitalWrite(LED_PIN, (now / 1000) & 1);  // slow heartbeat = latched off
      break;
  }

  static unsigned long last_dbg_ms = 0;
  if (now - last_dbg_ms >= 500) {
    last_dbg_ms = now;
    float i_min = i_meas[0], i_max = i_meas[0];
    for (int i = 1; i < NUM_CHANNELS; i++) {
      if (i_meas[i] < i_min) i_min = i_meas[i];
      if (i_meas[i] > i_max) i_max = i_meas[i];
    }
    Serial.printf("t=%lu phase=%d freq=%.1f | I[A]: A=%.2f B=%.2f C=%.2f D=%.2f | "
                  "duty[%%]: A=%.1f B=%.1f C=%.1f D=%.1f | spread=%.3f\n",
                  now, (int)phase, controller->getFrequency(),
                  i_meas[0], i_meas[1], i_meas[2], i_meas[3],
                  duty_out[0], duty_out[1], duty_out[2], duty_out[3],
                  i_max - i_min);
  }
}
