// Onboard current-balance PI controller. Policy (global, no board grouping):
// each tick, the LOWEST-current channel's target becomes the current MAX
// (push up); every other channel's target becomes the current MIN (pull
// down) via PI(+D). Bounded/autonomous per run: arms (3s, self-calibrate ADC
// zero) -> ramps drive frequency 1->210Hz over ramp_duration_ms -> holds at
// 210Hz for HOLD_MS -> ramps duty down -> latches off. Rotation direction
// and gains are runtime-adjustable (dir=/kp=/ki=/kd=/gains? over SerialComm)
// so tuning trials don't need a reflash; capture telemetry with
// tools/trigger_reset_log.py or tools/pid_autotune.py, plot with
// tools/plot_pid_log.py. Validation bar: max(i_meas)-min(i_meas) < 0.1A,
// sustained through HOLD.

#include <Arduino.h>
#include "PWMController.h"
#include "PWMSequencer.h"
#include "SerialComm.h"
#include "constants.h"
#include "CurrentSense.h"
#include "safety_startup.h"
#include "telemetry.h"

static inline float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// ============================== DRIVE ==============================
// Rotation direction is runtime-selectable (see dir=cw/dir=ccw in
// dispatchCommand). Default stays CCW (A -> C -> B -> D) -- this file's
// existing convention, used historically to test whether the persistent
// D-weakest pattern is coupling-driven (should flip/change with direction,
// since coupled real-power ~ sin(dphi) is odd) or a static hardware property.
const float PHASES_CW[NUM_CHANNELS] = {270.0, 90.0, 180.0, 0.0};
const float PHASES_CCW[NUM_CHANNELS] = {90.0, 270.0, 180.0, 0.0};
const float INITIAL_DUTY_CYCLES[NUM_CHANNELS] = {50.0, 50.0, 50.0, 50.0};
const float INITIAL_CARRIER_DUTY_CYCLES[NUM_CHANNELS] = {0, 0, 0, 0};
const float START_DUTY = 50.0f;  // all channels start equal; PI does the rest

const float start_freq = 1.0f;
const float end_freq = 210.0f;
const unsigned long ramp_duration_ms = 20000;

// ============================== CURRENT SENSE ==============================
// VNH5019 CS gain, A per V -- per-board calibration.
const float SENS[NUM_CHANNELS] = {15.26, 15.28, 15.57, 15.34};
CurrentSense currentSense(ADC_PINS, SENS, NUM_CHANNELS); // TAU_FILTER_MS=50 default -- see CurrentSense.h

// ============================== PI CONTROLLER ==============================
// Runtime-tunable via kp=/ki=/kd= (see dispatchCommand). Converged values
// (see project memory): KP=2.2, KI=0.10, KD=0.15.
float KP = 2.2f;                    // duty % per A of error
float KI = 0.10f;                   // duty % per A of error, per tick
float KD = 0.15f;
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

// Rate limit on how fast the latched-minimum channel ramps toward 100% duty.
// Snapping straight to 100% (previous behavior) can overshoot the group --
// see lesson: forcing a channel to 100% can make IT become the new leader
// before the argmin re-evaluation above has a chance to reassign idx_min to
// a genuinely different weakest channel. Ramping instead gives that
// reassignment time to happen mid-ramp.
float MIN_RAMP_PCT_PER_MS = 0.05f;  // duty %/ms at NOMINAL_TICK_MS rate -- untuned
                                     // placeholder, runtime-tunable via ramp=

float integrator[NUM_CHANNELS] = {START_DUTY, START_DUTY, START_DUTY, START_DUTY};
float duty_out[NUM_CHANNELS]   = {START_DUTY, START_DUTY, START_DUTY, START_DUTY};
float last_err[NUM_CHANNELS] = {0, 0, 0, 0};
int idx_min = 0;  // latched -- persists across ticks, see MIN_SWITCH_MARGIN_A

PWMController *controller;
PWMSequencer *seq;
SerialComm comm;
bool directionIsCcw = true; // default direction, matches this file's history

// Global min/max policy (literal, no board grouping) + plain PI, every channel,
// every tick. See the plan's "control policy" section for the full derivation.
void runControlTick(float dt_ms) {
  float rate_scale = dt_ms / NOMINAL_TICK_MS;
  float *i_meas = currentSense.i_meas;
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
      // Anchor the group by driving the minimum-current channel toward
      // 100% duty -- this is what maximizes achievable current, since
      // pushing the bottleneck channel as hard as possible is the only way
      // to raise the floor everyone else tracks. Ramped, not snapped: a
      // bounded rate gives the argmin re-evaluation above time to reassign
      // idx_min to a genuinely different, still-weaker channel before this
      // one overshoots the group.
      duty = clampf(duty_out[i] + MIN_RAMP_PCT_PER_MS * dt_ms, DUTY_MIN, DUTY_MAX);
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

// (Re)build controller/seq for the given direction. Cheap on ESP32; called
// once from setup() and again from dispatchCommand() on dir=cw/dir=ccw
// (only while ARMING/STOPPED -- never while a run is active).
void reinitController(bool ccw) {
  const float *phases = ccw ? PHASES_CCW : PHASES_CW;
  directionIsCcw = ccw;

  if (controller) delete controller;
  controller = new PWMController(PWM_PINS, phases, INITIAL_DUTY_CYCLES, NUM_CHANNELS);
  // Explicit non-zero starting frequency -- begin(0.0f) divides by zero
  // inside setGlobalFrequency() and permanently corrupts commutation timing.
  controller->begin(start_freq);
  controller->initCarrierPWM(CARRIER_PINS, PWM_FREQ, INITIAL_CARRIER_DUTY_CYCLES);
  allCoilsOff();

  if (seq) delete seq;
  seq = new PWMSequencer(controller);
  seq->addRampTask(start_freq, end_freq, ramp_duration_ms, TaskType::PWM_FREQ, TaskMode::EASE);
  seq->compile(25, 1.0f, INITIAL_DUTY_CYCLES, phases);
}

void printGains() {
  Serial.printf("KP=%.3f KI=%.3f KD=%.3f RAMP=%.4f\n", KP, KI, KD, MIN_RAMP_PCT_PER_MS);
}

void dispatchCommand(const String &raw) {
  String cmd = raw;
  cmd.trim();
  cmd.toLowerCase();

  bool running = (phase == RAMP_UP || phase == HOLD);

  if (cmd == "gains?" || cmd == "gains") {
    printGains();
  } else if (cmd.startsWith("dir=")) {
    if (phase != ARMING && phase != STOPPED) {
      Serial.println("ignored: running (stop first)");
      return;
    }
    String v = cmd.substring(4);
    if (v == "cw") {
      reinitController(false);
      Serial.println("direction=CW");
    } else if (v == "ccw") {
      reinitController(true);
      Serial.println("direction=CCW");
    } else {
      Serial.println("unknown dir (use dir=cw or dir=ccw)");
    }
  } else if (cmd.startsWith("kp=")) {
    if (running) { Serial.println("ignored: running (stop first)"); return; }
    KP = cmd.substring(3).toFloat();
    printGains();
  } else if (cmd.startsWith("ki=")) {
    if (running) { Serial.println("ignored: running (stop first)"); return; }
    KI = cmd.substring(3).toFloat();
    printGains();
  } else if (cmd.startsWith("kd=")) {
    if (running) { Serial.println("ignored: running (stop first)"); return; }
    KD = cmd.substring(3).toFloat();
    printGains();
  } else if (cmd.startsWith("ramp=")) {
    if (running) { Serial.println("ignored: running (stop first)"); return; }
    MIN_RAMP_PCT_PER_MS = cmd.substring(5).toFloat();
    printGains();
  } else {
    Serial.printf("unknown cmd '%s' (dir=cw/ccw  kp=/ki=/kd=/ramp=<val>  gains?)\n",
                  cmd.c_str());
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  // SAFE STARTUP: hold every gate-driver input LOW before constructing
  // anything, so the coils cannot energize during boot / LEDC attach if the
  // supply is already live.
  forceAllGatesLow();
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  currentSense.seed();

  controller = nullptr;
  seq = nullptr;
  reinitController(directionIsCcw); // default direction

  phase = ARMING;
  phase_start = millis();
  Serial.println("current_pid: arming 3s, then autonomous 1->210Hz ramp + PI, bounded run");
  Serial.println("commands: dir=cw|ccw  kp=/ki=/kd=/ramp=<val>  gains?");
}

void loop() {
  unsigned long now = millis();  // state-machine timing (seconds-scale, ms is plenty)

  String line = comm.step();
  if (line.length()) dispatchCommand(line);

  // ADC sampling is rate-limited -- NOT a software throttle, a real ESP32
  // ADC hardware constraint (see CurrentSense.cpp). Control computation
  // still runs every loop() iteration (unthrottled) against whatever the
  // latest valid sample is -- these are two different rates, tracked
  // separately.
  const unsigned long ADC_SAMPLE_MS = 1;
  static unsigned long last_adc_us = micros();
  unsigned long now_us = micros();
  float dt_since_adc_ms = (float)(now_us - last_adc_us) / 1000.0f;
  if (dt_since_adc_ms >= (float)ADC_SAMPLE_MS) {
    currentSense.update(dt_since_adc_ms);
    last_adc_us = now_us;
  }

  // dt for the control computation itself (KI/KD rate-scaling) -- this is
  // the real per-loop-iteration interval, separate from ADC_SAMPLE_MS above.
  static unsigned long last_control_us = now_us;
  float control_dt_ms = (float)(now_us - last_control_us) / 1000.0f;
  if (control_dt_ms <= 0.0f) control_dt_ms = 0.001f;  // guard div-by-zero only
  last_control_us = now_us;

  controller->step();
  if (phase == RAMP_UP || phase == HOLD) seq->step();

  switch (phase) {
    case ARMING:
      allCoilsOff();
      digitalWrite(LED_PIN, (now / 150) & 1);  // fast blink = arming
      currentSense.recalibrateZero();  // coils confirmed off here
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
    float *i_meas = currentSense.i_meas;
    float i_min = i_meas[0], i_max = i_meas[0];
    for (int i = 1; i < NUM_CHANNELS; i++) {
      if (i_meas[i] < i_min) i_min = i_meas[i];
      if (i_meas[i] > i_max) i_max = i_meas[i];
    }
    Serial.printf("t=%lu phase=%d freq=%.1f | ", now, (int)phase, controller->getFrequency());
    printCurrentAndDuty(i_meas, duty_out);
    Serial.printf(" | spread=%.3f dir=%d kp=%.2f ki=%.2f kd=%.2f ramp=%.4f\n",
                  i_max - i_min, directionIsCcw ? 1 : 0, KP, KI, KD, MIN_RAMP_PCT_PER_MS);
  }
}
