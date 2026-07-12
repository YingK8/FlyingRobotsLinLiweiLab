// Onboard current-balance controller using full-state LQR feedback instead
// of the per-channel heuristic PI in main_current_pid.cpp. Policy: every
// channel is driven toward a SHARED target current r (not the group's
// floating minimum) via u = u_ff(r) - K(x - x_ref), where K (lqr_gains.h) is
// designed offline from the coupled 4-channel plant model (plant_model.h,
// tools/build_state_space_model.py + tools/design_lqr_gains.py) so that
// cross-channel magnetic coupling is corrected FOR, not just reacted to.
// r ramps up over the run (like the old MIN_RAMP anchor, now driving a
// shared target) and freezes/backs off on saturation or overcurrent.
// Bounded/autonomous per run, same state machine and safety envelope as
// main_current_pid.cpp: arms (3s, self-calibrate ADC zero) -> ramps drive
// frequency 1->210Hz over ramp_duration_ms -> holds at 210Hz for HOLD_MS ->
// ramps duty down -> latches off. See the state-space plan for the full
// model derivation, system-ID prerequisites, and why this differs from PID.
//
// IMPORTANT: plant_model.h and lqr_gains.h are PLACEHOLDERS until the
// system-ID pipeline (tools/fit_rlc_model.py, tools/fit_mutual_inductance.py)
// is actually run on hardware -- see those files' comments before trusting
// this controller's behavior beyond "compiles and doesn't do anything
// aggressive" (LQR_K starts all-zero: pure feedforward, no coupling
// correction, until regenerated for real).

#include <Arduino.h>
#include "PhaseController.h"
#include "PhaseSequencer.h"
#include "SerialComm.h"
#include "constants.h"
#include "current_sense.h"
#include "safety_startup.h"
#include "telemetry.h"
#include "plant_model.h"
#include "lqr_gains.h"

static inline float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// ============================== DRIVE ==============================
const float PHASES_CW[NUM_CHANNELS] = {270.0, 90.0, 180.0, 0.0};
const float PHASES_CCW[NUM_CHANNELS] = {90.0, 270.0, 180.0, 0.0};
const float INITIAL_DUTY_CYCLES[NUM_CHANNELS] = {50.0, 50.0, 50.0, 50.0};
const float INITIAL_CARRIER_DUTY_CYCLES[NUM_CHANNELS] = {0, 0, 0, 0};
const float START_DUTY = 50.0f;

const float start_freq = 1.0f;
const float end_freq = 210.0f;
const unsigned long ramp_duration_ms = 20000;

// ============================== CURRENT SENSE ==============================
const float SENS[NUM_CHANNELS] = {15.26, 15.28, 15.57, 15.34};
CurrentSense currentSense(SENS); // TAU_FILTER_MS=50 default

// ============================== LQR CONTROLLER ==============================
const float DUTY_MIN = 5.0f;
const float DUTY_MAX = 100.0f;
const float I_MAX_A  = 12.0f;             // hard per-channel safety cap
const float OVERCURRENT_BACKOFF_A = 0.5f; // r step-back on overcurrent trip

// Shared target current r ramps up over the run (replaces the old
// anchor-to-minimum-channel ramp -- now every channel is driven toward the
// SAME target, with LQR_K correcting for coupling instead of a latched
// forced-100% channel). Runtime-tunable via rmax=/rrate= (see dispatchCommand).
float R_MAX_A = 8.0f;              // ceiling, kept below I_MAX_A for margin
float R_RAMP_A_PER_MS = 0.0003f;   // ~0.3A/s -- untuned placeholder, runtime-tunable
float r_target = 0.0f;             // current value of the shared target
const float NOMINAL_TICK_MS = 2.0f; // dt normalization for R_RAMP_A_PER_MS, matches main_current_pid.cpp's convention

float duty_out[NUM_CHANNELS] = {START_DUTY, START_DUTY, START_DUTY, START_DUTY};

PhaseController *controller;
PhaseSequencer *seq;
SerialComm comm;
bool directionIsCcw = true;

// u = u_ff(r) - K(x - x_ref), x_ref = r for every channel. u_ff is the
// steady-state duty->current relation (R_i * r = Kv_i * u_ff_i); coupling
// (L^-1 off-diagonal terms) only matters transiently, so it drops out of the
// feedforward and is instead handled by LQR_K acting on the live error.
void runControlTick() {
  float *i_meas = currentSense.i_meas;
  bool any_saturated_high = false;
  bool any_overcurrent = false;

  float err[NUM_CHANNELS];
  for (int i = 0; i < NUM_CHANNELS; i++) {
    err[i] = i_meas[i] - r_target;
    if (i_meas[i] > I_MAX_A) any_overcurrent = true;
  }

  for (int i = 0; i < NUM_CHANNELS; i++) {
    float u_ff = PLANT_R_OHM[i] * r_target / PLANT_KV[i];
    float correction = 0.0f;
    for (int j = 0; j < NUM_CHANNELS; j++) correction -= LQR_K[i][j] * err[j];
    float duty = clampf(u_ff + correction, DUTY_MIN, DUTY_MAX);
    if (duty >= DUTY_MAX) any_saturated_high = true;
    duty_out[i] = duty;
    controller->setCarrierDutyCycle(i, duty);
  }

  // Reference governor: only raise r while there's headroom; back off hard
  // on a real overcurrent trip, exactly mirroring main_current_pid.cpp's
  // OVERCURRENT_BACKOFF_PCT policy but acting on r instead of a duty
  // integrator (there's no integrator here -- LQR_K is a static gain).
  if (any_overcurrent) {
    r_target = clampf(r_target - OVERCURRENT_BACKOFF_A, 0.0f, R_MAX_A);
  } else if (!any_saturated_high) {
    r_target = clampf(r_target + R_RAMP_A_PER_MS * NOMINAL_TICK_MS, 0.0f, R_MAX_A);
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

void reinitController(bool ccw) {
  const float *phases = ccw ? PHASES_CCW : PHASES_CW;
  directionIsCcw = ccw;

  if (controller) delete controller;
  controller = new PhaseController(PWM_PINS, phases, INITIAL_DUTY_CYCLES, NUM_CHANNELS);
  controller->begin(start_freq);
  controller->initCarrierPWM(CARRIER_PINS, PWM_FREQ, INITIAL_CARRIER_DUTY_CYCLES);
  allCoilsOff();

  if (seq) delete seq;
  seq = new PhaseSequencer(controller);
  seq->addRampTask(start_freq, end_freq, ramp_duration_ms, TaskType::PWM_FREQ, TaskMode::EASE);
  seq->compile(25, 1.0f, INITIAL_DUTY_CYCLES, phases);
}

void printLqrParams() {
  Serial.printf("R_MAX=%.3f R_RAMP=%.5f\n", R_MAX_A, R_RAMP_A_PER_MS);
}

void dispatchCommand(const String &raw) {
  String cmd = raw;
  cmd.trim();
  cmd.toLowerCase();

  bool running = (phase == RAMP_UP || phase == HOLD);

  if (cmd == "gains?" || cmd == "gains") {
    printLqrParams();
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
  } else if (cmd.startsWith("rmax=")) {
    if (running) { Serial.println("ignored: running (stop first)"); return; }
    R_MAX_A = cmd.substring(5).toFloat();
    printLqrParams();
  } else if (cmd.startsWith("rrate=")) {
    if (running) { Serial.println("ignored: running (stop first)"); return; }
    R_RAMP_A_PER_MS = cmd.substring(6).toFloat();
    printLqrParams();
  } else {
    Serial.printf("unknown cmd '%s' (dir=cw/ccw  rmax=/rrate=<val>  gains?)\n",
                  cmd.c_str());
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  forceAllGatesLow();
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  currentSense.seed();

  controller = nullptr;
  seq = nullptr;
  reinitController(directionIsCcw);

  phase = ARMING;
  phase_start = millis();
  Serial.println("state_space: arming 3s, then autonomous 1->210Hz ramp + LQR, bounded run");
  Serial.println("commands: dir=cw|ccw  rmax=/rrate=<val>  gains?");
}

void loop() {
  unsigned long now = millis();

  String line = comm.handleSerialComm();
  if (line.length()) dispatchCommand(line);

  const unsigned long ADC_SAMPLE_MS = 1;
  static unsigned long last_adc_us = micros();
  unsigned long now_us = micros();
  float dt_since_adc_ms = (float)(now_us - last_adc_us) / 1000.0f;
  if (dt_since_adc_ms >= (float)ADC_SAMPLE_MS) {
    currentSense.update(dt_since_adc_ms);
    last_adc_us = now_us;
  }

  controller->run();
  if (phase == RAMP_UP || phase == HOLD) seq->run();

  switch (phase) {
    case ARMING:
      allCoilsOff();
      digitalWrite(LED_PIN, (now / 150) & 1);
      currentSense.recalibrateZero();
      if (now - phase_start >= ARM_MS) {
        r_target = 0.0f;
        for (int i = 0; i < NUM_CHANNELS; i++) {
          duty_out[i] = 0.0f;
          controller->setCarrierDutyCycle(i, 0.0f);
        }
        seq->start();
        phase = RAMP_UP;
        phase_start = now;
        Serial.println("ARMED -> ramping");
      }
      break;

    case RAMP_UP: {
      runControlTick();
      if (controller->getFrequency() >= end_freq - 0.5f) {
        phase = HOLD;
        phase_start = now;
      }
      break;
    }

    case HOLD: {
      runControlTick();
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
      digitalWrite(LED_PIN, (now / 1000) & 1);
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
    Serial.printf(" | spread=%.3f dir=%d r=%.3f rmax=%.3f rrate=%.5f\n",
                  i_max - i_min, directionIsCcw ? 1 : 0, r_target, R_MAX_A, R_RAMP_A_PER_MS);
  }
}
