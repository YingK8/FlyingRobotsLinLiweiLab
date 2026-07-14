// Generalized current-ratio PI controller: loads a JSON profile (SPIFFS,
// see task_sequences/pi_profile_*.json) that selects a per-channel target
// RATIO vector and a policy mode (shared_constraint -- generalizes
// main_current_pid.cpp's proven anchor-channel balance policy -- or
// independent, for rigs like a tilt experiment where each coil has its own
// supply/driver headroom). See src/RatioCurrentController.h for the control
// law and src/PiProfile.h for the JSON schema.
//
// main_current_pid.cpp is left UNTOUCHED as the proven fallback until this
// path is hardware-validated (see the plan's deferred Phase 6) -- do not
// retire it before then.
//
// Identical state machine to main_current_pid.cpp: arms (3s, self-calibrate
// ADC zero) -> waits for an explicit start command (never auto-starts, see
// lib/ExperimentPhase) -> ramps drive frequency 1->210Hz over
// ramp_duration_ms -> holds at 210Hz for HOLD_MS -> ramps duty down ->
// latches off. Rotation direction and gains are runtime-adjustable at any
// time (s=estop r=start dir=/kp=/ki=/kd=/ramp=/gains? over SerialComm), same
// UX as main_current_pid.cpp, so tools/pid_autotune.py-style tooling keeps
// working unmodified. LED_PIN flips on every commanded carrier-duty
// decrease for the whole RAMP_UP+HOLD run (see ledSignalDutyDecrease()).

#include <Arduino.h>
#include <FS.h>
#include <SPIFFS.h>
#include <string.h>

#include "ExperimentPhase.h"
#include "PWMController.h"
#include "PWMSequencer.h"
#include "PiProfile.h"
#include "RatioCurrentController.h"
#include "SerialComm.h"
#include "constants.h"
#include "CurrentSense.h"
#include "safety_startup.h"
#include "telemetry.h"

// ============================== DRIVE ==============================
const float PHASES_CW[NUM_CHANNELS] = {270.0, 90.0, 180.0, 0.0};
const float PHASES_CCW[NUM_CHANNELS] = {90.0, 270.0, 180.0, 0.0};
const float INITIAL_DUTY_CYCLES[NUM_CHANNELS] = {50.0, 50.0, 50.0, 50.0};
const float INITIAL_CARRIER_DUTY_CYCLES[NUM_CHANNELS] = {0, 0, 0, 0};
const float START_DUTY = 50.0f;

const float start_freq = 1.0f;
const float end_freq = 210.0f;
const unsigned long ramp_duration_ms = 20000;

const char *PROFILE_FILE = "/pi_profile.json";

// ============================== CURRENT SENSE ==============================
// SENS[] comes from constants.h (VNH5019 CS gain, per-board calibration).
CurrentSense currentSense(ADC_PINS, SENS, NUM_CHANNELS);

PWMController *controller;
PWMSequencer *seq;
SerialComm comm;
bool directionIsCcw = true;

RatioCurrentController::Config g_cfg;
RatioCurrentController *ratioController;
float duty_out[NUM_CHANNELS] = {START_DUTY, START_DUTY, START_DUTY, START_DUTY};

// Running-phase (RAMP_UP + HOLD) diagnostic: flips LED_PIN each tick any
// channel's commanded carrier duty drops vs. the previous tick (e.g. the PI
// backing off from an overcurrent trip or ratio-tracking overshoot) -- lets
// you see on the bench when the tilt controller is trimming duty back,
// without needing a serial log open. Covers the whole run, not just HOLD:
// duty can legitimately dip during RAMP_UP too as the independent-mode
// magnitude governor and per-channel PI settle against each other.
float duty_prev_run[NUM_CHANNELS];
bool run_led_state = false;

void ledSignalDutyDecrease() {
  bool duty_decreased = false;
  for (int i = 0; i < NUM_CHANNELS; i++) {
    if (duty_out[i] < duty_prev_run[i]) duty_decreased = true;
  }
  if (duty_decreased) {
    run_led_state = !run_led_state;
    digitalWrite(LED_PIN, run_led_state ? HIGH : LOW);
  }
  memcpy(duty_prev_run, duty_out, sizeof(duty_prev_run));
}

// ============================== STATE MACHINE ==============================
enum Phase { ARMING, WAITING, RAMP_UP, HOLD, ENDING, STOPPED, PROFILE_LOAD_FAILED };
Phase phase = ARMING;
unsigned long phase_start = 0;

// ARM_MS comes from constants.h (guaranteed coils-OFF period after boot).
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
  controller = new PWMController(PWM_PINS, phases, INITIAL_DUTY_CYCLES, NUM_CHANNELS);
  controller->begin(start_freq);
  controller->initCarrierPWM(CARRIER_PINS, PWM_FREQ, INITIAL_CARRIER_DUTY_CYCLES);
  allCoilsOff();

  if (seq) delete seq;
  seq = new PWMSequencer(controller);
  seq->addRampTask(start_freq, end_freq, ramp_duration_ms, TaskType::PWM_FREQ, TaskMode::EASE);
  seq->compile(25, 1.0f, INITIAL_DUTY_CYCLES, phases);
}

void printGains() {
  const RatioCurrentController::Config &c = ratioController->config();
  Serial.printf("KP=%.3f KI=%.3f KD=%.3f RAMP=%.4f mode=%s ratios=[%.2f,%.2f,%.2f,%.2f]\n",
                c.kp, c.ki, c.kd, c.rampPctPerMs,
                c.sharedConstraint ? "shared_constraint" : "independent",
                c.ratios[0], c.ratios[1], c.ratios[2], c.ratios[3]);
}

void dispatchCommand(const String &raw) {
  String cmd = raw;
  cmd.trim();
  cmd.toLowerCase();

  // E-stop first, unconditionally -- works in every phase, unlike the
  // tuning/start commands below which are only valid while parked.
  if (ExperimentPhase::isStopCommand(cmd)) {
    allCoilsOff();
    phase = STOPPED;
    Serial.println("ESTOP -> coils latched OFF");
    return;
  }

  if (phase == WAITING && ExperimentPhase::isStartCommand(cmd)) {
    ratioController->reset();
    for (int i = 0; i < NUM_CHANNELS; i++) controller->setCarrierDutyCycle(i, START_DUTY);
    seq->start();
    phase = RAMP_UP;
    phase_start = millis();
    memcpy(duty_prev_run, duty_out, sizeof(duty_prev_run));
    run_led_state = false;
    digitalWrite(LED_PIN, LOW);
    Serial.println("START -> ramping");
    return;
  }

  if (cmd == "gains?" || cmd == "gains") {
    printGains();
  } else if (cmd.startsWith("dir=")) {
    if (phase != ARMING && phase != WAITING && phase != STOPPED) {
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
    // PI controller always active: gains apply live, in any phase -- no
    // more racing the old ARMING window before RAMP_UP locked them out.
    g_cfg.kp = cmd.substring(3).toFloat();
    ratioController->setGains(g_cfg.kp, g_cfg.ki, g_cfg.kd);
    printGains();
  } else if (cmd.startsWith("ki=")) {
    g_cfg.ki = cmd.substring(3).toFloat();
    ratioController->setGains(g_cfg.kp, g_cfg.ki, g_cfg.kd);
    printGains();
  } else if (cmd.startsWith("kd=")) {
    g_cfg.kd = cmd.substring(3).toFloat();
    ratioController->setGains(g_cfg.kp, g_cfg.ki, g_cfg.kd);
    printGains();
  } else if (cmd.startsWith("ramp=")) {
    g_cfg.rampPctPerMs = cmd.substring(5).toFloat();
    ratioController->setRampPctPerMs(g_cfg.rampPctPerMs);
    printGains();
  } else {
    Serial.printf("unknown cmd '%s' (s=estop  r=start  dir=cw/ccw  kp=/ki=/kd=/ramp=<val>  gains?)\n",
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
  reinitController(directionIsCcw);

  SPIFFS.begin(true);
  if (!loadPiProfile(PROFILE_FILE, &g_cfg)) {
    // Never runs on garbage: stay latched off permanently, matching
    // main_experiment.cpp's JSON-load-failure handling.
    Serial.printf("FAILED to load %s -- staying latched off (never running "
                  "on a missing/invalid profile)\n", PROFILE_FILE);
    phase = PROFILE_LOAD_FAILED;
    return;
  }
  ratioController = new RatioCurrentController(g_cfg);

  phase = ARMING;
  phase_start = millis();
  Serial.printf("pi_profile: loaded %s, arming 3s, then waiting for a start "
                "command before the 1->210Hz ramp + ratio-tracking PI run\n", PROFILE_FILE);
  Serial.println("commands: s=estop  r=start  dir=cw|ccw  kp=/ki=/kd=/ramp=<val>  gains?");
  printGains();
}

void loop() {
  unsigned long now = millis();

  if (phase == PROFILE_LOAD_FAILED) {
    allCoilsOff();
    digitalWrite(LED_PIN, (now / 1000) & 1);
    return;
  }

  String line = comm.step();
  if (line.length()) dispatchCommand(line);

  const unsigned long ADC_SAMPLE_MS = 1;
  static unsigned long last_adc_us = micros();
  unsigned long now_us = micros();
  float dt_since_adc_ms = (float)(now_us - last_adc_us) / 1000.0f;
  if (dt_since_adc_ms >= (float)ADC_SAMPLE_MS) {
    currentSense.update(dt_since_adc_ms);
    last_adc_us = now_us;
  }

  static unsigned long last_control_us = now_us;
  float control_dt_ms = (float)(now_us - last_control_us) / 1000.0f;
  if (control_dt_ms <= 0.0f) control_dt_ms = 0.001f;
  last_control_us = now_us;

  controller->step();
  if (phase == RAMP_UP || phase == HOLD) seq->step();

  switch (phase) {
    case ARMING:
      if (ExperimentPhase::armingTick(now, phase_start, ARM_MS, controller,
                                       NUM_CHANNELS, currentSense, LED_PIN)) {
        phase = WAITING;
        phase_start = now;
        Serial.println("ARMED -> waiting for start (s=estop, r=start)");
      }
      break;

    case WAITING:
      ExperimentPhase::latchedOffTick(now, controller, NUM_CHANNELS, LED_PIN);
      break;

    case RAMP_UP:
      ratioController->computeTick(currentSense.i_meas, control_dt_ms, duty_out);
      for (int i = 0; i < NUM_CHANNELS; i++) controller->setCarrierDutyCycle(i, duty_out[i]);
      ledSignalDutyDecrease();
      if (controller->getFrequency() >= end_freq - 0.5f) {
        phase = HOLD;
        phase_start = now;
      }
      break;

    case HOLD:
      ratioController->computeTick(currentSense.i_meas, control_dt_ms, duty_out);
      for (int i = 0; i < NUM_CHANNELS; i++) controller->setCarrierDutyCycle(i, duty_out[i]);
      ledSignalDutyDecrease();
      if (now - phase_start >= HOLD_MS) {
        phase = ENDING;
        last_ramp_ms = now;
      }
      break;

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
      ExperimentPhase::latchedOffTick(now, controller, NUM_CHANNELS, LED_PIN);
      break;

    case PROFILE_LOAD_FAILED:
      break;  // handled at top of loop()
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
                  i_max - i_min, directionIsCcw ? 1 : 0, g_cfg.kp, g_cfg.ki, g_cfg.kd,
                  g_cfg.rampPctPerMs);
  }
}
