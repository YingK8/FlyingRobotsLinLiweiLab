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
#include "CurrentBalanceController.h"
#include "PhaseController.h"
#include "PhaseSequencer.h"
#include "SerialComm.h"
#include "constants.h"
#include "current_sense.h"
#include "safety_startup.h"
#include "telemetry.h"

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
CurrentSense currentSense(SENS); // TAU_FILTER_MS=50 default -- see current_sense.h

// ============================== PI CONTROLLER ==============================
// The balance law now lives in the shared CurrentBalanceController library
// (extracted verbatim from this file's former inline runControlTick). Here it
// runs with a flat 100% ceiling on every channel -- "push each channel to its
// max" -- so behavior is identical to the original. This file stays the
// reference tuning rig: KP/KI/KD/ramp remain runtime-tunable mirrors (kp=/ki=/
// kd=/ramp= over serial) that are pushed into the controller on change.
// Converged values: KP=2.2, KI=0.10, KD=0.15.
float KP = 2.2f;
float KI = 0.10f;
float KD = 0.15f;
float MIN_RAMP_PCT_PER_MS = 0.05f;

CurrentBalanceController balance;
const float CEILING_100[NUM_CHANNELS] = {100.0f, 100.0f, 100.0f, 100.0f};
float duty_out[NUM_CHANNELS] = {START_DUTY, START_DUTY, START_DUTY, START_DUTY};

PhaseController *controller;
PhaseSequencer *seq;
SerialComm comm;
bool directionIsCcw = true; // default direction, matches this file's history

// Balance tick: the shared controller with a flat 100% ceiling (see above).
// duty_out is kept for the debug telemetry print below.
void runControlTick(float dt_ms) {
  balance.computeTick(currentSense.i_meas, dt_ms, CEILING_100, duty_out);
  for (int i = 0; i < NUM_CHANNELS; i++)
    controller->setCarrierDutyCycle(i, duty_out[i]);
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
  controller = new PhaseController(PWM_PINS, phases, INITIAL_DUTY_CYCLES, NUM_CHANNELS);
  // Explicit non-zero starting frequency -- begin(0.0f) divides by zero
  // inside setGlobalFrequency() and permanently corrupts commutation timing.
  controller->begin(start_freq);
  controller->initCarrierPWM(CARRIER_PINS, PWM_FREQ, INITIAL_CARRIER_DUTY_CYCLES);
  allCoilsOff();

  if (seq) delete seq;
  seq = new PhaseSequencer(controller);
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
    balance.setGains(KP, KI, KD);
    printGains();
  } else if (cmd.startsWith("ki=")) {
    if (running) { Serial.println("ignored: running (stop first)"); return; }
    KI = cmd.substring(3).toFloat();
    balance.setGains(KP, KI, KD);
    printGains();
  } else if (cmd.startsWith("kd=")) {
    if (running) { Serial.println("ignored: running (stop first)"); return; }
    KD = cmd.substring(3).toFloat();
    balance.setGains(KP, KI, KD);
    printGains();
  } else if (cmd.startsWith("ramp=")) {
    if (running) { Serial.println("ignored: running (stop first)"); return; }
    MIN_RAMP_PCT_PER_MS = cmd.substring(5).toFloat();
    balance.setRamp(MIN_RAMP_PCT_PER_MS);
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

  // Seed the shared balance controller from the runtime-tunable mirrors (the
  // defaults already match, but keep this explicit so kp=/ramp= edits and the
  // controller never drift apart).
  balance.setGains(KP, KI, KD);
  balance.setRamp(MIN_RAMP_PCT_PER_MS);

  phase = ARMING;
  phase_start = millis();
  Serial.println("current_pid: arming 3s, then autonomous 1->210Hz ramp + PI, bounded run");
  Serial.println("commands: dir=cw|ccw  kp=/ki=/kd=/ramp=<val>  gains?");
}

void loop() {
  unsigned long now = millis();  // state-machine timing (seconds-scale, ms is plenty)

  String line = comm.handleSerialComm();
  if (line.length()) dispatchCommand(line);

  // ADC sampling is rate-limited -- NOT a software throttle, a real ESP32
  // ADC hardware constraint (see current_sense.cpp). Control computation
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

  controller->run();
  if (phase == RAMP_UP || phase == HOLD) seq->run();

  switch (phase) {
    case ARMING:
      allCoilsOff();
      digitalWrite(LED_PIN, (now / 150) & 1);  // fast blink = arming
      currentSense.recalibrateZero();  // coils confirmed off here
      if (now - phase_start >= ARM_MS) {
        balance.reset(START_DUTY);
        for (int i = 0; i < NUM_CHANNELS; i++)
          controller->setCarrierDutyCycle(i, START_DUTY);
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
