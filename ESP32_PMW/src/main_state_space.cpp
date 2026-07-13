// Dual-core current-balance controller. Core 0 (PwmActuator) owns the
// entire low-level, timing-critical domain: ADC sampling, PWM/carrier
// stepping, the bounded run state machine (ARMING->RAMP_UP->HOLD->
// ENDING->STOPPED), and an independent hard overcurrent trip -- it applies
// whatever duty is in SharedMemory's command region every tick, indifferent
// to freshness. This makes the bounded/safe-run guarantee independent of
// core 1 entirely: if the higher-level control law hangs, core 0 still
// completes its run and safely latches coils off.
//
// Core 1 (this file's setup()/loop(), Arduino's default core) owns the
// higher-level domain: serial comm, the control law
// (CurrentBalanceController -- coupling-aware feedforward + LQID feedback +
// constrained duty allocator, direction-selected), and the reference
// governor. See the state-space plan ("Coupling-aware feedforward +
// constrained duty allocator + dual-core split") for the full design
// rationale and tools/design_lqr_gains.py / tools/build_feedforward.py /
// tools/build_state_space_model.py for how the offline-fitted matrices this
// depends on were derived.

#include <Arduino.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include "CurrentBalanceController.h"
#include "PwmActuator.h"
#include "SerialComm.h"
#include "SharedMemory.h"
#include "constants.h"
#include "safety_startup.h"
#include "state_space_constants.h"
#include "telemetry.h"

SharedMemory sharedMemory;
PwmActuator pwmActuator(&sharedMemory);
CurrentBalanceController controller;
SerialComm comm;

// Core-1-local state: the control law's own view of things. directionIsCcw
// here optimistically mirrors what's been requested (dir= is only issued/
// honored while quiescent -- see PwmActuator::applyDirectionRequestIfSafe --
// so the two stay in sync in practice; core 0's copy remains authoritative
// for what hardware actually does).
bool directionIsCcw = true;
float R_MAX_A = 8.0f;
float R_RAMP_A_PER_MS = 0.0003f;
float r_target = 0.0f;
float duty_out[NUM_CHANNELS] = {START_DUTY, START_DUTY, START_DUTY, START_DUTY};
int last_seen_phase = PHASE_ARMING;
unsigned long last_control_ms = 0;

void pwmTaskFn(void *) {
  pwmActuator.begin();
  for (;;) {
    pwmActuator.run();
    vTaskDelay(1); // yield to IDLE0 every ~1 tick -- matches PwmActuator's own ADC pacing granularity
  }
}

void printLqrParams() {
  Serial.printf("R_MAX=%.3f R_RAMP=%.5f\n", R_MAX_A, R_RAMP_A_PER_MS);
}

void dispatchCommand(const String &raw) {
  String cmd = raw;
  cmd.trim();
  cmd.toLowerCase();

  bool running = (last_seen_phase == PHASE_RAMP_UP || last_seen_phase == PHASE_HOLD);

  if (cmd == "gains?" || cmd == "gains") {
    printLqrParams();
  } else if (cmd.startsWith("dir=")) {
    if (running) {
      Serial.println("ignored: running (stop first)");
      return;
    }
    String v = cmd.substring(4);
    if (v == "cw") {
      directionIsCcw = false;
      sharedMemory.requestDirection(false);
      Serial.println("direction=CW");
    } else if (v == "ccw") {
      directionIsCcw = true;
      sharedMemory.requestDirection(true);
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
    Serial.printf("unknown cmd '%s' (dir=cw/ccw  rmax=/rrate=<val>  gains?)\n", cmd.c_str());
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  forceAllGatesLow();
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  xTaskCreatePinnedToCore(pwmTaskFn, "pwmTask", 4096, nullptr, 2, nullptr, 0);

  Serial.println("state_space: dual-core, arming 3s, then autonomous 1->210Hz ramp + LQID, bounded run");
  Serial.println("commands: dir=cw|ccw  rmax=/rrate=<val>  gains?");
}

void loop() {
  unsigned long now = millis();

  String line = comm.step();
  if (line.length()) dispatchCommand(line);

  float i_meas[NUM_CHANNELS];
  int phase;
  float freq;
  sharedMemory.readMeasurement(i_meas, &phase, &freq);

  // Edge-detect ARMING->RAMP_UP: reset the control law's internal state
  // (z_integral/err_prev/derr_filt/duty_prev) exactly once per run, without
  // needing an explicit reset command from core 0.
  if (last_seen_phase == PHASE_ARMING && phase == PHASE_RAMP_UP) {
    controller.reset();
    r_target = 0.0f;
  }
  last_seen_phase = phase;

  if (phase == PHASE_RAMP_UP || phase == PHASE_HOLD) {
    if (now - last_control_ms >= CONTROL_TICK_MS) {
      last_control_ms = now;
      r_target = controller.computeTick(i_meas, r_target, R_MAX_A, R_RAMP_A_PER_MS,
                                         directionIsCcw, freq, duty_out);
      sharedMemory.publishDuty(duty_out);
    }
  }

  static unsigned long last_dbg_ms = 0;
  if (now - last_dbg_ms >= 500) {
    last_dbg_ms = now;
    float i_min = i_meas[0], i_max = i_meas[0];
    for (int i = 1; i < NUM_CHANNELS; i++) {
      if (i_meas[i] < i_min) i_min = i_meas[i];
      if (i_meas[i] > i_max) i_max = i_meas[i];
    }
    Serial.printf("t=%lu phase=%d freq=%.1f | ", now, phase, freq);
    printCurrentAndDuty(i_meas, duty_out);
    Serial.printf(" | spread=%.3f dir=%d r=%.3f rmax=%.3f rrate=%.5f\n",
                  i_max - i_min, directionIsCcw ? 1 : 0, r_target, R_MAX_A, R_RAMP_A_PER_MS);
  }
}
