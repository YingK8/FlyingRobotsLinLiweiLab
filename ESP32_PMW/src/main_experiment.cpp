// Generic JSON-driven experiment firmware: loads /experiment.json (SPIFFS,
// see spiffs_data/) and runs it autonomously, replacing the old per-
// experiment main_*.cpp files. ARMING (3s, coils forced off, self-calibrate
// ADC zero) -> RUNNING (execute /experiment.json, printing "t=.. label=.."
// on every step-label change for scope-capture correlation, see
// tools/coupling_matrix.py --segments-log) -> DONE (coils latched off,
// including on a JSON parse failure, so it never runs on garbage or loops
// forever).

#include <Arduino.h>
#include <FS.h>
#include <SPIFFS.h>

#include "JsonPhaseSequencer.h"
#include "PhaseController.h"
#include "constants.h"
#include "current_sense.h"
#include "safety_startup.h"
#include "telemetry.h"

// Default boot phases/frequency -- experiment.json's "setDirection" overrides
// phases at runtime; drive frequency stays constant (matches the project's
// coupling-sweep convention) unless the JSON itself ramps it.
const float INITIAL_PHASES[NUM_CHANNELS] = {270.0, 90.0, 180.0, 0.0}; // CW
const float INITIAL_DUTY_CYCLES[NUM_CHANNELS] = {50.0, 50.0, 50.0, 50.0};
const float INITIAL_CARRIER_DUTY_CYCLES[NUM_CHANNELS] = {0, 0, 0, 0};
const float DRIVE_FREQ = 190.0f;

const unsigned long ARM_MS = 3000; // guaranteed coils-OFF period after boot
const char *EXPERIMENT_FILE = "/experiment.json";

// VNH5019 CS gain, A per V -- per-board calibration (see main_current_pid.cpp
// tuning history); measured current is reported in telemetry as a redundant
// cross-check against the PicoScope capture, not used for control here.
const float SENS[NUM_CHANNELS] = {15.26, 15.28, 15.57, 15.34};

PhaseController *controller;
JsonPhaseSequencer *seq;
CurrentSense currentSense(SENS);

enum ExpPhase { ARMING, RUNNING, DONE };
ExpPhase phase = ARMING;
unsigned long phase_start = 0;
String lastLabel; // "" until the first labeled step runs

void allCoilsOff() {
  for (int i = 0; i < NUM_CHANNELS; i++)
    controller->setCarrierDutyCycle(i, 0.0f);
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  // SAFE STARTUP: gates forced LOW before constructing anything.
  forceAllGatesLow();
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  currentSense.seed();

  controller = new PhaseController(PWM_PINS, INITIAL_PHASES,
                                    INITIAL_DUTY_CYCLES, NUM_CHANNELS);
  seq = new JsonPhaseSequencer(controller);

  // Explicit non-zero starting frequency -- begin(0.0f) (the default) divides
  // by zero inside setGlobalFrequency() and permanently corrupts the
  // commutation timing (root-caused during main_current_pid.cpp's bring-up).
  controller->begin(DRIVE_FREQ);
  controller->initCarrierPWM(CARRIER_PINS, PWM_FREQ, INITIAL_CARRIER_DUTY_CYCLES);
  allCoilsOff(); // explicit: carriers latched at 0 after LEDC attach

  SPIFFS.begin(true);

  phase = ARMING;
  phase_start = millis();
  Serial.printf("main_experiment: arming %lums, then loading %s\n", ARM_MS,
                EXPERIMENT_FILE);
}

void loop() {
  controller->run();
  unsigned long now = millis();

  // ADC sampling paced separately from everything else -- see
  // current_sense.cpp for why (ESP32 ADC mux/S&H settling requirement).
  static unsigned long last_adc_us = micros();
  unsigned long now_us = micros();
  const float ADC_SAMPLE_MS = 1.0f;
  float dt_adc_ms = (float)(now_us - last_adc_us) / 1000.0f;
  if (dt_adc_ms >= ADC_SAMPLE_MS) {
    currentSense.update(dt_adc_ms);
    last_adc_us = now_us;
  }

  switch (phase) {
  case ARMING:
    allCoilsOff();
    digitalWrite(LED_PIN, (now / 150) & 1); // fast blink = arming/settling
    currentSense.recalibrateZero();         // coils confirmed off here
    if (now - phase_start >= ARM_MS) {
      bool ok = seq->loadFromJsonFile(EXPERIMENT_FILE, 25, DRIVE_FREQ,
                                       INITIAL_DUTY_CYCLES, INITIAL_PHASES);
      if (!ok) {
        Serial.println("ARMED -> FAILED to load experiment.json -- staying "
                        "latched off (never energizing on a parse failure)");
        phase = DONE;
      } else {
        seq->start();
        phase = RUNNING;
        Serial.println("ARMED -> running experiment");
      }
      phase_start = now;
    }
    break;

  case RUNNING: {
    seq->run();
    String label = seq->labelForStep(seq->currentIndex());
    if (label != lastLabel) {
      lastLabel = label;
      float dutyPct[NUM_CHANNELS];
      for (int i = 0; i < NUM_CHANNELS; i++)
        dutyPct[i] = controller->getCarrierDutyCycle(i);
      Serial.printf("t=%lu label=%s | ", now, label.c_str());
      printCurrentAndDuty(currentSense.i_meas, dutyPct);
      Serial.println();
    }
    if (seq->isDone()) {
      allCoilsOff();
      phase = DONE;
      Serial.println("experiment complete -> DONE (coils latched OFF)");
    }
    break;
  }

  case DONE:
    // SAFE SHUTDOWN: coils latched OFF permanently; slow heartbeat so the
    // board still reads as alive without ever re-energizing.
    allCoilsOff();
    digitalWrite(LED_PIN, (now / 1000) & 1);
    break;
  }
}
