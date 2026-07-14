#include <Arduino.h>
#include <FS.h>
#include <SPIFFS.h>
#include <math.h>

#include "ExperimentPhase.h"
#include "JsonPWMSequencer.h"
#include "PiProfile.h"
#include "PWMController.h"
#include "RatioCurrentController.h"
#include "SerialComm.h"
#include "constants.h"
#include "CurrentSense.h"
#include "safety_startup.h"
#include "telemetry.h"

const float INITIAL_PHASES[NUM_CHANNELS] = {270.0, 90.0, 180.0, 0.0}; // CW
const float INITIAL_DUTY_CYCLES[NUM_CHANNELS] = {50.0, 50.0, 50.0, 50.0};
const float INITIAL_CARRIER_DUTY_CYCLES[NUM_CHANNELS] = {0, 0, 0, 0};
const float DRIVE_FREQ = 190.0f;

const char *EXPERIMENT_FILE = "/experiment.json";
// Optional PI compensation on top of the JSON schedule's carrier-duty
// commands (see the "pi=on"/"pi=off" dispatch in loop()). Opt-in: staged by
// tools/run_experiment.py's --pi-compensate --pi-profile <file>, which
// reuses main_pi_profile.cpp's PiProfile.h/RatioCurrentController schema
// and validate+stage step. Absent or invalid file -> PI compensation stays
// unavailable, firmware runs fully open-loop as before.
const char *PI_PROFILE_FILE = "/pi_profile.json";

// Hard overcurrent trip -- unlike main_current_pid.cpp this firmware is
// fully open-loop (JsonPWMSequencer just plays back a schedule) with no
// other protection. Needed for system-ID sweeps that deliberately hunt for
// an a-priori-unknown resonance peak (see tools/gen_solo_sweep_experiment.py)
// where current could spike well past the ~2A RMS coil rating (PCB doc) if
// left unwatched.
const float I_SAFETY_MAX_A = 8.0f;

PWMController *controller;
JsonPWMSequencer *seq;
CurrentSense currentSense(ADC_PINS, SENS, NUM_CHANNELS);
SerialComm comm;

// PI compensation state -- see PI_PROFILE_FILE above and the "pi=on"/
// "pi=off" dispatch in loop(). ratioController is constructed only if
// PI_PROFILE_FILE loads successfully at setup() (piProfileLoaded); piOn
// gates whether it's actually applied each RUNNING tick (default off, so
// a normal --fw experiment run with no staged profile is unaffected).
RatioCurrentController::Config g_piCfg;
RatioCurrentController *ratioController = nullptr;
bool piProfileLoaded = false;
bool piOn = false;

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

  controller = new PWMController(PWM_PINS, INITIAL_PHASES,
                                    INITIAL_DUTY_CYCLES, NUM_CHANNELS);
  seq = new JsonPWMSequencer(controller);

  // Explicit non-zero starting frequency -- begin(0.0f) (the default) divides
  // by zero inside setGlobalFrequency() and permanently corrupts the
  // commutation timing (root-caused during main_current_pid.cpp's bring-up).
  controller->begin(DRIVE_FREQ);
  controller->initCarrierPWM(CARRIER_PINS, PWM_FREQ, INITIAL_CARRIER_DUTY_CYCLES);
  allCoilsOff(); // explicit: carriers latched at 0 after LEDC attach

  SPIFFS.begin(true);

  // Optional -- absent/invalid file just means PI compensation stays
  // unavailable (piOn can never be set to true; "pi=on" is rejected).
  piProfileLoaded = loadPiProfile(PI_PROFILE_FILE, &g_piCfg);
  if (piProfileLoaded) {
    ratioController = new RatioCurrentController(g_piCfg);
    Serial.printf("loaded %s -- PI compensation available (send 'pi=on' to "
                  "engage)\n", PI_PROFILE_FILE);
  }

  phase = ARMING;
  phase_start = millis();
  Serial.printf("main_experiment: arming %lums, then autonomously loading %s "
                "(s=e-stop at any time)\n", ARM_MS, EXPERIMENT_FILE);
}

void loop() {
  controller->step();
  unsigned long now = millis();

  // ADC sampling paced separately from everything else -- see
  // CurrentSense.cpp for why (ESP32 ADC mux/S&H settling requirement).
  static unsigned long last_adc_us = micros();
  unsigned long now_us = micros();
  const float ADC_SAMPLE_MS = 1.0f;
  float dt_adc_ms = (float)(now_us - last_adc_us) / 1000.0f;
  if (dt_adc_ms >= ADC_SAMPLE_MS) {
    currentSense.update(dt_adc_ms);
    last_adc_us = now_us;
  }

  // dt for the PI compensation tick (KI/KD rate-scaling), same pattern as
  // main_current_pid.cpp/main_pi_profile.cpp's control_dt_ms.
  static unsigned long last_control_us = now_us;
  float control_dt_ms = (float)(now_us - last_control_us) / 1000.0f;
  if (control_dt_ms <= 0.0f) control_dt_ms = 0.001f;  // guard div-by-zero only
  last_control_us = now_us;

  String line = comm.step();
  if (line.length()) {
    line.trim();
    line.toLowerCase();
    if (ExperimentPhase::isStopCommand(line)) {
      allCoilsOff();
      phase = DONE;
      Serial.println("ESTOP -> coils latched OFF");
    } else if (line == "pi=on") {
      if (!piProfileLoaded) {
        Serial.printf("ignored: no valid %s loaded -- stage one via "
                      "run_experiment.py --pi-profile\n", PI_PROFILE_FILE);
      } else {
        piOn = true;
        Serial.println("PI compensation ON -- carrier duty now closed-loop "
                        "(ratio-tracking around the JSON schedule's reference)");
      }
    } else if (line == "pi=off") {
      piOn = false;
      Serial.println("PI compensation OFF -- carrier duty follows the JSON "
                      "schedule directly");
    }
  }

  switch (phase) {
  case ARMING:
    if (ExperimentPhase::armingTick(now, phase_start, ARM_MS, controller,
                                     NUM_CHANNELS, currentSense, LED_PIN)) {
      bool ok = seq->loadFromJsonFile(EXPERIMENT_FILE, 25, DRIVE_FREQ,
                                       INITIAL_DUTY_CYCLES, INITIAL_PHASES);
      if (!ok) {
        Serial.println("ARMED -> FAILED to load experiment.json -- staying "
                        "latched off (never energizing on a parse failure)");
        phase = DONE;
      } else {
        if (ratioController) ratioController->reset();
        seq->start();
        phase = RUNNING;
        Serial.println("ARMED -> running experiment");
      }
      phase_start = now;
    }
    break;

  case RUNNING: {
    seq->step();

    // PI compensation (opt-in, see "pi=on"): channel A's SCHEDULED carrier
    // duty (the sequencer's own intended state -- see
    // PWMSequencer::scheduledCarrierDutyCycle()'s comment; NOT
    // controller->getCarrierDutyCycle(), which reads back whatever this
    // same block last wrote and would create a self-referential feedback
    // loop, confirmed on hardware) is the common reference every channel's
    // target is ratio[i] * reference of (matches this project's existing
    // ratios convention, e.g. task_sequences/pi_profile_tilt.json's A=1.0).
    // NAN if the schedule hasn't commanded any carrier duty yet -- skip
    // compensation that tick rather than feeding a bogus reference.
    if (piOn && ratioController) {
      float reference = seq->scheduledCarrierDutyCycle(0);
      if (!isnan(reference)) {
        float duty_out[NUM_CHANNELS];
        ratioController->computeTickWithReference(currentSense.i_meas, control_dt_ms,
                                                  reference, duty_out);
        for (int i = 0; i < NUM_CHANNELS; i++)
          controller->setCarrierDutyCycle(i, duty_out[i]);
      }
    }

    for (int i = 0; i < NUM_CHANNELS; i++) {
      if (currentSense.i_meas[i] > I_SAFETY_MAX_A) {
        allCoilsOff();
        phase = DONE;
        Serial.printf("SAFETY: channel %d overcurrent (%.2fA) -- latching off\n",
                      i, currentSense.i_meas[i]);
        break;
      }
    }
    if (phase != RUNNING) break;

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

    // Periodic telemetry (in addition to the on-label-change print above):
    // system-ID sweeps need the SETTLED current at each frequency, but the
    // on-change print fires at the *start* of a dwell, before the current has
    // settled. This samples throughout each dwell so a fit script can average
    // the tail of it.
    static unsigned long last_periodic_ms = 0;
    const unsigned long PERIODIC_TELEMETRY_MS = 200;
    if (now - last_periodic_ms >= PERIODIC_TELEMETRY_MS) {
      last_periodic_ms = now;
      float dutyPct[NUM_CHANNELS];
      for (int i = 0; i < NUM_CHANNELS; i++)
        dutyPct[i] = controller->getCarrierDutyCycle(i);
      Serial.printf("t=%lu label=%s | ", now, lastLabel.c_str());
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
    ExperimentPhase::latchedOffTick(now, controller, NUM_CHANNELS, LED_PIN);
    break;
  }
}
