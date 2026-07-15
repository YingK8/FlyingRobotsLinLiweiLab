#include "balanced_experiment.h"

#include <FS.h>
#include <SPIFFS.h>

#include "CurrentBalanceController.h"
#include "JsonPhaseSequencer.h"
#include "PhaseController.h"
#include "constants.h"
#include "current_sense.h"
#include "safety_startup.h"
#include "telemetry.h"

namespace {

// Project rotation conventions (coil order A,B,C,D). A JSON setDirection
// overrides these at runtime; they seed the controller at construction.
const float PHASES_CW[NUM_CHANNELS] = {270.0f, 90.0f, 180.0f, 0.0f};
const float PHASES_CCW[NUM_CHANNELS] = {90.0f, 270.0f, 180.0f, 0.0f};
const float INITIAL_DUTY_CYCLES[NUM_CHANNELS] = {50.0f, 50.0f, 50.0f, 50.0f};
const float INITIAL_CARRIER_DUTY_CYCLES[NUM_CHANNELS] = {0, 0, 0, 0};
const float START_DUTY = 50.0f; // channels start equal; the PI does the rest

// VNH5019 CS gain, A per V -- per-board calibration (shared across every
// experiment; historically duplicated in each main).
const float SENS[NUM_CHANNELS] = {15.26f, 15.28f, 15.57f, 15.34f};

const unsigned long ARM_MS = 3000; // guaranteed coils-OFF period after boot
const unsigned long TELEMETRY_MS = 500;
const float ADC_SAMPLE_MS = 1.0f;

enum Phase { ARMING, RUNNING, DONE };

ExperimentConfig g_cfg;
Phase g_phase = ARMING;
unsigned long g_phaseStart = 0;
unsigned long g_lastAdcUs = 0;
unsigned long g_lastControlUs = 0;
unsigned long g_lastTelemetryMs = 0;

PhaseController *g_controller = nullptr;
JsonPhaseSequencer *g_seq = nullptr;
CurrentSense g_currentSense(SENS);
CurrentBalanceController g_balance;
float g_dutyOut[NUM_CHANNELS] = {0, 0, 0, 0};

void allCoilsOff() {
  for (int i = 0; i < NUM_CHANNELS; i++)
    g_controller->setCarrierDutyCycle(i, 0.0f);
}

void emitTelemetry(unsigned long now) {
  float *iMeas = g_currentSense.i_meas;
  float iMin = iMeas[0], iMax = iMeas[0];
  for (int i = 1; i < NUM_CHANNELS; i++) {
    if (iMeas[i] < iMin)
      iMin = iMeas[i];
    if (iMeas[i] > iMax)
      iMax = iMeas[i];
  }
  // Same line format as main_current_pid.cpp so tools/pid_metrics.py (and
  // tools/tilt_metrics.py) parse it unchanged: "... phase=N freq=F | I[A]: ...
  // | duty[%]: ... | spread=S".
  Serial.printf("t=%lu phase=%d freq=%.1f | ", now, (int)g_phase,
                g_controller->getFrequency());
  printCurrentAndDuty(iMeas, g_dutyOut);
  Serial.printf(" | spread=%.3f pi=%d hold=%d ihold=%.2f\n", iMax - iMin,
                g_cfg.piEnabled ? 1 : 0, g_balance.holdFrozen() ? 1 : 0,
                g_balance.holdTarget());
}

} // namespace

void experimentSetup(const ExperimentConfig &cfg) {
  g_cfg = cfg;

  Serial.begin(115200);
  delay(1000);

  // SAFE STARTUP: gates forced LOW before constructing anything.
  forceAllGatesLow();
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  g_currentSense.seed();

  const float *phases = cfg.ccwDefault ? PHASES_CCW : PHASES_CW;
  g_controller =
      new PhaseController(PWM_PINS, phases, INITIAL_DUTY_CYCLES, NUM_CHANNELS);
  // Explicit non-zero starting frequency -- begin(0.0f) divides by zero inside
  // setGlobalFrequency() and permanently corrupts commutation timing.
  g_controller->begin(cfg.driveFreq);
  g_controller->initCarrierPWM(CARRIER_PINS, PWM_FREQ,
                               INITIAL_CARRIER_DUTY_CYCLES);
  allCoilsOff();

  g_seq = new JsonPhaseSequencer(g_controller);

  SPIFFS.begin(true);

  g_phase = ARMING;
  g_phaseStart = millis();
  g_lastAdcUs = micros();
  g_lastControlUs = g_lastAdcUs;
  Serial.printf("balanced_experiment: arming %lums, then loading %s (pi=%d)\n",
                ARM_MS, cfg.jsonFile, cfg.piEnabled ? 1 : 0);
}

void experimentLoop() {
  g_controller->run();
  unsigned long now = millis();

  // ADC sampling paced separately (ESP32 ADC mux/S&H settling requirement).
  unsigned long nowUs = micros();
  float dtAdcMs = (float)(nowUs - g_lastAdcUs) / 1000.0f;
  if (dtAdcMs >= ADC_SAMPLE_MS) {
    g_currentSense.update(dtAdcMs);
    g_lastAdcUs = nowUs;
  }

  // Real per-loop interval for the control computation (KI/KD rate-scaling).
  float controlDtMs = (float)(nowUs - g_lastControlUs) / 1000.0f;
  if (controlDtMs <= 0.0f)
    controlDtMs = 0.001f;
  g_lastControlUs = nowUs;

  if (g_phase == RUNNING)
    g_seq->run();

  switch (g_phase) {
  case ARMING:
    allCoilsOff();
    digitalWrite(LED_PIN, (now / 150) & 1); // fast blink = arming/settling
    g_currentSense.recalibrateZero();       // coils confirmed off here
    if (now - g_phaseStart >= ARM_MS) {
      const float *phases = g_cfg.ccwDefault ? PHASES_CCW : PHASES_CW;
      bool ok = g_seq->loadFromJsonFile(g_cfg.jsonFile, 25, g_cfg.driveFreq,
                                        INITIAL_DUTY_CYCLES, phases);
      if (!ok) {
        Serial.printf("ARMED -> FAILED to load %s -- staying latched off\n",
                      g_cfg.jsonFile);
        g_phase = DONE;
      } else {
        g_balance.reset(START_DUTY);
        g_seq->start();
        g_phase = RUNNING;
        Serial.println("ARMED -> running experiment");
      }
      g_phaseStart = now;
    }
    break;

  case RUNNING: {
    // The sequencer's commanded carrier is the per-channel ceiling. In PI mode
    // the balance loop redistributes duty beneath it; in passthrough mode the
    // sequencer's write to the controller stands and we only mirror it for
    // telemetry.
    float ceiling[NUM_CHANNELS];
    for (int i = 0; i < NUM_CHANNELS; i++)
      ceiling[i] = g_seq->getCommandedCarrier(i);

    if (g_cfg.piEnabled) {
      g_balance.step(g_currentSense.i_meas, controlDtMs, ceiling, g_dutyOut);
      for (int i = 0; i < NUM_CHANNELS; i++)
        g_controller->setCarrierDutyCycle(i, g_dutyOut[i]);
    } else {
      for (int i = 0; i < NUM_CHANNELS; i++)
        g_dutyOut[i] = g_controller->getCarrierDutyCycle(i);
    }

    for (int i = 0; i < NUM_CHANNELS; i++) {
      if (g_currentSense.i_meas[i] > g_cfg.iSafetyMax) {
        allCoilsOff();
        g_phase = DONE;
        Serial.printf("SAFETY: channel %d overcurrent (%.2fA) -- latching off\n",
                      i, g_currentSense.i_meas[i]);
        break;
      }
    }
    if (g_phase != RUNNING)
      break;

    if (g_seq->isDone()) {
      allCoilsOff();
      g_phase = DONE;
      Serial.println("experiment complete -> DONE (coils latched OFF)");
    }
    break;
  }

  case DONE:
    allCoilsOff();
    digitalWrite(LED_PIN, (now / 1000) & 1); // slow heartbeat = latched off
    break;
  }

  if (now - g_lastTelemetryMs >= TELEMETRY_MS) {
    g_lastTelemetryMs = now;
    emitTelemetry(now);
  }
}
