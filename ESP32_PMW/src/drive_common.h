#pragma once

#include <Arduino.h>
#include <SPIFFS.h>

#include "JsonPwmSequencer.h"
#include "PwmController.h"
#include "constants.h"
#include "reset_button.h"
#include "safety_startup.h"
#include "telemetry.h"

static const float PHASES_CW[NUM_CHANNELS] = {270.0f, 90.0f, 180.0f, 0.0f};
static const float PHASES_CCW[NUM_CHANNELS] = {90.0f, 270.0f, 180.0f, 0.0f};
static const float INITIAL_DUTY[NUM_CHANNELS] = {50.0f, 50.0f, 50.0f, 50.0f};
static const float CARRIER_ZERO[NUM_CHANNELS] = {0.0f, 0.0f, 0.0f, 0.0f};

// VNH5019 CS gain, A per V -- per-board calibration (shared across experiments).
static const float SENS[NUM_CHANNELS] = {15.26f, 15.28f, 15.57f, 15.34f};

// Per-channel series resonance, from `coil_phase.py --measure`. The robot responds to the
// phase of the coil CURRENT, which these two numbers per channel fully determine:
//
//     theta_k(f) = atan( Q_k * (f/f0_k - f0_k/f) )
//
// `PwmController::setPhaseTrim` subtracts that from the command so the four currents come
// out level. See `controller/control/theory.md` 22.
//
// ALL ZERO / Q ZERO = UNCALIBRATED, and the trim stays off. That is deliberate: a guessed
// trim rotates the field by a number nobody measured, which is worse than no trim at all.
// Fill these in only from a probe sweep on the bank actually fitted -- f0 goes as C^-1/2,
// so they are void the moment the capacitor bank changes (it went 400 -> 800 uF on
// 2026-09-01, which is what invalidated every fitted constant before that date).
static const float COIL_F0_HZ[NUM_CHANNELS] = {0.0f, 0.0f, 0.0f, 0.0f};
static const float COIL_Q[NUM_CHANNELS] = {0.0f, 0.0f, 0.0f, 0.0f};

// Boot: serial + force every gate LOW before any driver exists, LED on. Call
// first in setup(), before ctl.begin(), so the coils can't glitch on and the
// ADC zero (captured by enableCurrentSense) is taken against a true-off baseline.
inline void driveBoot() {
  Serial.begin(SERIAL_BAUD);
  delay(1000);
  forceAllGatesLow();
  initResetButton();
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH); // active indicator; goes LOW when blocked

  if (!SPIFFS.begin(/*formatOnFail*/ false))
    Serial.println("[driveBoot] SPIFFS mount FAILED -- run `pio run -t uploadfs` to update json changes");
}

// Shared 2 Hz telemetry line, same field layout the ai/ log parsers expect:
// "t=.. freq=.. | I[A]: .. | duty[%]: .. | spread=.. bal=.. trip=..".
inline void driveTelemetry(PwmController &c) {
  checkResetButton(); // poll every loop (before the 500 ms print throttle below)

  static unsigned long last = 0;
  unsigned long now = millis();
  if (now - last < 500)
    return;
  last = now;

  const float *im = c.measuredCurrents();
  float duty[NUM_CHANNELS];
  for (int i = 0; i < NUM_CHANNELS; i++)
    duty[i] = c.getCarrierDutyCycle(i);

  float imin = im ? im[0] : 0.0f, imax = im ? im[0] : 0.0f;
  if (im)
    for (int i = 1; i < NUM_CHANNELS; i++) {
      if (im[i] < imin) imin = im[i];
      if (im[i] > imax) imax = im[i];
    }

  Serial.printf("t=%lu freq=%.1f | ", now, c.getFrequency());
  if (im)
    printCurrentAndDuty(im, duty);
  Serial.printf(" | spread=%.3f bal=%d trip=%d\n", imax - imin,
                c.balanceActive() ? 1 : 0, c.overcurrentTripped() ? 1 : 0);
}
