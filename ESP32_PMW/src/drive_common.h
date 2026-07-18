#pragma once

#include <Arduino.h>
#include <SPIFFS.h>

#include "JsonPhaseSequencer.h"
#include "PwmController.h"
#include "constants.h"
#include "safety_startup.h"
#include "telemetry.h"

// Shared DATA + two small utilities for the per-experiment mains. The mains
// instantiate PwmController + JsonPhaseSequencer directly and call their methods
// in setup() (begin / initCarrierPWM / enableCurrentSense / enableCurrentBalance
// / loadFromJsonFile / start) -- there is no construction wrapper to hide that.
// This header only holds the things genuinely shared across experiments: the
// const pin-phase/calibration arrays, the boot sequence, and the telemetry line.
// There is NO arming gate: each main calls seq.start() at the end of setup().
//
// The PI current-balance loop is folded into PwmController (opt-in): lift
// experiments call ctl.enableCurrentBalance(); characterization sweeps
// (comp/coupling/dc/ceiling) skip it so their commanded carriers pass through.

// Project rotation conventions (coil order A,B,C,D). A JSON setDirection in the
// schedule overrides these at runtime; they seed the controller at construction.
static const float PHASES_CW[NUM_CHANNELS] = {270.0f, 90.0f, 180.0f, 0.0f};
static const float PHASES_CCW[NUM_CHANNELS] = {90.0f, 270.0f, 180.0f, 0.0f};
static const float INITIAL_DUTY[NUM_CHANNELS] = {50.0f, 50.0f, 50.0f, 50.0f};
static const float CARRIER_ZERO[NUM_CHANNELS] = {0.0f, 0.0f, 0.0f, 0.0f};

// VNH5019 CS gain, A per V -- per-board calibration (shared across experiments).
static const float SENS[NUM_CHANNELS] = {15.26f, 15.28f, 15.57f, 15.34f};

// Boot: serial + force every gate LOW before any driver exists, LED off. Call
// first in setup(), before ctl.begin(), so the coils can't glitch on and the
// ADC zero (captured by enableCurrentSense) is taken against a true-off baseline.
inline void driveBoot() {
  Serial.begin(115200);
  delay(1000);
  forceAllGatesLow();
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  // Mount the SPIFFS partition that holds the JSON schedule. Without this,
  // loadFromJsonFile()'s SPIFFS.open() fails, the queue stays empty, and the
  // coils just idle at DC -- so a mount failure must be loud, not silent.
  // formatOnFail=false: never auto-wipe an uploaded schedule on a transient
  // mount hiccup; a genuinely empty partition just needs `pio run -t uploadfs`.
  if (!SPIFFS.begin(/*formatOnFail*/ false))
    Serial.println("[driveBoot] SPIFFS mount FAILED -- run `pio run -t uploadfs`");
}

// Shared 2 Hz telemetry line, same field layout the ai/ log parsers expect:
// "t=.. freq=.. | I[A]: .. | duty[%]: .. | spread=.. bal=.. trip=..".
inline void driveTelemetry(PwmController &c) {
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
