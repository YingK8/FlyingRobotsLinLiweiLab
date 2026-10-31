#pragma once

#include <Arduino.h>
#include <SPIFFS.h>

#include "JsonPwmSequencer.h"
#include "PwmController.h"
#include "SyncLedPin.h"
#include "constants.h"
#include "reset_button.h"
#include "safety_startup.h"
#include "telemetry.h"

using sync_led::led;

// A/C and B/D are opposite: reversing direction swaps B/D
static const float PHASES_CW[NUM_CHANNELS] = {0.0f, 90.0f, 180.0f, 270.0f};
static const float PHASES_CCW[NUM_CHANNELS] = {0.0f, 270.0f, 180.0f, 90.0f};
static const float INITIAL_DUTY[NUM_CHANNELS] = {50.0f, 50.0f, 50.0f, 50.0f};
static const float CARRIER_ZERO[NUM_CHANNELS] = {0.0f, 0.0f, 0.0f, 0.0f};

static const float SENS[NUM_CHANNELS] = {15.26f, 15.28f, 15.57f, 15.34f};

// all zero = uncalibrated and the trim stays off; void when the capacitor bank changes
static const float COIL_F0_HZ[NUM_CHANNELS] = {0.0f, 0.0f, 0.0f, 0.0f};
static const float COIL_Q[NUM_CHANNELS] = {0.0f, 0.0f, 0.0f, 0.0f};

// call first in setup(), before ctl.begin(), so no gate glitches on
inline void driveBoot(bool activeLed = true) {
  // first, before the delay: the carrier pins float from reset until this line, and a
  // floating carrier drove all four bridges to full on 2026-10-30
  forceAllGatesLow();
  // 1 kB RX buffer: an overflow is silent and the coils hold their last command
  Serial.setRxBufferSize(1024);
  Serial.begin(SERIAL_BAUD);
  delay(1000);
  initResetButton();
  led().attachTo(LED_PIN);
  if (activeLed) led().holdOn();

  if (!SPIFFS.begin(false))
    Serial.println("[driveBoot] SPIFFS mount FAILED -- run `pio run -t uploadfs` to update json changes");
}

// telemetry field layout is protocol: the ai/ log parsers read it by name
inline void driveTelemetry(PwmController &c) {
  // call every loop or a button press is never seen
  checkResetButton();
  led().advance();

  static unsigned long last = 0;
  unsigned long now = millis();
  if (now - last < 500)
    return;
  last = now;

  const float *im = c.measuredCurrents();
  float duty[NUM_CHANNELS];
  for (int i = 0; i < NUM_CHANNELS; i++)
    duty[i] = c.getCarrierDutyCycle(i);

  float imin = im ? im[0] : 0.0f;
  float imax = im ? im[0] : 0.0f;
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
