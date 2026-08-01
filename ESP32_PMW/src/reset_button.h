#pragma once

#include "constants.h"
#include "safety_startup.h"
#include <Arduino.h>

// Momentary block/unblock button on RESET_BUTTON_PIN (GPIO14), wired to drive
// the pin to 3V3 when pressed (active HIGH). The internal pulldown holds it LOW
// at rest, so no external resistor is needed. Toggle semantics:
//   press while active  -> BLOCK: every gate forced low, LED off, firmware halts
//   press while blocked -> ESP.restart(): clean gates-low boot, LED back on,
//                          experiment restarts (see driveBoot())
// Shared via drive_common.h. The LED (GPIO2, onboard) is ON while active.

// Wait for one debounced press (HIGH held RESET_DEBOUNCE_MS) then release.
inline void waitForPress() {
  const unsigned long RESET_DEBOUNCE_MS = 30;
  for (;;) {
    while (digitalRead(RESET_BUTTON_PIN) == LOW)
      delay(5);
    unsigned long start = millis();
    while (digitalRead(RESET_BUTTON_PIN) == HIGH)
      delay(5);
    if (millis() - start >= RESET_DEBOUNCE_MS)
      return; // held long enough -- confirmed press, now released
  }
}

// Configure the pin. Call once in setup() (folded into driveBoot()).
inline void initResetButton() {
  pinMode(RESET_BUTTON_PIN, INPUT_PULLDOWN); // idle LOW; HIGH only while pressed
}

// Poll every loop iteration (folded into driveTelemetry()). Debounces the press:
// the pin must read HIGH continuously for RESET_DEBOUNCE_MS before we act. On a
// confirmed press it forces the coils off, turns the LED off, and parks here
// (blocked) until the next confirmed press, which reboots -- setup() runs again
// and the experiment restarts from a clean boot.
inline void checkResetButton() {
  static const unsigned long RESET_DEBOUNCE_MS = 30;
  static unsigned long highSince = 0; // 0 = not currently held high

  if (digitalRead(RESET_BUTTON_PIN) == HIGH) {
    unsigned long now = millis();
    if (highSince == 0)
      highSince = (now == 0) ? 1 : now; // avoid the 0 "not held" sentinel
    else if (now - highSince >= RESET_DEBOUNCE_MS) {
      forceAllGatesLow(); // instant hard-off
      digitalWrite(LED_PIN, LOW);
      Serial.println("[block] button pressed -- gates off, press again to restart");
      while (digitalRead(RESET_BUTTON_PIN) == HIGH)
        delay(5);     // wait out the blocking press
      waitForPress(); // park here until the next press
      Serial.println("[block] button pressed -- restarting");
      Serial.flush();
      ESP.restart();
    }
  } else {
    highSince = 0; // released before debounce elapsed -- restart the timer
  }
}
