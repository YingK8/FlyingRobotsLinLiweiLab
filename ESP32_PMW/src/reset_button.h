#pragma once

#include "constants.h"
#include "safety_startup.h"
#include <Arduino.h>

// Momentary reset button on RESET_BUTTON_PIN (GPIO14), wired to drive the pin to
// 3V3 when pressed (active HIGH). The internal pulldown holds it LOW at rest, so
// no external resistor is needed. A confirmed press instantly cuts every gate
// driver and reboots the MCU, which re-runs setup() from a clean, gates-low boot
// (see driveBoot()) -- i.e. it restarts the experiment. Shared via drive_common.h.

// Configure the pin. Call once in setup() (folded into driveBoot()).
inline void initResetButton() {
  pinMode(RESET_BUTTON_PIN, INPUT_PULLDOWN); // idle LOW; HIGH only while pressed
}

// Poll every loop iteration (folded into driveTelemetry()). Debounces the press:
// the pin must read HIGH continuously for RESET_DEBOUNCE_MS before we act, so
// contact bounce or noise can't trigger a spurious reset. On a confirmed press
// it forces the coils off and reboots immediately -- setup() runs again and the
// experiment restarts from a clean boot.
inline void checkResetButton() {
  static const unsigned long RESET_DEBOUNCE_MS = 30;
  static unsigned long highSince = 0; // 0 = not currently held high

  if (digitalRead(RESET_BUTTON_PIN) == HIGH) {
    unsigned long now = millis();
    if (highSince == 0)
      highSince = (now == 0) ? 1 : now; // avoid the 0 "not held" sentinel
    else if (now - highSince >= RESET_DEBOUNCE_MS) {
      forceAllGatesLow(); // instant hard-off before the reboot
      Serial.println("[reset] button pressed -- restarting");
      Serial.flush();
      ESP.restart();
    }
  } else {
    highSince = 0; // released before debounce elapsed -- restart the timer
  }
}
