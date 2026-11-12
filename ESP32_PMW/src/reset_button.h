#pragma once

#include "SyncLedPin.h"
#include "constants.h"
#include "safety_startup.h"
#include <Arduino.h>

// press while active blocks every gate; press again reboots
inline void waitForPress() {
  const unsigned long RESET_DEBOUNCE_MS = 30;
  for (;;) {
    while (digitalRead(RESET_BUTTON_PIN) == LOW)
      delay(5);
    unsigned long start = millis();
    while (digitalRead(RESET_BUTTON_PIN) == HIGH)
      delay(5);
    if (millis() - start >= RESET_DEBOUNCE_MS)
      return;
  }
}

inline void initResetButton() {
  pinMode(RESET_BUTTON_PIN, INPUT_PULLDOWN);
}

// call every loop or a press is never seen
inline void checkResetButton() {
  static const unsigned long RESET_DEBOUNCE_MS = 30;
  static unsigned long highSince = 0;

  if (digitalRead(RESET_BUTTON_PIN) == HIGH) {
    unsigned long now = millis();
    if (highSince == 0)
      highSince = (now == 0) ? 1 : now;
    else if (now - highSince >= RESET_DEBOUNCE_MS) {
      forceAllGatesLow();
      sync_led::led().holdOff();
      Serial.println("[block] button pressed -- gates off, press again to restart");
      while (digitalRead(RESET_BUTTON_PIN) == HIGH)
        delay(5);
      waitForPress();
      Serial.println("[block] button pressed -- restarting");
      Serial.flush();
      ESP.restart();
    }
  } else {
    highSince = 0;
  }
}
