// Straight DC drive — all PWM and carrier pins held HIGH (no PWM, no carrier,
// no phase toggling). Each H-bridge is full-on in one direction, so a constant
// DC current flows through every coil. Used for DC current-sense calibration
// (supply current == coil current, CS reads a steady value).
//
// WARNING: this is continuous full drive. With the resonant cap in circuit the
// coil is just R at DC, so current = V_supply / R_coil (~3.2 ohm) — limit the
// bench supply current/voltage accordingly before powering.

#include <Arduino.h>
#include "constants.h"

void setup() {
  Serial.begin(115200);
  delay(1000);

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH);  // solid on = DC drive active

  // Drive every PWM and carrier pin to a constant HIGH (straight DC).
  for (int i = 0; i < NUM_CHANNELS; i++) {
    pinMode(PWM_PINS[i], OUTPUT);
    pinMode(CARRIER_PINS[i], OUTPUT);
    digitalWrite(PWM_PINS[i], HIGH);
    digitalWrite(CARRIER_PINS[i], HIGH);
  }

  Serial.println("DC drive: all PWM + carrier pins held HIGH");
}

void loop() {
}
