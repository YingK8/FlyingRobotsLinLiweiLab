// Bare CS readout: does a PHASE pin's logic level leak into the CS nets?
// (cs_cal 2026-11-03_133409_B: B and D read ~1.3 V whenever GPIO25 sat HIGH, carriers 0.)
// No PwmController, no ISR, no zero subtraction, no SENS: raw mV only.
// NOT a drive path: the carriers stay LOW from forceAllGatesLow() and are never touched,
// so nothing energises a coil even with the PSU on.
//
// Serial (921600, any monitor):  H<ch> / L<ch>  set phase pin ch (0..3 = A..D) HIGH / LOW
//                                S              all phase pins LOW
// Prints every 200 ms:  ph=ABCD levels | mV A B C D (mean of 64 reads each)
#include "drive_common.h"

static int level[NUM_CHANNELS] = {0, 0, 0, 0};

void setup() {
  driveBoot();  // all gates LOW, LED, button
  analogReadResolution(12);
  for (int i = 0; i < NUM_CHANNELS; i++) analogSetPinAttenuation(ADC_PINS[i], ADC_11db);
}

void loop() {
  checkResetButton();
  led().advance();

  if (Serial.available()) {
    String c = Serial.readStringUntil('\n');
    c.trim();
    int ch = c.substring(1).toInt();
    if (c == "S") {
      for (int i = 0; i < NUM_CHANNELS; i++) level[i] = 0;
    } else if ((c[0] == 'H' || c[0] == 'L') && ch >= 0 && ch < NUM_CHANNELS) {
      level[ch] = c[0] == 'H';
    }
    for (int i = 0; i < NUM_CHANNELS; i++) digitalWrite(PWM_PINS[i], level[i]);
  }

  static unsigned long last = 0;
  if (millis() - last < 200) return;
  last = millis();
  float mv[NUM_CHANNELS];
  for (int i = 0; i < NUM_CHANNELS; i++) {
    analogReadMilliVolts(ADC_PINS[i]);  // mux settling, as in current_sense.cpp
    uint32_t s = 0;
    for (int k = 0; k < 64; k++) s += analogReadMilliVolts(ADC_PINS[i]);
    mv[i] = s / 64.0f;
  }
  Serial.printf("ph=%d%d%d%d | mV %6.1f %6.1f %6.1f %6.1f\n", level[0], level[1], level[2],
                level[3], mv[0], mv[1], mv[2], mv[3]);
}
