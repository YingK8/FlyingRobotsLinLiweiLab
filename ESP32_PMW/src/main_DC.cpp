// Straight DC drive — all PWM and carrier pins held HIGH (no PWM, no carrier,
// no phase toggling). Each H-bridge is full-on in one direction, so a constant
// DC current flows through every coil. Used for DC current-sense calibration
// (supply current == coil current, CS reads a steady value).
//
// WARNING: this is continuous full drive. With the resonant cap in circuit the
// coil is just R at DC, so current = V_supply / R_coil (~3.2 ohm) — limit the
// bench supply current/voltage accordingly before powering.

#include <Arduino.h>

const int NUM_CHANNELS = 4;

const gpio_num_t PWM_PINS[NUM_CHANNELS] = {GPIO_NUM_32, GPIO_NUM_23,
                                           GPIO_NUM_27, GPIO_NUM_25};
const gpio_num_t CARRIER_PINS[NUM_CHANNELS] = {GPIO_NUM_33, GPIO_NUM_13,
                                               GPIO_NUM_14, GPIO_NUM_26};

const int LED_PIN = 2;

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
