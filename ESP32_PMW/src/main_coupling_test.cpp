// Coupling / drive-imbalance test rig — AUTOMATED sweep (no serial needed).
//
// Common-ground + no USB isolator means we can't talk to the ESP32 while the
// coils are powered. So this runs a fixed timed sequence on boot:
//
//   gap(off) -> solo A -> gap -> solo B -> gap -> solo C -> gap -> solo D
//   -> gap -> ALL four -> gap -> (repeat forever)
//
// Flash over USB (coils unpowered), disconnect USB, power the system, then
// capture one long PicoScope recording (>= one full loop). The analyzer
// `pico/coupling_sweep_analyze.py` auto-detects each burst by which channels
// are active, so the recording start time does not need to be aligned.
//
// Each coil draws several amps when driven — the off-gaps also give the coils a
// moment to cool between segments. Constant 190 Hz, 100% carrier when active.

#include <Arduino.h>
#include "PhaseController.h"

const int NUM_CHANNELS = 4;

const gpio_num_t A_PWM_PIN = GPIO_NUM_32;
const gpio_num_t B_PWM_PIN = GPIO_NUM_23;
const gpio_num_t C_PWM_PIN = GPIO_NUM_27;
const gpio_num_t D_PWM_PIN = GPIO_NUM_25;

const gpio_num_t A_CARRIER_PIN = GPIO_NUM_33;
const gpio_num_t B_CARRIER_PIN = GPIO_NUM_13;
const gpio_num_t C_CARRIER_PIN = GPIO_NUM_14;
const gpio_num_t D_CARRIER_PIN = GPIO_NUM_26;

const gpio_num_t PWM_PINS[NUM_CHANNELS] = {A_PWM_PIN, B_PWM_PIN, C_PWM_PIN, D_PWM_PIN};
const gpio_num_t CARRIER_PINS[NUM_CHANNELS] = {A_CARRIER_PIN, B_CARRIER_PIN,
                                              C_CARRIER_PIN, D_CARRIER_PIN};

const float INITIAL_PHASES[NUM_CHANNELS] = {270.0, 90.0, 180.0, 0.0};
const float INITIAL_DUTY_CYCLES[NUM_CHANNELS] = {50.0, 50.0, 50.0, 50.0};

const int PWM_FREQ = 15000;          // carrier frequency (Hz)
const float DRIVE_FREQ = 190.0f;     // coil drive (commutation) frequency (Hz)
const float ON_DUTY = 100.0f;        // carrier duty when a channel is active
const float INITIAL_CARRIER_DUTY_CYCLES[NUM_CHANNELS] = {0, 0, 0, 0};

// Sweep timing.
const unsigned long ACTIVE_MS = 4000;   // drive each segment this long
const unsigned long GAP_MS = 2000;      // off-gap between segments (burst separator)

// Active set per segment, as a channel bitmask (bit i = channel i).
const uint8_t SEQ[] = {0b0001, 0b0010, 0b0100, 0b1000, 0b1111}; // A, B, C, D, ALL
const int N_SEQ = sizeof(SEQ) / sizeof(SEQ[0]);

const int LED_PIN = 2;

PhaseController *controller;
uint8_t last_mask = 0xFF;  // force first apply

void applyMask(uint8_t mask) {
  for (int i = 0; i < NUM_CHANNELS; i++)
    controller->setCarrierDutyCycle(i, (mask >> i) & 1 ? ON_DUTY : 0.0f);
  digitalWrite(LED_PIN, mask ? HIGH : LOW);
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  controller = new PhaseController(PWM_PINS, INITIAL_PHASES, INITIAL_DUTY_CYCLES,
                                   NUM_CHANNELS);
  controller->begin();
  controller->initCarrierPWM(CARRIER_PINS, PWM_FREQ, INITIAL_CARRIER_DUTY_CYCLES);
  controller->setGlobalFrequency(DRIVE_FREQ);

  Serial.printf("coupling sweep: %d segments, %lums active / %lums gap, loop %lums\n",
                N_SEQ, ACTIVE_MS, GAP_MS, N_SEQ * (ACTIVE_MS + GAP_MS));
}

void loop() {
  controller->run();

  // Stateless timeline: each slot is GAP_MS off then ACTIVE_MS driving SEQ[idx].
  const unsigned long slot = GAP_MS + ACTIVE_MS;
  unsigned long pos = millis() % (N_SEQ * slot);
  int idx = pos / slot;
  unsigned long within = pos % slot;
  uint8_t mask = (within < GAP_MS) ? 0 : SEQ[idx];

  if (mask != last_mask) {
    applyMask(mask);
    last_mask = mask;
    Serial.printf("t=%lu  segment %d  mask=0x%X\n", millis(), idx, mask);
  }
}
