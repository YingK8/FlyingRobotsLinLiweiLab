// Coupling / drive-imbalance test rig — AUTOMATED, BOUNDED, SAFE sweep.
//
// Purpose: quantify magnetic coupling between coils. Drive ONE coil at a time
// at a fixed frequency and record all four current-sense channels on the scope;
// the current that appears on the *undriven* coils is the mutually-coupled
// current, so |CS_other| / |CS_driven| gives the coupling ratio per pair.
//
// Sequence (constant 190 Hz, 100% carrier when active):
//   ARM (coils OFF) -> solo A,B,C,D -> the six PAIRS (A+B,A+C,A+D,B+C,B+D,C+D)
//   -> ALL four -> (repeat NUM_LOOPS times) -> STOP (OFF), gap between each.
//
// The solos give each coil's uncoupled current; in each pair, a coil's current
// shift from its solo value is caused by the mutual EMF of its partner, so the
// solo+pair currents solve for the mutual inductance M_ij of every pair
// (analysis: tools/analyze_coupling.py). ALL is kept as a cross-check.
//
// Hardware note: common-ground + no USB isolator means we can't talk to the
// ESP32 while the coils are powered, so this runs a fixed timed sequence on boot
// (no serial commands needed during the run). The scope burst-detector keys off
// which channels are active, so the recording start need not be time-aligned.
//
// SAFE STARTUP:  every gate-driver input is force-held LOW before anything else,
// and the carriers stay at 0 through a guaranteed ARM_MS off-period, so the coils
// cannot glitch on during boot / LEDC attach — even if the coil supply is already
// live when the board resets.
// SAFE SHUTDOWN: the sweep is BOUNDED to NUM_LOOPS. When it finishes it latches
// every coil OFF permanently (DONE state, slow heartbeat) instead of looping
// forever — so an unattended board ends de-energized and the coils never cook.
//
// Workflow: flash over USB with the coil supply OFF, disconnect USB, power the
// coils, then capture one PicoScope recording of at least the printed duration.

#include <Arduino.h>
#include "PhaseController.h"
#include "constants.h"

const float INITIAL_PHASES[NUM_CHANNELS] = {270.0, 90.0, 180.0, 0.0};
const float INITIAL_DUTY_CYCLES[NUM_CHANNELS] = {50.0, 50.0, 50.0, 50.0};

const float DRIVE_FREQ = 190.0f;     // coil drive (commutation) frequency (Hz)
const float ON_DUTY = 100.0f;        // carrier duty when a channel is active
const float INITIAL_CARRIER_DUTY_CYCLES[NUM_CHANNELS] = {0, 0, 0, 0};

// Timing.
const unsigned long ARM_MS = 3000;      // guaranteed coils-OFF period after boot
const unsigned long ACTIVE_MS = 3000;   // drive each segment this long
const unsigned long GAP_MS = 2000;      // off-gap between segments (burst separator)
const int NUM_LOOPS = 1;                // BOUNDED: run this many full sweeps, then STOP

// Active set per segment, as a channel bitmask (bit i = channel i).
// 4 solos + 6 pairs + ALL. Pairs are what quantify M_ij (see header).
const uint8_t SEQ[] = {
    0b0001, 0b0010, 0b0100, 0b1000,             // solo A, B, C, D
    0b0011, 0b0101, 0b1001, 0b0110, 0b1010, 0b1100, // A+B,A+C,A+D,B+C,B+D,C+D
    0b1111};                                    // ALL
const int N_SEQ = sizeof(SEQ) / sizeof(SEQ[0]);

PhaseController *controller;
uint8_t last_mask = 0xFF;  // force first apply

enum Phase { ARMING, SWEEPING, DONE };
Phase phase = ARMING;
unsigned long sweep_start_ms = 0;

void allCoilsOff() {
  for (int i = 0; i < NUM_CHANNELS; i++) controller->setCarrierDutyCycle(i, 0.0f);
}

void applyMask(uint8_t mask) {
  for (int i = 0; i < NUM_CHANNELS; i++)
    controller->setCarrierDutyCycle(i, (mask >> i) & 1 ? ON_DUTY : 0.0f);
  digitalWrite(LED_PIN, mask ? HIGH : LOW);
}

void setup() {
  Serial.begin(115200);

  // SAFE STARTUP: hold every gate-driver input LOW before constructing anything,
  // so the coils cannot energize during boot / LEDC attach if the supply is live.
  const gpio_num_t ALL_PINS[] = {A_PWM_PIN, B_PWM_PIN, C_PWM_PIN, D_PWM_PIN,
                                 A_CARRIER_PIN, B_CARRIER_PIN, C_CARRIER_PIN,
                                 D_CARRIER_PIN};
  for (gpio_num_t p : ALL_PINS) { pinMode(p, OUTPUT); digitalWrite(p, LOW); }
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  delay(1000);

  controller = new PhaseController(PWM_PINS, INITIAL_PHASES, INITIAL_DUTY_CYCLES,
                                   NUM_CHANNELS);
  controller->begin();
  controller->initCarrierPWM(CARRIER_PINS, PWM_FREQ, INITIAL_CARRIER_DUTY_CYCLES);
  controller->setGlobalFrequency(DRIVE_FREQ);
  allCoilsOff();  // explicit: carriers latched at 0 after LEDC attach

  unsigned long total =
      ARM_MS + (unsigned long)NUM_LOOPS * N_SEQ * (GAP_MS + ACTIVE_MS);
  Serial.printf("coupling sweep (SAFE): arm %lums, %d loops x %d seg "
                "(%lums active / %lums gap), drive %.0fHz. Total ~%lums — "
                "record >= %lums.\n",
                ARM_MS, NUM_LOOPS, N_SEQ, ACTIVE_MS, GAP_MS, DRIVE_FREQ, total,
                total + GAP_MS);
}

void loop() {
  controller->run();
  unsigned long now = millis();

  switch (phase) {
    case ARMING:
      // Coils guaranteed OFF; fast LED blink = arming/settling.
      allCoilsOff();
      digitalWrite(LED_PIN, (now / 150) & 1);
      if (now >= ARM_MS) {
        phase = SWEEPING;
        sweep_start_ms = now;
        last_mask = 0xFF;   // force re-apply on entry
        Serial.println("ARMED -> sweeping");
      }
      break;

    case SWEEPING: {
      unsigned long elapsed = now - sweep_start_ms;
      const unsigned long slot = GAP_MS + ACTIVE_MS;
      const unsigned long loop_len = (unsigned long)N_SEQ * slot;
      if (elapsed >= (unsigned long)NUM_LOOPS * loop_len) {
        applyMask(0);       // ensure OFF before latching
        phase = DONE;
        Serial.println("sweep complete -> DONE (coils latched OFF)");
        break;
      }
      unsigned long pos = elapsed % loop_len;
      int idx = pos / slot;
      unsigned long within = pos % slot;
      uint8_t mask = (within < GAP_MS) ? 0 : SEQ[idx];
      if (mask != last_mask) {
        applyMask(mask);
        last_mask = mask;
        Serial.printf("t=%lu  segment %d  mask=0x%X\n", now, idx, mask);
      }
      break;
    }

    case DONE:
      // SAFE SHUTDOWN: all coils latched OFF permanently; slow heartbeat so the
      // board still reads as alive without ever re-energizing.
      allCoilsOff();
      digitalWrite(LED_PIN, (now / 1000) & 1);
      break;
  }
}
