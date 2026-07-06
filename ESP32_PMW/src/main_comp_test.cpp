// Coupling-compensation A/B test v2 — FEEDFORWARD trims, scope-in-the-loop.
//
// v1 tried an on-board closed loop on the CS->ADC taps, but the MAP phase
// proved those taps are no longer connected (solo drive moved the ADC reads by
// ~0 mV while the scope saw full 170 mV bursts). So the ESP32 is blind to its
// own currents; the compensation is instead a per-channel FEEDFORWARD duty trim
// computed offline from the previous PicoScope capture, iterated:
//   run -> measure per-channel CS RMS on scope -> trim_i *= mean/rms_i ->
//   re-flash -> repeat until the spread stops shrinking.
//
// Each run is one self-contained A/B capture (constant 190 Hz rotating field,
// phases A/B/C/D = 270/90/180/0):
//   ARM (coils OFF) -> BASELINE: all four at equal BASE_DUTY (uncompensated)
//   -> gap -> TRIMMED: all four at TRIM[i]*BASE_DUTY (compensated)
//   -> DONE (coils latched OFF, slow heartbeat).
//
// Baseline spread vs trimmed spread in the same capture = how much the trims
// cancel the coupling-driven current redistribution.
//
// TRIM values are in the FIRMWARE channel frame (scope frame has B<->D swapped:
// firmware B -> scope D, firmware D -> scope B; reconfirmed by v1's solos).
// Iteration history (all-4 CS RMS, firmware frame, floor-corrected):
//   iter 0 (equal duty, contaminated by v1 duty slam): A=63 B=115 C=115 D=94
//   iter 1 (comp_ff_iter1.csv): baseline A=107 B=201 C=199 D=164 (max/min 1.88)
//     trims {1.444,0.795,0.793,0.968} -> A=203 B=155 C=178 D=168 (max/min 1.31)
//     A overshot low->high: plant gain ~1.7x super-linear -> damped update ^0.6.
//
// Safety: no ADC feedback exists, so protection is structural — SAFE STARTUP
// gate hold, bounded segments, trims clamped in code, mean trim normalized to
// 1.0 (keeps total draw at the measured ~6 A, inside the 7 V / 10 A supply).

#include <Arduino.h>
#include "PhaseController.h"
#include "constants.h"

// ---- drive pins: PWM_PINS / CARRIER_PINS / NUM_CHANNELS come from constants.h ----
// CCW rotation map (matches main_tilt 2026-07-04 CCW config). Trims are
// direction-specific: coupled power transfer ~ sin(dphi) flips with rotation.
const float INITIAL_PHASES[NUM_CHANNELS] = {90.0, 270.0, 180.0, 0.0};
const float INITIAL_DUTY_CYCLES[NUM_CHANNELS] = {50.0, 50.0, 50.0, 50.0};

const float DRIVE_FREQ = 190.0f;  // rotating-field rate (Hz)
const float INITIAL_CARRIER_DUTY_CYCLES[NUM_CHANNELS] = {0, 0, 0, 0};

// ---- the compensation (firmware frame A,B,C,D) ----
const float BASE_DUTY = 50.0f;
// CCW iter 2: (mean/rms)^0.6 from comp_ccw_iter1.csv trimmed segment
// (iter1 {0.850,1.288,0.975,0.887}: spread 1.99 -> 1.14).
// CCW baseline fw frame A=302 B=151 C=240 D=281 — redistribution flipped vs
// CW as predicted by sin(dphi) reversal (CW had A=107 lowest).
// (Converged CW trims were {1.337, 0.866, 0.794, 1.003}, comp_ff_iter2.csv.)
const float TRIM[NUM_CHANNELS] = {0.839f, 1.331f, 0.982f, 0.848f};
const float TRIM_MIN = 0.30f, TRIM_MAX = 1.80f;  // keep duty in [15%, 90%]

// ---- timing (all bounded) ----
const unsigned long ARM_MS      = 3000;
const unsigned long BASELINE_MS = 10000;
const unsigned long GAP_MS      = 2000;
const unsigned long TRIMMED_MS  = 10000;

PhaseController *controller;

enum Phase { ARMING, BASELINE, MID_GAP, TRIMMED, DONE };
Phase phase = ARMING;
unsigned long phase_start = 0;

static inline float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

void allCoilsOff() {
  for (int i = 0; i < NUM_CHANNELS; i++) controller->setCarrierDutyCycle(i, 0.0f);
}

void applyDuties(bool trimmed) {
  for (int i = 0; i < NUM_CHANNELS; i++) {
    float d = trimmed ? BASE_DUTY * clampf(TRIM[i], TRIM_MIN, TRIM_MAX)
                      : BASE_DUTY;
    controller->setCarrierDutyCycle(i, d);
  }
  digitalWrite(LED_PIN, HIGH);
}

void enterPhase(Phase p, unsigned long now, const char *name) {
  phase = p;
  phase_start = now;
  Serial.printf("t=%lu -> %s\n", now, name);
}

void setup() {
  Serial.begin(115200);

  // SAFE STARTUP: hold every gate-driver input LOW before constructing anything,
  // so the coils cannot energize during boot / LEDC attach if the supply is live.
  for (int i = 0; i < NUM_CHANNELS; i++) {
    pinMode(PWM_PINS[i], OUTPUT);
    digitalWrite(PWM_PINS[i], LOW);
    pinMode(CARRIER_PINS[i], OUTPUT);
    digitalWrite(CARRIER_PINS[i], LOW);
  }
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  delay(1000);

  controller = new PhaseController(PWM_PINS, INITIAL_PHASES, INITIAL_DUTY_CYCLES,
                                   NUM_CHANNELS);
  controller->begin();
  controller->initCarrierPWM(CARRIER_PINS, PWM_FREQ, INITIAL_CARRIER_DUTY_CYCLES);
  controller->setGlobalFrequency(DRIVE_FREQ);
  allCoilsOff();

  Serial.printf("comp A/B v2 (feedforward): baseline %lums @ %.0f%%, trimmed "
                "%lums @ %.1f/%.1f/%.1f/%.1f%%. Total ~%lums.\n",
                BASELINE_MS, BASE_DUTY, TRIMMED_MS,
                BASE_DUTY * TRIM[0], BASE_DUTY * TRIM[1],
                BASE_DUTY * TRIM[2], BASE_DUTY * TRIM[3],
                ARM_MS + BASELINE_MS + GAP_MS + TRIMMED_MS);
}

void loop() {
  controller->run();
  unsigned long now = millis();

  switch (phase) {
    case ARMING:
      allCoilsOff();
      digitalWrite(LED_PIN, (now / 150) & 1);  // fast blink = arming
      if (now >= ARM_MS) {
        applyDuties(false);
        enterPhase(BASELINE, now, "BASELINE (equal duty)");
      }
      break;

    case BASELINE:
      if (now - phase_start >= BASELINE_MS) {
        allCoilsOff();
        digitalWrite(LED_PIN, LOW);
        enterPhase(MID_GAP, now, "GAP");
      }
      break;

    case MID_GAP:
      if (now - phase_start >= GAP_MS) {
        applyDuties(true);
        enterPhase(TRIMMED, now, "TRIMMED (compensated)");
      }
      break;

    case TRIMMED:
      if (now - phase_start >= TRIMMED_MS) {
        allCoilsOff();
        enterPhase(DONE, now, "DONE (coils latched OFF)");
      }
      break;

    case DONE:
      // SAFE SHUTDOWN: coils latched OFF permanently; slow heartbeat = alive.
      allCoilsOff();
      digitalWrite(LED_PIN, (now / 1000) & 1);
      break;
  }
}
