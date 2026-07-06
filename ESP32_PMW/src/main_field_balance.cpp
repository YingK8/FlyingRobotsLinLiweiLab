// Low-level current-balance feedback controller for a balanced rotating field.
//
// Four coils driven at a constant 190 Hz form a 2-axis rotating field:
//   axis 1 = opposite pair A & C,  axis 2 = opposite pair B & D.
// A balanced (circular) rotating field needs all four coil-current amplitudes
// equal. This closes a per-channel integral loop on the measured coil current
// (VNH5019 CS shunt -> ESP32 ADC), trimming each channel's carrier duty so every
// coil holds the same target current. That keeps each opposite pair matched
// (A==C, B==D) and the field balanced, automatically compensating per-channel /
// per-board drive differences and drift.
//
// Per-coil CS sense (CS pin -> pulldown shunt -> ADC1, input-only pins):
//   A=GPIO36(VP)  B=GPIO39(VN)  C=GPIO34  D=GPIO35
// (32/33 are taken by A's PWM/carrier; ADC2 is unusable with Wi-Fi.)
//
// SAFETY: per-channel hard current cap backs the duty off fast on overcurrent.

#include <Arduino.h>
#include "PhaseController.h"
#include "constants.h"

const float INITIAL_PHASES[NUM_CHANNELS] = {270.0, 90.0, 180.0, 0.0};
const float INITIAL_DUTY_CYCLES[NUM_CHANNELS] = {50.0, 50.0, 50.0, 50.0};

const float DRIVE_FREQ = 190.0f;  // coil drive / rotating-field rate (Hz)

// ---- current sense: ADC pin + calibration per coil (index 0..3 = A,B,C,D) ----
// SENS/OFFSET are from the multipoint CS calibration. RE-CALIBRATE for the new
// per-coil shunts if they differ from the ~2.5 kOhm used before.
const int   ADC_PINS[NUM_CHANNELS]  = {36, 39, 34, 35};            // A,B,C,D
const float SENS[NUM_CHANNELS]      = {15.26, 15.28, 15.57, 15.34}; // A per V
const float OFFSET_MV[NUM_CHANNELS] = {0.0, -40.4, 0.0, -40.8};     // CS zero-current (mV)

// ---- controller tuning ----
const float TARGET_A   = 5.0f;    // current setpoint per coil (A)
const float KI         = 0.30f;   // integral gain: %duty per (A * update)
const float DUTY_MIN   = 5.0f;    // keep a little drive so CS is observable
const float DUTY_MAX   = 99.0f;   // stay in LEDC PWM mode (100% = park-high path)
const float I_MAX_A    = 12.0f;   // hard per-channel safety cap (A)
const float ADC_ALPHA  = 0.02f;   // CS smoothing (EMA of raw mV)
const unsigned long LOOP_MS = 20; // control update period (50 Hz)
const float START_DUTY = 70.0f;   // initial guess before the loop converges

PhaseController *controller;
float cs_mv[NUM_CHANNELS] = {0, 0, 0, 0};     // smoothed CS (mV)
float duty[NUM_CHANNELS]  = {START_DUTY, START_DUTY, START_DUTY, START_DUTY};
float i_meas[NUM_CHANNELS] = {0, 0, 0, 0};    // last measured current (A)

static inline float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  pinMode(LED_PIN, OUTPUT);

  analogReadResolution(12);
  for (int i = 0; i < NUM_CHANNELS; i++) {
    analogSetPinAttenuation(ADC_PINS[i], ADC_11db);   // ~0..3.1 V range
    cs_mv[i] = analogReadMilliVolts(ADC_PINS[i]);     // seed the EMA
  }

  controller = new PhaseController(PWM_PINS, INITIAL_PHASES, INITIAL_DUTY_CYCLES,
                                   NUM_CHANNELS);
  controller->begin();
  controller->initCarrierPWM(CARRIER_PINS, PWM_FREQ, INITIAL_DUTY_CYCLES);
  controller->setGlobalFrequency(DRIVE_FREQ);
  for (int i = 0; i < NUM_CHANNELS; i++)
    controller->setCarrierDutyCycle(i, duty[i]);

  Serial.printf("field-balance: target=%.2f A, KI=%.2f, %luHz loop\n",
                TARGET_A, KI, 1000 / LOOP_MS);
}

void loop() {
  controller->run();

  // Continuously smooth each CS channel (averages out 190 Hz / 15 kHz ripple).
  for (int i = 0; i < NUM_CHANNELS; i++)
    cs_mv[i] += ADC_ALPHA * (analogReadMilliVolts(ADC_PINS[i]) - cs_mv[i]);

  static unsigned long last = 0;
  unsigned long now = millis();
  if (now - last < LOOP_MS)
    return;
  last = now;

  // Per-channel integral control toward the common current setpoint.
  for (int i = 0; i < NUM_CHANNELS; i++) {
    i_meas[i] = SENS[i] * (cs_mv[i] - OFFSET_MV[i]) / 1000.0f;  // A
    if (i_meas[i] > I_MAX_A) {
      duty[i] -= 5.0f;                       // fast back-off on overcurrent
    } else {
      duty[i] += KI * (TARGET_A - i_meas[i]); // integrator (state = duty)
    }
    duty[i] = clampf(duty[i], DUTY_MIN, DUTY_MAX);
    controller->setCarrierDutyCycle(i, duty[i]);
  }

  // Status + pair-balance error (A-C, B-D) at ~2 Hz.
  static unsigned long last_dbg = 0;
  if (now - last_dbg >= 500) {
    last_dbg = now;
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
    Serial.printf("I[A]: A=%.2f C=%.2f (dAC=%+.2f) | B=%.2f D=%.2f (dBD=%+.2f)  "
                  "duty: %.0f %.0f %.0f %.0f\n",
                  i_meas[0], i_meas[2], i_meas[0] - i_meas[2],
                  i_meas[1], i_meas[3], i_meas[1] - i_meas[3],
                  duty[0], duty[1], duty[2], duty[3]);
  }
}
