#include <Arduino.h>
#include "PhaseController.h"
#include "PhaseSequencer.h"
#include "CoilBalancer.h"

// ============================================================================
//  main_balance.cpp  --  bench test for the low-level coil current-balance loop.
//
//  Spins the field up to a fixed hold frequency, then enables CoilBalancer, which
//  reads the 4 current-sense (CS) ADC channels and trims carrier duties so each
//  opposing pair (A=B, C=D) carries equal current at the highest common level.
//
//  Telemetry (CSV @ ~10 Hz):
//    B,<ms>,<Ia>,<Ib>,<Ic>,<Id>,<dutyA>,<dutyB>,<dutyC>,<dutyD>,<errAB>,<errCD>
//  where I* are current proxies (mV/ohm) and duties are applied carrier duty %.
// ============================================================================

const int NUM_CHANNELS = 4;

// --- GOLDEN, UNTOUCHABLE pin map (see canonical-gpio-pin-map). -------------
const gpio_num_t A_PWM_PIN = GPIO_NUM_32;
const gpio_num_t B_PWM_PIN = GPIO_NUM_23;
const gpio_num_t C_PWM_PIN = GPIO_NUM_27;
const gpio_num_t D_PWM_PIN = GPIO_NUM_25;

const gpio_num_t A_CARRIER_PIN = GPIO_NUM_33;
const gpio_num_t B_CARRIER_PIN = GPIO_NUM_13;
const gpio_num_t C_CARRIER_PIN = GPIO_NUM_14;
const gpio_num_t D_CARRIER_PIN = GPIO_NUM_26;

const gpio_num_t PWM_PINS[NUM_CHANNELS] =     {A_PWM_PIN,     B_PWM_PIN,     C_PWM_PIN,     D_PWM_PIN};
const gpio_num_t CARRIER_PINS[NUM_CHANNELS] = {A_CARRIER_PIN, B_CARRIER_PIN, C_CARRIER_PIN, D_CARRIER_PIN};

// rotation is counter-clockwise: A -> C -> B -> D
const float INITIAL_PHASES[NUM_CHANNELS] = {0.0, 180.0, 90.0, 270.0};
const float INITIAL_DUTY_CYCLES[NUM_CHANNELS] = {50.0, 50.0, 50.0, 50.0};

const int   PWM_FREQ = 15000;
const float carrier_duty = 100.0;
const float INITIAL_CARRIER_DUTY_CYCLES[NUM_CHANNELS] = {
    carrier_duty, carrier_duty, carrier_duty, carrier_duty};

// --- spin-up -> hold ------------------------------------------------------
const float HOLD_FREQ = 185.0f;
const unsigned long SPINUP_DURATION_MS = 6000;

// --- CURRENT-SENSE / BALANCER CONFIG --------------------------------------
//  CS ADC pins, channel order A,B,C,D.  *** classic ESP32 ADC NOTE ***
//  Your wiring (GPIO 19:A, 18:B, 5:C, 21:D) is NOT ADC-capable on the classic
//  ESP-WROOM-32: only ADC1 (32-39) / ADC2 (0,2,4,12-15,25-27) can analogRead.
//  Moved to the four free input-only ADC1 pins. Re-route the CS wires to match:
//      A -> GPIO36   B -> GPIO39   C -> GPIO34   D -> GPIO35
const CoilBalancerConfig BAL_CFG = {
  .adcPins     = {36, 39, 34, 35},                 // A, B, C, D
  .shuntOhms   = {2532.0f, 2530.0f, 2542.0f, 2540.0f}, // your measured shunts
  .ceilingDuty = {100.0f, 100.0f, 100.0f, 100.0f}, // weak coil held here (full on)
  .dutyFloor   = 50.0f,                             // never starve a coil below 50%
  .trimMax     = 40.0f,                             // strong coil may drop to 60%
  .pairs       = {{0, 1}, {2, 3}},                  // (A,B) and (C,D)
  .kp          = 0.0f,    // start pure-I; add small Kp if convergence is too slow
  .ki          = 8.0f,    // duty% per (current-unit * s); tune for ~1-2 s settle
  .deadband    = 0.5f,    // current-proxy units; freeze integrator within this
  .updatePeriodMs = 5,    // 200 Hz control update
  .adcOversample  = 16,   // average 16 samples/read to knock down ADC noise
};

const int LED_PIN = 2;
bool led_state = false;

PhaseController* controller;
PhaseSequencer*  seq;
CoilBalancer*    balancer;

enum BalState { SPINUP, BALANCE };
BalState state = SPINUP;

unsigned long last_telem_ms = 0;
const unsigned long TELEM_PERIOD_MS = 100;  // ~10 Hz

void sendTelemetry() {
  Serial.printf("B,%lu,%.2f,%.2f,%.2f,%.2f,%.1f,%.1f,%.1f,%.1f,%.2f,%.2f\n",
                millis(),
                balancer->getCurrent(0), balancer->getCurrent(1),
                balancer->getCurrent(2), balancer->getCurrent(3),
                balancer->getDutyApplied(0), balancer->getDutyApplied(1),
                balancer->getDutyApplied(2), balancer->getDutyApplied(3),
                balancer->getPairError(0), balancer->getPairError(1));
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(20);
  delay(1000);

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  controller = new PhaseController(PWM_PINS, INITIAL_PHASES, INITIAL_DUTY_CYCLES, NUM_CHANNELS);
  seq = new PhaseSequencer(controller);

  controller->begin();
  controller->initCarrierPWM(CARRIER_PINS, PWM_FREQ, INITIAL_CARRIER_DUTY_CYCLES);

  balancer = new CoilBalancer(controller, BAL_CFG);
  balancer->begin();   // configures ADC; leaves balancing DISABLED until spun up

  // Eased spin-up to the hold frequency, then we hand off to the balancer.
  seq->addEaseRampTask(1.0f, HOLD_FREQ, SPINUP_DURATION_MS);
  seq->compile(25, 1.0f, INITIAL_DUTY_CYCLES, INITIAL_PHASES);
  seq->start();

  last_telem_ms = millis();
}

void loop() {
  controller->run();

  if (state == SPINUP) {
    seq->run();
    if (seq->isDone()) {
      controller->setGlobalFrequency(HOLD_FREQ);
      balancer->setEnabled(true);   // start balancing only once aloft/spun-up
      state = BALANCE;
    }
  } else {
    balancer->update();             // rate-limited inside; reads CS + trims duty
  }

  unsigned long now = millis();
  if (now - last_telem_ms >= TELEM_PERIOD_MS) {
    last_telem_ms = now;
    sendTelemetry();
    led_state = !led_state;
    digitalWrite(LED_PIN, led_state ? HIGH : LOW);
  }
}
