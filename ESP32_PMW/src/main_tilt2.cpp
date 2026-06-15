#include <Arduino.h>
#include "PhaseController.h"
#include "PhaseSequencer.h"
#include <FS.h>
#include <SPIFFS.h>

// CONFIGURATION
const int NUM_CHANNELS = 4;

const gpio_num_t A_PWM_PIN = GPIO_NUM_32;
const gpio_num_t B_PWM_PIN = GPIO_NUM_23;
const gpio_num_t C_PWM_PIN = GPIO_NUM_27;
const gpio_num_t D_PWM_PIN = GPIO_NUM_25;

const gpio_num_t A_CARRIER_PIN = GPIO_NUM_33;
const gpio_num_t B_CARRIER_PIN = GPIO_NUM_13; 
const gpio_num_t C_CARRIER_PIN = GPIO_NUM_14; 
const gpio_num_t D_CARRIER_PIN = GPIO_NUM_26; 

const gpio_num_t PWM_PINS[NUM_CHANNELS] =     {A_PWM_PIN,     B_PWM_PIN,      C_PWM_PIN,      D_PWM_PIN};
const gpio_num_t CARRIER_PINS[NUM_CHANNELS] = {A_CARRIER_PIN, B_CARRIER_PIN,  C_CARRIER_PIN,  D_CARRIER_PIN};

// rotation is counter-clockwise: A -> C -> B -> D
const float INITIAL_PHASES[NUM_CHANNELS] = {0.0, 180.0, 90.0, 270.0};
const float INITIAL_DUTY_CYCLES[NUM_CHANNELS] = {50.0, 50.0, 50.0, 50.0};
const gpio_num_t SYNC_PIN = GPIO_NUM_8;

const int PWM_FREQ = 15000;
// All channels start at 100% so disk spins up at full power before sweep begins
const float INITIAL_CARRIER_DUTY_CYCLES[NUM_CHANNELS] = {100.0, 100.0, 100.0, 100.0};

const float start_freq = 1.0f;
const float end_freq = 200.0f;
const unsigned long ramp_duration_ms = 40000;

// --- INDICATOR LED CONFIGURATION ---
const int LED_PIN = 2;
bool led_state = false;

PhaseController* controller;
PhaseSequencer* seq;

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(20);
  delay(1000);
  SPIFFS.begin(true);

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  // sync frames
  for (int i = 0; i < 4; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(200);
    digitalWrite(LED_PIN, LOW);
  }

  controller = new PhaseController(PWM_PINS, INITIAL_PHASES, INITIAL_DUTY_CYCLES, NUM_CHANNELS);
  seq = new PhaseSequencer(controller);

  controller->begin();
  controller->initCarrierPWM(CARRIER_PINS, PWM_FREQ, INITIAL_CARRIER_DUTY_CYCLES);

  seq->addEaseRampTask(start_freq, end_freq, ramp_duration_ms);
  seq->compile(25, 1.0f, INITIAL_DUTY_CYCLES, INITIAL_PHASES);

  seq->start();
}

// --- STATE MACHINE: sweep A (ch0) and C (ch2) from 100% down to 0% after ramp ---
// Hold 100% through spin-up, then step down; B and D stay at 100%.
float carrier_duty_sweep = 100.0f;
const float d_duty_step = 10.0f;

const unsigned long wait_time_ms = 3000;
unsigned long wait_start_time = 0;

bool ramp_finished = false;

void loop() {
  controller->run();
  seq->run();

  unsigned long current_millis = millis();

  if (!ramp_finished && controller->getFrequency() >= end_freq - 0.5f) {
    ramp_finished = true;
    wait_start_time = current_millis;
  }

  // NOTE: do NOT drop A/C carrier during the frequency ramp. The disk must spin
  // up at full power on all four channels, or the field is asymmetric (A/C weaker
  // than B/D) and destabilises before reaching 200 Hz. The tilt happens only after
  // spin-up, via the stepped sweep below.

  // Sweep A and C from 100% down to 0% (after spin-up); B and D remain at 100%
  if (ramp_finished && carrier_duty_sweep > 0.0f) {

    if (current_millis - wait_start_time >= wait_time_ms) {
      carrier_duty_sweep -= d_duty_step;

      if (carrier_duty_sweep < 0.0f) {
        carrier_duty_sweep = 0.0f;
      }

      controller->setCarrierDutyCycle(0, carrier_duty_sweep); // A
      controller->setCarrierDutyCycle(2, carrier_duty_sweep); // C

      led_state = !led_state;
      digitalWrite(LED_PIN, led_state ? HIGH : LOW);

      wait_start_time = current_millis;
    }
  }
}
