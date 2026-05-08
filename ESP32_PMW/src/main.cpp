#include <Arduino.h>
#include "PhaseController.h"
#include "PhaseSequencer.h"

// CONFIGURATION
const int NUM_CHANNELS = 4;

const gpio_num_t A_PWM_PIN = GPIO_NUM_32;
const gpio_num_t B_PWM_PIN = GPIO_NUM_25;
const gpio_num_t C_PWM_PIN = GPIO_NUM_27;
// FIXED: Changed from 35 (Input Only) to 23 (Safe Output)
const gpio_num_t D_PWM_PIN = GPIO_NUM_23; 

const gpio_num_t A_CARRIER_PIN = GPIO_NUM_33;
const gpio_num_t B_CARRIER_PIN = GPIO_NUM_26;
const gpio_num_t C_CARRIER_PIN = GPIO_NUM_14;
const gpio_num_t D_CARRIER_PIN = GPIO_NUM_13;

const gpio_num_t PWM_PINS[NUM_CHANNELS] =     {A_PWM_PIN,     B_PWM_PIN,      C_PWM_PIN,      D_PWM_PIN};
const gpio_num_t CARRIER_PINS[NUM_CHANNELS] = {A_CARRIER_PIN, B_CARRIER_PIN,  C_CARRIER_PIN,  D_CARRIER_PIN};

// rotation is counter-clockwise: A -> C -> B -> D
const float INITIAL_PHASES[NUM_CHANNELS] = {0.0, 180.0, 90.0, 270.0};
const float INITIAL_DUTY_CYCLES[NUM_CHANNELS] = {50.0, 50.0, 50.0, 50.0};
const gpio_num_t SYNC_PIN = GPIO_NUM_8;

const int PWM_FREQ = 15000; 
const float carrier_duty = 100.0;
const float INITIAL_CARRIER_DUTY_CYCLES[NUM_CHANNELS] = {carrier_duty, carrier_duty, carrier_duty, carrier_duty};

const float start_freq = 1.0f;
const float end_freq = 200.0f;
const unsigned long ramp_duration_ms = 10000;

// --- INDICATOR LED CONFIGURATION ---
const int LED_PIN = 2; 
bool led_state = false; 

// declare as global POINTERS (do not initialize yet)
PhaseController* controller;
PhaseSequencer* seq;

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(20);
  delay(1000);
  SPIFFS.begin(true); 
  
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW); 
  
  // instantiate the objects dynamically NOW that the OS has booted
  controller = new PhaseController(PWM_PINS, INITIAL_PHASES, INITIAL_DUTY_CYCLES, NUM_CHANNELS);
  seq = new PhaseSequencer(controller);

  controller->begin();
  controller->initCarrierPWM(CARRIER_PINS, PWM_FREQ, INITIAL_CARRIER_DUTY_CYCLES);

  seq->addEaseRampTask(start_freq, end_freq, ramp_duration_ms); 
  seq->compile(25, 1.0f, INITIAL_DUTY_CYCLES, INITIAL_PHASES);

  seq->start();
}

// --- STATE MACHINE VARIABLES ---
float carrier_duty_sweep = 100.0f;
const float d_duty_step = 10.0f; 

const unsigned long wait_time_ms = 1000; 
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

  if (ramp_finished && carrier_duty_sweep > 0.0f) { 

    if (current_millis - wait_start_time >= wait_time_ms) {
      carrier_duty_sweep -= d_duty_step;

      if (carrier_duty_sweep < 0.0f) {
        carrier_duty_sweep = 0.0f;
      }

      controller->setCarrierDutyCycle(0, carrier_duty_sweep);
      controller->setCarrierDutyCycle(3, carrier_duty_sweep);
      
      led_state = !led_state; 
      digitalWrite(LED_PIN, led_state ? HIGH : LOW); 

      wait_start_time = current_millis; 
    }
  }
}