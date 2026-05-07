#include "PhaseController.h"
#include "PhaseSequencer.h"
#include <Arduino.h>
#include <FS.h>
#include <SPIFFS.h>

#ifndef LEDC_APB_CLK_HZ
#define LEDC_APB_CLK_HZ 80000000UL
#endif

// CONFIGURATION
const int NUM_CHANNELS = 4;

const gpio_num_t A_PWM_PIN = GPIO_NUM_23;
const gpio_num_t B_PWM_PIN = GPIO_NUM_18;
const gpio_num_t C_PWM_PIN = GPIO_NUM_21;
const gpio_num_t D_PWM_PIN = GPIO_NUM_5;

const gpio_num_t A_CARRIER_PIN = GPIO_NUM_22;
const gpio_num_t B_CARRIER_PIN = GPIO_NUM_4;
const gpio_num_t C_CARRIER_PIN = GPIO_NUM_19;
const gpio_num_t D_CARRIER_PIN = GPIO_NUM_15;

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
const unsigned long ramp_duration_ms = 40000;

// --- INDICATOR LED CONFIGURATION ---
const int LED_PIN = 2; // Common built-in LED pin for ESP32
bool led_state = false; // Track the current state of the LED

PhaseController controller(PWM_PINS, INITIAL_PHASES, INITIAL_DUTY_CYCLES, NUM_CHANNELS);
PhaseSequencer seq(&controller);

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(20);
  delay(1000);
  SPIFFS.begin(true);
  
  // Set up the LED pin
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW); // Ensure it starts off
  
  // controller.enableSync(SYNC_PIN);
  controller.begin();
  
  // Initialize carrier PWM for each channel
  controller.initCarrierPWM(CARRIER_PINS, PWM_FREQ, INITIAL_CARRIER_DUTY_CYCLES);

  // Dry Takeoff Configuration
  seq.addEaseRampTask(start_freq, end_freq, ramp_duration_ms); // 1Hz to 200Hz over 40000ms
  seq.compile(25, 1.0f, INITIAL_DUTY_CYCLES, INITIAL_PHASES);

  seq.start();
}

// --- STATE MACHINE VARIABLES ---
float carrier_duty_sweep = 100.0f;
const float d_duty_step = 10.0f; 

// Timing variables for pausing
const unsigned long wait_time_ms = 1500; // Time between steps (1.5 seconds)
unsigned long wait_start_time = 0;

// NEW: Latch to track if the frequency ramp is completely done
bool ramp_finished = false;

void loop() {
  controller.run(); // hardware timer drift compensation
  seq.run();        // state machine queue

  unsigned long current_millis = millis();

  // 1. Detect ramp completion cleanly
  // Using 199.5f accounts for floating point inaccuracy and hardware timer drift
  if (!ramp_finished && controller.getFrequency() >= end_freq - 0.5f) {
    ramp_finished = true;
    wait_start_time = current_millis; // Start the timer ONCE when ramp finishes
  }

  // 2. Sweep logic
  // Changed to > 0.0f so it actually stops executing when it hits 0
  if (ramp_finished && carrier_duty_sweep > 0.0f) { 

    if (current_millis - wait_start_time >= wait_time_ms) {
      // Apply the step down
      carrier_duty_sweep -= d_duty_step;

      // Clamp to 0.0f to prevent sending negative duty cycles to the controller
      if (carrier_duty_sweep < 0.0f) {
        carrier_duty_sweep = 0.0f;
      }

      controller.setCarrierDutyCycle(1, carrier_duty_sweep);
      controller.setCarrierDutyCycle(3, carrier_duty_sweep);
      
      // TOGGLE THE LED
      led_state = !led_state; // Flip the boolean state
      digitalWrite(LED_PIN, led_state ? HIGH : LOW); // Write the new state to the pin

      wait_start_time = current_millis; // Reset timer for the next 10% decrement
    }
  }
}