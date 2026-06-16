#include <Arduino.h>
#include "PhaseController.h"
#include "PhaseSequencer.h"
#include <FS.h>
#include <SPIFFS.h>

// #include <gpio_viewer.h> // Must me the first include in your project
// GPIOViewer gpio_viewer;

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
// const float INITIAL_PHASES[NUM_CHANNELS] = {0.0, 180.0, 90.0, 270.0};
// rotation is clockwise: D -> B -> C -> A
const float INITIAL_PHASES[NUM_CHANNELS] = {270.0, 90.0, 180.0, 0.0};
const float INITIAL_DUTY_CYCLES[NUM_CHANNELS] = {50.0, 50.0, 50.0, 50.0};
const gpio_num_t SYNC_PIN = GPIO_NUM_8;

const int PWM_FREQ = 15000; 
const float carrier_duty = 100.0;
const float INITIAL_CARRIER_DUTY_CYCLES[NUM_CHANNELS] = {carrier_duty, carrier_duty, carrier_duty, carrier_duty};

const float start_freq = 1.0f;
const float end_freq = 190.0f;
const unsigned long ramp_duration_ms = 30000;

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

  Serial.println("running main tilt");

  // gpio_viewer.connectToWifi("Your SSID network", "Your WiFi Password");
  
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
enum SweepState { RAMP_UP, HOLD, TRANSITION, DONE };
SweepState sweep_state = RAMP_UP;

float duty_start = 100.0f;   // duty at the beginning of the current transition
float duty_target = 100.0f;  // duty at the end of the current transition
const float d_duty_step = 10.0f;

const unsigned long hold_time_ms = 2500;        // hold at each duty level
const unsigned long transition_time_ms = 500;   // cubic ramp between levels (3s total per step)
unsigned long state_start_time = 0;

// only channels 0(A) and 2(C) sweep down; 1(B) and 3(D) stay at 100%
void setSweepCarrierDuty(float duty) {
  controller->setCarrierDutyCycle(1, duty);
  controller->setCarrierDutyCycle(2, duty);
}

void loop() {
  controller->run();
  seq->run();

  unsigned long current_millis = millis();

  // TEMP DEBUG: 1Hz status print. ledc readback = duty the hardware actually
  // has latched (0..4095 at 12-bit); 0 means LEDC is not driving that pin.
  static unsigned long last_dbg_ms = 0;
  if (current_millis - last_dbg_ms >= 1000) {
    last_dbg_ms = current_millis;
    Serial.printf("state=%d freq=%.1f target=%.1f ledc0=%u ledc3=%u\n",
                  (int)sweep_state, controller->getFrequency(), duty_target,
                  (unsigned)ledc_get_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0),
                  (unsigned)ledc_get_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_3));
  }

  switch (sweep_state) {
    case RAMP_UP:
      if (controller->getFrequency() >= end_freq - 0.5f) {
        sweep_state = HOLD;
        state_start_time = current_millis;
      }
      break;

    case HOLD:
      if (current_millis - state_start_time >= hold_time_ms) {
        if (duty_target <= 0.0f) {
          sweep_state = DONE;
          break;
        }
        duty_start = duty_target;
        duty_target = duty_target - d_duty_step;
        if (duty_target < 0.0f) {
          duty_target = 0.0f;
        }

        led_state = !led_state;
        digitalWrite(LED_PIN, led_state ? HIGH : LOW);

        sweep_state = TRANSITION;
        state_start_time = current_millis;
      }
      break;

    case TRANSITION: {
      float t = (float)(current_millis - state_start_time) / (float)transition_time_ms;
      if (t >= 1.0f) {
        setSweepCarrierDuty(duty_target);
        sweep_state = HOLD;
        state_start_time = current_millis;
      } else {
        // cubic smoothstep: ease in/out between duty levels
        float ease = t * t * (3.0f - 2.0f * t);
        setSweepCarrierDuty(duty_start + (duty_target - duty_start) * ease);
      }
      break;
    }

    case DONE:
      break;
  }
}