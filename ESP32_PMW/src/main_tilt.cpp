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
// FIXED: Changed from 35 (Input Only) to 23 (Safe Output)
const gpio_num_t D_PWM_PIN = GPIO_NUM_25; 

const gpio_num_t A_CARRIER_PIN = GPIO_NUM_33;
const gpio_num_t B_CARRIER_PIN = GPIO_NUM_13;
const gpio_num_t C_CARRIER_PIN = GPIO_NUM_14;
const gpio_num_t D_CARRIER_PIN = GPIO_NUM_26;

const gpio_num_t PWM_PINS[NUM_CHANNELS] =     {A_PWM_PIN,     B_PWM_PIN,      C_PWM_PIN,      D_PWM_PIN};
const gpio_num_t CARRIER_PINS[NUM_CHANNELS] = {A_CARRIER_PIN, B_CARRIER_PIN,  C_CARRIER_PIN,  D_CARRIER_PIN};

// rotation is counter-clockwise: A -> C -> B -> D
const float INITIAL_PHASES[NUM_CHANNELS] = {0.0, 180.0, 90.0, 270.0};

// clockwise rotation: D -> B -> C -> A
// const float INITIAL_PHASES[NUM_CHANNELS] = {270.0, 90.0, 180.0, 0.0};
const float INITIAL_DUTY_CYCLES[NUM_CHANNELS] = {50.0, 50.0, 50.0, 50.0};
const gpio_num_t SYNC_PIN = GPIO_NUM_8;

const int PWM_FREQ = 20000; 
const float carrier_duty = 100.0;
// Per-channel MAX carrier duty (the ceiling each channel starts at / is swept from).
// Carrier pins are driven DIRECTLY from GPIO (no inverter): 100 = constant HIGH = full
// on, <100 = LEDC PWM at PWM_FREQ. (Relies on the idle-HIGH fix in PhaseController.cpp.)
const float CARRIER_MAX_DUTY[NUM_CHANNELS] = {95.0, 100.0, 100.0, 95.0};
const float INITIAL_CARRIER_DUTY_CYCLES[NUM_CHANNELS] = {
    CARRIER_MAX_DUTY[0], CARRIER_MAX_DUTY[1], CARRIER_MAX_DUTY[2], CARRIER_MAX_DUTY[3]};

const float start_freq = 1.0f;
const float end_freq = 250.0f;
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
enum SweepState { RAMP_UP, HOLD, TRANSITION, DONE };
SweepState sweep_state = RAMP_UP;

// Channels whose carrier is swept down: B = 1, D = 3. A and C stay at full.
const int SWEEP_CHANNELS[] = {1, 2};
const int NUM_SWEEP_CHANNELS = sizeof(SWEEP_CHANNELS) / sizeof(SWEEP_CHANNELS[0]);

// The sweep tracks a FRACTION of each channel's max (%). Each step subtracts
// d_duty_step % of max, so the duty decreases linearly and reaches 0.
const float d_duty_step = 10.0f;       // step size, as % of each channel's max
float duty_target = 100.0f;            // fraction of max we are settling toward (%)
float duty_start = 100.0f;             // fraction of max we are easing away from (%)

const unsigned long hold_time_ms = 1500;        // dwell at each level
const unsigned long transition_time_ms = 1000;  // smoothstep glide between levels
unsigned long state_start_time = 0;

// Apply a sweep fraction (0..100 % of max) to every swept channel. Each channel
// lands at CARRIER_MAX_DUTY[ch] * fraction/100, so B and D scale down from their
// reduced ceiling while A and C are left untouched.
void setSweepCarrierDuty(float fractionPct) {
  for (int i = 0; i < NUM_SWEEP_CHANNELS; i++) {
    int ch = SWEEP_CHANNELS[i];
    controller->setCarrierDutyCycle(ch, CARRIER_MAX_DUTY[ch] * fractionPct / 100.0f);
  }
}

void loop() {
  controller->run();
  seq->run();

  unsigned long current_millis = millis();

  // TEMP DEBUG: 1Hz per-channel DELIVERED duty, measured in the ISR. If one
  // channel slowly drops while the others hold, firmware timing is the cause;
  // if all four hold steady (~50%) the cause is hardware/tank, not the code.
  static unsigned long last_dbg_ms = 0;
  if (current_millis - last_dbg_ms >= 1000) {
    last_dbg_ms = current_millis;
    Serial.printf("state=%d freq=%.1f | dutyA=%.1f dutyB=%.1f dutyC=%.1f dutyD=%.1f\n",
                  (int)sweep_state, controller->getFrequency(),
                  controller->getMeasuredDuty(0), controller->getMeasuredDuty(1),
                  controller->getMeasuredDuty(2), controller->getMeasuredDuty(3));
    controller->resetMeasuredCounts();
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