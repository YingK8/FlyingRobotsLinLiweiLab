#include "JsonPhaseSequencer.h"
#include "PhaseController.h"
#include <Arduino.h>
#include <FS.h>
#include <SPIFFS.h>

#ifndef LEDC_APB_CLK_HZ
#define LEDC_APB_CLK_HZ 80000000UL
#endif

// CONFIGURATION
const int NUM_CHANNELS = 4;
                                          //    A            B            C            D
const gpio_num_t PWM_PINS[NUM_CHANNELS] = {GPIO_NUM_23, GPIO_NUM_15, GPIO_NUM_21, GPIO_NUM_18};
const gpio_num_t CARRIER_PINS[NUM_CHANNELS] = {GPIO_NUM_22, GPIO_NUM_4, GPIO_NUM_19, GPIO_NUM_5};

// rotation is counter-clockwise: A -> C -> B -> D
const float INITIAL_PHASES[NUM_CHANNELS] = {0.0, 180.0, 90.0, 270.0};
const float INITIAL_DUTY_CYCLES[NUM_CHANNELS] = {50.0,50.0,50.0,50.0};
const gpio_num_t SYNC_PIN = GPIO_NUM_8;

const int PWM_FREQ = 2000; // 20kHz
const float carrier_duty = 100.0;
const float INITIAL_CARRIER_DUTY_CYCLES[NUM_CHANNELS] = {carrier_duty, carrier_duty, carrier_duty, carrier_duty};

PhaseController controller(PWM_PINS, INITIAL_PHASES, INITIAL_DUTY_CYCLES, NUM_CHANNELS);
JsonPhaseSequencer seq(&controller);

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(20);
  delay(1000);
  SPIFFS.begin(true);
  // controller.enableSync(SYNC_PIN);
  controller.begin(1.0f);
  // Initialize carrier PWM for each channel
  controller.initCarrierPWM(CARRIER_PINS, PWM_FREQ, INITIAL_CARRIER_DUTY_CYCLES);

  // Water Takeoff Configuration
  // seq.addWaitTask(1000); // 5 second delay before starting
  // seq.addEaseRampTask(1.0f, 150.0f, 30000); // 1Hz to 250Hz over 60000ms
  // (60s) seq.addEaseRampTask(150.0f, 270.0f, 15000); // 1Hz to 250Hz over
  // 60000ms (60s)

  // Dry Takeoff Configuration
  seq.addEaseRampTask(1.0f, 250.0f, 25000); // 1Hz to 250Hz over 25000ms
  seq.compile(25, 1.0f, INITIAL_DUTY_CYCLES, INITIAL_PHASES);

  seq.start();
}

void loop() {
  controller.run(); // hardware timer drift compensation
  seq.run();        // state machine queue
}
