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
const gpio_num_t PWM_PINS[NUM_CHANNELS] = {GPIO_NUM_19, GPIO_NUM_33,
                                           GPIO_NUM_27, GPIO_NUM_32};
const float INITIAL_PHASES[NUM_CHANNELS] = {0.0, 90.0, 180.0, 270.0};
const float INITIAL_DUTY_CYCLES[NUM_CHANNELS] = {50.0, 50.0, 50.0, 50.0};
const gpio_num_t SYNC_PIN = GPIO_NUM_4;

// === PWM PIN 21 CONFIGURATION ===
const gpio_num_t CARRIER_PINS[NUM_CHANNELS] = {GPIO_NUM_21, GPIO_NUM_22,
                                               GPIO_NUM_26, GPIO_NUM_31};
const int PWM_FREQ = 50000; // 20kHz
const float INITIAL_CARRIER_DUTY_CYCLES[NUM_CHANNELS] = {100.0, 100.0, 100.0,
                                                         100.0};

PhaseController controller(PWM_PINS, INITIAL_PHASES, INITIAL_DUTY_CYCLES,
                           NUM_CHANNELS);
JsonPhaseSequencer seq(&controller);

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(20);
  delay(1000);
  SPIFFS.begin(true);
  controller.enableSync(SYNC_PIN);
  controller.begin(300.0f);
  // Initialize carrier PWM for each channel
  controller.initCarrierPWM(CARRIER_PINS, PWM_FREQ,
                            INITIAL_CARRIER_DUTY_CYCLES);

  // Water Takeoff Configuration¨
  // seq.addWaitTask(1000); // 5 second delay before starting
  // seq.addEaseRampTask(1.0f, 150.0f, 30000); // 1Hz to 250Hz over 60000ms
  // (60s) seq.addEaseRampTask(150.0f, 270.0f, 15000); // 1Hz to 250Hz over
  // 60000ms (60s)

  // Dry Takeoff Configuration
  // seq.addEaseRampTask(1.0f, 100.0f, 15000); // 1Hz to 250Hz over 20000ms
  // (20s) seq.compile(25, 1.0f, INITIAL_DUTY_CYCLES, INITIAL_PHASES);
  seq.compile(25, 300.0f, INITIAL_DUTY_CYCLES, INITIAL_PHASES);

  seq.start();
}

void loop() {
  controller.run(); // hardware timer drift compensation
  seq.run();        // state machine queue

  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    if (command.length() > 0) {
      float duty = constrain(command.toFloat(), 0.0f, 100.0f);
      for (int ch = 0; ch < NUM_CHANNELS; ch++) {
        controller.setDutyCycle(ch, duty);
      }
      Serial.printf("Duty set to %.2f%% on %d channels\n", duty, NUM_CHANNELS);
    } else if (command.equalsIgnoreCase("p")) {
      for (int ch = 0; ch < NUM_CHANNELS; ch++) {
        controller.setDutyCycle(ch, 0.0f);
      }
      Serial.printf("turned off all channels\n");
    }
  }
}
