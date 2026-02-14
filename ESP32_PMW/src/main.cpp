#include <Arduino.h>
#include "PhaseController.h"

// === CONFIGURATION ===
const int NUM_CHANNELS = 4;
// ESP32 Pins (Ensure these are valid output pins on your specific board)
const gpio_num_t PWM_PINS[NUM_CHANNELS] = {GPIO_NUM_12, GPIO_NUM_27, GPIO_NUM_33, GPIO_NUM_15}; 
const float INITIAL_PHASES[NUM_CHANNELS] = {0.0, 90.0, 180.0, 270.0};
const gpio_num_t SYNC_PIN = GPIO_NUM_4;

// Ramp Constants
const float START_FREQ = 200.0f;
const float END_FREQ = 200.0f; // Set to same as start for constant freq, or change for ramp
const float RAMP_DURATION_MS = 120000.0f; // 120 seconds
const float SLOPE = (END_FREQ - START_FREQ) / (RAMP_DURATION_MS / 1000.0f); // Hz per second
// === CONFIGURATION ===

PhaseController controller(PWM_PINS, INITIAL_PHASES, NUM_CHANNELS);

unsigned long startTime;
unsigned long lastPrintTime = 0;
bool rampFinished = false;

void setup() {
    Serial.begin(115200);
    // while(!Serial); // Optional: Wait for Serial on native USB boards
    delay(1000);
    Serial.println("ESP32 PhaseController: Duty Cycle Setter Example");

    // 1. Configure Sync
    controller.enableSync(SYNC_PIN);
    float duty = 45.0f;
    controller.setDutyCycle(0, duty);
    controller.setDutyCycle(1, duty);
    controller.setDutyCycle(2, duty);
    controller.setDutyCycle(3, duty);

    // 2. Begin Generation
    controller.begin(START_FREQ);
    
    startTime = millis();
}

void loop() {
    // Required housekeeping for the timer-based controller
    // This compensates for drift if external sync is used
    controller.run();
}