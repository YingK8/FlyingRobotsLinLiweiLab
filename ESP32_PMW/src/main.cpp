#include <Arduino.h>
#include "PhaseController.h"

// === CONFIGURATION ===
const int NUM_CHANNELS = 4;
const gpio_num_t PWM_PINS[NUM_CHANNELS] = {GPIO_NUM_12, GPIO_NUM_27, GPIO_NUM_33, GPIO_NUM_15}; 
const float INITIAL_PHASES[NUM_CHANNELS] = {0.0, 90.0, 180.0, 270.0};
const gpio_num_t SYNC_PIN = GPIO_NUM_4;

// Ramp Constants
const float START_FREQ = 0.0f;
const float END_FREQ = 200.0f;
const float RAMP_DURATION_MS = 120000.0f; // 30 seconds
const float SLOPE = (END_FREQ - START_FREQ) / (RAMP_DURATION_MS / 1000.0f); // Hz per second
// === CONFIGURATION ===

PhaseController controller(PWM_PINS, INITIAL_PHASES, NUM_CHANNELS);

unsigned long startTime;
unsigned long lastPrintTime = 0;
bool rampFinished = false;

void setup() {
    Serial.begin(115200);
    while(!Serial); // Wait for Serial on some boards
    Serial.println("ESP32 PhaseController: Ramping 1Hz to 300Hz...");

    // 1. Configure Sync
    controller.enableSync(SYNC_PIN);

    // 2. Begin Generation at 1Hz
    controller.begin(START_FREQ);
    
    startTime = millis();
}

void loop() {
    // Required housekeeping for the timer-based controller
    controller.run();

    unsigned long elapsed = millis() - startTime;

    if (elapsed < RAMP_DURATION_MS) {
        // Linear Ramp Calculation
        float elapsedSeconds = elapsed / 1000.0f;
        float currentFreq = START_FREQ + (SLOPE * elapsedSeconds);
        
        // Safety bound
        if (currentFreq > END_FREQ) currentFreq = END_FREQ;
        
        controller.setGlobalFrequency(currentFreq);

        // Print frequency every 1000ms for debugging
        if (millis() - lastPrintTime > 1000) {
            Serial.print("Ramping... Current Freq: ");
            Serial.print(currentFreq);
            Serial.println(" Hz");
            lastPrintTime = millis();
        }
    } 
    else if (!rampFinished) {
        // Finalize at 300Hz
        controller.setGlobalFrequency(END_FREQ);
        Serial.print("Ramp Complete. Target reached: ");
        Serial.print(END_FREQ);
        Serial.println(" Hz");
        rampFinished = true;
    }
}