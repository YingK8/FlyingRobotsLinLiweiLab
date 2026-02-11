#include <Arduino.h>
#include "PhaseController.h"

// === CONFIGURATION ===
// NOTE: Channel number must be < 8 for efficient handling, though strict limit depends on pin count
const int NUM_CHANNELS = 4;
const gpio_num_t PWM_PINS[NUM_CHANNELS] = {GPIO_NUM_12, GPIO_NUM_27, GPIO_NUM_33, GPIO_NUM_15}; 
const float INITIAL_PHASES[NUM_CHANNELS] = {0.0, 90.0, 180.0, 270.0};
// === CONFIGURATION ===

// Pin used for synchronization
const gpio_num_t SYNC_PIN = GPIO_NUM_4;

PhaseController controller(PWM_PINS, INITIAL_PHASES, NUM_CHANNELS);

// State tracking variables
unsigned long startTime;
bool rampFinished = false;

void setup() {
    Serial.begin(115200);
    Serial.println("Starting ESP-IDF Timer PhaseController...");

    // 1. Configure Sync
    controller.enableSync(SYNC_PIN);

    // 2. Begin Generation
    // Start at initial frequency of 10Hz
    controller.begin(10.0);
    
    // Record start time
    startTime = millis();
}

void loop() {
    // Housekeeping for phase controller
    controller.run();

    // Logic Timing
    unsigned long elapsed = millis() - startTime;

    // --- FREQUENCY RAMP LOGIC ---
    
    // Phase 1: Hold 10Hz for first 8 seconds (0s - 8s)
    if (elapsed < 8000) {
        // Do nothing, already set to 10Hz in setup()
    } 
    // Phase 2: Ramp from 10Hz to 250Hz over 10 seconds (8s - 18s)
    else if (elapsed < 18000) {
        // Calculate progress (0.0 to 10.0 seconds)
        float rampTimeSeconds = (elapsed - 8000) / 1000.0f;
        
        // Linear Ramp Equation: Freq = Start + (Slope * Time)
        // Slope = (250Hz - 10Hz) / 10s = 24Hz/s
        float currentFreq = 10.0f + (24.0f * rampTimeSeconds);
        
        controller.setGlobalFrequency(currentFreq);
    } 
    // Phase 3: Hold 250Hz (18s+)
    else if (!rampFinished) {
        controller.setGlobalFrequency(250.0);
        Serial.println("Ramp Complete. Holding 250Hz.");
        rampFinished = true;
    }
}