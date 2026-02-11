#include <Arduino.h>
#include "PhaseController.h"

// === CONFIGURATION ===
// NOTE: Channel number must be < 8 for efficient handling, though strict limit depends on pin count
// These settings only apply if your device is a client. a server will ignore these!
const int NUM_CHANNELS = 2;
const gpio_num_t PWM_PINS[NUM_CHANNELS] = {GPIO_NUM_18, GPIO_NUM_32}; 
const float INITIAL_PHASES[NUM_CHANNELS] = {90.0, 180.0};
// === CONFIGURATION ===

// Pin used for synchronization
const gpio_num_t SYNC_PIN = GPIO_NUM_4;

PhaseController controller(PWM_PINS, INITIAL_PHASES, NUM_CHANNELS);

void setup() {
    Serial.begin(115200);
    Serial.println("Starting ESP-IDF Timer PhaseController...");

    // 1. Configure Sync
    controller.enableSync(SYNC_PIN);

    // 2. Begin Generation
    // The high-resolution timer starts automatically inside begin()
    controller.begin(50.0);
}

void loop() {
    // The waveform generation is now handled by a 25us periodic hardware timer interrupt.
    // It runs completely independently of this loop.
    
    // We only need to call run() to handle housekeeping (re-calculating params based on average freq)
    controller.run();

    // // Example: Print status safely
    // static unsigned long lastPrint = 0;
    // if (millis() - lastPrint > 1000) {
    //     lastPrint = millis();
    //     if (!isMaster) {
    //          // Print the measured frequency derived from the sync pin
    //          Serial.printf("Synced Freq: %.2f Hz\n", controller.getFrequency(0));
    //     } else {
    //          Serial.println("Master Mode Running");
    //     }
    // }
    
    // You can now use delay() or other blocking code without affecting the waveform
    // delay(50); 
}