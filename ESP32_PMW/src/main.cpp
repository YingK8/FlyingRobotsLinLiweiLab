#include <Arduino.h>
#include "PhasePWM.h" 

// ================= CONFIGURATION =================

// Total Channels: 1 Master (Hidden) + 4 User Channels
#define TOTAL_CHANNELS 5 

// Pin Definitions
// Index 0 is the Hidden Master (Pin 14) - connects to SYNC_INPUT_PIN
// Indices 1-4 are the User Pins (Oscilloscope probes go here)
const int CHANNEL_PINS[TOTAL_CHANNELS] = {14, 15, 32, 33, 27};

// Sync Wiring: Connect Pin 14 -> Pin 4 physically!
#define SYNC_INPUT_PIN 4  

// PWM Settings
#define PWM_FREQ_HZ 350 // 350 Hz

// PWM generator
PhasePWM generator(CHANNEL_PINS, TOTAL_CHANNELS, SYNC_INPUT_PIN, PWM_FREQ_HZ);

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("Initializing 4-Channel Fixed Phase PWM...");

    if (!generator.begin()) {
        Serial.println("Failed to start PWM.");
        while(1);
    }

    // initialise channel (four)
    float phase = 0.0;
    float phase_diff = 360.0 / (TOTAL_CHANNELS - 1);

    for (int i = 0; i < TOTAL_CHANNELS - 1; i++) {
        generator.setPhase(i, phase);
        generator.setDuty(i, 50.0);
        phase += phase_diff;
    }
}

void loop() {
    delay(1000);
}