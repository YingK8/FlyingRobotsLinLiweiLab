#include <Arduino.h>
#include "PhaseController.h"
#include "PhaseSequencer.h"

// === CONFIGURATION ===
const int NUM_CHANNELS = 4;
const gpio_num_t PWM_PINS[NUM_CHANNELS] = {GPIO_NUM_12, GPIO_NUM_27, GPIO_NUM_33, GPIO_NUM_15}; 
const float INITIAL_PHASES[NUM_CHANNELS] = {0.0, 90.0, 180.0, 270.0};
const float INITIAL_DUTY_CYCLES[NUM_CHANNELS] = {50.0, 50.0, 50.0, 50.0};
const gpio_num_t SYNC_PIN = GPIO_NUM_4;

PhaseController controller(PWM_PINS, INITIAL_PHASES, INITIAL_DUTY_CYCLES, NUM_CHANNELS);
PhaseSequencer seq(&controller);

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("ESP32 Sequence Controller Example");

    // 1. Init phase controller
    controller.enableSync(SYNC_PIN);
    
    // Note: Start at a nominal frequency, not 0! 
    // We achieve 0Hz DC by setting duty cycles to 100/0, not by stopping the frequency.
    controller.begin(5.0f); 

    // === COMPILE THE TAKEOFF QUEUE ===
    
    // Step 1: 0Hz DC Pole Alignment
    // Output static HIGH on Ch 0/2, LOW on Ch 1/3 (modify array pattern to fit your specific motor)
    seq.addDutyCycleTask(100.0, 0.0, 100.0, 0.0); 
    seq.addWaitTask(1000); // Hold DC for 1 second

    // Step 2: Restore normal running duty cycles
    seq.addDutyCycleTask(INITIAL_DUTY_CYCLES, NUM_CHANNELS); 

    // Step 3: Ease-In Ease-Out Ramp Profile
    // Ramps from 5Hz to 220Hz over 10000ms (10 seconds)
    seq.addEaseRampTask(5.0, 220.0, 10000); 

    // === COMPILE TRAJECTORY ===
    // This turns all the tasks above into an explicit, pre-calculated 
    // high-speed Look-Up Table (LUT) so the ESP32 hot loop does zero math.
    // resolution = 10ms step sizes
    Serial.println("Compiling trajectory...");
    float initDuty[NUM_CHANNELS] = {50.0, 50.0, 50.0, 50.0};
    seq.compile(10, 10.0f, initDuty, INITIAL_PHASES);

    // Kickoff the sequence
    Serial.println("Starting Takeoff Sequence...");
    seq.start();
}

void loop() {
    controller.run(); // Maintain hardware timer drift compensation
    seq.run(); // Evaluate state machine queue (Now running purely on the pre-compiled LUT)
}