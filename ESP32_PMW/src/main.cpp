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

void addBlockSpin(PhaseSequencer& seq, int revolutions, float startFreqHz, float endFreqHz);

void setup() {
    Serial.begin(115200);
    delay(1000);

    controller.enableSync(SYNC_PIN);     
    controller.begin(1.0f);      

    // Water Takeoff Configuration
    seq.addLinearRampTask(1.0f, 250.0f, 60000); // 1Hz to 250Hz over 60000ms (60s)
    seq.compile(10, 1.0f, INITIAL_DUTY_CYCLES, INITIAL_PHASES); 

    // Dry Takeoff Configuration
    // seq.addEaseRampTask(1.0f, 250.0, 20000); // 1Hz to 250Hz over 20000ms (20s)
    // seq.compile(10, 1.0f, INITIAL_DUTY_CYCLES, INITIAL_PHASES); 
    
    seq.start();                                                
}

void loop() {
    controller.run(); // hardware timer drift compensation
    seq.run(); // state machine queue
}

void addBlockSpin(PhaseSequencer& seq, int revolutions, float startFreqHz, float endFreqHz) {
    if (startFreqHz <= 0.0f || endFreqHz <= 0.0f || revolutions <= 0) return;
    
    const float vertical_on[4]   = {100.0, 0.0, 100.0, 0.0};
    const float horizontal_on[4] = {0.0, 100.0, 0.0, 100.0};
    const float phase_up[4]      = {0.0, 0.0, 180.0, 180.0}; 
    const float phase_right[4]   = {180.0, 0.0, 0.0, 180.0}; 
    const float phase_down[4]    = {180.0, 180.0, 0.0, 0.0}; 
    const float phase_left[4]    = {0.0, 180.0, 180.0, 0.0}; 

    int totalSteps = revolutions * 4;

    for (int step = 0; step < totalSteps; step++) {
        // Interpolate the frequency for the current step (linear acceleration)
        float t = totalSteps > 1 ? (float)step / (totalSteps - 1) : 1.0f;
        float currentFreq = startFreqHz + t * (endFreqHz - startFreqHz);
        
        // Calculate precise wait time for this specific 90-degree step
        uint32_t stepTimeMs = (uint32_t)(1000.0f / (currentFreq * 4.0f));

        int phaseState = step % 4;

        if (phaseState == 0) {
            seq.addDutyCycleTask(vertical_on, 4);
            seq.addPhaseTask(phase_up, 4); 
        } else if (phaseState == 1) {
            seq.addDutyCycleTask(horizontal_on, 4);
            seq.addPhaseTask(phase_right, 4);
        } else if (phaseState == 2) {
            seq.addDutyCycleTask(vertical_on, 4);
            seq.addPhaseTask(phase_down, 4);
        } else if (phaseState == 3) {
            seq.addDutyCycleTask(horizontal_on, 4);
            seq.addPhaseTask(phase_left, 4);
        }
        seq.addWaitTask(stepTimeMs);
    }
}