#include <Arduino.h>
#include "PhaseController.h"
#include "PhaseSequencer.h"

// === CONFIGURATION ===
const int NUM_CHANNELS = 4;
// const gpio_num_t PWM_PINS[NUM_CHANNELS] = {GPIO_NUM_15, GPIO_NUM_12, GPIO_NUM_27, GPIO_NUM_33}; 
// const float INITIAL_PHASES[NUM_CHANNELS] = {0.0, 90.0, 180.0, 270.0};
const gpio_num_t PWM_PINS[NUM_CHANNELS] = {GPIO_NUM_26, GPIO_NUM_33, GPIO_NUM_27, GPIO_NUM_32};
const float INITIAL_PHASES[NUM_CHANNELS] = {0.0, 180, 90.0, 270.0};

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
    // seq.addWaitTask(1000); // 5 second delay before starting
    // seq.addEaseRampTask(1.0f, 150.0f, 30000); // 1Hz to 250Hz over 60000ms (60s)
    // seq.addEaseRampTask(150.0f, 270.0f, 15000); // 1Hz to 250Hz over 60000ms (60s)
    

    // Dry Takeoff Configuration
    seq.addEaseRampTask(1.0f, 270.0f, 15000); // 1Hz to 250Hz over 20000ms (20s)
    // seq.compile(10, 1.0f, INITIAL_DUTY_CYCLES, INITIAL_PHASES); 
    seq.compile(10, 1.0f, INITIAL_DUTY_CYCLES, INITIAL_PHASES); 
    
    seq.start();                                                
}

void loop() {
    controller.run(); // hardware timer drift compensation
    seq.run(); // state machine queue
}