#include <Arduino.h>
#include "PhaseController.h"
#include "PhaseSequencer.h"

#ifndef LEDC_APB_CLK_HZ
#define LEDC_APB_CLK_HZ 80000000UL
#endif

// === CONFIGURATION ===
const int NUM_CHANNELS = 4;
// const gpio_num_t PWM_PINS[NUM_CHANNELS] = {GPIO_NUM_15, GPIO_NUM_12, GPIO_NUM_27, GPIO_NUM_33}; 
// const float INITIAL_PHASES[NUM_CHANNELS] = {0.0, 90.0, 180.0, 270.0};
const gpio_num_t PWM_PINS[NUM_CHANNELS] = {GPIO_NUM_19, GPIO_NUM_33, GPIO_NUM_27, GPIO_NUM_32};
const float INITIAL_PHASES[NUM_CHANNELS] = {0.0, 90.0, 180.0, 270.0};
const float INITIAL_DUTY_CYCLES[NUM_CHANNELS] = {50.0, 50.0, 50.0, 50.0};
const gpio_num_t SYNC_PIN = GPIO_NUM_4;

// === PWM PIN 21 CONFIGURATION ===
const gpio_num_t PWM_PIN = GPIO_NUM_21;
const int PWM_FREQ = 50000; // 20kHz
float pwm = 50.0;

PhaseController controller(PWM_PINS, INITIAL_PHASES, INITIAL_DUTY_CYCLES, NUM_CHANNELS);
PhaseSequencer seq(&controller);

void setup() {
    Serial.begin(115200);
    delay(1000);

    controller.enableSync(SYNC_PIN);     
    controller.begin(1.0f);    
    controller.initCarrierPWM(PWM_PIN, PWM_FREQ, pwm);
    
    // Water Takeoff Configuration¨
    // seq.addWaitTask(1000); // 5 second delay before starting
    // seq.addEaseRampTask(1.0f, 150.0f, 30000); // 1Hz to 250Hz over 60000ms (60s)
    // seq.addEaseRampTask(150.0f, 270.0f, 15000); // 1Hz to 250Hz over 60000ms (60s)
    
    // Dry Takeoff Configuration
    seq.addEaseRampTask(1.0f, 190.0f, 15000); // 1Hz to 250Hz over 20000ms (20s)
    // seq.compile(25, 1.0f, INITIAL_DUTY_CYCLES, INITIAL_PHASES); 
    seq.compile(25, 1.0f, INITIAL_DUTY_CYCLES, INITIAL_PHASES); 
    
    seq.start();                                                
}

void loop() {
    controller.run(); // hardware timer drift compensation
    seq.run(); // state machine queue

    // Triangle ramp: 10s up + 10s down = 20s full cycle.
    static unsigned long lastDutyUpdateMs = 0;
    const unsigned long nowMs = millis();
    if (nowMs - lastDutyUpdateMs >= 20) {
        lastDutyUpdateMs = nowMs;

        const float dutyMin = 0.0f;
        const float dutyMax = 100.0f;
        const unsigned long cycleMs = 20000UL;
        const unsigned long halfCycleMs = cycleMs / 2UL;
        const unsigned long t = nowMs % cycleMs;

        if (t < halfCycleMs) {
            const float progress = (float)t / (float)halfCycleMs;
            pwm = dutyMin + (dutyMax - dutyMin) * progress;
        } else {
            const float progress = (float)(t - halfCycleMs) / (float)halfCycleMs;
            pwm = dutyMax - (dutyMax - dutyMin) * progress;
        }

        controller.setCarrierDutyCycle(pwm);
    }
}
