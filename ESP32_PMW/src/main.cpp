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
const int PWM_PIN_21 = 21;
const int PWM_FREQ = 20000; // 20kHz
const int PWM_MAX_REQUESTED_RESOLUTION = 8;
const int DUTY_STEP_DIVISOR = 50; // ~2% step (1/50 of full scale)
int pwmResolutionBits = PWM_MAX_REQUESTED_RESOLUTION;
int pwmMaxDuty = (1 << PWM_MAX_REQUESTED_RESOLUTION) - 1;
int dutyCycle = pwmMaxDuty / 2; // Start at 50%

int clampResolutionForFrequency(int requestedResolution, uint32_t frequencyHz) {
    int maxSupportedResolution = requestedResolution;

    while (maxSupportedResolution > 1) {
        uint32_t maxFreqAtResolution = LEDC_APB_CLK_HZ / (1UL << maxSupportedResolution);
        if (frequencyHz <= maxFreqAtResolution) {
            break;
        }
        maxSupportedResolution--;
    }

    if (maxSupportedResolution < 1) {
        maxSupportedResolution = 1;
    }
    return maxSupportedResolution;
}

int dutyStep() {
    return max(1, pwmMaxDuty / DUTY_STEP_DIVISOR);
}

void applyDutyCycleDelta(int delta, const char* actionLabel) {
    dutyCycle = constrain(dutyCycle + delta, 0, pwmMaxDuty);
    ledcWrite(PWM_PIN_21, dutyCycle);
    Serial.printf(
        "Duty cycle %s: %d/%d (%.1f%%)\n",
        actionLabel,
        dutyCycle,
        pwmMaxDuty,
        (dutyCycle * 100.0f) / pwmMaxDuty
    );
}

void handleDutyCycleCommand(char cmd) {
    if (cmd == 'a' || cmd == 'A') {
        applyDutyCycleDelta(dutyStep(), "increased");
    } else if (cmd == 'd' || cmd == 'D') {
        applyDutyCycleDelta(-dutyStep(), "decreased");
    }
}

PhaseController controller(PWM_PINS, INITIAL_PHASES, INITIAL_DUTY_CYCLES, NUM_CHANNELS);
PhaseSequencer seq(&controller);

void setup() {
    Serial.begin(115200);
    delay(1000);

    controller.enableSync(SYNC_PIN);     
    controller.begin(1.0f);    
    
    // For high frequency PWM, reduce resolution if needed to keep clocking valid.
    pwmResolutionBits = clampResolutionForFrequency(PWM_MAX_REQUESTED_RESOLUTION, PWM_FREQ);
    pwmMaxDuty = (1 << pwmResolutionBits) - 1;
    dutyCycle = pwmMaxDuty / 2;

    bool pwmAttachOk = ledcAttach(PWM_PIN_21, PWM_FREQ, pwmResolutionBits);
    if (!pwmAttachOk) {
        Serial.printf("PWM attach failed on pin %d\n", PWM_PIN_21);
    }
    ledcWrite(PWM_PIN_21, dutyCycle);

    Serial.printf(
        "PWM pin=%d freq=%dHz resolution=%dbit maxDuty=%d\n",
        PWM_PIN_21,
        PWM_FREQ,
        pwmResolutionBits,
        pwmMaxDuty
    );
    
    // Water Takeoff Configuration
    // seq.addWaitTask(1000); // 5 second delay before starting
    // seq.addEaseRampTask(1.0f, 150.0f, 30000); // 1Hz to 250Hz over 60000ms (60s)
    // seq.addEaseRampTask(150.0f, 270.0f, 15000); // 1Hz to 250Hz over 60000ms (60s)
    
    // Dry Takeoff Configuration
    seq.addEaseRampTask(1.0f, 200.0f, 10000); // 1Hz to 250Hz over 20000ms (20s)
    // seq.compile(25, 1.0f, INITIAL_DUTY_CYCLES, INITIAL_PHASES); 
    seq.compile(25, 1.0f, INITIAL_DUTY_CYCLES, INITIAL_PHASES); 
    
    seq.start();                                                
}

void loop() {
    controller.run(); // hardware timer drift compensation
    seq.run(); // state machine queue
    
    // Handle serial input for PWM duty cycle control
    if (Serial.available() > 0) {
        handleDutyCycleCommand(Serial.read());
    }
}
