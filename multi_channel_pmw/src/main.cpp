#include <Arduino.h>
#include "MegaPhasePwm.h"

MegaPhasePwm pwmDriver;

void setup() {
    Serial.begin(115200);

    pwmDriver.begin(300.0); // 300Hz
    Serial.println("Debug: PWM initialized at 300Hz");

    // Test with higher duty cycles to make waves more visible
    pwmDriver.setDutyCycle(0, 0.5); // Channel 0 - 50% duty
	pwmDriver.setDutyCycle(5, 0.5); // Channel 5 - 50% duty
    pwmDriver.setDutyCycle(6, 0.5); // Channel 6 - 50% duty
    Serial.println("Debug: Duty cycles set");

	pwmDriver.setPhase(0, 0.0);
    pwmDriver.setPhase(5, 0.0 + 60);
	pwmDriver.setPhase(6, 0.0 + 180);
}

void loop() {
    // delay(100); // Slower to observe
}