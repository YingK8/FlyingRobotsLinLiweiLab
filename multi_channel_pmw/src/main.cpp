#include <Arduino.h>
#include "MegaPhasePwm.h"

MegaPhasePwm pwmDriver;

void setup() {
    Serial.begin(115200);
    while (!Serial); // Wait for serial connection
    Serial.println("Debug: Opposite Wave Test");
    
    pwmDriver.begin(300.0);
    Serial.println("Debug: PWM initialized at 300Hz");

    // Test with higher duty cycles to make waves more visible
    pwmDriver.setDutyCycle(0, 0.4); // Channel 0 - 50% duty
	pwmDriver.setDutyCycle(5, 0.5); // Channel 1 - 50% duty
    pwmDriver.setDutyCycle(6, 0.6); // Channel 2 - 50% duty
    Serial.println("Debug: Duty cycles set");

	pwmDriver.setPhase(0, 0.0);
    pwmDriver.setPhase(6, 0.0 + 60);
	pwmDriver.setPhase(5, 0.0 + 180);

    
    delay(1000); // Give time to see initial state
}

float phase = 0.0;

void loop() {
    delay(100); // Slower to observe
}