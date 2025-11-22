#include <Arduino.h>
#include "ESP32_PWM.h"

// Define an array of pins for the 8 channels
int pwmPins[] = {2, 4, 5, 12, 13, 14, 15, 16}; // Adjust to your GPIO pins

const float frequency = 300.0;
const float duty = 0.5;

// Create a PWM controller instance
ESP32_PWM pwm;

void setup() {
  Serial.begin(115200);
  
  // Initialize all pins as outputs
  for (int i = 0; i < 8; i++) {
    int pmwPin = pwmPins[i];
    pinMode(pmwPin, OUTPUT);
    pwm.setPWM(pmwPin, frequency);
  }
  
  // Set up 8 channels with progressive phase shifts
  for (int channel = 0; channel < 8; channel++) {
    // Calculate phase delay in microseconds
    float phaseShiftDegrees = channel * 45.0;
    int phaseDelayUs = (int)((1000000.0 / 400) * (phaseShiftDegrees / 360.0));
    
    // Attach the pin to a PWM channel with phase shift
    pwm.attachPin(pwmPins[channel], channel, phaseDelayUs);
    
    // Set duty cycle to 50%
    pwm.setDutyCycle(channel, 128); // 128/256 = 50%
    
    Serial.print("Channel ");
    Serial.print(channel);
    Serial.print(" on Pin ");
    Serial.print(pwmPins[channel]);
    Serial.print(": Phase = ");
    Serial.print(phaseShiftDegrees);
    Serial.println(" degrees");
  }
}

void loop() {
  // Your main code can adjust parameters here
  // Example: pwm.setDutyCycle(channel, newDuty);
  delay(100);
}