#include <Arduino.h>
#include "pmw_config.h"
#include "pwm_controller.h"
#include "interrupts.h"

void setup() {
#ifdef DEBUG
    Serial.begin(115200);
    Serial.println(F("Debug Active"));
#endif

    pwmController.initialize();
    setupInterrupts();

#ifdef DEBUG
    Serial.println(F("System Ready"));
#endif
}

void loop() {
    pwmController.handleRequests();
    pwmController.handleDutyAdjustment();
    
    // Add your custom code here
}