#include <Arduino.h>
#include <PMW_Generator.h>

void setup() {
    Serial.begin(115200);
    setup_pwm(50.0, PWM_FREQ);
    pinMode(FAULT_PIN, INPUT_PULLDOWN); // Set FAULT pin as input
    // initialise PMWs
    mcpwm_set_duty(MCPWM_UNIT_0, MCPWM_TIMER_0, MCPWM_OPR_A, 50);
    mcpwm_set_duty(MCPWM_UNIT_0, MCPWM_TIMER_0, MCPWM_OPR_B, 50);
    set_phase_degrees(90.0, 80000000UL / PWM_FREQ);
}

void loop() {
  if (handle_fault()) {
    while (1);
  }
  delay(10);
}