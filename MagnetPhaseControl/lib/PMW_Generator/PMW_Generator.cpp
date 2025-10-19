#include <Arduino.h>
#include "driver/mcpwm.h"
#include "PMW_Generator.h"

void setup_pwm() {
    // Configure MCPWM
    mcpwm_config_t pwm_config = {
        .frequency = PWM_FREQ,
        .cmpr_a = 50.0,  // 50% duty
        .cmpr_b = 50.0,  // 50% duty  
        .duty_mode = MCPWM_DUTY_MODE_0,
        .counter_mode = MCPWM_UP_COUNTER,
    };

    // Initialize
    mcpwm_gpio_init(MCPWM_UNIT_0, MCPWM0A, PWM1_GPIO);
    mcpwm_gpio_init(MCPWM_UNIT_0, MCPWM0B, PWM2_GPIO);
    printf("Dual PWM ready on GPIO %d and %d, Fault pin %d\n", PWM1_GPIO, PWM2_GPIO, FAULT_PIN);
}

bool handle_fault() {
    if (digitalRead(FAULT_PIN) == LOW) {
      mcpwm_set_duty(MCPWM_UNIT_0, MCPWM_TIMER_0, MCPWM_OPR_A, 0);
      mcpwm_set_duty(MCPWM_UNIT_0, MCPWM_TIMER_0, MCPWM_OPR_B, 0);
      Serial.println("ERROR: Fault detected! PWM outputs disabled.");
      return true;
    }
    return false;
}